"""
energy_integrating_detector.py
==============================
GPU-Accelerated Energy-Integrating Detector (EID) Module.

Models the physical detection chain of a multi-row CT detector:
  1. Scintillator X-ray Absorption & Quantum Detection Efficiency (QDE) eta(E)
  2. Polychromatic Energy Integration: Deposited energy to electron conversion
  3. Optical Scintillator Spread: 2D Point Spread Function (PSF) optical blur
  4. Photo-Diode PRNU (Gain Non-Uniformity) & Defective Pixels
  5. Quantum Poisson Noise & Electronic Readout Noise (sigma = 500 e-)
  6. Calibration flat-field air reference computation (I_0)

Supports batched GPU arrays (B, H, W) via CuPy for real-time 16,728-view throughput.
"""

from __future__ import annotations

import json
import pathlib
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

# Primary GPU acceleration engine
try:
    import cupy as cp
    _ = cp.cuda.runtime.getDeviceProperties(0)
    from cupyx.scipy.ndimage import gaussian_filter as cupy_gaussian_filter
    HAS_CUPY = True
except Exception:
    import numpy as cp
    from scipy.ndimage import gaussian_filter as cupy_gaussian_filter
    HAS_CUPY = False

from scipy.ndimage import gaussian_filter as scipy_gaussian_filter


class DetectorMaterial:
    """Represents detector scintillator material and attenuation properties."""

    def __init__(self, name: str, density_g_cm3: float, mu_rho_table_cm2_g: Dict[float, float]):
        self.name = name
        self.density_g_cm3 = float(density_g_cm3)
        self.mu_rho_table_cm2_g = {float(k): float(v) for k, v in mu_rho_table_cm2_g.items()}
        self._energies = np.array(list(self.mu_rho_table_cm2_g.keys()), dtype=float)
        self._mu_rho = np.array(list(self.mu_rho_table_cm2_g.values()), dtype=float)

    def mass_attenuation_coefficient(self, energy_keV: float) -> float:
        return float(np.interp(energy_keV, self._energies, self._mu_rho))

    def linear_attenuation_coefficient(self, energy_keV: float) -> float:
        """Returns linear attenuation coefficient in cm^-1."""
        return self.mass_attenuation_coefficient(energy_keV) * self.density_g_cm3


MATERIALS: Dict[str, DetectorMaterial] = {
    "CsI": DetectorMaterial(
        "Cesium Iodide", 4.51,
        {20: 5.4, 30: 2.2, 40: 1.6, 50: 1.1, 60: 0.65, 80: 0.36, 100: 0.26, 120: 0.20, 140: 0.17}
    ),
    "GOS": DetectorMaterial(
        "Gadolinium Oxysulfide", 7.34,
        {20: 8.5, 30: 3.4, 40: 2.5, 50: 1.6, 60: 1.0, 80: 0.55, 100: 0.40, 120: 0.31, 140: 0.26}
    ),
    "aSe": DetectorMaterial(
        "Amorphous Selenium", 4.28,
        {20: 3.1, 30: 1.2, 40: 0.9, 50: 0.55, 60: 0.38, 80: 0.24, 100: 0.18, 120: 0.14, 140: 0.12}
    ),
    "CdTe": DetectorMaterial(
        "Cadmium Telluride", 5.85,
        {20: 13.0, 30: 5.5, 40: 4.0, 50: 2.5, 60: 1.7, 80: 0.95, 100: 0.68, 120: 0.53, 140: 0.44}
    ),
}

DEFAULT_DETECTOR_CONFIG: Dict[str, Any] = {
    "detector_rows": 64,
    "detector_channels": 888,
    "detector_element_width_mm": 0.625,
    "detector_element_height_mm": 0.625,
    "detector_width_mm": 555.0,
    "detector_height_mm": 40.0,
    "detector_type": "energy_integrating",
    "detector_material": "scintillator_photodiode",
    "collimation_mm": 40.0,
}

MATERIAL_NAME_MAP = {
    "scintillator_photodiode": "CsI",
}


class EnergyIntegratingDetector:
    """
    GPU-accelerated CT Energy-Integrating Detector (EID) with CsI scintillator physics.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        thickness_cm: float = 0.06,  # 0.6 mm CsI scintillator
        fill_factor: float = 0.80,
        conversion_gain_e_per_keV: float = 50.0,
        electronic_noise_e: float = 500.0,
        psf_sigma_px: float = 0.8,
        gain_nonuniformity_fraction: float = 0.02,
        dead_pixel_fraction: float = 0.001,
        random_seed: int = 42,
        use_cupy: bool = True,
    ):
        self.config = config or DEFAULT_DETECTOR_CONFIG
        self.rows = int(self.config.get("detector_rows", 64))
        self.channels = int(self.config.get("detector_channels", 888))
        self.element_width_mm = float(self.config.get("detector_element_width_mm", 0.625))
        self.element_height_mm = float(self.config.get("detector_element_height_mm", 0.625))
        self.array_width_mm = float(self.config.get("detector_width_mm", 555.0))
        self.array_height_mm = float(self.config.get("detector_height_mm", 40.0))

        raw_mat = self.config.get("detector_material", "CsI")
        mat_key = MATERIAL_NAME_MAP.get(raw_mat, raw_mat)
        self.material = MATERIALS.get(mat_key, MATERIALS["CsI"])

        self.thickness_cm = float(thickness_cm)
        self.fill_factor = float(fill_factor)
        self.conversion_gain = float(conversion_gain_e_per_keV)
        self.electronic_noise_e = float(electronic_noise_e)
        self.psf_sigma_px = float(psf_sigma_px)
        self.gain_nonuniformity_fraction = float(gain_nonuniformity_fraction)
        self.dead_pixel_fraction = float(dead_pixel_fraction)

        self.use_cupy = use_cupy and HAS_CUPY
        self.xp = cp if self.use_cupy else np

        # Fixed detector calibration maps (PRNU gain map and dead pixel mask)
        rng_host = np.random.default_rng(random_seed)
        gain_np = rng_host.normal(1.0, self.gain_nonuniformity_fraction, size=(self.rows, self.channels)).astype(np.float32)
        dead_np = (rng_host.random((self.rows, self.channels)) > self.dead_pixel_fraction).astype(np.float32)

        if self.use_cupy:
            self.gain_map = cp.asarray(gain_np)
            self.dead_pixel_map = cp.asarray(dead_np)
        else:
            self.gain_map = gain_np
            self.dead_pixel_map = dead_np

    @classmethod
    def from_config_file(
        cls,
        json_path: Union[str, pathlib.Path] = "ct_system_output.json",
        use_cupy: bool = True,
    ) -> "EnergyIntegratingDetector":
        """Initializes detector from system JSON configuration."""
        path = pathlib.Path(json_path)
        if not path.is_file():
            alt = path.parent / "system_config" / path.name
            if alt.is_file():
                path = alt

        cfg = DEFAULT_DETECTOR_CONFIG.copy()
        if path.is_file():
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if "detector_config" in data:
                cfg.update(data["detector_config"])
            elif "3_detector_team" in data:
                cfg.update(data["3_detector_team"])
        return cls(config=cfg, use_cupy=use_cupy)

    def quantum_efficiency(self, energy_keV: float) -> float:
        """
        Computes quantum detection efficiency eta(E):
            eta(E) = (1.0 - exp(-mu(E) * thickness)) * fill_factor
        """
        mu = self.material.linear_attenuation_coefficient(energy_keV)  # cm^-1
        absorption_fraction = 1.0 - np.exp(-mu * self.thickness_cm)
        return float(absorption_fraction * self.fill_factor)

    def get_quantum_efficiency_table(self, energy_bins_keV: Any) -> Any:
        """Computes QDE eta(E) array for all energy bins on GPU or CPU."""
        xp = self.xp
        if isinstance(energy_bins_keV, cp.ndarray) if HAS_CUPY else False:
            e_np = cp.asnumpy(energy_bins_keV)
        else:
            e_np = np.asarray(energy_bins_keV, dtype=float)

        eta_np = np.array([self.quantum_efficiency(float(e)) for e in e_np], dtype=np.float32)
        return xp.asarray(eta_np) if self.use_cupy else eta_np

    def apply_spread(self, image: Any) -> Any:
        """Applies 2D Gaussian optical spread (PSF) on GPU or CPU."""
        if self.psf_sigma_px <= 0:
            return image

        if image.ndim == 2:
            # Single view (H, W)
            if self.use_cupy and HAS_CUPY:
                return cupy_gaussian_filter(image, sigma=self.psf_sigma_px)
            return scipy_gaussian_filter(image, sigma=self.psf_sigma_px)
        elif image.ndim == 3:
            # Batch of views (B, H, W)
            if self.use_cupy and HAS_CUPY:
                return cupy_gaussian_filter(image, sigma=(0, self.psf_sigma_px, self.psf_sigma_px))
            return scipy_gaussian_filter(image, sigma=(0, self.psf_sigma_px, self.psf_sigma_px))
        else:
            raise ValueError(f"Unsupported array shape for optical spread: {image.shape}")

    def detect(
        self,
        photon_fluence_per_bin: Any,
        energy_bins_keV: Any,
        add_noise: bool = True,
    ) -> Dict[str, Any]:
        """
        Processes transmitted spectral photon fluence into measured detector electronic signal.

        Arguments:
            photon_fluence_per_bin: (B, H, W, N_bins) or (H, W, N_bins) transmitted fluence on GPU
            energy_bins_keV:        (N_bins,) energy bin centers in keV
            add_noise:              Whether to inject quantum Poisson & electronic noise
        Returns:
            Dictionary containing:
                - measured_signal_electrons: (B, H, W) or (H, W) realistic measured signal
                - ideal_signal_electrons:    (B, H, W) or (H, W) noise-free blurred signal
                - quantum_noise_std:         Standard deviation of quantum noise
                - gain_map:                  PRNU map
                - dead_pixel_map:            Defective pixel mask
        """
        xp = self.xp
        is_batched = photon_fluence_per_bin.ndim == 4
        if not is_batched:
            # Shape (H, W, N_bins) -> expand to (1, H, W, N_bins)
            fluence_4d = photon_fluence_per_bin[None, ...]
        else:
            fluence_4d = photon_fluence_per_bin

        B, H, W, N_bins = fluence_4d.shape

        # 1. Quantum efficiency and energy weighting tensors
        eta = self.get_quantum_efficiency_table(energy_bins_keV)  # (N_bins,)
        if not isinstance(energy_bins_keV, xp.ndarray):
            e_tensor = xp.asarray(energy_bins_keV, dtype=xp.float32)
        else:
            e_tensor = energy_bins_keV.astype(xp.float32)

        # Weights mapping photons per bin to signal electrons and noise variance:
        # S(E) = Phi(E) * eta(E) * E * G_conv
        # Var(E) = Phi(E) * eta(E) * E^2 * G_conv^2
        w_signal = (eta * e_tensor * self.conversion_gain).reshape(1, 1, 1, N_bins)
        w_var    = (eta * (e_tensor**2) * (self.conversion_gain**2)).reshape(1, 1, 1, N_bins)

        # 2. Integrate across energy bins: (B, H, W)
        signal_electrons = xp.sum(fluence_4d * w_signal, axis=-1)
        quantum_var = xp.sum(fluence_4d * w_var, axis=-1)
        quantum_std = xp.sqrt(xp.maximum(quantum_var, 1e-12))

        # 3. Apply scintillator optical spread (PSF)
        signal_blurred = self.apply_spread(signal_electrons)
        quantum_std_blurred = self.apply_spread(quantum_std)

        # 4. Apply PRNU gain variation and dead pixel defects
        gain = self.gain_map.reshape(1, H, W)
        dead = self.dead_pixel_map.reshape(1, H, W)

        ideal_signal = signal_blurred * gain * dead

        if add_noise:
            # Gaussian electronic readout noise
            elec_noise = xp.random.normal(0.0, self.electronic_noise_e, size=ideal_signal.shape).astype(xp.float32)
            # Poisson quantum noise realization
            quant_noise = xp.random.normal(0.0, 1.0, size=ideal_signal.shape).astype(xp.float32) * quantum_std_blurred

            measured_signal = (ideal_signal + elec_noise + quant_noise) * dead
        else:
            measured_signal = ideal_signal

        # Non-negative physical constraint
        measured_signal = xp.maximum(measured_signal, 0.0)

        if not is_batched:
            measured_signal = measured_signal[0]
            ideal_signal = ideal_signal[0]
            quantum_std_blurred = quantum_std_blurred[0]

        return {
            "measured_signal_electrons": measured_signal.astype(xp.float32),
            "ideal_signal_electrons": ideal_signal.astype(xp.float32),
            "quantum_noise_std_electrons": quantum_std_blurred.astype(xp.float32),
            "gain_map": self.gain_map,
            "dead_pixel_map": self.dead_pixel_map,
        }

    def compute_air_reference_signal(
        self,
        incident_fluence_1d: Any,
        energy_bins_keV: Any,
    ) -> Any:
        """
        Precomputes unattenuated flat-field air reference projection (I_0) for sinogram synthesis.
        Returns 2D array of shape (H, W) on GPU.
        """
        xp = self.xp
        H, W = self.rows, self.channels
        N_bins = len(energy_bins_keV)

        # Broadcast nominal unattenuated fluence: (1, H, W, N_bins)
        inc_4d = xp.broadcast_to(incident_fluence_1d.reshape(1, 1, 1, N_bins), (1, H, W, N_bins))
        res = self.detect(inc_4d, energy_bins_keV, add_noise=False)
        return res["ideal_signal_electrons"][0]

    def mtf(self, spatial_frequency_lp_per_mm: np.ndarray) -> np.ndarray:
        """Computes theoretical presampling Modulation Transfer Function (MTF)."""
        sigma_mm = self.psf_sigma_px * self.element_width_mm
        return np.exp(-2.0 * (np.pi ** 2) * (sigma_mm ** 2) * (spatial_frequency_lp_per_mm ** 2))


if __name__ == "__main__":
    det = EnergyIntegratingDetector.from_config_file("ct_system_output.json", use_cupy=True)
    print(f"EnergyIntegratingDetector initialized: {det.rows} rows x {det.channels} channels")
    print(f"Material: {det.material.name} (Density: {det.material.density_g_cm3} g/cm3, Thickness: {det.thickness_cm*10:.2f} mm)")
    print(f"Fill factor: {det.fill_factor:.2f} | Conversion Gain: {det.conversion_gain} e-/keV | Elec Noise: {det.electronic_noise_e} e-")

    xp = cp if HAS_CUPY else np
    e_test = xp.arange(10.0, 121.0, 1.0, dtype=xp.float32)
    qde_60 = det.quantum_efficiency(60.0)
    print(f"QDE at 60 keV: {qde_60 * 100:.2f}%")

    # Test single-view detection
    fake_fluence = xp.ones((det.rows, det.channels, len(e_test)), dtype=xp.float32) * 200.0
    out = det.detect(fake_fluence, e_test, add_noise=True)
    sig = out["measured_signal_electrons"]
    sig_mean = float(cp.asnumpy(sig).mean()) if HAS_CUPY else float(sig.mean())
    print(f"Output signal shape: {sig.shape} (Type: {type(sig)}) | Mean Signal: {sig_mean:.1f} e-")
