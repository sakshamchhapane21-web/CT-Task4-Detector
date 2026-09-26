"""
ct_detector_simulator.py
=========================

Physically-grounded simulation of a multi-row, energy-integrating CT detector
array (scintillator + photodiode), matching the following kind of geometry
specification:

    {
      "detector_rows": 64,
      "detector_channels": 888,
      "detector_element_width_mm": 0.625,
      "detector_element_height_mm": 0.625,
      "detector_width_mm": 555.0,
      "detector_height_mm": 40.0,
      "detector_type": "energy_integrating",
      "detector_material": "scintillator_photodiode",
      "collimation_mm": 40.0
    }

What is modeled
----------------
1. Geometry        : rows, channels, pixel pitch, active area, collimation.
2. Material physics : Beer-Lambert X-ray absorption efficiency (quantum
                       efficiency, QE), light yield, Swank (secondary quantum
                       sink) factor, afterglow/lag.
3. DQE(0, q)        : Detective Quantum Efficiency derived from a cascaded
                       linear-systems noise model (quantum + Swank noise +
                       additive electronic noise), as a function of exposure.
4. MTF(f) / spread  : pixel-aperture MTF combined with an optical/charge
                       spread (blur) MTF -> point-spread function used to
                       convolve (cross-talk) neighbouring pixels.
5. Electronic noise : additive Gaussian read noise, given as an RMS electron
                       count (or ADU) standard deviation.
6. Imperfections    : dead/stuck pixel map, pixel-to-pixel gain
                       non-uniformity (fixed-pattern noise), fixed offset
                       (dark) map -- all reproducible via a seed so a given
                       "detector unit" always has the same defect signature.

Physical model notes
---------------------
This is a *representative*, tunable engineering model, not a vendor-exact
detector model (real scintillator thickness, dopant, and electronics design
are proprietary). Material constants in MATERIAL_DB are approximate,
literature-typical values for common CT scintillators and are meant to be
overridden with measured values when available.

DQE(0) derivation (cascaded systems analysis)
----------------------------------------------
Let:
    q        = mean number of incident X-ray quanta per detector pixel
    eta      = quantum efficiency (probability of absorption)
    A_swank  = Swank factor (0 < A_swank <= 1): excess-noise factor caused by
               the variability in the amount of light produced per absorbed
               X-ray (the "secondary quantum sink").
    g        = conversion gain, mean electrons produced per absorbed photon
    sigma_e  = electronic (additive) noise, RMS electrons per pixel

Mean output signal:            S = eta * q * g
Quantum+Swank noise variance:  Var_q = eta * q * g^2 / A_swank
Electronic noise variance:     Var_e = sigma_e^2

    SNR_out^2 = S^2 / (Var_q + Var_e)
    SNR_in^2  = q                                  (ideal photon counter)

    DQE(0; q) = SNR_out^2 / SNR_in^2
              = eta^2 * A_swank * g^2 * q / (eta * g^2 * q + A_swank * sigma_e^2)

As q -> infinity:  DQE(0) -> eta * A_swank   (quantum-noise-limited ceiling)
As q -> 0:         DQE(0) -> 0               (electronic-noise-limited)

Frequency dependence is applied as the common simplifying approximation
    DQE(f; q) = DQE(0; q) * MTF(f)^2
which is exact when electronic noise is white and the quantum-noise NPS is
assumed to scale like MTF(f)^2 as well (a standard first-order treatment).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.ndimage import gaussian_filter1d, convolve1d


# ---------------------------------------------------------------------------
# 1. Material database
# ---------------------------------------------------------------------------
# mu_per_cm       : linear attenuation coefficient at an assumed CT effective
#                   energy (~65-70 keV), 1/cm
# density_g_cm3   : material density
# light_yield_photons_per_kev : optical photons produced per keV absorbed
# swank_factor    : secondary quantum sink excess-noise factor (<=1)
# afterglow_pct   : residual signal (%) persisting one frame after exposure
# default_thickness_mm : typical scintillator thickness used in CT detectors
MATERIAL_DB = {
    "GOS_photodiode": {  # Gadolinium oxysulfide (Gd2O2S:Pr), the most common
        "display_name": "Gadolinium Oxysulfide (GOS/GADOX) + Photodiode",
        "mu_per_cm": 14.0,
        "density_g_cm3": 7.34,
        "light_yield_photons_per_kev": 60.0,
        "swank_factor": 0.97,
        "afterglow_pct": 0.10,
        "default_thickness_mm": 1.4,
        "photodiode_qe": 0.80,
        "optical_coupling_efficiency": 0.55,
    },
    "CsI_photodiode": {  # Cesium Iodide, thallium doped
        "display_name": "CsI:Tl + Photodiode",
        "mu_per_cm": 10.0,
        "density_g_cm3": 4.51,
        "light_yield_photons_per_kev": 54.0,
        "swank_factor": 0.94,
        "afterglow_pct": 0.30,
        "default_thickness_mm": 1.8,
        "photodiode_qe": 0.75,
        "optical_coupling_efficiency": 0.55,
    },
    "CdWO4_photodiode": {  # Cadmium Tungstate
        "display_name": "CdWO4 (CWO) + Photodiode",
        "mu_per_cm": 11.5,
        "density_g_cm3": 7.90,
        "light_yield_photons_per_kev": 15.0,
        "swank_factor": 0.98,
        "afterglow_pct": 0.02,
        "default_thickness_mm": 1.5,
        "photodiode_qe": 0.80,
        "optical_coupling_efficiency": 0.55,
    },
    # Generic alias matching the reference config's "scintillator_photodiode"
    # label -- defaults to GOS-like behaviour (the industry-standard choice
    # for multi-row energy-integrating CT detectors).
    "scintillator_photodiode": {
        "display_name": "Generic Scintillator + Photodiode (GOS-like default)",
        "mu_per_cm": 14.0,
        "density_g_cm3": 7.34,
        "light_yield_photons_per_kev": 60.0,
        "swank_factor": 0.97,
        "afterglow_pct": 0.10,
        "default_thickness_mm": 1.4,
        "photodiode_qe": 0.80,
        "optical_coupling_efficiency": 0.55,
    },
}


# ---------------------------------------------------------------------------
# 2. Configuration dataclasses
# ---------------------------------------------------------------------------
@dataclass
class DetectorGeometry:
    rows: int
    channels: int
    element_width_mm: float
    element_height_mm: float
    width_mm: float
    height_mm: float
    collimation_mm: float
    detector_type: str
    material_key: str

    @classmethod
    def from_config(cls, cfg: dict) -> "DetectorGeometry":
        return cls(
            rows=cfg["detector_rows"],
            channels=cfg["detector_channels"],
            element_width_mm=cfg["detector_element_width_mm"],
            element_height_mm=cfg["detector_element_height_mm"],
            width_mm=cfg["detector_width_mm"],
            height_mm=cfg["detector_height_mm"],
            collimation_mm=cfg["collimation_mm"],
            detector_type=cfg.get("detector_type", "energy_integrating"),
            material_key=cfg.get("detector_material", "scintillator_photodiode"),
        )


@dataclass
class MaterialModel:
    key: str
    thickness_mm: Optional[float] = None      # override default thickness
    effective_energy_kev: float = 65.0        # assumed CT effective energy

    def __post_init__(self):
        if self.key not in MATERIAL_DB:
            raise ValueError(
                f"Unknown material '{self.key}'. Available: {list(MATERIAL_DB)}"
            )
        props = MATERIAL_DB[self.key]
        self.display_name = props["display_name"]
        self.mu_per_cm = props["mu_per_cm"]
        self.density_g_cm3 = props["density_g_cm3"]
        self.light_yield_photons_per_kev = props["light_yield_photons_per_kev"]
        self.swank_factor = props["swank_factor"]
        self.afterglow_pct = props["afterglow_pct"]
        self.photodiode_qe = props["photodiode_qe"]
        self.optical_coupling_efficiency = props["optical_coupling_efficiency"]
        if self.thickness_mm is None:
            self.thickness_mm = props["default_thickness_mm"]

    @property
    def quantum_efficiency(self) -> float:
        """Beer-Lambert X-ray absorption efficiency (fraction of photons
        that interact in the scintillator)."""
        thickness_cm = self.thickness_mm / 10.0
        return 1.0 - np.exp(-self.mu_per_cm * thickness_cm)

    @property
    def conversion_gain_e_per_photon(self) -> float:
        """Mean electrons generated per ABSORBED X-ray photon, combining
        scintillator light yield, optical coupling efficiency and photodiode
        quantum efficiency."""
        optical_photons = self.light_yield_photons_per_kev * self.effective_energy_kev
        return (
            optical_photons
            * self.optical_coupling_efficiency
            * self.photodiode_qe
        )


@dataclass
class ElectronicNoiseModel:
    electronic_noise_std_e: float = 35.0   # RMS electrons, read/electronic noise
    # CT integrating front-ends use large charge-integration capacitors to
    # cope with the very high, wide-dynamic-range photon fluence seen across
    # a scan (open-beam vs. deep-attenuation paths). ~2.5e8 e- (~40 pC) is a
    # representative order of magnitude; tune to match a real ASIC's
    # full-scale charge if known.
    full_well_capacity_e: float = 2.5e8    # electrons, saturation charge
    adc_bit_depth: int = 16                # ADC resolution


@dataclass
class ImperfectionModel:
    dead_pixel_fraction: float = 0.002     # fraction of pixels permanently dead
    gain_nonuniformity_std: float = 0.01   # fractional pixel-to-pixel gain sigma
    offset_std_e: float = 15.0             # fixed-pattern dark-offset sigma, electrons
    blur_sigma_mm: float = 0.35            # optical/charge spread sigma, mm


# ---------------------------------------------------------------------------
# 3. Main simulator
# ---------------------------------------------------------------------------
class CTDetectorSimulator:
    def __init__(
        self,
        config: dict,
        material: Optional[MaterialModel] = None,
        electronics: Optional[ElectronicNoiseModel] = None,
        imperfections: Optional[ImperfectionModel] = None,
        seed: Optional[int] = 42,
    ):
        self.geometry = DetectorGeometry.from_config(config)
        self.material = material or MaterialModel(self.geometry.material_key)
        self.electronics = electronics or ElectronicNoiseModel()
        self.imperfections = imperfections or ImperfectionModel()
        self.rng = np.random.default_rng(seed)

        self._build_defect_maps()

    # -- geometry / defect maps -------------------------------------------
    def _build_defect_maps(self):
        rows, cols = self.geometry.rows, self.geometry.channels
        n_pixels = rows * cols

        # Dead / stuck pixel mask (True = dead)
        n_dead = int(round(self.imperfections.dead_pixel_fraction * n_pixels))
        dead_flat = np.zeros(n_pixels, dtype=bool)
        dead_idx = self.rng.choice(n_pixels, size=n_dead, replace=False)
        dead_flat[dead_idx] = True
        self.dead_pixel_map = dead_flat.reshape(rows, cols)

        # Pixel-to-pixel gain non-uniformity (multiplicative, mean 1.0)
        self.gain_map = 1.0 + self.rng.normal(
            0.0, self.imperfections.gain_nonuniformity_std, size=(rows, cols)
        )
        self.gain_map = np.clip(self.gain_map, 0.5, 1.5)

        # Fixed-pattern dark offset (additive, electrons)
        self.offset_map = self.rng.normal(
            0.0, self.imperfections.offset_std_e, size=(rows, cols)
        )

    # -- physics summaries ---------------------------------------------------
    @property
    def quantum_efficiency(self) -> float:
        return self.material.quantum_efficiency

    def dqe0(self, mean_incident_quanta: float) -> float:
        """DQE at zero spatial frequency, for a given mean incident photon
        fluence per pixel (see module docstring for derivation)."""
        eta = self.quantum_efficiency
        A = self.material.swank_factor
        g = self.material.conversion_gain_e_per_photon
        sigma_e = self.electronics.electronic_noise_std_e
        q = mean_incident_quanta

        numerator = (eta ** 2) * A * (g ** 2) * q
        denominator = eta * (g ** 2) * q + A * (sigma_e ** 2)
        if denominator <= 0:
            return 0.0
        return float(numerator / denominator)

    def mtf(self, frequencies_cy_per_mm: np.ndarray) -> np.ndarray:
        """Modulation Transfer Function: pixel-aperture sinc term combined
        with a Gaussian optical/charge-spread term."""
        f = np.asarray(frequencies_cy_per_mm, dtype=float)
        a = self.geometry.element_width_mm
        aperture_mtf = np.abs(np.sinc(f * a))  # numpy sinc = sin(pi x)/(pi x)
        sigma = self.imperfections.blur_sigma_mm
        blur_mtf = np.exp(-2.0 * (np.pi * sigma * f) ** 2)
        return aperture_mtf * blur_mtf

    def dqe(self, frequencies_cy_per_mm: np.ndarray, mean_incident_quanta: float) -> np.ndarray:
        """DQE(f) = DQE(0) * MTF(f)^2 (standard simplifying approximation,
        see module docstring)."""
        return self.dqe0(mean_incident_quanta) * self.mtf(frequencies_cy_per_mm) ** 2

    def nyquist_frequency(self) -> float:
        return 1.0 / (2.0 * self.geometry.element_width_mm)

    # -- image formation pipeline --------------------------------------------
    def _apply_blur(self, image: np.ndarray) -> np.ndarray:
        """Apply the optical/charge spread as a separable Gaussian blur in
        pixel units (cross-talk between neighbouring detector elements)."""
        sigma_px_x = self.imperfections.blur_sigma_mm / self.geometry.element_width_mm
        sigma_px_y = self.imperfections.blur_sigma_mm / self.geometry.element_height_mm
        blurred = gaussian_filter1d(image, sigma=sigma_px_y, axis=0, mode="nearest")
        blurred = gaussian_filter1d(blurred, sigma=sigma_px_x, axis=1, mode="nearest")
        return blurred

    def simulate_projection(
        self,
        incident_photon_image: np.ndarray,
        apply_blur: bool = True,
        apply_electronic_noise: bool = True,
        apply_imperfections: bool = True,
        digitize: bool = True,
    ) -> dict:
        """
        Run the full detector physics pipeline on an incident-photon-fluence
        image (photons per pixel, shape = (rows, channels)).

        Returns a dict with the intermediate and final images so the effect
        of each stage can be inspected:
            ideal_signal_e      : noiseless, blur-free signal (electrons)
            quantum_noisy_e     : + Poisson + Swank (quantum) noise
            blurred_e           : + optical/charge spread
            with_electronic_e   : + additive Gaussian electronic noise
            with_imperfections_e: + dead pixels / gain / offset
            digital_counts      : final ADC output (if digitize=True)
        """
        image = np.asarray(incident_photon_image, dtype=float)
        if image.shape != (self.geometry.rows, self.geometry.channels):
            raise ValueError(
                f"incident_photon_image must have shape "
                f"{(self.geometry.rows, self.geometry.channels)}, got {image.shape}"
            )

        eta = self.quantum_efficiency
        A = self.material.swank_factor
        g = self.material.conversion_gain_e_per_photon

        # 1. Ideal (noiseless) signal in electrons
        ideal_signal_e = image * eta * g

        # 2. Quantum + Swank noise:
        #    - number of absorbed photons is Poisson(eta * incident)
        #    - each absorbed photon's light output has excess variance
        #      captured by the Swank factor A (<=1): equivalent variance
        #      per photon = g^2 / A instead of g^2.
        n_absorbed = self.rng.poisson(np.clip(image * eta, 0, None)).astype(float)
        # Represent the per-photon gain spread as a Gamma-distributed multiplier
        # with mean 1 and variance (1/A - 1), consistent with the Swank factor.
        variance_per_photon = max(1.0 / A - 1.0, 1e-6)
        shape_k = 1.0 / variance_per_photon
        gain_fluctuation = self.rng.gamma(shape=shape_k, scale=1.0 / shape_k, size=n_absorbed.shape)
        quantum_noisy_e = n_absorbed * g * gain_fluctuation

        # 3. Optical/charge spread (blur / cross-talk between pixels)
        blurred_e = self._apply_blur(quantum_noisy_e) if apply_blur else quantum_noisy_e.copy()

        # 4. Additive electronic (Gaussian) read noise
        if apply_electronic_noise:
            electronic_noise = self.rng.normal(
                0.0, self.electronics.electronic_noise_std_e, size=blurred_e.shape
            )
            with_electronic_e = blurred_e + electronic_noise
        else:
            with_electronic_e = blurred_e.copy()

        # 5. Detector imperfections: gain non-uniformity, fixed offset, dead pixels
        if apply_imperfections:
            with_imperfections_e = with_electronic_e * self.gain_map + self.offset_map
            with_imperfections_e[self.dead_pixel_map] = 0.0
        else:
            with_imperfections_e = with_electronic_e.copy()

        result = {
            "ideal_signal_e": ideal_signal_e,
            "quantum_noisy_e": quantum_noisy_e,
            "blurred_e": blurred_e,
            "with_electronic_e": with_electronic_e,
            "with_imperfections_e": with_imperfections_e,
        }

        if digitize:
            result["digital_counts"] = self._digitize(with_imperfections_e)

        return result

    def _digitize(self, signal_e: np.ndarray) -> np.ndarray:
        max_code = 2 ** self.electronics.adc_bit_depth - 1
        scaled = np.clip(signal_e, 0, self.electronics.full_well_capacity_e)
        codes = np.round(scaled / self.electronics.full_well_capacity_e * max_code)
        return codes.astype(np.uint32 if self.electronics.adc_bit_depth > 16 else np.uint16)

    def simulate_flat_field(self, mean_photon_fluence: float, n_frames: int = 1) -> np.ndarray:
        """Convenience helper: simulate `n_frames` uniform flat-field
        acquisitions and return the stack (n_frames, rows, channels) of
        final digital images -- useful for measuring real-world noise,
        offset and gain statistics from the simulated detector."""
        rows, cols = self.geometry.rows, self.geometry.channels
        frames = np.empty((n_frames, rows, cols), dtype=float)
        flat_input = np.full((rows, cols), mean_photon_fluence, dtype=float)
        for i in range(n_frames):
            out = self.simulate_projection(flat_input, digitize=False)
            frames[i] = out["with_imperfections_e"]
        return frames

    # -- reporting -----------------------------------------------------------
    def summary(self, mean_incident_quanta: float = 1.0e5) -> str:
        eta = self.quantum_efficiency
        g = self.material.conversion_gain_e_per_photon
        dqe0 = self.dqe0(mean_incident_quanta)
        nyq = self.nyquist_frequency()
        lines = [
            "CT Detector Simulator - Configuration Summary",
            "=" * 48,
            f"Geometry:",
            f"  Rows x Channels        : {self.geometry.rows} x {self.geometry.channels}",
            f"  Pixel pitch (W x H)    : {self.geometry.element_width_mm} mm x "
            f"{self.geometry.element_height_mm} mm",
            f"  Active area (W x H)    : {self.geometry.width_mm} mm x {self.geometry.height_mm} mm",
            f"  Collimation            : {self.geometry.collimation_mm} mm",
            f"  Detector type          : {self.geometry.detector_type}",
            "",
            f"Material: {self.material.display_name}",
            f"  Thickness              : {self.material.thickness_mm} mm",
            f"  Effective energy       : {self.material.effective_energy_kev} keV",
            f"  Quantum efficiency (eta): {eta:.4f} ({eta*100:.2f} %)",
            f"  Swank factor            : {self.material.swank_factor:.3f}",
            f"  Conversion gain (g)     : {g:.1f} e-/absorbed photon",
            "",
            f"Electronics:",
            f"  Electronic noise (sigma_e): {self.electronics.electronic_noise_std_e:.1f} e- RMS",
            f"  Full well capacity      : {self.electronics.full_well_capacity_e:.2e} e-",
            f"  ADC bit depth           : {self.electronics.adc_bit_depth} bit",
            "",
            f"Imperfections:",
            f"  Dead pixels             : {self.dead_pixel_map.sum()} "
            f"({self.dead_pixel_map.mean()*100:.3f} % of array)",
            f"  Gain non-uniformity sig : {self.imperfections.gain_nonuniformity_std*100:.2f} %",
            f"  Fixed offset sigma      : {self.imperfections.offset_std_e:.1f} e-",
            f"  Optical/charge spread   : {self.imperfections.blur_sigma_mm:.3f} mm (1-sigma)",
            "",
            f"Performance @ {mean_incident_quanta:.2e} photons/pixel:",
            f"  DQE(0)                  : {dqe0:.4f}",
            f"  Nyquist frequency       : {nyq:.3f} cy/mm",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# 4. Convenience loader matching the reference JSON structure
# ---------------------------------------------------------------------------
def load_from_reference_config(path_or_dict) -> CTDetectorSimulator:
    """Accepts either a path to a JSON file or a dict containing a
    top-level "detector" key matching the structure given in the prompt,
    and returns a ready CTDetectorSimulator."""
    if isinstance(path_or_dict, dict):
        cfg = path_or_dict
    else:
        with open(path_or_dict, "r") as f:
            cfg = json.load(f)
    detector_cfg = cfg["detector"] if "detector" in cfg else cfg
    return CTDetectorSimulator(detector_cfg)
