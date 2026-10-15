"""
xray.py
=======
GPU-Accelerated X-Ray Source Physics & Spectrum Generation Module.

Generates the polychromatic tungsten bremsstrahlung energy spectrum,
models aluminum filtration, calculates incident photon fluence, applies
tube current modulation (TCM), and computes ray positions and unit directions
on the GPU (CuPy) aligned with 'ct_system_output.json'.

Key Interfaces:
    - XRaySource: Configurable physical class encapsulating tube parameters.
    - generate_spectrum(): Kramers bremsstrahlung + Al filtration + W characteristic peaks.
    - generate_rays_gpu(): Vectorized GPU ray generator producing (origins, directions).
"""

from __future__ import annotations

import json
import pathlib
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

# Primary GPU acceleration engine
try:
    import cupy as cp
    # Verify that a CUDA-capable device is accessible
    _ = cp.cuda.runtime.getDeviceProperties(0)
    HAS_CUPY = True
except Exception:
    import numpy as cp
    HAS_CUPY = False


def generate_spectrum(
    kVp: float = 120.0,
    Z: int = 74,
    filter_thickness_mm: float = 1.5,
    energy_min_keV: float = 10.0,
    energy_step_keV: float = 1.0,
    use_cupy: bool = True,
) -> Tuple[Any, Any]:
    """
    Computes polychromatic bremsstrahlung X-ray spectrum with aluminum filtration.

    Physics:
        Unfiltered intensity: I_unfilt(E) = Z * (kVp - E) / E
        Al filtration:        T_Al(E) = exp( -t_Al * (25.0 / E)^3 )
        Filtered spectrum:    S(E) = max(0, I_unfilt(E)) * T_Al(E)

    Returns:
        energies_keV: (N_bins,) grid of photon energies
        spectrum:     (N_bins,) normalized relative photon fluence distribution
    """
    xp = cp if (use_cupy and HAS_CUPY) else np

    energies = xp.arange(energy_min_keV, kVp + energy_step_keV, energy_step_keV, dtype=xp.float32)

    # Kramers bremsstrahlung approximation
    unfiltered = float(Z) * (float(kVp) - energies) / xp.maximum(energies, 1e-3)
    unfiltered = xp.maximum(unfiltered, 0.0)

    # Empirical aluminum K-edge attenuation curve
    attenuation_exponent = float(filter_thickness_mm) * (25.0 / xp.maximum(energies, 1.0)) ** 3
    transmission = xp.exp(-attenuation_exponent)

    spectrum = unfiltered * transmission

    # Characteristic Tungsten K-shell peaks (K-alpha ~ 59.3 keV, K-beta ~ 67.2 keV)
    k_alpha_mask = xp.exp(-0.5 * ((energies - 59.3) / 1.2) ** 2)
    k_beta_mask = xp.exp(-0.5 * ((energies - 67.2) / 1.2) ** 2)
    characteristic = (0.25 * k_alpha_mask + 0.10 * k_beta_mask) * float(kVp > 70.0)
    spectrum = spectrum + characteristic * xp.max(spectrum)

    # Normalize integral to unity
    total_area = xp.sum(spectrum)
    if total_area > 0:
        spectrum = spectrum / total_area

    return energies, spectrum


def generate_rays_gpu(
    source_position: Any,
    detector_pixels: Any,
    use_cupy: bool = True,
) -> Tuple[Any, Any]:
    """
    Generates ray origin and normalized direction vectors for each detector pixel on the GPU.

    Arguments:
        source_position: (B, 1, 1, 3) or (3,) source position in scanner coordinates
        detector_pixels: (B, H, W, 3) or (N_pixels, 3) detector element coordinates
    Returns:
        origins:    Broadcasting ray origin array of matching shape (B, H, W, 3)
        directions: Unit direction vectors pointing from source to detector element (B, H, W, 3)
    """
    xp = cp if (use_cupy and HAS_CUPY) else np

    if not isinstance(source_position, xp.ndarray):
        source_position = xp.asarray(source_position, dtype=xp.float32)
    if not isinstance(detector_pixels, xp.ndarray):
        detector_pixels = xp.asarray(detector_pixels, dtype=xp.float32)

    # Ensure source_position broadcasts with detector_pixels
    while source_position.ndim < detector_pixels.ndim:
        source_position = source_position[:, None, ...] if source_position.ndim > 1 else source_position[None, ...]

    # Vector difference: pointing from source to detector pixel
    diff = detector_pixels - source_position

    # Normalize to unit direction
    inv_norm = 1.0 / xp.maximum(xp.linalg.norm(diff, axis=-1, keepdims=True), 1e-12)
    directions = diff * inv_norm

    # Broadcast origins to match detector shape
    origins = xp.broadcast_to(source_position, detector_pixels.shape)

    return origins, directions


class XRaySource:
    """
    Models the CT X-ray tube physics: tube potential (kVp), tube current (mA),
    exposure timing, tube current modulation (TCM), bowtie filter, and ray geometry.
    """

    def __init__(
        self,
        kvp: float = 120.0,
        tube_current_ma: float = 200.0,
        exposure_time_s: float = 1.0,
        filter_thickness_mm: float = 1.5,
        energy_step_keV: float = 1.0,
        tcm_enabled: bool = True,
        tcm_depth: float = 0.25,
        tcm_phase_deg: float = 0.0,
        focal_spot_size_mm: Tuple[float, float] = (0.4, 0.5),
        use_cupy: bool = True,
    ):
        self.kvp = float(kvp)
        self.tube_current_ma = float(tube_current_ma)
        self.exposure_time_s = float(exposure_time_s)
        self.filter_thickness_mm = float(filter_thickness_mm)
        self.energy_step_keV = float(energy_step_keV)
        self.tcm_enabled = bool(tcm_enabled)
        self.tcm_depth = float(tcm_depth)
        self.tcm_phase_deg = float(tcm_phase_deg)
        self.focal_spot_size_mm = focal_spot_size_mm
        self.use_cupy = use_cupy and HAS_CUPY

        # Generate base polychromatic spectrum
        self.energies, self.spectrum_weights = generate_spectrum(
            kVp=self.kvp,
            Z=74,
            filter_thickness_mm=self.filter_thickness_mm,
            energy_min_keV=10.0,
            energy_step_keV=self.energy_step_keV,
            use_cupy=self.use_cupy,
        )
        self.num_bins = len(self.energies)

        # Baseline incident photon fluence scale per detector pixel (~2.5e5 photons/pixel at 200 mA)
        self.photons_per_mas_pixel = 1250.0

    @classmethod
    def from_config_file(
        cls,
        json_path: Union[str, pathlib.Path] = "ct_system_output.json",
        use_cupy: bool = True,
    ) -> "XRaySource":
        """Initializes XRaySource using parameters from 'ct_system_output.json'."""
        path = pathlib.Path(json_path)
        if not path.is_file():
            # Check system_config subfolder
            alt_path = path.parent / "system_config" / path.name
            if alt_path.is_file():
                path = alt_path

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        sys_cfg = data.get("system_config", {})
        tcm_cfg = data.get("tcm", {})
        src_cfg = data.get("source_config", {})

        return cls(
            kvp=sys_cfg.get("kvp", 120.0),
            tube_current_ma=sys_cfg.get("tube_current_ma", 200.0),
            exposure_time_s=sys_cfg.get("exposure_time_s", 1.0),
            filter_thickness_mm=src_cfg.get("al_filtration_mm", 1.5),
            energy_step_keV=src_cfg.get("e_step_kev", 1.0),
            tcm_enabled=tcm_cfg.get("enabled", True),
            tcm_depth=tcm_cfg.get("modulation_depth", 0.25),
            tcm_phase_deg=tcm_cfg.get("phase_deg", 0.0),
            use_cupy=use_cupy,
        )

    def get_tcm_factor(self, theta_rad: Any) -> Any:
        """
        Computes angular tube current modulation scaling factor:
            scale(theta) = 1.0 + depth * cos(2 * theta + phase)
        Double-angle variation accounts for lateral vs AP body thickness variation.
        """
        xp = cp if self.use_cupy else np
        if not self.tcm_enabled:
            return xp.ones_like(theta_rad, dtype=xp.float32)

        phase_rad = float(np.radians(self.tcm_phase_deg))
        tcm_factor = 1.0 + self.tcm_depth * xp.cos(2.0 * theta_rad + phase_rad)
        return xp.clip(tcm_factor, 0.3, 2.0).astype(xp.float32)

    def calculate_incident_fluence(
        self,
        batch_size: int,
        detector_rows: int = 64,
        detector_channels: int = 888,
        theta_rad: Optional[Any] = None,
    ) -> Any:
        """
        Calculates incident spectral photon fluence Phi_0(E) per detector pixel for a batch.

        Returns:
            fluence_per_bin: (B, H, W, N_bins) GPU array representing incident photons per energy bin.
        """
        xp = cp if self.use_cupy else np

        # Base nominal photons per pixel
        nominal_photons_per_pixel = self.tube_current_ma * self.exposure_time_s * self.photons_per_mas_pixel
        nominal_photons_per_pixel /= 1000.0  # Normalized to per-view exposure fraction

        if theta_rad is not None:
            tcm_factor = self.get_tcm_factor(theta_rad).reshape(batch_size, 1, 1, 1)
        else:
            tcm_factor = 1.0

        # Base photon count per bin: (1, 1, 1, N_bins)
        spectrum_1d = (self.spectrum_weights * nominal_photons_per_pixel).reshape(1, 1, 1, self.num_bins)

        # Broadcast across batch and detector grid: (B, H, W, N_bins)
        ones_grid = xp.ones((batch_size, detector_rows, detector_channels, 1), dtype=xp.float32)
        fluence = (ones_grid * spectrum_1d) * tcm_factor

        return fluence.astype(xp.float32)

    def generate_batch_rays(
        self,
        source_positions: Any,
        pixel_positions: Any,
    ) -> Tuple[Any, Any]:
        """
        Generates ray origins and directions for a batch of views on the GPU.

        Arguments:
            source_positions: (B, 1, 1, 3) source coordinates
            pixel_positions:  (B, H, W, 3) detector element coordinates
        Returns:
            origins:          (B, H, W, 3)
            directions:       (B, H, W, 3)
        """
        return generate_rays_gpu(source_positions, pixel_positions, use_cupy=self.use_cupy)

    def sample_photon_energies(self, n_samples: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """Backwards compatibility helper: samples discrete photon energies from spectrum."""
        rng = rng or np.random.default_rng()
        e_np = cp.asnumpy(self.energies) if HAS_CUPY and isinstance(self.energies, cp.ndarray) else np.asarray(self.energies)
        w_np = cp.asnumpy(self.spectrum_weights) if HAS_CUPY and isinstance(self.spectrum_weights, cp.ndarray) else np.asarray(self.spectrum_weights)
        w_np = w_np / w_np.sum()
        return rng.choice(e_np, size=n_samples, p=w_np)


def run_source_simulation(
    source_pos: Any,
    det_pos: Any,
    detector_pixels: Any,
    cfg: Optional[Any] = None,
    rng_seed: int = 0,
) -> Dict[str, Any]:
    """
    Backwards compatibility function for per-view ray bundle generation.
    """
    xp = np
    source_pos = np.asarray(source_pos, dtype=np.float32)
    det_pos = np.asarray(det_pos, dtype=np.float32)
    detector_pixels = np.asarray(detector_pixels, dtype=np.float32)

    diff = detector_pixels - source_pos.reshape(1, 3)
    directions = diff / np.maximum(np.linalg.norm(diff, axis=-1, keepdims=True), 1e-12)

    n_samples = len(detector_pixels)
    e_grid, s_weights = generate_spectrum(120.0, 74, 1.5, 10.0, 1.0, use_cupy=False)

    rng = np.random.default_rng(rng_seed)
    sampled_energies = rng.choice(e_grid, size=n_samples, p=s_weights)
    weights = np.ones(n_samples, dtype=np.float32)

    data_matrix = np.column_stack([
        np.broadcast_to(source_pos, detector_pixels.shape),
        directions,
        sampled_energies,
        weights,
    ])

    return {
        "data_matrix": data_matrix,
        "energies_grid": e_grid,
        "photon_positions": np.broadcast_to(source_pos, detector_pixels.shape),
        "photon_directions": directions,
        "photon_energies": sampled_energies,
        "photon_weights": weights,
    }


if __name__ == "__main__":
    src = XRaySource.from_config_file("ct_system_output.json")
    print(f"XRaySource configured: {src.kvp} kVp | {src.tube_current_ma} mA | TCM: {src.tcm_enabled}")
    print(f"Spectrum bins: {src.num_bins} (Range: {float(src.energies[0])} to {float(src.energies[-1])} keV)")

    # Test batch ray generation
    xp = cp if HAS_CUPY else np
    s_test = xp.array([[[[541.0, 0.0, 0.0]]]], dtype=xp.float32)
    p_test = xp.array([[[[-408.0, 0.0, 0.0]]]], dtype=xp.float32)
    orig, d = generate_rays_gpu(s_test, p_test, use_cupy=HAS_CUPY)
    d_host = cp.asnumpy(d) if HAS_CUPY else d
    print(f"Test Ray Direction: {d_host[0, 0, 0]} (Expected pointing along -X)")
    fluence = src.calculate_incident_fluence(1, 64, 888)
    print(f"Incident Fluence Tensor Shape: {fluence.shape} (Type: {type(fluence)})")
