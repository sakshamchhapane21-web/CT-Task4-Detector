import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt


class DetectorMaterial:
    def __init__(self, name, density_g_cm3, mu_rho_table_cm2_g):
        self.name = name
        self.density_g_cm3 = density_g_cm3
        self.mu_rho_table_cm2_g = mu_rho_table_cm2_g

    def mass_attenuation_coefficient(self, energy_keV):
        energies = np.array(list(self.mu_rho_table_cm2_g.keys()), dtype=float)
        mu_rho_values = np.array(list(self.mu_rho_table_cm2_g.values()), dtype=float)
        return np.interp(energy_keV, energies, mu_rho_values)

    def linear_attenuation_coefficient(self, energy_keV):
        return self.mass_attenuation_coefficient(energy_keV) * self.density_g_cm3


MATERIALS = {
    "CsI": DetectorMaterial("Cesium Iodide", 4.51,
        {20: 5.4, 30: 2.2, 40: 1.6, 50: 1.1, 60: 0.65, 80: 0.36, 100: 0.26, 120: 0.20, 140: 0.17}),
    "GOS": DetectorMaterial("Gadolinium Oxysulfide", 7.34,
        {20: 8.5, 30: 3.4, 40: 2.5, 50: 1.6, 60: 1.0, 80: 0.55, 100: 0.40, 120: 0.31, 140: 0.26}),
    "aSe": DetectorMaterial("Amorphous Selenium", 4.28,
        {20: 3.1, 30: 1.2, 40: 0.9, 50: 0.55, 60: 0.38, 80: 0.24, 100: 0.18, 120: 0.14, 140: 0.12}),
    "CdTe": DetectorMaterial("Cadmium Telluride", 5.85,
        {20: 13.0, 30: 5.5, 40: 4.0, 50: 2.5, 60: 1.7, 80: 0.95, 100: 0.68, 120: 0.53, 140: 0.44}),
}


DETECTOR_CONFIG = {
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
    "scintillator_photodiode": "GOS",
}


class EnergyIntegratingDetector:
    def __init__(self, config=DETECTOR_CONFIG, thickness_cm=0.06,
                 fill_factor=0.8, conversion_gain_e_per_keV=50.0,
                 electronic_noise_e=500.0, psf_sigma_px=0.8,
                 gain_nonuniformity_fraction=0.02, dead_pixel_fraction=0.001,
                 random_seed=42):
        self.config = config
        self.rows = config["detector_rows"]
        self.channels = config["detector_channels"]
        self.element_width_mm = config["detector_element_width_mm"]
        self.element_height_mm = config["detector_element_height_mm"]
        self.array_width_mm = config["detector_width_mm"]
        self.array_height_mm = config["detector_height_mm"]
        self.detector_type = config["detector_type"]
        self.collimation_mm = config["collimation_mm"]
        material_key = MATERIAL_NAME_MAP.get(config["detector_material"], config["detector_material"])
        self.material = MATERIALS[material_key]
        self.thickness_cm = thickness_cm
        self.pixel_size_mm = self.element_width_mm
        self.fill_factor = fill_factor
        self.conversion_gain = conversion_gain_e_per_keV
        self.electronic_noise_e = electronic_noise_e
        self.psf_sigma_px = psf_sigma_px
        self.gain_nonuniformity_fraction = gain_nonuniformity_fraction
        self.dead_pixel_fraction = dead_pixel_fraction
        self.rng = np.random.default_rng(random_seed)

    def quantum_efficiency(self, energy_keV):
        mu = self.material.linear_attenuation_coefficient(energy_keV)
        absorption_fraction = 1.0 - np.exp(-mu * self.thickness_cm)
        return absorption_fraction * self.fill_factor

    def apply_spread(self, image):
        return gaussian_filter(image, sigma=self.psf_sigma_px)

    def generate_gain_map(self, shape):
        return self.rng.normal(1.0, self.gain_nonuniformity_fraction, size=shape)

    def generate_dead_pixel_map(self, shape):
        mask = self.rng.random(shape) > self.dead_pixel_fraction
        return mask.astype(float)

    def mtf(self, spatial_frequency_lp_per_mm):
        sigma_mm = self.psf_sigma_px * self.pixel_size_mm
        return np.exp(-2.0 * (np.pi ** 2) * (sigma_mm ** 2) * (spatial_frequency_lp_per_mm ** 2))

    def dqe(self, spatial_frequency_lp_per_mm, mean_energy_keV, mean_photon_fluence_per_pixel):
        eta = self.quantum_efficiency(mean_energy_keV)
        quantum_variance = mean_photon_fluence_per_pixel * eta * (mean_energy_keV ** 2) * (self.conversion_gain ** 2)
        electronic_variance = self.electronic_noise_e ** 2
        noise_fraction = quantum_variance / (quantum_variance + electronic_variance)
        mtf_values = self.mtf(spatial_frequency_lp_per_mm)
        return eta * noise_fraction * (mtf_values ** 2)

    def detect(self, photon_fluence_per_bin, energy_bins_keV):
        height, width, n_bins = photon_fluence_per_bin.shape
        signal_energy_keV = np.zeros((height, width))
        quantum_energy_variance = np.zeros((height, width))

        for bin_index in range(n_bins):
            energy_keV = energy_bins_keV[bin_index]
            eta = self.quantum_efficiency(energy_keV)
            detected_photons = photon_fluence_per_bin[:, :, bin_index] * eta
            signal_energy_keV += detected_photons * energy_keV
            quantum_energy_variance += detected_photons * (energy_keV ** 2)

        signal_electrons = signal_energy_keV * self.conversion_gain
        quantum_noise_electrons = np.sqrt(quantum_energy_variance) * self.conversion_gain

        signal_blurred = self.apply_spread(signal_electrons)
        quantum_noise_blurred = self.apply_spread(quantum_noise_electrons)

        gain_map = self.generate_gain_map((height, width))
        dead_pixel_map = self.generate_dead_pixel_map((height, width))

        ideal_signal = signal_blurred * gain_map * dead_pixel_map

        electronic_noise_sample = self.rng.normal(0.0, self.electronic_noise_e, size=(height, width))
        quantum_noise_sample = self.rng.normal(0.0, 1.0, size=(height, width)) * quantum_noise_blurred

        measured_signal = ideal_signal + quantum_noise_sample + electronic_noise_sample
        measured_signal = measured_signal * dead_pixel_map

        total_std = np.sqrt(quantum_noise_blurred ** 2 + self.electronic_noise_e ** 2)

        return {
            "measured_signal_electrons": measured_signal,
            "ideal_signal_electrons": ideal_signal,
            "quantum_noise_std_electrons": quantum_noise_blurred,
            "electronic_noise_std_electrons": np.full((height, width), self.electronic_noise_e),
            "total_std_electrons": total_std,
            "gain_map": gain_map,
            "dead_pixel_map": dead_pixel_map,
        }


def generate_synthetic_input(height=64, width=888, energy_bins_keV=(40, 60, 80, 100, 120),
                              spectrum_weights=(0.35, 0.30, 0.18, 0.10, 0.07),
                              incident_fluence_per_pixel=2.0e5,
                              material_thickness_cm=15.0,
                              mu_water_cm_inv=None):
    if mu_water_cm_inv is None:
        mu_water_cm_inv = {40: 0.27, 60: 0.21, 80: 0.18, 100: 0.17, 120: 0.16}

    energy_bins_keV = np.array(energy_bins_keV, dtype=float)
    weights = np.array(spectrum_weights, dtype=float)
    weights = weights / weights.sum()

    phantom = np.ones((height, width))
    phantom[height // 4: 3 * height // 4, width // 4: 3 * width // 4] = 1.35

    photon_fluence_per_bin = np.zeros((height, width, len(energy_bins_keV)))
    for bin_index, energy_keV in enumerate(energy_bins_keV):
        mu_water = mu_water_cm_inv[int(energy_keV)]
        attenuated_fraction = np.exp(-mu_water * material_thickness_cm * phantom)
        incident_photons = incident_fluence_per_pixel * weights[bin_index]
        photon_fluence_per_bin[:, :, bin_index] = incident_photons * attenuated_fraction

    return photon_fluence_per_bin, energy_bins_keV


def summarize_detector_performance(detector, photon_fluence_per_bin, energy_bins_keV, result):
    bin_means = [np.mean(photon_fluence_per_bin[:, :, i]) for i in range(len(energy_bins_keV))]
    mean_energy_keV = np.average(energy_bins_keV, weights=bin_means)
    mean_fluence_per_pixel = np.sum(photon_fluence_per_bin, axis=2).mean()

    efficiency_by_energy = {int(e): float(detector.quantum_efficiency(e)) for e in energy_bins_keV}

    spatial_frequencies = np.linspace(0.0, 5.0, 50)
    mtf_curve = detector.mtf(spatial_frequencies)
    dqe_curve = detector.dqe(spatial_frequencies, mean_energy_keV, mean_fluence_per_pixel)

    summary = {
        "material": detector.material.name,
        "thickness_cm": detector.thickness_cm,
        "efficiency_by_energy_keV": efficiency_by_energy,
        "mean_quantum_efficiency": float(np.mean(list(efficiency_by_energy.values()))),
        "psf_sigma_px": detector.psf_sigma_px,
        "mtf_at_1_lp_mm": float(np.interp(1.0, spatial_frequencies, mtf_curve)),
        "dqe_at_0_lp_mm": float(dqe_curve[0]),
        "dqe_at_1_lp_mm": float(np.interp(1.0, spatial_frequencies, dqe_curve)),
        "electronic_noise_std_e": detector.electronic_noise_e,
        "mean_quantum_noise_std_e": float(result["quantum_noise_std_electrons"].mean()),
        "mean_total_std_e": float(result["total_std_electrons"].mean()),
        "dead_pixel_fraction_actual": float(1.0 - result["dead_pixel_map"].mean()),
        "gain_nonuniformity_std": float(result["gain_map"].std()),
        "spatial_frequencies_lp_mm": spatial_frequencies,
        "mtf_curve": mtf_curve,
        "dqe_curve": dqe_curve,
        "detector_rows": detector.rows,
        "detector_channels": detector.channels,
        "element_width_mm": detector.element_width_mm,
        "element_height_mm": detector.element_height_mm,
        "array_width_mm": detector.array_width_mm,
        "array_height_mm": detector.array_height_mm,
        "collimation_mm": detector.collimation_mm,
        "detector_type": detector.detector_type,
    }
    return summary


def plot_results(result, summary, save_path="detector_output.png"):
    fig, axes = plt.subplots(2, 3, figsize=(18, 8))

    axes[0, 0].imshow(result["ideal_signal_electrons"], cmap="gray", aspect="auto")
    axes[0, 0].set_title("Ideal Signal (electrons)")

    axes[0, 1].imshow(result["measured_signal_electrons"], cmap="gray", aspect="auto")
    axes[0, 1].set_title("Measured Signal with Noise")

    axes[0, 2].imshow(result["gain_map"], cmap="viridis", aspect="auto")
    axes[0, 2].set_title("Gain Nonuniformity Map")

    axes[1, 0].imshow(result["dead_pixel_map"], cmap="gray", aspect="auto")
    axes[1, 0].set_title("Dead Pixel Map")

    axes[1, 1].plot(summary["spatial_frequencies_lp_mm"], summary["mtf_curve"])
    axes[1, 1].set_title("MTF vs Spatial Frequency")
    axes[1, 1].set_xlabel("lp/mm")
    axes[1, 1].set_ylabel("MTF")

    axes[1, 2].plot(summary["spatial_frequencies_lp_mm"], summary["dqe_curve"])
    axes[1, 2].set_title("DQE vs Spatial Frequency")
    axes[1, 2].set_xlabel("lp/mm")
    axes[1, 2].set_ylabel("DQE")

    plt.tight_layout()
    plt.savefig(save_path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    detector = EnergyIntegratingDetector(
        config=DETECTOR_CONFIG,
        thickness_cm=0.06,
        fill_factor=0.8,
        conversion_gain_e_per_keV=50.0,
        electronic_noise_e=500.0,
        psf_sigma_px=0.8,
        gain_nonuniformity_fraction=0.02,
        dead_pixel_fraction=0.001,
    )

    photon_fluence_per_bin, energy_bins_keV = generate_synthetic_input(
        height=detector.rows, width=detector.channels
    )

    result = detector.detect(photon_fluence_per_bin, energy_bins_keV)
    summary = summarize_detector_performance(detector, photon_fluence_per_bin, energy_bins_keV, result)

    print("Detector Type:", summary["detector_type"])
    print("Detector Rows x Channels:", summary["detector_rows"], "x", summary["detector_channels"])
    print("Element Size (mm):", summary["element_width_mm"], "x", summary["element_height_mm"])
    print("Array Size (mm):", summary["array_width_mm"], "x", summary["array_height_mm"])
    print("Collimation (mm):", summary["collimation_mm"])
    print("Detector Material:", summary["material"], "Thickness (cm):", summary["thickness_cm"])
    print("Quantum Efficiency by Energy (keV -> eta):", summary["efficiency_by_energy_keV"])
    print("Mean Quantum Efficiency:", round(summary["mean_quantum_efficiency"], 4))
    print("PSF Sigma (pixels):", summary["psf_sigma_px"])
    print("MTF at 1 lp/mm:", round(summary["mtf_at_1_lp_mm"], 4))
    print("DQE at 0 lp/mm:", round(summary["dqe_at_0_lp_mm"], 4))
    print("DQE at 1 lp/mm:", round(summary["dqe_at_1_lp_mm"], 4))
    print("Electronic Noise Std (electrons):", summary["electronic_noise_std_e"])
    print("Mean Quantum Noise Std (electrons):", round(summary["mean_quantum_noise_std_e"], 2))
    print("Mean Total Noise Std (electrons):", round(summary["mean_total_std_e"], 2))
    print("Actual Dead Pixel Fraction:", round(summary["dead_pixel_fraction_actual"], 5))
    print("Gain Nonuniformity Std:", round(summary["gain_nonuniformity_std"], 4))

    plot_results(result, summary, save_path="C:\\Users\\Saksham Chhapane\\Downloads\\CT\\detector_output.png")
