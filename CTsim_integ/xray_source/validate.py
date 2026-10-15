"""
Validation checks + plots. Kept concise: prints PASS/FAIL and produces the
required plots.
"""

import numpy as np
import matplotlib.pyplot as plt


def validate(results, cfg, save_dir="."):
    checks = []

    checks.append(("Spectrum max energy <= kVp",
                    results["energies_grid"].max() < cfg.KVP))

    sampled = results["photon_energies"]
    checks.append(("Sampled energies within spectrum range",
                    sampled.min() >= cfg.E_MIN_KEV and sampled.max() < cfg.KVP))

    fs_w, fs_h = cfg.FOCAL_SPOT_SIZE
    offsets = results["photon_positions"] - cfg.SOURCE_POSITION
    dist_from_source = np.linalg.norm(offsets, axis=1)
    checks.append(("Focal spot positions within expected extent",
                    dist_from_source.max() <= 0.5 * np.hypot(fs_w, fs_h) + 1e-6))

    norms = np.linalg.norm(results["photon_directions"], axis=1)
    central = cfg.DETECTOR_POSITION - cfg.SOURCE_POSITION
    central = central / np.linalg.norm(central)
    cos_angle = np.clip(results["photon_directions"] @ central, -1, 1)
    angle_from_center_deg = np.rad2deg(np.arccos(cos_angle))
    checks.append(("Direction norms approx 1", np.allclose(norms, 1.0, atol=1e-6)))
    checks.append(("All directions within cone_angle",
                    angle_from_center_deg.max() <= cfg.CONE_ANGLE_DEG + 1e-6))

    unfiltered = results["intensity_unfiltered"]
    filtered = results["intensity_filtered"]
    low_e_mask = results["energies_grid"] < 30
    checks.append(("Filtration reduces low-energy spectrum",
                    filtered[low_e_mask].sum() < unfiltered[low_e_mask].sum()))

    thickness = results["bowtie_thickness"]
    center_mask = angle_from_center_deg < np.percentile(angle_from_center_deg, 10)
    edge_mask = angle_from_center_deg > np.percentile(angle_from_center_deg, 90)
    checks.append(("Bowtie thicker at edge than at center",
                    thickness[edge_mask].mean() > thickness[center_mask].mean()))

    print("\n--- VALIDATION ---")
    for name, passed in checks:
        print(f"[{'PASS' if passed else 'FAIL'}] {name}")

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    axes[0, 0].plot(results["energies_grid"], results["intensity_unfiltered"], label="unfiltered")
    axes[0, 0].plot(results["energies_grid"], results["intensity_filtered"], label="filtered")
    axes[0, 0].set_title("Spectrum (unfiltered vs filtered)")
    axes[0, 0].set_xlabel("Energy (keV)")
    axes[0, 0].legend()

    bin_width = results["energies_grid"][1] - results["energies_grid"][0]
    spectrum_density = filtered / filtered.sum() / bin_width
    axes[0, 1].hist(sampled, bins=60, density=True, alpha=0.6, label="sampled photons")
    axes[0, 1].plot(results["energies_grid"], spectrum_density, "r-", label="filtered spectrum shape")
    axes[0, 1].set_title("Sampled photon energies vs spectrum")
    axes[0, 1].set_xlabel("Energy (keV)")
    axes[0, 1].legend()

    n_show = min(2000, len(offsets))
    idx = np.random.choice(len(offsets), size=n_show, replace=False)
    axes[0, 2].scatter(offsets[idx, 0], offsets[idx, 1], s=2, alpha=0.4)
    axes[0, 2].set_title("Focal spot positions (local x-y)")
    axes[0, 2].set_xlabel("mm")
    axes[0, 2].set_ylabel("mm")
    axes[0, 2].set_aspect("equal")

    axes[1, 0].scatter(angle_from_center_deg, thickness, s=2, alpha=0.3)
    axes[1, 0].set_title("Bowtie thickness vs angle from center")
    axes[1, 0].set_xlabel("Angle from center (deg)")
    axes[1, 0].set_ylabel("Thickness (mm)")

    image, _, _ = results["flat_field"]
    im = axes[1, 1].imshow(image.T, origin="lower", cmap="gray")
    axes[1, 1].set_title("Flat-field / no-object intensity map")
    plt.colorbar(im, ax=axes[1, 1])

    axes[1, 2].axis("off")
    axes[1, 2].text(0.05, 0.5,
                     "See mAs_vs_weight.png\nfor current/exposure\nscaling checks.",
                     fontsize=10)

    plt.tight_layout()
    out_path = f"{save_dir}/validation_overview.png"
    plt.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Saved plots to {out_path}")


def validate_current_and_exposure_scaling(calculate_photon_weights_fn, save_dir="."):
    """Checks: total weight scales ~linearly with mA, exposure_time, and mAs."""
    n_samples = 1000  # small sample count is fine, we only need the totals

    mA_values = np.linspace(20, 1300, 15)
    weight_vs_mA = [calculate_photon_weights_fn(mA, 0.5, n_samples).sum() for mA in mA_values]

    exposure_values = np.linspace(0.1, 2.0, 15)
    weight_vs_exposure = [calculate_photon_weights_fn(500.0, t, n_samples).sum() for t in exposure_values]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(mA_values, weight_vs_mA, "o-")
    axes[0].set_xlabel("mA")
    axes[0].set_ylabel("Total source weight")
    axes[0].set_title("mA vs total weight (exposure_time fixed)")

    axes[1].plot(mA_values * 0.5, weight_vs_mA, "o-", label="via mA sweep")
    axes[1].plot(exposure_values * 500.0, weight_vs_exposure, "x--", label="via exposure sweep")
    axes[1].set_xlabel("mAs")
    axes[1].set_ylabel("Total source weight")
    axes[1].set_title("mAs vs total weight")
    axes[1].legend()

    plt.tight_layout()
    out_path = f"{save_dir}/mAs_vs_weight.png"
    plt.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Saved plots to {out_path}")
