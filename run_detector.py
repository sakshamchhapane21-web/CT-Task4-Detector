"""
run_detector_demo.py
=====================
Demonstrates ct_detector_simulator.CTDetectorSimulator using the reference
64-row x 888-channel, 0.625 mm pixel, scintillator/photodiode CT detector.

Produces:
  1. detector_characterization.png : MTF, DQE(f) at several exposures,
     gain/defect map, flat-field noise histogram.
  2. phantom_projection.png        : a synthetic water-cylinder projection
     (parallel-beam line-integral approximation) pushed through the full
     detector pipeline at a "normal dose" and a "low dose" exposure, shown
     both as 2D projection images and as central-row line profiles in both
     transmitted-signal and reconstructed attenuation (line-integral) space.
  3. summary.txt                   : text summary of detector parameters.

Note on the phantom: a uniform-density cylinder with a parallel-beam,
constant-path-length-per-channel model is used purely to produce a
realistic-looking, spatially varying photon fluence to exercise the
detector model. It is a geometric simplification (true fan-beam CT
projection/reconstruction is out of scope here) -- the physics under test
is the DETECTOR response, not the imaging geometry.
"""

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")   # render to file, no display window needed
import matplotlib.pyplot as plt

from ct_detector_simulator import CTDetectorSimulator, MaterialModel, \
    ElectronicNoiseModel, ImperfectionModel

# Output folder: "detector_sim_outputs" created next to this script,
# wherever it's run from (Windows, Mac, Linux all fine).
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "detector_sim_outputs")
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Build the simulator from the reference detector configuration
# ---------------------------------------------------------------------------
REFERENCE_CONFIG = {
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

sim = CTDetectorSimulator(
    REFERENCE_CONFIG,
    material=MaterialModel("scintillator_photodiode", thickness_mm=1.4, effective_energy_kev=65.0),
    electronics=ElectronicNoiseModel(electronic_noise_std_e=35.0, full_well_capacity_e=2.5e8, adc_bit_depth=16),
    imperfections=ImperfectionModel(
        dead_pixel_fraction=0.002,
        gain_nonuniformity_std=0.01,
        offset_std_e=15.0,
        blur_sigma_mm=0.35,
    ),
    seed=42,
)

print(sim.summary(mean_incident_quanta=1.0e5))
with open(f"C:\\Users\\Saksham Chhapane\\Downloads\\files\\summary.txt", "w") as f:
    f.write(sim.summary(mean_incident_quanta=1.0e5) + "\n")

# ---------------------------------------------------------------------------
# 2. Characterization figure: MTF, DQE(f), defect map, noise histogram
# ---------------------------------------------------------------------------
freqs = np.linspace(0, sim.nyquist_frequency(), 200)
mtf_curve = sim.mtf(freqs)

# A second simulator, identical except for much noisier electronics, used
# purely as a comparison case: with the reference detector's high per-photon
# conversion gain (~1716 e-/absorbed photon), 35 e- RMS read noise is
# equivalent to under 1/1000 of a photon, so the reference detector is
# essentially ALWAYS quantum-noise-limited across any realistic CT dose
# (this is a real, well-known property of scintillator/photodiode CT
# detectors). To actually visualize the electronic-noise-limited regime,
# a "degraded electronics" variant (5000 e- RMS) is shown alongside it.
sim_noisy_elec = CTDetectorSimulator(
    REFERENCE_CONFIG,
    material=MaterialModel("scintillator_photodiode", thickness_mm=1.4, effective_energy_kev=65.0),
    electronics=ElectronicNoiseModel(electronic_noise_std_e=5000.0,
                                      full_well_capacity_e=2.5e8, adc_bit_depth=16),
    imperfections=ImperfectionModel(dead_pixel_fraction=0.002, gain_nonuniformity_std=0.01,
                                     offset_std_e=15.0, blur_sigma_mm=0.35),
    seed=42,
)

eta_, A_, g_ = sim.quantum_efficiency, sim.material.swank_factor, sim.material.conversion_gain_e_per_photon
plateau_dqe = eta_ * A_
floor_default = A_ * sim.electronics.electronic_noise_std_e ** 2 / (eta_ * g_ ** 2)
floor_noisy = A_ * sim_noisy_elec.electronics.electronic_noise_std_e ** 2 / (eta_ * g_ ** 2)

fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# (a) MTF
ax = axes[0, 0]
ax.plot(freqs, mtf_curve, lw=2, color="tab:blue")
ax.axvline(sim.nyquist_frequency(), color="gray", ls="--", lw=1, label="Nyquist")
ax.set_xlabel("Spatial frequency (cy/mm)")
ax.set_ylabel("MTF")
ax.set_title("Modulation Transfer Function\n(pixel aperture x optical/charge spread)")
ax.set_ylim(0, 1.02)
ax.grid(alpha=0.3)
ax.legend()

# (b) DQE(f) at high vs. low dose, default vs. degraded electronics
ax = axes[0, 1]
q_high, q_low = 2.0e5, 2.0e2
ax.plot(freqs, sim.dqe(freqs, q_high), lw=2, color="tab:blue",
        label=f"Reference electronics, high dose (q={q_high:.0e})")
ax.plot(freqs, sim.dqe(freqs, q_low), lw=2, color="tab:blue", ls="--",
        label=f"Reference electronics, low dose (q={q_low:.0e})")
ax.plot(freqs, sim_noisy_elec.dqe(freqs, q_high), lw=2, color="tab:red",
        label=f"Degraded electronics, high dose (q={q_high:.0e})")
ax.plot(freqs, sim_noisy_elec.dqe(freqs, q_low), lw=2, color="tab:red", ls="--",
        label=f"Degraded electronics, low dose (q={q_low:.0e})")
ax.set_xlabel("Spatial frequency (cy/mm)")
ax.set_ylabel("DQE(f)")
ax.set_title("DQE vs. frequency, dose and electronic noise")
ax.grid(alpha=0.3)
ax.legend(fontsize=7)

# (c) DQE(0) vs. mean exposure (log-x): quantum-limited plateau + knee
ax = axes[0, 2]
q_range = np.logspace(-3, 6, 300)
dqe0_default = [sim.dqe0(q) for q in q_range]
dqe0_noisy = [sim_noisy_elec.dqe0(q) for q in q_range]
ax.plot(q_range, dqe0_default, lw=2, color="tab:blue", label="Reference electronics (35 e- RMS)")
ax.plot(q_range, dqe0_noisy, lw=2, color="tab:red", label="Degraded electronics (5000 e- RMS)")
ax.axhline(plateau_dqe, color="gray", ls=":", lw=1, label=f"Quantum-limited ceiling (eta*A={plateau_dqe:.3f})")
ax.axvline(floor_noisy, color="tab:red", ls=":", lw=1, alpha=0.6)
ax.set_xscale("log")
ax.set_xlabel("Mean incident quanta per pixel, q")
ax.set_ylabel("DQE(0)")
ax.set_title("DQE(0) vs. exposure\n(shows quantum-limited plateau and electronic-noise 'knee')")
ax.grid(alpha=0.3, which="both")
ax.legend(fontsize=7)

# (d) Defect / gain non-uniformity map
ax = axes[1, 0]
gain_display = sim.gain_map.copy()
im = ax.imshow(gain_display, cmap="coolwarm", aspect="auto", vmin=0.95, vmax=1.05)
dead_rows, dead_cols = np.where(sim.dead_pixel_map)
ax.scatter(dead_cols, dead_rows, s=4, color="black", marker="x", label="Dead pixel")
ax.set_xlabel("Channel")
ax.set_ylabel("Row")
ax.set_title(f"Pixel gain map & dead pixels "
             f"({sim.dead_pixel_map.sum()} dead / {sim.dead_pixel_map.size})")
ax.legend(loc="upper right", fontsize=8)
plt.colorbar(im, ax=ax, label="Relative gain")

# (e) Flat-field noise histogram at normal dose
ax = axes[1, 1]
flat_frames = sim.simulate_flat_field(mean_photon_fluence=2.0e4, n_frames=8)
noise_e = (flat_frames - flat_frames.mean()).ravel()
ax.hist(noise_e, bins=80, color="tab:green", alpha=0.8)
ax.set_xlabel("Deviation from mean signal (electrons)")
ax.set_ylabel("Pixel count")
ax.set_title("Flat-field noise histogram\n(quantum + electronic + fixed-pattern), 8 frames @ q=2e4")
ax.grid(alpha=0.3)

# (f) Noise budget bar chart at a representative operating exposure
ax = axes[1, 2]
q_op = 2.0e4
quantum_sigma_e = np.sqrt(eta_ * q_op * g_ ** 2 / A_)
electronic_sigma_e = sim.electronics.electronic_noise_std_e
total_sigma_e = np.sqrt(quantum_sigma_e ** 2 + electronic_sigma_e ** 2)
bars = ax.bar(
    ["Quantum\n(+Swank)", "Electronic", "Total"],
    [quantum_sigma_e, electronic_sigma_e, total_sigma_e],
    color=["tab:orange", "tab:purple", "tab:gray"],
)
ax.set_ylabel("Noise (electrons RMS)")
ax.set_yscale("log")
ax.set_title(f"Noise budget @ operating point q={q_op:.0e} ph/px\n"
             "(reference electronics)")
for b in bars:
    ax.annotate(f"{b.get_height():.1f}", (b.get_x() + b.get_width() / 2, b.get_height()),
                ha="center", va="bottom", fontsize=8)
ax.grid(alpha=0.3, axis="y")

fig.suptitle("CT Detector Characterization -- 64 x 888, 0.625 mm pixels, "
             "scintillator/photodiode", fontsize=13, y=1.0)
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/detector_characterization.png", dpi=150, bbox_inches="tight")
plt.close(fig)

print(f"\nQuantum-limited DQE ceiling (eta*Swank): {plateau_dqe:.4f}")
print(f"Electronic-noise-equivalent floor, reference electronics : {floor_default:.2e} photons/pixel")
print(f"Electronic-noise-equivalent floor, degraded electronics  : {floor_noisy:.2e} photons/pixel")

# ---------------------------------------------------------------------------
# 3. Phantom projection: water cylinder, parallel-beam approximation
# ---------------------------------------------------------------------------
rows, cols = sim.geometry.rows, sim.geometry.channels
pixel_w = sim.geometry.element_width_mm
channel_positions_mm = (np.arange(cols) - (cols - 1) / 2.0) * pixel_w

cylinder_radius_mm = 150.0   # ~ abdomen-sized water phantom
mu_water_per_cm = 0.20       # approx linear attenuation coefficient of water @ ~65 keV

path_length_mm = np.zeros_like(channel_positions_mm)
inside = np.abs(channel_positions_mm) < cylinder_radius_mm
path_length_mm[inside] = 2.0 * np.sqrt(cylinder_radius_mm**2 - channel_positions_mm[inside]**2)
true_attenuation = mu_water_per_cm * (path_length_mm / 10.0)   # unitless line integral

I0_high = 2.0e5   # open-beam photon fluence per pixel, "high/normal dose"
I0_low = 2.0e4    # "low dose" acquisition (10x less flux -> ~3.16x more projection noise)

incident_high = I0_high * np.exp(-true_attenuation)
incident_low = I0_low * np.exp(-true_attenuation)

# replicate across all detector rows (thin 40 mm collimation -> treat as ~uniform along z)
incident_image_high = np.tile(incident_high, (rows, 1))
incident_image_low = np.tile(incident_low, (rows, 1))

out_high = sim.simulate_projection(incident_image_high, digitize=False)
out_low = sim.simulate_projection(incident_image_low, digitize=False)

central_row = rows // 2

fig, axes = plt.subplots(2, 2, figsize=(13, 9))

# (a) 2D projection image, high dose
ax = axes[0, 0]
im = ax.imshow(out_high["with_imperfections_e"], cmap="gray", aspect="auto")
ax.set_title("Simulated projection image (signal, e-)\nNormal dose, q0=2e5 ph/px")
ax.set_xlabel("Channel")
ax.set_ylabel("Row")
plt.colorbar(im, ax=ax, fraction=0.03)

# (b) 2D projection image, low dose
ax = axes[0, 1]
im = ax.imshow(out_low["with_imperfections_e"], cmap="gray", aspect="auto")
ax.set_title("Simulated projection image (signal, e-)\nLow dose, q0=2e4 ph/px")
ax.set_xlabel("Channel")
ax.set_ylabel("Row")
plt.colorbar(im, ax=ax, fraction=0.03)

# (c) central-row transmitted signal profile: ideal vs noisy, both doses
ax = axes[1, 0]
ax.plot(channel_positions_mm, out_high["ideal_signal_e"][central_row], color="black",
        lw=1.5, label="Ideal (noiseless)")
ax.plot(channel_positions_mm, out_high["with_imperfections_e"][central_row], color="tab:red",
        lw=0.8, alpha=0.8, label="Normal dose (measured)")
ax.plot(channel_positions_mm, out_low["with_imperfections_e"][central_row], color="tab:blue",
        lw=0.8, alpha=0.8, label="Low dose (measured)")
ax.set_xlabel("Channel position (mm)")
ax.set_ylabel("Detector signal (electrons)")
ax.set_title(f"Central row (row {central_row}) transmitted signal")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# (d) recovered line-integral (attenuation) profile: this is the noise that
# actually matters for CT reconstruction quality.
def recovered_attenuation(signal_e, I0):
    eta = sim.quantum_efficiency
    g = sim.material.conversion_gain_e_per_photon
    ideal_full_scale_e = I0 * eta * g
    ratio = np.clip(signal_e, 1e-6, None) / ideal_full_scale_e
    return -np.log(ratio)

recovered_high = recovered_attenuation(out_high["with_imperfections_e"][central_row], I0_high)
recovered_low = recovered_attenuation(out_low["with_imperfections_e"][central_row], I0_low)

ax = axes[1, 1]
ax.plot(channel_positions_mm, true_attenuation, color="black", lw=1.5, label="True line integral")
ax.plot(channel_positions_mm, recovered_high, color="tab:red", lw=0.8, alpha=0.8,
        label="Recovered, normal dose")
ax.plot(channel_positions_mm, recovered_low, color="tab:blue", lw=0.8, alpha=0.8,
        label="Recovered, low dose")
# Note: this detector has an intentionally uncorrected dead pixel (see the
# gain/defect map figure) which, if it falls near the phantom edge where
# transmitted counts are naturally near zero, produces a large -log()
# spike -- a realistic illustration of why real scanners apply a dead-pixel
# correction map before reconstruction. The y-axis is clipped so this single
# spike doesn't dominate the plot; it is annotated instead.
dead_cols_central_row = np.where(sim.dead_pixel_map[central_row])[0]
y_max = float(np.nanmax(true_attenuation)) * 1.6
ax.set_ylim(-0.3, y_max)
for dc in dead_cols_central_row:
    x_pos = channel_positions_mm[dc]
    if abs(x_pos) < channel_positions_mm.max():
        ax.annotate("dead pixel\nartifact", xy=(x_pos, y_max * 0.97), xytext=(x_pos, y_max * 0.75),
                    ha="center", fontsize=7, color="dimgray",
                    arrowprops=dict(arrowstyle="->", color="dimgray", lw=0.8))
ax.set_xlabel("Channel position (mm)")
ax.set_ylabel("Line integral (attenuation, unitless)")
ax.set_title("Recovered projection-domain attenuation profile\n(this noise propagates into reconstructed images)")
ax.legend(fontsize=8, loc="upper left")
ax.grid(alpha=0.3)

fig.suptitle("Water-cylinder phantom projection through the simulated detector",
             fontsize=13, y=1.0)
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/phantom_projection.png", dpi=150, bbox_inches="tight")
plt.close(fig)

print("\nSaved:")
print(f"  {OUT_DIR}/detector_characterization.png")
print(f"  {OUT_DIR}/phantom_projection.png")
print(f"  {OUT_DIR}/summary.txt")