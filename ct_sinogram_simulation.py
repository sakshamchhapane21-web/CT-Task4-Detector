"""
ct_sinogram_simulation.py
===========================
Extends ct_detector_simulator.py (unchanged) to:

  1. Build a RANDOM tissue phantom -- a body outline plus a random number of
     random-sized/positioned/rotated tissue inclusions, each drawn from a
     small reference table of approximate tissue X-ray attenuation
     coefficients (TISSUE_DB).
  2. Forward-project that phantom analytically (parallel-beam, ellipse/line
     intersection) at many rotation angles to build a full sinogram.
  3. Push every view through the SAME detector physics pipeline used before
     (quantum efficiency, quantum+Swank noise, blur, electronic noise,
     imperfections) so every projection includes realistic detector error.
  4. Keep the per-view detector readout in exactly the flat format you are
     using elsewhere:

        detector_signal = np.array([signal_0, signal_1, ..., signal_N])

     where N = detector_rows * detector_channels (64*888 = 56832) and each
     entry is the energy-weighted (electron-domain) signal of one ray/pixel
     for that view. All views are stacked into `all_detector_signals` with
     shape (n_views, N).
  5. Reconstruct an image from the noisy sinogram with filtered
     back-projection (skimage.transform.iradon) and compare it against the
     known ground-truth phantom.

Material note: the detector material ("scintillator_photodiode" in the
reference config) defaults to a Gadolinium Oxysulfide (GOS/GADOX) +
photodiode model -- see MATERIAL_DB in ct_detector_simulator.py. Swap in
"CsI_photodiode" or "CdWO4_photodiode" there to compare materials.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skimage.transform import iradon

from ct_detector_simulator import (
    CTDetectorSimulator, MaterialModel, ElectronicNoiseModel, ImperfectionModel,
)

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "detector_sim_outputs")
os.makedirs(OUT_DIR, exist_ok=True)

RNG_SEED = 7
rng = np.random.default_rng(RNG_SEED)

# ---------------------------------------------------------------------------
# 1. Tissue attenuation reference table
# ---------------------------------------------------------------------------
# Approximate linear attenuation coefficients (1/cm) at a representative CT
# effective energy (~65 keV). These are textbook/NIST-typical, illustrative
# values -- not a substitute for measured spectral data.
TISSUE_DB = {
    "air":              0.0002,
    "lung":             0.05,
    "fat":              0.17,
    "soft_tissue":      0.21,   # used as the body background
    "muscle":           0.215,
    "blood":            0.22,
    "liver":            0.225,
    "cyst_fluid":       0.20,
    "bone_cancellous":  0.30,
    "calcification":    0.45,
    "bone_cortical":    0.55,
    "iodine_contrast":  0.65,
}
BACKGROUND_TISSUE = "soft_tissue"
# Tissue types allowed as random inclusions (exclude the background itself)
INCLUSION_TISSUES = [t for t in TISSUE_DB if t != BACKGROUND_TISSUE]


# ---------------------------------------------------------------------------
# 2. Random tissue phantom: body circle + random non-overlapping inclusions
# ---------------------------------------------------------------------------
class RandomTissuePhantom:
    def __init__(self, body_radius_mm=150.0, n_inclusions=None, seed=None):
        self.rng = np.random.default_rng(seed)
        self.body_radius_mm = body_radius_mm
        self.body_mu = TISSUE_DB[BACKGROUND_TISSUE]
        self.n_inclusions = n_inclusions if n_inclusions is not None else self.rng.integers(4, 8)
        self.inclusions = self._generate_inclusions()

    def _generate_inclusions(self, max_attempts=500):
        inclusions = []
        attempts = 0
        while len(inclusions) < self.n_inclusions and attempts < max_attempts:
            attempts += 1
            tissue = self.rng.choice(INCLUSION_TISSUES)
            a = self.rng.uniform(8.0, 35.0)          # semi-axis, mm
            b = self.rng.uniform(8.0, 35.0)
            phi = self.rng.uniform(0, np.pi)
            max_r = max(a, b)
            margin = self.body_radius_mm - max_r - 5.0
            if margin <= 0:
                continue
            radius_pos = self.rng.uniform(0, margin)
            angle_pos = self.rng.uniform(0, 2 * np.pi)
            cx = radius_pos * np.cos(angle_pos)
            cy = radius_pos * np.sin(angle_pos)

            # reject if it overlaps an existing inclusion (bounding-circle test)
            ok = True
            for inc in inclusions:
                dist = np.hypot(cx - inc["cx"], cy - inc["cy"])
                if dist < (max_r + max(inc["a"], inc["b"]) + 3.0):
                    ok = False
                    break
            if not ok:
                continue

            inclusions.append({
                "tissue": tissue, "cx": cx, "cy": cy,
                "a": a, "b": b, "phi": phi, "mu": TISSUE_DB[tissue],
            })
        return inclusions

    # -- analytic forward projection --------------------------------------
    @staticmethod
    def _chord_length_ellipse(cx, cy, a, b, phi, theta, u):
        """Vectorized (over u) chord length of the ray (angle theta, detector
        offset u) through an ellipse centered at (cx,cy), semi-axes (a,b),
        rotation phi. Ray: (x,y) = (u*cos(theta) - t*sin(theta),
                                     u*sin(theta) + t*cos(theta)).
        Returns chord length array (0 where the ray misses the ellipse)."""
        x0 = u * np.cos(theta) - cx
        y0 = u * np.sin(theta) - cy
        dx = -np.sin(theta)
        dy = np.cos(theta)

        cphi, sphi = np.cos(phi), np.sin(phi)
        xr0 = x0 * cphi + y0 * sphi
        yr0 = -x0 * sphi + y0 * cphi
        xrd = dx * cphi + dy * sphi
        yrd = -dx * sphi + dy * cphi

        A = (xrd ** 2) / a ** 2 + (yrd ** 2) / b ** 2
        B = 2.0 * (xr0 * xrd / a ** 2 + yr0 * yrd / b ** 2)
        C = (xr0 ** 2) / a ** 2 + (yr0 ** 2) / b ** 2 - 1.0

        disc = B ** 2 - 4 * A * C
        chord = np.zeros_like(u, dtype=float)
        hit = disc > 0
        sqrt_disc = np.sqrt(np.clip(disc, 0, None))
        t1 = (-B - sqrt_disc) / (2 * A)
        t2 = (-B + sqrt_disc) / (2 * A)
        chord[hit] = np.abs(t2 - t1)[hit]
        return chord

    def line_integral(self, theta_rad, u_mm):
        """Total attenuation line integral (unitless) for a single view
        angle `theta_rad`, evaluated at all detector offsets `u_mm`
        (array, mm). Body chord uses background mu; each inclusion
        contributes (mu_inclusion - body_mu) * chord_inclusion (valid since
        inclusions are fully inside the body and mutually non-overlapping)."""
        u = np.asarray(u_mm, dtype=float)
        body_chord_mm = self._chord_length_ellipse(0.0, 0.0, self.body_radius_mm,
                                                     self.body_radius_mm, 0.0, theta_rad, u)
        total_cm = self.body_mu * (body_chord_mm / 10.0)
        for inc in self.inclusions:
            chord_mm = self._chord_length_ellipse(inc["cx"], inc["cy"], inc["a"], inc["b"],
                                                    inc["phi"], theta_rad, u)
            total_cm += (inc["mu"] - self.body_mu) * (chord_mm / 10.0)
        return total_cm  # unitless line integral (mu * length, cm^-1 * cm)

    # -- rasterized ground-truth image, for display only --------------------
    def rasterize(self, extent_mm=200.0, n=400):
        xs = np.linspace(-extent_mm, extent_mm, n)
        ys = np.linspace(-extent_mm, extent_mm, n)
        X, Y = np.meshgrid(xs, ys)
        mu_map = np.zeros_like(X)
        mu_map[(X ** 2 + Y ** 2) <= self.body_radius_mm ** 2] = self.body_mu
        for inc in self.inclusions:
            cphi, sphi = np.cos(inc["phi"]), np.sin(inc["phi"])
            Xr = (X - inc["cx"]) * cphi + (Y - inc["cy"]) * sphi
            Yr = -(X - inc["cx"]) * sphi + (Y - inc["cy"]) * cphi
            mask = (Xr / inc["a"]) ** 2 + (Yr / inc["b"]) ** 2 <= 1.0
            mu_map[mask] = inc["mu"]
        return mu_map, (-extent_mm, extent_mm, -extent_mm, extent_mm)


# ---------------------------------------------------------------------------
# 3. Build detector simulator (reference geometry, unchanged physics)
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
    imperfections=ImperfectionModel(dead_pixel_fraction=0.002, gain_nonuniformity_std=0.01,
                                     offset_std_e=15.0, blur_sigma_mm=0.35),
    seed=42,
)

ROWS, CHANNELS = sim.geometry.rows, sim.geometry.channels
N = ROWS * CHANNELS  # 64 * 888 = 56832 rays per view
channel_positions_mm = (np.arange(CHANNELS) - (CHANNELS - 1) / 2.0) * sim.geometry.element_width_mm
I0 = 2.0e5  # open-beam photon fluence per pixel (constant "tube output" across views)

print(f"Rays per view (N = rows*channels): {N}")

# ---------------------------------------------------------------------------
# 4. Random phantom + multi-angle acquisition loop
# ---------------------------------------------------------------------------
phantom = RandomTissuePhantom(body_radius_mm=150.0, seed=RNG_SEED)
print(f"Generated {len(phantom.inclusions)} random tissue inclusions:")
for i, inc in enumerate(phantom.inclusions):
    print(f"  [{i}] {inc['tissue']:<16s} mu={inc['mu']:.3f}/cm  "
          f"center=({inc['cx']:.1f},{inc['cy']:.1f}) mm  axes=({inc['a']:.1f},{inc['b']:.1f}) mm")

N_VIEWS = 180
angles_deg = np.linspace(0.0, 180.0, N_VIEWS, endpoint=False)
angles_rad = np.deg2rad(angles_deg)

all_detector_signals = np.empty((N_VIEWS, N), dtype=float)          # <- your requested flat format, stacked
true_line_integral_sino = np.empty((N_VIEWS, CHANNELS), dtype=float)  # ground truth, for reference
recovered_line_integral_sino = np.empty((N_VIEWS, CHANNELS), dtype=float)

eta_ = sim.quantum_efficiency
g_ = sim.material.conversion_gain_e_per_photon
ideal_full_scale_e = I0 * eta_ * g_
central_row = ROWS // 2

for v, theta in enumerate(angles_rad):
    line_integral = phantom.line_integral(theta, channel_positions_mm)   # shape (CHANNELS,)
    true_line_integral_sino[v] = line_integral

    incident_fluence = I0 * np.exp(-line_integral)                       # shape (CHANNELS,)
    incident_image = np.tile(incident_fluence, (ROWS, 1))                # shape (ROWS, CHANNELS)

    out = sim.simulate_projection(incident_image, digitize=False)
    signal_image = out["with_imperfections_e"]                           # shape (ROWS, CHANNELS)

    # ---- your exact requested output format, per view ----
    detector_signal = signal_image.flatten()   # np.array([signal_0, signal_1, ..., signal_{N-1}])
    all_detector_signals[v] = detector_signal

    # For the reconstruction demo, pull the central row back out of the
    # flattened array and convert it back to line-integral (attenuation)
    # space -- this is what a real reconstruction pipeline does per slice.
    central_row_signal = detector_signal.reshape(ROWS, CHANNELS)[central_row]
    ratio = np.clip(central_row_signal, 1e-6, None) / ideal_full_scale_e
    recovered_line_integral_sino[v] = -np.log(ratio)

print(f"\nall_detector_signals shape: {all_detector_signals.shape}  "
      f"(n_views={N_VIEWS}, N={N})")
print(f"Example: view 0, first 5 signal values (electrons): "
      f"{all_detector_signals[0, :5]}")

# ---------------------------------------------------------------------------
# 4b. Known-bad-channel correction (standard CT preprocessing step)
# ---------------------------------------------------------------------------
# The detector's dead-pixel map is fixed in *detector* space, so a dead
# channel corrupts the SAME channel in every view -> a bright vertical
# streak in the sinogram, which turns into a strong ring artifact after
# back-projection. Real scanners carry a known bad-channel map and correct
# it before reconstruction (typically by interpolating from neighbouring,
# healthy channels); we do the same here using the simulator's own
# dead_pixel_map for the central row, so the effect of the defect is fully
# modeled but not left uncorrected.
raw_sino_uncorrected = recovered_line_integral_sino.copy()  # kept for comparison plot
dead_cols = np.where(sim.dead_pixel_map[central_row])[0]
if len(dead_cols) > 0:
    good_cols = np.setdiff1d(np.arange(CHANNELS), dead_cols)
    for v in range(N_VIEWS):
        recovered_line_integral_sino[v, dead_cols] = np.interp(
            dead_cols, good_cols, recovered_line_integral_sino[v, good_cols]
        )
print(f"\nCorrected {len(dead_cols)} known dead channel(s) at index/indices {dead_cols.tolist()} "
      f"by linear interpolation before reconstruction.")

# ---------------------------------------------------------------------------
# 5. Filtered back-projection reconstruction from the noisy sinogram
# ---------------------------------------------------------------------------
# skimage.transform.iradon expects shape (n_detector_channels, n_angles)
sinogram_for_recon = recovered_line_integral_sino.T
reconstruction = iradon(sinogram_for_recon, theta=angles_deg, filter_name="ramp", circle=True)

# Ground-truth raster for visual comparison (independent of the forward
# projector -- built directly from the phantom's shape list)
truth_extent_mm = 1.3 * phantom.body_radius_mm
mu_map, extent = phantom.rasterize(extent_mm=truth_extent_mm, n=reconstruction.shape[0])

# ---------------------------------------------------------------------------
# 6. Plots
# ---------------------------------------------------------------------------
sino_extent = [channel_positions_mm.min(), channel_positions_mm.max(), angles_deg.max(), angles_deg.min()]
li_vmax = np.percentile(true_line_integral_sino, 99.5)

fig, axes = plt.subplots(2, 3, figsize=(17, 11))

ax = axes[0, 0]
im = ax.imshow(mu_map, cmap="gray", extent=extent, origin="lower")
ax.set_title(f"Ground-truth random tissue phantom\n({len(phantom.inclusions)} inclusions, seed={RNG_SEED})")
ax.set_xlabel("x (mm)")
ax.set_ylabel("y (mm)")
plt.colorbar(im, ax=ax, label="mu (1/cm)", fraction=0.046)

ax = axes[0, 1]
im = ax.imshow(true_line_integral_sino, cmap="gray", aspect="auto", extent=sino_extent, vmin=0, vmax=li_vmax)
ax.set_title("Ground-truth sinogram (noiseless line integral)")
ax.set_xlabel("Detector channel position (mm)")
ax.set_ylabel("View angle (deg)")
plt.colorbar(im, ax=ax, label="Line integral", fraction=0.046)

ax = axes[0, 2]
im = ax.imshow(raw_sino_uncorrected, cmap="gray", aspect="auto", extent=sino_extent, vmin=0, vmax=li_vmax)
ax.set_title("Simulated (noisy) sinogram, UNCORRECTED\nnote the dead-channel streak")
ax.set_xlabel("Detector channel position (mm)")
ax.set_ylabel("View angle (deg)")
plt.colorbar(im, ax=ax, label="Line integral", fraction=0.046)

ax = axes[1, 0]
im = ax.imshow(recovered_line_integral_sino, cmap="gray", aspect="auto", extent=sino_extent, vmin=0, vmax=li_vmax)
ax.set_title("Simulated sinogram after detector physics\n+ dead-channel correction (used for recon)")
ax.set_xlabel("Detector channel position (mm)")
ax.set_ylabel("View angle (deg)")
plt.colorbar(im, ax=ax, label="Line integral", fraction=0.046)

ax = axes[1, 1]
recon_vmax = np.percentile(reconstruction, 99.5)
im = ax.imshow(reconstruction, cmap="gray", origin="lower", vmin=0, vmax=recon_vmax)
ax.set_title("Filtered back-projection reconstruction\n(from the corrected noisy sinogram)")
ax.set_xlabel("pixel")
ax.set_ylabel("pixel")
plt.colorbar(im, ax=ax, label="Reconstructed attenuation (a.u.)", fraction=0.046)

ax = axes[1, 2]
# central horizontal profile through the reconstruction vs. ground truth,
# for a direct quantitative comparison of recovered vs. true structure
recon_center = reconstruction.shape[0] // 2
truth_center = mu_map.shape[0] // 2
recon_x_mm = np.linspace(-truth_extent_mm, truth_extent_mm, reconstruction.shape[1])
truth_x_mm = np.linspace(-truth_extent_mm, truth_extent_mm, mu_map.shape[1])
ax.plot(truth_x_mm, mu_map[truth_center], color="black", lw=1.5, label="Ground truth (mu, 1/cm)")
ax.plot(recon_x_mm, reconstruction[recon_center] * (phantom.body_mu / np.median(
    reconstruction[recon_center][reconstruction[recon_center] > recon_vmax * 0.3])),
        color="tab:red", lw=1, alpha=0.8, label="FBP reconstruction (scaled)")
ax.set_xlabel("x (mm)")
ax.set_ylabel("mu (1/cm)")
ax.set_title("Central horizontal profile:\nground truth vs. reconstruction")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

fig.suptitle("Random tissue phantom -> detector simulation -> sinogram -> FBP reconstruction",
             fontsize=13, y=1.0)
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/sinogram_reconstruction.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# Persist the raw per-view detector_signal dataset for downstream use
np.save(f"{OUT_DIR}/all_detector_signals.npy", all_detector_signals)

print("\nSaved:")
print(f"  {OUT_DIR}/sinogram_reconstruction.png")
print(f"  {OUT_DIR}/all_detector_signals.npy  (shape {all_detector_signals.shape})")
