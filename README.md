# CT-Task04-Detector

Energy Integrating Detector (EID) simulation for the CT pipeline. This module takes attenuated photon fluence from the tissue-interaction stage (Task03) and simulates a realistic energy-integrating detector response, producing calibrated, noisy sinogram data for the image-reconstruction stage.

## Pipeline Context

```
Task01 (X-ray Source) -> Task03 (Tissue Interaction) -> Task04 (Detector, this repo) -> Reconstruction
```

This repo owns the detector physics: converting incident photon fluence per energy bin into a measured electrical signal, including detector efficiency, blur, and all noise sources. It also defines the data contract handed off to the Image Reconstruction team.

## Detector Configuration

Physical geometry, from `DETECTOR_CONFIG` in `energy_integrating_detector.py`:

| Parameter | Value | Description |
|---|---|---|
| `detector_rows` | 64 | Rows along the z-axis; number of simultaneous CT slices per rotation |
| `detector_channels` | 888 | Elements across the fan angle (in-plane channels) |
| `detector_element_width_mm` | 0.625 | Element width; sets pixel pitch and Nyquist resolution limit |
| `detector_element_height_mm` | 0.625 | Element height along z |
| `detector_width_mm` | 555.0 | Total in-plane array width |
| `detector_height_mm` | 40.0 | Total z-extent of the array |
| `detector_type` | `energy_integrating` | Sums total absorbed energy per pixel (not photon-counting) |
| `detector_material` | `scintillator_photodiode` | Mapped to GOS (Gadolinium Oxysulfide) internally |
| `collimation_mm` | 40.0 | Beam collimation width at the detector |

Physics parameters, set on `EnergyIntegratingDetector`:

| Parameter | Default | Description |
|---|---|---|
| `thickness_cm` | 0.06 | Scintillator thickness; controls absorption depth |
| `fill_factor` | 0.8 | Light-sensitive fraction of each pixel's area |
| `conversion_gain_e_per_keV` | 50.0 | Electrons produced per keV absorbed |
| `electronic_noise_e` | 500.0 | Fixed readout noise floor, in electrons |
| `psf_sigma_px` | 0.8 | Point-spread function width, in pixels |
| `gain_nonuniformity_fraction` | 0.02 | Per-pixel random gain variation (std. dev.) |
| `dead_pixel_fraction` | 0.001 | Fraction of permanently dead pixels |

## Files

### `energy_integrating_detector.py`
Core detector model.

- `DetectorMaterial` / `MATERIALS` — mass attenuation coefficient tables for CsI, GOS, aSe, CdTe
- `EnergyIntegratingDetector` — the detector object; key methods:
  - `quantum_efficiency(energy_keV)` — Beer-Lambert absorption fraction x fill factor
  - `mtf(frequency)` — analytic MTF from the Gaussian PSF
  - `dqe(frequency, mean_energy, mean_fluence)` — detective quantum efficiency vs spatial frequency
  - `detect(photon_fluence_per_bin, energy_bins_keV)` — full signal chain: absorption, energy integration, conversion gain, blur, gain/dead-pixel imperfections, quantum + electronic noise
- `generate_synthetic_input` / `generate_random_tissue_interaction_output` / `generate_chest_tissue_interaction_output` — standalone test inputs (uniform phantom, random anatomy, anatomically structured chest with lungs/spine/ribs) for running this module without a live Task03 feed
- `summarize_detector_performance` / `plot_results` — reporting and visualization

Run directly:
```
python energy_integrating_detector.py
```
Prints efficiency, DQE, noise, and tissue path-length statistics, and writes `detector_output.png`.

### `ct_reconstruction.py`
Full acquisition + reconstruction demo. Rotates a 2D chest phantom through many projection angles, runs each through the same detector physics as above, assembles a sinogram, and reconstructs the image via Filtered Back Projection (Ram-Lak filter + backprojection), no external CT library required.

Run directly:
```
python ct_reconstruction.py
```
Prints correlation and RMSE against the ground-truth phantom, and writes `reconstruction_output.png`.

### `send_detector_output_to_reconstruction.py`
Defines and exports the data contract for the Image Reconstruction team.

Run directly:
```
python send_detector_output_to_reconstruction.py
```
Writes:
- `detector_handoff_package.npz` — numeric arrays
- `detector_handoff_package_metadata.json` — human-readable metadata

## Output Data Contract (for Image Reconstruction Team)

**`.npz` arrays:**

| Key | Shape | Description |
|---|---|---|
| `measured_signal_electrons` | `(n_angles, n_channels)` | Raw noisy detector signal; this is the sinogram |
| `reference_signal_electrons` | `(n_channels,)` | Open-beam calibration signal, for log-normalization |
| `gain_map` | `(n_channels,)` | Per-channel gain nonuniformity (for flat-field correction) |
| `dead_pixel_map` | `(n_channels,)` | Per-channel dead-pixel mask |
| `angles_deg` | `(n_angles,)` | Projection angle for each sinogram row |

**`_metadata.json`:** full detector config, material, physics constants (conversion gain, noise std, PSF sigma), energy bins, and a `notes` field documenting the exact normalization convention:

```
line_integral = -ln(measured_signal_electrons / reference_signal_electrons)
```

Load with:
```python
from send_detector_output_to_reconstruction import load_detector_output_package
arrays, metadata = load_detector_output_package("detector_handoff_package")
```

## Physics Summary

1. **Efficiency**: `eta(E) = (1 - exp(-mu(E) * thickness)) * fill_factor`
2. **Energy integration**: signal accumulates `detected_photons(E) * E` summed across all energy bins (distinguishes this from a photon-counting detector)
3. **Conversion**: absorbed energy (keV) to electrons via `conversion_gain_e_per_keV`
4. **Spread**: Gaussian PSF blur (`psf_sigma_px`), whose Fourier transform gives the MTF
5. **DQE**: `eta * (quantum_variance / (quantum_variance + electronic_variance)) * MTF^2`
6. **Noise**: quantum (Poisson, signal-dependent) + electronic (fixed Gaussian) combined in quadrature for `total_std_electrons`
7. **Imperfections**: multiplicative per-pixel gain map + binary dead-pixel mask

## Dependencies

```
numpy
scipy
matplotlib
```

## Known Limitations / Open Items

- Isocenter-referenced z-coverage and in-plane resolution require the gantry's source-to-detector and source-to-axis distances (owned by the geometry/source team), not derivable from this config alone.
- `TISSUE_MATERIALS` and `MATERIALS` attenuation tables are illustrative approximations of real NIST/ICRU values, sufficient for pipeline integration testing but not for dosimetric accuracy.
- Reconstruction module uses a monochromatic-equivalent line integral; polychromatic beam-hardening correction is not implemented.

## Team

CT-Task04-Detector — used by `detector_team` and `projection_team`.
