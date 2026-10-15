# Tissue interaction and CT cascade, phase 1

This folder is a self-contained, isolated phase-1 pipeline. The original team
folders are not modified.

## Which file should I run?

Run [main.py](./main.py). It is the entry point that cascades:

```text
ct_system_output.json
        ↓
source spectrum + per-view CT geometry
        ↓
detector_geometry.py (detector pixels, rays, beam angle)
        ↓
tissue_interaction.py (Beer-Lambert transmission)
        ↓
energy_integrating_detector.py
        ↓
one CSV and one NPY detector image per view
```

From a PowerShell terminal, run all configured views:

```powershell
cd "C:\Users\Sridhar\OneDrive\Desktop\Sri\code\Python\Medical_img_lab\code_combine_12_10_26\ti_new"
python main.py --photons-per-ray 1
```

The default command processes all `16,728` views from the supplied
configuration. Each view generates `64 x 888 = 56,832` detector rays and one
CSV plus one NPY detector image.

For a smaller test run, select specific view indices:

```powershell
python main.py --views 0 1 2 --photons-per-ray 1
```

Use `--config path\to\ct_system_output.json` to select another configuration
with the same `system_config` and `trajectory.views` structure.

## Files in this folder

| File | Responsibility |
|---|---|
| `main.py` | Runnable source → tissue → detector pipeline and per-view CSV/NPY output |
| `config.py` | Loads `ct_system_output.json` and extracts per-view geometry |
| `spectrum.py` | Generates the phase-1 analytical filtered source spectrum |
| `detector_geometry.py` | Creates detector pixels, rays, central direction, and beam half-angle |
| `tissue_interaction.py` | Traverses voxels and calculates energy-resolved transmission |
| `material.py` | Uses xraylib NIST material data and returns attenuation in `mm^-1` |
| `cylinder_voxel.py` | Creates the phase-1 cylindrical material phantom |
| `energy_integrating_detector.py` | Converts transmitted energy bins to detector electrons |
| `ct_system_output.json` | Local copy of the supplied CT system and trajectory configuration |
| `units_n_dimensions.md` | Units, shapes, and contracts through the detector stage |
| `LIBRARIES_USED.md` | Installed Python packages, purposes, and import verification |
| `changes.md` | Files changed and the reason for each change |
| `remaining_work.md` | Remaining source, tissue-interaction, and detector work |

The local copy of `ct_system_output.json` was taken from the project root
`ct_system_output.json`. The configuration is therefore available in this
folder without requiring imports from another team folder.

## Do the source and detector rotate?

Yes. The source and detector rotate around the Y-axis in the XZ rotation
plane, matching the geometry flow previously represented by `syncer.py`.
`main.py` does not independently rotate a fixed geometry; it reads the
precomputed per-view trajectory from:

```text
ct_system_output.json
  -> trajectory
  -> views
  -> source / detector / angle_deg
```

For view `i`, the configuration supplies a different `angle_deg`,
`source_position`, and `detector_position`. The source and detector therefore
move around the object for each projection. The table position also changes
along Y for the helical scan through `table_y_mm`.

The tissue and detector stages use those positions to create new
source-to-detector rays for every view. `detector_geometry.py` calculates the
fan/cone beam directions and `main.py` reports the configured view angle and
calculated half-cone angle.

## Pipeline boundary

```text
source rays + source spectrum -> tissue_interaction -> detector.detect(...)
```

```python
from tissue_interaction import tissue_interaction

transmitted = tissue_interaction(
    rays,                  # (n_rays, 6), positions/directions in mm
    voxel_material,        # (ny, nz, nx), integer material IDs
    voxel_size_mm,         # mm
    energies_keV,          # (n_energy_bins,), keV
    source_spectrum,       # spectral density or per-bin fluence
    photons_per_ray=1.0,   # calibrated incident photons for each ray
    spectrum_is_density=True,
)

photon_fluence_per_bin = transmitted.reshape(
    detector_rows, detector_channels, len(energies_keV)
)
detector_result = detector.detect(photon_fluence_per_bin, energies_keV)
```

The returned array is **photons / ray / energy bin**. With the default
`photons_per_ray=1.0`, it is a normalized per-ray result, not an absolute
scanner fluence. Set `photons_per_ray` from a source calibration to obtain
quantitative detector input. The detector receives the energy-resolved array,
not an energy-integrated signal.

## Beam angle

Beam angle is accounted for in `detector_geometry.py`. For every configured
view, detector pixel centers are generated using the source position,
detector position, detector row/channel pitch, and detector orientation.
Each pixel produces its own normalized ray direction. The code computes the
maximum angle between the central source-to-detector direction and all pixel
ray directions as `half_cone_angle_deg`. The per-view angle from
`ct_system_output.json` is also written to every CSV row as `angle_deg`.

This means the beam is not treated as a single parallel beam. It is a
deterministic cone/fan beam over the detector pixels, consistent with the
source and detector positions in the supplied trajectory.

## Per-view output

For each selected view, `main.py` writes:

```text
photon_output/view_0000_detector.csv
photon_output/view_0000_detector.npy
```

The CSV contains view index, gantry angle, table position, detector
row/channel, ray origin and direction, the energy-bin count/range, and the
measured detector signal in electrons. The final signal column is named
`detector_signal_electrons_energy_integrated` because this detector sums the
contribution from all energy bins; it is not a separate signal for one bin.
The NPY file contains the detector image with shape
`(detector_rows, detector_channels)`.

The CSV signal is obtained from the detector stage:

```text
tissue_interaction
  -> transmitted photons per ray and energy bin
  -> detector.detect(...)
  -> measured_signal_electrons
  -> CSV detector_signal_electrons_energy_integrated
```

The detector applies quantum efficiency, energy weighting, conversion gain,
blur, gain/dead-pixel effects, and noise. Since noise can mathematically
produce a negative readout even though physical electrons cannot be negative,
`clamp_negative_signal=True` is enabled by default and stores negative
readouts as zero. Set it to `False` only when studying the raw signed noise
model.

## Confirmed phase-1 limitations

* The existing source spectrum is an analytical shape and is not an absolute
  photon calibration. The new API therefore requires an explicit
  `photons_per_ray` value and documents the resulting units.
* The model uses Beer-Lambert attenuation for primary photons only.
* Absolute detector values remain dependent on source calibration and detector
  calibration.

## Recommendations

* Calibrate photons per ray (or photons per detector pixel) against a measured
  air scan before using results for quantitative dosimetry.
* Validate material data and scanner geometry against a trusted system
  specification.
* Add Monte Carlo photoelectric, Compton, and Rayleigh scattering in a later
  phase. The API reserves `scatter_model="monte_carlo"` and fails explicitly
  until that model exists.

## Dependency

All code files required by the local cascade are in this folder. Install the
Python dependencies with:

```powershell
python -m pip install numpy scipy matplotlib xraylib
```

`numpy` and `xraylib` are required by tissue interaction. `scipy` and
`matplotlib` are used by the copied detector module. The original
`cylinder_voxel.py` layout is supported: `(ny, nz, nx)` with indexing
`[y, z, x]`.
