# `ti_new` change log

This file records the changes made to the self-contained tissue-interaction
pipeline and the reason for each change.

## Current implementation

| File | Change | Reason |
|---|---|---|
| `main.py` | Added the runnable source → tissue → detector entry point. | Provides one file to run the complete local pipeline. |
| `main.py` | Added per-view CSV and NPY output. | Preserves detector results for every selected CT view and supports later reconstruction. |
| `main.py` | Removed the default three-view restriction; omitted `--views` now processes all configured views. | Ensures the normal run covers all 16,728 trajectory views while retaining subset selection for testing. |
| `main.py` | Added energy-bin count and minimum/maximum energy metadata to each CSV row. | Makes clear which energy range contributed to the energy-integrated detector signal. |
| `main.py` | Uses `detector_signal_electrons_energy_integrated` as the CSV signal column name. | Distinguishes the detector output from the tissue-stage energy-resolved photon fluence. |
| `config.py` | Added loading and validation of `ct_system_output.json`. | Keeps the required CT configuration inside `ti_new` and avoids dependencies on another team folder. |
| `spectrum.py` | Added the local analytical filtered source spectrum. | Supplies the source energy bins and spectral shape required by tissue interaction. |
| `detector_geometry.py` | Added detector pixel generation and ray construction. | Produces one source-to-detector ray for each detector pixel. |
| `detector_geometry.py` | Calculates the per-view half-cone angle. | Accounts for the detector fan/cone beam geometry instead of treating the beam as parallel. |
| `tissue_interaction.py` | Added Beer-Lambert primary-photon attenuation. | Calculates energy-resolved transmitted photons through the voxel phantom. |
| `tissue_interaction.py` | Added input validation and explicit units/shapes. | Prevents incompatible source, phantom, and detector data from silently propagating. |
| `tissue_interaction.py` | Corrected voxel indexing to support `(ny, nz, nx)` with `[y, z, x]` access. | Matches the existing cylinder phantom layout. |
| `tissue_interaction.py` | Added `scatter_model="monte_carlo"` as an explicit future extension point. | Keeps scattering out of phase 1 without silently claiming it is implemented. |
| `material.py` | Added NIST material lookup through `xraylib`. | Converts material data into linear attenuation coefficients in `mm^-1`. |
| `cylinder_voxel.py` | Kept the local phase-1 cylinder phantom generator. | Provides a self-contained test phantom for the pipeline. |
| `energy_integrating_detector.py` | Copied the detector implementation into `ti_new`. | Ensures all runtime code required by the cascade is available in one folder. |
| `energy_integrating_detector.py` | Added `clamp_negative_signal=True`. | Physical electron counts cannot be negative; noise-generated negative readouts are stored as zero. |
| `README.md` | Added execution instructions and pipeline documentation. | Explains which file to run, how views are selected, and where outputs are stored. |
| `units_n_dimensions.md` | Added the unit and array-shape contract from source through detector. | Documents the meaning and dimensions of every important pipeline variable. |
| `LIBRARIES_USED.md` | Added the dependency list, package purposes, installation commands, and import check. | Makes the required Python libraries explicit and reproducible. |
| `ct_system_output.json` | Added a local copy of the supplied CT system configuration. | Provides per-view source positions, detector positions, gantry angles, and table positions locally. |
| `__init__.py` | Added package exports for the tissue-interaction API. | Allows the module to be imported as a package as well as run through `main.py`. |

## Output behavior

The normal command is:

```powershell
python main.py --views 0 1 2
```

Outputs are written to `photon_output/` unless `--output-dir` is supplied:

```text
view_0000_detector.csv
view_0000_detector.npy
```

The CSV detector value is produced by the detector stage after tissue
interaction. Tissue interaction returns photons per ray per energy bin; the
energy-integrating detector combines all bins into one signal per detector
pixel.

The source and detector positions are different for each configured view.
Their XZ positions represent rotation around the Y-axis, while
`table_y_mm` represents helical translation along Y.
