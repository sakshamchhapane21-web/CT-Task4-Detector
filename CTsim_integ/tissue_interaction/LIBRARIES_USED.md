# Libraries used by `ti_new`

The `ti_new` source-to-detector pipeline uses Python 3.13 and the following
packages:

| Package | Used by | Purpose |
|---|---|---|
| `numpy` | `tissue_interaction.py`, `detector_geometry.py`, `spectrum.py`, `cylinder_voxel.py` | Arrays, numerical calculations, ray geometry, spectra, and voxel phantoms |
| `xraylib` | `material.py` | NIST material densities and mass attenuation coefficients |
| `scipy` | `energy_integrating_detector.py` | Detector point-spread-function Gaussian filtering |
| `matplotlib` | `energy_integrating_detector.py` | Detector demo and plotting helpers included in the copied detector module |

## Installation

The packages have been installed into the project virtual environment:

```powershell
cd "C:\Users\Sridhar\OneDrive\Desktop\Sri\code\Python\Medical_img_lab\code_combine_12_10_26"
.\.venv\Scripts\python.exe -m pip install numpy scipy matplotlib xraylib
```

Run the pipeline with the same environment:

```powershell
cd "C:\Users\Sridhar\OneDrive\Desktop\Sri\code\Python\Medical_img_lab\code_combine_12_10_26\ti_new"
..\.venv\Scripts\python.exe main.py
```

The executable path from the `ti_new` directory is:

```powershell
..\.venv\Scripts\python.exe main.py
```

## Import verification

Verify all required imports with:

```powershell
..\.venv\Scripts\python.exe -c "import numpy, scipy, matplotlib, xraylib; print('All ti_new libraries imported successfully')"
```
