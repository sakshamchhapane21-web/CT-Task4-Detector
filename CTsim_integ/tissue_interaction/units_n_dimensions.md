# Units and dimensions: source to detector

This document is the data contract for the files in `ti_new`.

## Coordinate and geometry conventions

| Variable | Shape | Unit | Meaning |
|---|---:|---|---|
| `source_position_mm` | `(3,)` | `mm` | Source center `[x, y, z]` |
| `detector_position_mm` | `(3,)` | `mm` | Detector center `[x, y, z]` |
| `detector_pixels_mm` | `(n_rays, 3)` | `mm` | Detector pixel centers `[x, y, z]` |
| `central_direction` | `(3,)` | dimensionless | Unit vector from source to detector center |
| `rays` | `(n_rays, 6)` | mixed | `[origin_x_mm, origin_y_mm, origin_z_mm, direction_x, direction_y, direction_z]` |
| `ray_origin` | `(3,)` | `mm` | Starting point of one ray |
| `ray_direction` | `(3,)` | dimensionless | Unit direction vector; norm must be 1 |
| `angle_deg` | scalar | `degree` | Gantry/view angle from `ct_system_output.json` |
| `table_y_mm` | scalar | `mm` | Helical table position for the view |
| `half_cone_angle_deg` | scalar | `degree` | Maximum pixel-ray angle from the central beam direction |
| `pixel_width_mm` | scalar | `mm` | Detector channel pitch |
| `pixel_height_mm` | scalar | `mm` | Detector row pitch |

The trajectory uses the XZ plane for rotation and Y for table translation.
The source and detector positions are read from each entry of
`ct_system_output.json -> trajectory -> views`.

## Source spectrum

| Variable | Shape | Unit | Meaning |
|---|---:|---|---|
| `kvp` | scalar | `kV` | Tube potential; maximum source energy is approximately `kvp keV` |
| `energies_keV` | `(n_energy_bins,)` | `keV` | Strictly increasing energy-bin centers |
| `spectrum` | `(n_energy_bins,)` | arbitrary spectral density | Analytical filtered source shape before calibration |
| `photons_per_ray` | scalar | `photons / ray` | Explicit source fluence calibration |
| `incident_photons_per_bin` | `(n_energy_bins,)` | `photons / ray / bin` | Calibrated source fluence after spectrum normalization |

The analytical spectrum is not an absolute scanner measurement. The
`photons_per_ray` argument supplies the calibration needed to produce
quantitative detector input.

## Tissue phantom and attenuation

| Variable | Shape | Unit | Meaning |
|---|---:|---|---|
| `voxel_material` | `(ny, nz, nx)` | integer ID | Material ID at `[y, z, x]` |
| `voxel_density` | `(ny, nz, nx)` | `g/cm^3` in the helper | Optional density volume from phantom generation |
| `voxel_size_mm` | scalar | `mm` | Cubic voxel edge length |
| `material_lengths` | dictionary | `material ID -> mm` | Total path length through each material for one ray |
| `mu_over_rho` | scalar/array | `cm^2/g` | xraylib mass attenuation coefficient |
| `density` | scalar | `g/cm^3` | xraylib material density |
| `mu` | scalar/array | `mm^-1` | Linear attenuation coefficient used by Beer-Lambert |
| `attenuation` | `(n_energy_bins,)` | dimensionless | `sum(mu_material_mm^-1 * path_length_mm)` |
| `transmission` | `(n_energy_bins,)` | dimensionless | `exp(-attenuation)` |

## Tissue-interaction output

| Variable | Shape | Unit | Meaning |
|---|---:|---|---|
| `transmitted_photons` | `(n_rays, n_energy_bins)` | `photons / ray / bin` | Primary photons remaining after tissue attenuation |
| `photon_fluence_per_bin` | `(detector_rows, detector_channels, n_energy_bins)` | `photons / pixel / bin` for one-pixel-per-ray mapping | Detector input after reshape |

The current phase models primary attenuation only. Scattered photons are not
included. The public API reserves `scatter_model="monte_carlo"` for a future
Monte Carlo implementation.

## Detector output

| Variable | Shape | Unit | Meaning |
|---|---:|---|---|
| `energy_bins_keV` | `(n_energy_bins,)` | `keV` | Detector energy-bin centers |
| `signal_energy_keV` | `(detector_rows, detector_channels)` | `keV` equivalent | Energy absorbed by detector pixels before conversion |
| `conversion_gain` | scalar | `electrons / keV` | Detector conversion gain |
| `measured_signal_electrons` | `(detector_rows, detector_channels)` | `electrons` | Detector output after efficiency, blur, gain variation, dead pixels, and noise |
| `detector_signal_electrons_energy_integrated` in CSV | scalar per row | `electrons` | One detector pixel's detector output after summing all energy bins |
| `energy_bin_count` in CSV | scalar per row | count | Number of source/detector energy bins used to produce the integrated signal |
| `energy_min_keV`, `energy_max_keV` in CSV | scalar per row | `keV` | Energy range included in the integrated detector signal |

For the supplied configuration:

```text
detector_rows = 64
detector_channels = 888
n_rays = 64 * 888 = 56,832
```

If the source spectrum has `239` bins, the main array shapes are:

```text
rays                    = (56832, 6)
transmitted_photons     = (56832, 239)
photon_fluence_per_bin  = (64, 888, 239)
detector_signal          = (64, 888)
```

## CSV fields

`view_XXXX_detector.csv` contains:

```text
view_index                 integer
angle_deg                  degree
table_y_mm                 mm
detector_row               integer
detector_channel           integer
ray_origin_x_mm            mm
ray_origin_y_mm            mm
ray_origin_z_mm            mm
ray_direction_x/y/z        dimensionless
energy_bin_count            count
energy_min_keV              keV
energy_max_keV              keV
detector_signal_electrons_energy_integrated  electrons
```

The CSV signal is produced by `energy_integrating_detector.py` after
`tissue_interaction.py` has generated the transmitted energy-resolved photon
fluence. It is therefore not a tissue-interaction value and it does not
belong to one individual energy bin.
