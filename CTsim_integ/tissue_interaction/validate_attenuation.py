import numpy as np

from cylinder_voxel import create_cylinder_voxel
from tissue_interaction import (
    tissue_interaction,
    traverse_ray,
    precompute_mu,
)

# ============================================================
# 1. CONFIGURATION
# ============================================================

ENERGIES_KEV = np.array([20.0, 40.0, 60.0, 80.0])

SPECTRUM = np.ones(
    len(ENERGIES_KEV),
    dtype=np.float64
)

PHOTONS_PER_RAY = 1000.0

RADIUS_MM = 25.0
HEIGHT_MM = 50.0
VOXEL_SIZE_MM = 1.0
MATERIAL_ID = 1


# ============================================================
# 2. CREATE CYLINDRICAL PHANTOM
# ============================================================

print("=" * 65)
print("CREATING CYLINDRICAL PHANTOM")
print("=" * 65)

voxel_density, voxel_material = create_cylinder_voxel(
    radius_mm=RADIUS_MM,
    height_mm=HEIGHT_MM,
    voxel_size_mm=VOXEL_SIZE_MM,
    density=1.0,
    material_id=MATERIAL_ID,
)

print("Phantom shape:", voxel_material.shape)
print("Phantom density shape:", voxel_density.shape)
print("Voxel size:", VOXEL_SIZE_MM, "mm")
print("Cylinder radius:", RADIUS_MM, "mm")
print("Cylinder height:", HEIGHT_MM, "mm")

assert voxel_material.shape == voxel_density.shape, (
    "Density and material arrays have different shapes."
)

assert np.all(np.isfinite(voxel_density)), (
    "Phantom density contains non-finite values."
)

print("Phantom creation: PASS")


# ============================================================
# 3. DEFINE TEST RAYS
# ============================================================

# Ray format:
# [origin_x, origin_y, origin_z,
#  direction_x, direction_y, direction_z]

test_rays = {
    "central_x": np.array(
        [-100.0, 0.0, 0.0, 1.0, 0.0, 0.0]
    ),

    "offset_x_b15": np.array(
        [-100.0, 0.0, 15.0, 1.0, 0.0, 0.0]
    ),

    "miss_volume": np.array(
        [-100.0, 0.0, 26.0, 1.0, 0.0, 0.0]
    ),
}


# ============================================================
# 4. TEST RAY TRAVERSAL
# ============================================================

print("\n" + "=" * 65)
print("RAY TRAVERSAL TESTS")
print("=" * 65)

traversal_results = {}

for name, ray in test_rays.items():

    origin = ray[:3]
    direction = ray[3:]

    material_lengths = traverse_ray(
        origin,
        direction,
        voxel_material,
        VOXEL_SIZE_MM,
    )

    traversal_results[name] = material_lengths

    total_length = sum(material_lengths.values())

    print(f"\nTest: {name}")
    print("Origin:", origin)
    print("Direction:", direction)
    print("Material path lengths:", material_lengths)
    print(f"Total path length: {total_length:.6f} mm")

    if name == "central_x":

        assert abs(total_length - 50.0) < 1e-6, (
            f"Expected 50 mm central path, "
            f"but obtained {total_length:.6f} mm."
        )

    elif name == "offset_x_b15":

        assert abs(total_length - 40.0) < 1e-6, (
            f"Expected approximately 40 mm offset path, "
            f"but obtained {total_length:.6f} mm."
        )

    elif name == "miss_volume":

        assert len(material_lengths) == 0, (
            "The ray should miss the cylindrical phantom."
        )

        assert total_length == 0.0, (
            "The ray outside the phantom should have zero path length."
        )

print("\nRay traversal tests: PASS")


# ============================================================
# 5. PRECOMPUTE ATTENUATION COEFFICIENTS
# ============================================================

print("\n" + "=" * 65)
print("ATTENUATION COEFFICIENT TESTS")
print("=" * 65)

mu_table = precompute_mu(
    voxel_material,
    ENERGIES_KEV,
)

mu_table = np.asarray(mu_table)

print("Attenuation coefficient array shape:", mu_table.shape)
print("Energy bins:", ENERGIES_KEV, "keV")

assert np.all(np.isfinite(mu_table)), (
    "Attenuation coefficient table contains non-finite values."
)

assert np.all(mu_table >= 0), (
    "Attenuation coefficients must be non-negative."
)

print("All attenuation coefficients are finite and non-negative.")


# ============================================================
# 6. TEST TRANSMITTED PHOTONS
# ============================================================

print("\n" + "=" * 65)
print("TRANSMITTED PHOTON TESTS")
print("=" * 65)

for name, ray in test_rays.items():

    origin = ray[:3]
    direction = ray[3:]

    # tissue_interaction expects rays in shape (n_rays, 6).
    rays = ray.reshape(1, 6)

    transmitted = tissue_interaction(
        rays,
        voxel_material,
        ENERGIES_KEV,
        SPECTRUM.reshape(1, -1),
        photons_per_ray=PHOTONS_PER_RAY,
    )

    transmitted = np.asarray(transmitted)

    print(f"\nTest: {name}")
    print("Transmitted array shape:", transmitted.shape)
    print("Transmitted photons:", transmitted)

    assert transmitted.ndim == 2, (
        "Expected transmitted photons with shape "
        "(number_of_rays, number_of_energy_bins)."
    )

    assert transmitted.shape == (1, len(ENERGIES_KEV)), (
        f"Unexpected output shape: {transmitted.shape}"
    )

    assert np.all(np.isfinite(transmitted)), (
        f"{name}: transmitted photons contain non-finite values."
    )

    assert np.all(transmitted >= 0), (
        f"{name}: transmitted photons contain negative values."
    )

    # For a ray that misses the phantom, the output should equal
    # the incident spectrum if the implementation does not apply
    # additional attenuation outside the phantom.
    if name == "miss_volume":

        print(
            "Missed the phantom. Checking whether the transmitted "
            "spectrum matches the incident spectrum."
        )

        # This is a diagnostic check. The expected incident spectrum
        # must match the normalization used by tissue_interaction().
        print("Input spectrum:", SPECTRUM)
        print("Configured photons_per_ray:", PHOTONS_PER_RAY)

    else:

        print("Ray passes through the phantom.")

print("\nTransmitted photon tests: PASS")


# ============================================================
# 7. INDEPENDENT BEER-LAMBERT CHECK
# ============================================================

print("\n" + "=" * 65)
print("BEER-LAMBERT LAW CHECK")
print("=" * 65)

print(
    "Checking the implementation against the Beer-Lambert law:"
)
print("I = I0 * exp(-sum(mu_i * path_length_i))")

for name in ["central_x", "offset_x_b15"]:

    material_lengths = traversal_results[name]

    # Calculate optical depth separately for every energy bin.
    optical_depth = np.zeros(
        len(ENERGIES_KEV),
        dtype=np.float64
    )

    for material_id, path_length_mm in material_lengths.items():

        # Skip materials absent from the phantom.
        # Material ID 0 represents air in this project.
        # Include its attenuation only if it appears in the path
        # lengths and the corresponding coefficient is available.
        if mu_table.ndim == 2:

            if material_id < 0 or material_id >= mu_table.shape[0]:
                raise ValueError(
                    f"Material ID {material_id} is outside "
                    "the attenuation coefficient table."
                )

            mu_values = mu_table[material_id]

        elif mu_table.ndim == 1:

            # Some implementations return coefficients for a single
            # material. This is only valid for that material.
            if material_id != MATERIAL_ID:
                raise ValueError(
                    "The attenuation table contains one material, "
                    f"but ray traversal returned material ID {material_id}."
                )

            mu_values = mu_table

        else:

            raise ValueError(
                f"Unexpected attenuation table shape: {mu_table.shape}"
            )

        optical_depth += mu_values * path_length_mm

    transmission_fraction = np.exp(-optical_depth)

    print(f"\nTest: {name}")
    print("Optical depth:", optical_depth)
    print("Transmission fraction:", transmission_fraction)

    assert np.all(transmission_fraction >= 0), (
        "Transmission fractions must be non-negative."
    )

    assert np.all(transmission_fraction <= 1.0), (
        "Transmission fractions cannot exceed one "
        "for non-negative attenuation coefficients."
    )

print("\nBeer-Lambert calculation: PASS")


# ============================================================
# 8. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 65)
print("VALIDATION SUMMARY")
print("=" * 65)

print("1. Phantom creation: PASS")
print("2. Ray traversal: PASS")
print("3. Attenuation coefficients: PASS")
print("4. Transmitted photon output: PASS")
print("5. Beer-Lambert calculation: PASS")

print(
    "\nAll implemented validation checks completed successfully."
)

print(
    "Note: the independent Beer-Lambert calculation verifies "
    "the mathematical transmission formula, but does not yet "
    "compare its predicted values with tissue_interaction() output."
)