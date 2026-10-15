import numpy as np

try:
    from .material import linear_attenuation_coefficient
except ImportError:  # standalone execution from this directory
    from material import linear_attenuation_coefficient


def ray_box_intersection(
    origin,
    direction,
    box_min,
    box_max
):

    t_min = -np.inf
    t_max = np.inf

    for axis in range(3):

        if abs(direction[axis]) < 1e-12:

            if (
                origin[axis] < box_min[axis]
                or origin[axis] > box_max[axis]
            ):
                return None

            continue

        t1 = (
            box_min[axis] - origin[axis]
        ) / direction[axis]

        t2 = (
            box_max[axis] - origin[axis]
        ) / direction[axis]

        t_near = min(t1, t2)
        t_far = max(t1, t2)

        t_min = max(t_min, t_near)
        t_max = min(t_max, t_far)

        if t_min > t_max:
            return None

    return t_min, t_max


def traverse_ray(
    origin,
    direction,
    voxel_material,
    voxel_size_mm
):

    nz, ny, nx = voxel_material.shape

    volume_size = np.array([
        nx * voxel_size_mm,
        ny * voxel_size_mm,
        nz * voxel_size_mm
    ])

    volume_min = -volume_size / 2.0
    volume_max = volume_size / 2.0

    intersection = ray_box_intersection(
        origin,
        direction,
        volume_min,
        volume_max
    )

    if intersection is None:
        return {}

    t_entry, t_exit = intersection

    if t_exit < 0:
        return {}
    
    # make sure the distance travelled isn't -ve
    t = max(t_entry, 0.0)

    # point of entry of the ray into the material (refer LAST PAGE of NOTEBOOK)
    point = origin + t * direction

    # converts this physical position into a voxel inde
    voxel = np.floor(
        (point - volume_min) / voxel_size_mm
    ).astype(int)

    voxel[0] = np.clip(voxel[0], 0, nx - 1)
    voxel[1] = np.clip(voxel[1], 0, ny - 1)
    voxel[2] = np.clip(voxel[2], 0, nz - 1)

    # identifies current voxel
    ix, iy, iz = voxel

    step_x = 1 if direction[0] > 0 else -1
    step_y = 1 if direction[1] > 0 else -1
    step_z = 1 if direction[2] > 0 else -1

   ## -------- key part in handling voxel to voxel movement --------
   #  Difficult to understand without an example 
    ''' 
    (t_max_x,t_max_y,t_max_z) --> The next point along the ray where the ray will cross a voxel boundary in X, Y or Z.
    '''
    if direction[0] != 0:
        next_x = (
            volume_min[0]
            + (ix + (step_x > 0)) * voxel_size_mm
        )
        t_max_x = (next_x - origin[0]) / direction[0]
        t_delta_x = abs(voxel_size_mm / direction[0])  # how much t much increase to complete one voxel in that direction
    else:
        t_max_x = np.inf
        t_delta_x = np.inf

    if direction[1] != 0:
        next_y = (
            volume_min[1]
            + (iy + (step_y > 0)) * voxel_size_mm
        )
        t_max_y = (
            next_y - origin[1]
        ) / direction[1]
        t_delta_y = abs(
            voxel_size_mm / direction[1]
        )
    else:
        t_max_y = np.inf
        t_delta_y = np.inf

    if direction[2] != 0:
        next_z = (
            volume_min[2]
            + (iz + (step_z > 0)) * voxel_size_mm
        )
        t_max_z = (
            next_z - origin[2]
        ) / direction[2]
        t_delta_z = abs(
            voxel_size_mm / direction[2]
        )
    else:
        t_max_z = np.inf
        t_delta_z = np.inf

    material_lengths = {}

    while (
        0 <= ix < nx  # x coordinate within range
        and 0 <= iy < ny  # y coordinate withtin the range 
        and 0 <= iz < nz # z coordinate within range
        and t <= t_exit
    ):

        material_id = int(
            voxel_material[iz, iy, ix]
        )

        # find the next boundary 
        next_t = min(
            t_max_x,
            t_max_y,
            t_max_z,
            t_exit
        )

        # path travelled by the ray
        path_length = max(
            0.0,
            next_t - t
        )

        material_lengths[material_id] = (
            material_lengths.get(material_id, 0.0)
            + path_length
        )

        t = next_t

        if t >= t_exit:
            break

        if t_max_x <= t_max_y and t_max_x <= t_max_z:
            ix += step_x
            t_max_x += t_delta_x

        elif t_max_y <= t_max_x and t_max_y <= t_max_z:
            iy += step_y
            t_max_y += t_delta_y

        else:
            iz += step_z
            t_max_z += t_delta_z

    return material_lengths


def precompute_mu(
    material_ids,
    energies
):
    
    # returns a mu_table in dict format of size 2
    ## first key gives array of 'mu' of 1st material for all energies provided (1,1.5, 2, 2.5, ...)
    ## second key gives array of 'mu' of 2nd material for all energies
    mu_table = {}

    for material_id in material_ids:

        mu_table[material_id] = np.array([
            linear_attenuation_coefficient(
                material_id,
                energy
            )
            for energy in energies
        ])

    return mu_table


# def calculate_spectral_transmission(
#     material_lengths,
#     energies,
#     spectrum,
#     mu_table
# ):

#     # Initiating total attenuation exponent of each material
#     # in future we do 
  
#     atten_expo = np.zeros(
#         len(energies)
#     )

#     for material_id, path_length in material_lengths.items():
#         '''
#         for each material_id --> each material that is seen during a ray traversal
#         for path_length --> contains the total path length of a material
#                             despite encountering other materials in between
#         (see the example at the bottom to understand)
#         '''
        
        
#         # Ignore background air for Phase 1 of implementation
#         if material_id == 0:
#             continue

#         # A(E)=∑_m ​μ_m​(E) * L_m
#         #  L_m --> path length through material m, unit: mm 
#         # ​μ_m = attenuation coefficient of material m, mm⁻¹ 
#         atten_expo += (
#             mu_table[material_id]
#             * path_length
#         )

#     # -------------- Here comes Beer-Lambert -------------------
#     # out of loop of the materials
#     # calculate the I/I0 (or) N/N0 fraction 
#     transmission = np.exp(-atten_expo)

#     # Energy-integrating detector - produces an energy-integrated detector signal for each detector pixel
#     ##    ------- ROBLEM IDENTIFIED ----
#     detector_signal = np.sum(
#         spectrum
#         * energies
#         * transmission
#     )
    

#     return detector_signal


def calculate_transmitted_photons(
    material_lengths,
    energies,
    spectrum,
    mu_table
):

    attenuation = np.zeros(len(energies))

    for material_id, path_length in material_lengths.items():
        '''
         for each material_id --> each material that is seen during a ray traversal
         for path_length --> contains the total path length of a material
                             despite encountering other materials in between
         (see the example at the bottom to understand)
        '''

        # Ignore background air for Phase 1
        if material_id == 0:
            continue

        # A(E)=∑_m ​μ_m​(E) * L_m
        #  L_m --> path length through material m, unit: mm 
        # ​μ_m = attenuation coefficient of material m, mm⁻¹ 
        attenuation += mu_table[material_id]* path_length

      # -------------- Here comes Beer-Lambert -------------------
      # out of loop of the materials
      # calculate the I/I0 (or) N/N0 fraction 
    transmission = np.exp(-attenuation)

    # Energy-resolved photons reaching the detector
    transmitted_photons = spectrum * transmission

    return transmitted_photons


def tissue_interaction(data, voxel_material, voxel_size_mm):
    """Calculate energy-resolved transmitted fluence from source output only.

    Parameters
    ----------
    data : dict
        The complete dictionary returned by ``Source_team_setup.main``.
        ``data["data_matrix"]`` has columns [x, y, z, dx, dy, dz,
        sampled_energy_keV, weight], and ``data["energies_grid"]`` defines
        the energy bins consumed by scattering and the detector.
    voxel_material : ndarray
        3-D material-ID phantom.
    voxel_size_mm : float
        Isotropic voxel spacing in millimetres.

    Returns
    -------
    ndarray
        Shape (number_of_rays, number_of_energy_bins). Each source ray is
        assigned to its sampled energy bin, with the corresponding weighted
        fluence attenuated using Beer-Lambert law. Other bins for that ray are
        zero; this avoids applying the source spectrum a second time because
        the sampled ray energy already came from the filtered spectrum.
    """
    if not isinstance(data, dict):
        raise TypeError("data must be the dictionary returned by run_source_simulation")
    if "data_matrix" not in data or "energies_grid" not in data:
        raise ValueError("Source data must contain data_matrix and energies_grid")

    rays = np.asarray(data["data_matrix"], dtype=float)
    energy_grid = np.asarray(data["energies_grid"], dtype=float)
    voxel_material = np.asarray(voxel_material)
    if rays.ndim != 2 or rays.shape[1] != 8:
        raise ValueError("data['data_matrix'] must have shape (N, 8)")
    if energy_grid.ndim != 1 or energy_grid.size == 0:
        raise ValueError("data['energies_grid'] must be a non-empty 1-D array")
    if not np.all(np.isfinite(rays)) or not np.all(np.isfinite(energy_grid)):
        raise ValueError("Source data and energy grid must contain only finite values")
    if np.any(energy_grid <= 0) or np.any(np.diff(energy_grid) <= 0):
        raise ValueError("Energy grid must be positive and strictly increasing")
    if np.any(rays[:, 7] < 0) or np.any(rays[:, 6] <= 0):
        raise ValueError("Photon weights must be non-negative and energies positive")
    if voxel_material.ndim != 3:
        raise ValueError("voxel_material must be a 3-D array")
    if not np.isfinite(voxel_size_mm) or voxel_size_mm <= 0:
        raise ValueError("voxel_size_mm must be positive and finite")

    sampled_energies = rays[:, 6]
    energy_indices = np.searchsorted(energy_grid, sampled_energies)
    valid = energy_indices < energy_grid.size
    clipped_indices = np.clip(energy_indices, 0, energy_grid.size - 1)
    matches = valid & np.isclose(energy_grid[clipped_indices], sampled_energies, rtol=0.0, atol=1e-8)
    if not np.all(matches):
        bad = int(np.flatnonzero(~matches)[0])
        raise ValueError(
            f"Ray {bad} energy {sampled_energies[bad]} keV is not present in data['energies_grid']"
        )

    # Precompute material attenuation for the shared grid once, then select
    # the sampled bin for each ray. This keeps xraylib calls out of the ray loop.
    material_ids = np.unique(voxel_material)
    material_ids = material_ids[material_ids != 0]
    mu_table = precompute_mu(material_ids, energy_grid) if material_ids.size else {}
    output = np.zeros((rays.shape[0], energy_grid.size), dtype=np.float64)

    for ray_index, ray in enumerate(rays):
        origin = ray[0:3]
        direction = ray[3:6]
        direction_norm = np.linalg.norm(direction)
        if direction_norm <= 0:
            raise ValueError(f"Ray {ray_index} has a zero direction vector")
        direction = direction / direction_norm
        material_lengths = traverse_ray(origin, direction, voxel_material, float(voxel_size_mm))
        energy_index = int(energy_indices[ray_index])
        attenuation = 0.0
        for material_id, path_length in material_lengths.items():
            if material_id == 0:
                continue  # background air for the current phase-1 model
            if material_id not in mu_table:
                raise KeyError(f"No attenuation data for material ID {material_id}")
            attenuation += float(mu_table[material_id][energy_index]) * float(path_length)
        output[ray_index, energy_index] = float(ray[7]) * np.exp(-attenuation)

    if not np.all(np.isfinite(output)) or np.any(output < 0):
        raise ValueError("Transmitted fluence contains invalid values")
    return output

