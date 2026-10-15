"""
Photon weight model.

Relates physical tube parameters (mA, exposure_time) to how much "share of
the total source output" each simulated ray carries, then lets the bowtie
filter (and optionally the heel effect) further modify those weights.
"""

import numpy as np
try:
    from .spectrum import mu_aluminum_per_mm
except ImportError:  # standalone execution from this directory
    from spectrum import mu_aluminum_per_mm


def calculate_photon_weights(mA, exposure_time, n_samples):
    """
    Physical output ~ mAs = mA * exposure_time.

    N_samples stays fixed -- it's a simulation choice, not a physical
    quantity -- so every ray gets an equal share of the total output.
    Increasing mA or exposure_time scales every weight up linearly; it
    never changes the number of simulated rays.

    No absolute photon calibration is available from the reference data,
    so total_weight is in arbitrary relative units (proportional to mAs).
    """
    mAs = mA * exposure_time
    total_weight = mAs
    weight_per_photon = total_weight / n_samples
    return np.full(n_samples, weight_per_photon)


def bowtie_thickness_mm(angle_from_center_rad, cone_angle_rad,
                         min_thickness_mm, max_thickness_mm):
    """Simple analytical bowtie: thin at center, thicker at the edges."""
    fraction = np.clip(angle_from_center_rad / cone_angle_rad, 0, 1)
    return min_thickness_mm + (max_thickness_mm - min_thickness_mm) * fraction ** 2


def apply_bowtie(directions, central_direction, energies, weights,
                  cone_angle_deg, min_thickness_mm, max_thickness_mm):
    """Reduce each ray's weight by the transmission through the bowtie."""
    central_direction = central_direction / np.linalg.norm(central_direction)
    cos_angle = np.clip(directions @ central_direction, -1.0, 1.0)
    angle_from_center = np.arccos(cos_angle)
    cone_angle_rad = np.deg2rad(cone_angle_deg)

    thickness = bowtie_thickness_mm(angle_from_center, cone_angle_rad,
                                     min_thickness_mm, max_thickness_mm)
    transmission = np.exp(-mu_aluminum_per_mm(energies) * thickness)

    new_weights = weights * transmission
    return new_weights, thickness, transmission


def apply_heel_effect(directions, central_direction, weights,
                       anode_angle_deg, heel_axis, enable=False, strength=0.3):
    """
    OPTIONAL simple heel-effect approximation: intensity tilts slightly
    along one transverse axis (lower toward the anode side, higher toward
    the cathode side). Disabled by default -- the core project does not
    depend on this.
    """
    if not enable:
        return weights

    heel_axis = heel_axis / np.linalg.norm(heel_axis)
    projection = directions @ heel_axis
    heel_factor = np.clip(1.0 + strength * projection, 0.3, 1.7)
    return weights * heel_factor


def add_poisson_noise(weights, enable=False, rng=None):
    """
    OPTIONAL source-level Poisson statistics on photon NUMBER (distinct
    from the random ENERGY sampling done elsewhere). Each ray's weight is
    treated as the expected number of real photons that ray-bundle stands
    for; if enabled, that expectation is redrawn from Poisson(weight).
    Disabled by default -- see the short note in main.py on why this is
    usually unnecessary with a fixed N_samples + weight model.
    """
    if not enable:
        return weights
    if rng is None:
        rng = np.random.default_rng()
    return rng.poisson(weights).astype(float)
