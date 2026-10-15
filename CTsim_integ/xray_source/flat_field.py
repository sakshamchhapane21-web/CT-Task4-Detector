"""
No-object ("flat field") projection: propagate rays from the source to the
detector plane with nothing in between, and accumulate their weights on a
simple 2D grid. This visualizes the SOURCE's spatial intensity pattern --
it is NOT a detector simulation (no detector response/material/noise).
"""

import numpy as np
from geometry import orthonormal_basis


def generate_flat_field(photon_positions, photon_directions, weights,
                         source_position, detector_position,
                         grid_size_mm=(400.0, 400.0), bins=(200, 200)):
    """
    Intersect each ray with the detector plane (through detector_position,
    perpendicular to the source->detector axis) and histogram the photon
    weights on a 2D grid.
    """
    central_direction = detector_position - source_position
    central_direction = central_direction / np.linalg.norm(central_direction)
    u, v = orthonormal_basis(central_direction)

    denom = photon_directions @ central_direction
    denom = np.where(np.abs(denom) < 1e-9, 1e-9, denom)
    t = ((detector_position - photon_positions) @ central_direction) / denom

    hit_points = photon_positions + t[:, None] * photon_directions

    rel = hit_points - detector_position
    coord_u = rel @ u
    coord_v = rel @ v

    half_w, half_h = grid_size_mm[0] / 2, grid_size_mm[1] / 2
    image, x_edges, y_edges = np.histogram2d(
        coord_u, coord_v,
        bins=bins,
        range=[[-half_w, half_w], [-half_h, half_h]],
        weights=weights,
    )
    return image, x_edges, y_edges
