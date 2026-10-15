"""Focal-spot sampling and detector-synchronized cone-beam geometry."""
import numpy as np


def orthonormal_basis(central_direction):
    direction = np.asarray(central_direction, dtype=float)
    norm = np.linalg.norm(direction)
    if norm == 0:
        raise ValueError("central_direction must be non-zero")
    direction = direction / norm
    # Prefer the detector's vertical axis as the helper so that, for the
    # standard X-axis CT trajectory, the focal-spot width axis aligns with
    # the detector channel direction (rather than swapping width/height).
    helper = np.array([0.0, 1.0, 0.0])
    if abs(np.dot(helper, direction)) > 0.9:
        helper = np.array([0.0, 0.0, 1.0])
    u = np.cross(direction, helper)
    u /= np.linalg.norm(u)
    v = np.cross(direction, u)
    return u, v


def sample_focal_spot(source_position, focal_spot_size_mm, central_direction, n_samples, rng=None):
    rng = rng or np.random.default_rng()
    source_position = np.asarray(source_position, dtype=float)
    u, v = orthonormal_basis(central_direction)
    width, height = focal_spot_size_mm
    du = rng.uniform(-width / 2, width / 2, size=n_samples)
    dv = rng.uniform(-height / 2, height / 2, size=n_samples)
    return source_position + du[:, None] * u + dv[:, None] * v


def generate_pixel_directions(focal_spot_positions, detector_pixels):
    """Return unit vectors from each sampled focal point to its detector pixel."""
    origins = np.asarray(focal_spot_positions, dtype=float)
    pixels = np.asarray(detector_pixels, dtype=float)
    if origins.shape != pixels.shape or origins.ndim != 2 or origins.shape[1] != 3:
        raise ValueError("focal_spot_positions and detector_pixels must both have shape (N, 3)")
    vectors = pixels - origins
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norms <= 0):
        raise ValueError("A focal spot position coincides with a detector pixel")
    return vectors / norms


def sample_directions(central_direction, cone_angle_deg, n_samples, rng=None):
    """Legacy random cone sampler retained for standalone source validation."""
    rng = rng or np.random.default_rng()
    central_direction = np.asarray(central_direction, dtype=float)
    central_direction /= np.linalg.norm(central_direction)
    u_vec, v_vec = orthonormal_basis(central_direction)
    cone = np.deg2rad(cone_angle_deg)
    cos_theta = 1 - rng.uniform(0, 1, n_samples) * (1 - np.cos(cone))
    phi = rng.uniform(0, 2 * np.pi, n_samples)
    sin_theta = np.sqrt(np.maximum(0, 1 - cos_theta ** 2))
    directions = (sin_theta * np.cos(phi))[:, None] * u_vec + (sin_theta * np.sin(phi))[:, None] * v_vec + cos_theta[:, None] * central_direction
    return directions / np.linalg.norm(directions, axis=1, keepdims=True)
