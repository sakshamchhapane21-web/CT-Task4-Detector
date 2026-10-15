"""Per-view CT geometry and detector pixel coordinate generation."""
import numpy as np


def load_geometry(config):
    system = config["system_config"]
    trajectory = config["trajectory"]
    views = trajectory.get("views", [])
    if not views:
        raise ValueError("Configuration trajectory.views is empty")
    sid = float(system.get("sid_mm", trajectory.get("source_radius_mm", 0.0)))
    sdd = float(system.get("sdd_mm", sid + trajectory.get("detector_radius_mm", 0.0)))
    angles = np.array([float(v["angle_deg"]) for v in views], dtype=float)
    y_positions = np.array([float(v["table_y_mm"]) for v in views], dtype=float)
    total = min(int(system.get("total_views", len(views))), len(views))
    return {
        "SID": sid, "SDD": sdd,
        "detector_radius": float(trajectory.get("detector_radius_mm", max(0.0, sdd - sid))),
        "views_per_rotation": int(system.get("num_views_per_rotation", len(views))),
        "total_views": total,
        "view_indices": np.arange(total),
        "theta": np.deg2rad(angles[:total]),
        "y_positions": y_positions[:total],
        "trajectory_views": views[:total],
    }


def get_view_geometry(geometry, view_index):
    if not 0 <= view_index < geometry["total_views"]:
        raise IndexError(f"Invalid view index: {view_index}")
    view = geometry["trajectory_views"][view_index]
    source = np.array([view["source"]["x_mm"], view["source"]["y_mm"], view["source"]["z_mm"]], dtype=float)
    detector = np.array([view["detector"]["x_mm"], view["detector"]["y_mm"], view["detector"]["z_mm"]], dtype=float)
    direction = detector - source
    norm = np.linalg.norm(direction)
    if norm <= 0:
        raise ValueError(f"Source and detector coincide in view {view_index}")
    return source, detector, direction / norm


def detector_basis(source_pos, det_pos):
    """Return detector width, height, and normal unit vectors."""
    normal = np.asarray(source_pos, dtype=float) - np.asarray(det_pos, dtype=float)
    norm = np.linalg.norm(normal)
    if norm <= 0:
        raise ValueError("Source and detector positions must differ")
    normal /= norm
    height_axis = np.array([0.0, 1.0, 0.0])
    width_axis = np.cross(height_axis, normal)
    width_norm = np.linalg.norm(width_axis)
    if width_norm <= 1e-12:
        raise ValueError("Cannot determine detector width axis from this geometry")
    width_axis /= width_norm
    height_axis = np.cross(normal, width_axis)
    height_axis /= np.linalg.norm(height_axis)
    return width_axis, height_axis, normal


def generate_detector_pixels(det_pos, source_pos, rows, channels, w_mm, h_mm):
    """Generate detector pixel centers in row-major order (row, channel)."""
    rows, channels = int(rows), int(channels)
    if rows <= 0 or channels <= 0 or w_mm <= 0 or h_mm <= 0:
        raise ValueError("Detector dimensions and pixel pitches must be positive")
    u, v, _ = detector_basis(source_pos, det_pos)
    u_coords = (np.arange(channels) - (channels - 1) / 2.0) * float(w_mm)
    v_coords = (np.arange(rows) - (rows - 1) / 2.0) * float(h_mm)
    U, V = np.meshgrid(u_coords, v_coords, indexing="xy")
    pixels = np.asarray(det_pos, dtype=float) + U[..., None] * u + V[..., None] * v
    return pixels.reshape(rows * channels, 3)
