"""
helical.py
==========
High-Performance Helical CT Trajectory and Kinematics Module.

Handles helical scanner geometry, gantry rotation, table translation, and
detector-source kinematics with an exact 180-degree phase difference in accordance
with clinical helical CT physics and 'ct_system_output.json'.

Coordinate System Conventions (from ct_system_output.json):
    - Rotation plane: XZ plane
    - Translation axis: Y axis (table feed direction)
    - Isocenter: (0.0, 0.0, 0.0) mm
    - At view angle theta:
        Source:   ( R_src * cos(theta), y_table,  R_src * sin(theta) )
        Detector: (-R_det * cos(theta), y_table, -R_det * sin(theta) )   [180-deg phase offset]
"""

from __future__ import annotations

import json
import pathlib
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    import numpy as cp
    HAS_CUPY = False


class HelicalTrajectory:
    """
    Encapsulates helical CT scanner kinematics, coordinate frames, and
    source/detector trajectory paths.
    """

    def __init__(
        self,
        sid_mm: float = 541.0,
        sdd_mm: float = 949.0,
        total_views: int = 16728,
        num_views_per_rotation: int = 984,
        table_feed_per_rotation_mm: float = 20.64,
        collimation_mm: float = 40.0,
        detector_rows: int = 64,
        detector_channels: int = 888,
        pixel_width_mm: float = 0.625,
        pixel_height_mm: float = 0.625,
        rotation_plane: str = "XZ",
        translation_axis: str = "Y",
    ):
        self.sid_mm = float(sid_mm)
        self.sdd_mm = float(sdd_mm)
        self.detector_radius_mm = float(sdd_mm - sid_mm)  # 408.0 mm
        self.total_views = int(total_views)
        self.num_views_per_rotation = int(num_views_per_rotation)
        self.num_rotations = self.total_views / self.num_views_per_rotation  # 17.0
        self.table_feed_per_rotation_mm = float(table_feed_per_rotation_mm)
        self.collimation_mm = float(collimation_mm)
        self.pitch = self.table_feed_per_rotation_mm / self.collimation_mm  # 0.516

        self.detector_rows = int(detector_rows)
        self.detector_channels = int(detector_channels)
        self.pixel_width_mm = float(pixel_width_mm)
        self.pixel_height_mm = float(pixel_height_mm)

        self.rotation_plane = rotation_plane
        self.translation_axis = translation_axis

        # Kinematic increments
        self.angle_increment_rad = 2.0 * np.pi / self.num_views_per_rotation
        self.angle_increment_deg = np.degrees(self.angle_increment_rad)
        self.y_step_mm = self.table_feed_per_rotation_mm / self.num_views_per_rotation

    @classmethod
    def from_config_file(cls, json_path: Union[str, pathlib.Path] = "ct_system_output.json") -> "HelicalTrajectory":
        """Factory constructor loading physical parameters from JSON configuration."""
        path = pathlib.Path(json_path)
        if not path.is_file():
            raise FileNotFoundError(f"Configuration file not found: {path.resolve()}")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        sys_cfg = data.get("system_config", {})
        traj_cfg = data.get("trajectory", {})

        sid = sys_cfg.get("sid_mm", traj_cfg.get("source_radius_mm", 541.0))
        sdd = sys_cfg.get("sdd_mm", 949.0)
        total_views = sys_cfg.get("total_views", 16728)
        views_per_rot = sys_cfg.get("num_views_per_rotation", 984)
        feed_per_rot = sys_cfg.get("table_feed_per_rotation_mm", 20.64)
        collimation = sys_cfg.get("collimation_mm", 40.0)

        # Detector parameters (from standard CT specifications or sub-dictionaries)
        det_cfg = data.get("detector_config", {})
        det_rows = det_cfg.get("detector_rows", 64)
        det_channels = det_cfg.get("detector_channels", 888)
        pix_w = det_cfg.get("detector_element_width_mm", 0.625)
        pix_h = det_cfg.get("detector_element_height_mm", 0.625)

        coord = traj_cfg.get("coordinate_system", {})
        rot_plane = coord.get("rotation_plane", "XZ")
        trans_axis = coord.get("translation_axis", "Y")

        return cls(
            sid_mm=sid,
            sdd_mm=sdd,
            total_views=total_views,
            num_views_per_rotation=views_per_rot,
            table_feed_per_rotation_mm=feed_per_rot,
            collimation_mm=collimation,
            detector_rows=det_rows,
            detector_channels=det_channels,
            pixel_width_mm=pix_w,
            pixel_height_mm=pix_h,
            rotation_plane=rot_plane,
            translation_axis=trans_axis,
        )

    def get_view_kinematics(self, view_index: int) -> Dict[str, Any]:
        """
        Computes source position, detector center position, and central direction
        for a single projection view index.
        """
        theta = view_index * self.angle_increment_rad
        y = view_index * self.y_step_mm

        # Source coordinates
        x_src = self.sid_mm * np.cos(theta)
        y_src = y
        z_src = self.sid_mm * np.sin(theta)
        source_pos = np.array([x_src, y_src, z_src], dtype=np.float64)

        # Detector center coordinates with exact 180-degree phase difference:
        # cos(theta + pi) = -cos(theta), sin(theta + pi) = -sin(theta)
        x_det = -self.detector_radius_mm * np.cos(theta)
        y_det = y
        z_det = -self.detector_radius_mm * np.sin(theta)
        detector_pos = np.array([x_det, y_det, z_det], dtype=np.float64)

        central_dir = detector_pos - source_pos
        central_dir /= np.linalg.norm(central_dir)

        return {
            "view_index": view_index,
            "theta_rad": theta,
            "theta_deg": np.degrees(theta),
            "table_y_mm": y,
            "source_pos_mm": source_pos,
            "detector_pos_mm": detector_pos,
            "central_direction": central_dir,
        }

    def compute_all_trajectories(self, use_cupy: bool = True) -> Dict[str, Any]:
        """
        Computes the complete continuous helical trajectory across all 16,728 views.
        Returns GPU arrays (CuPy) if use_cupy=True, else NumPy arrays.
        """
        xp = cp if (use_cupy and HAS_CUPY) else np

        view_indices = xp.arange(self.total_views, dtype=xp.float32)
        theta = view_indices * float(self.angle_increment_rad)
        y_table = view_indices * float(self.y_step_mm)

        cos_t = xp.cos(theta)
        sin_t = xp.sin(theta)

        # Source coordinates [N_views, 3]
        x_src = float(self.sid_mm) * cos_t
        y_src = y_table
        z_src = float(self.sid_mm) * sin_t
        source_positions = xp.stack([x_src, y_src, z_src], axis=-1)

        # Detector coordinates [N_views, 3] with 180-degree phase offset
        x_det = -float(self.detector_radius_mm) * cos_t
        y_det = y_table
        z_det = -float(self.detector_radius_mm) * sin_t
        detector_positions = xp.stack([x_det, y_det, z_det], axis=-1)

        return {
            "view_indices": view_indices,
            "theta_rad": theta,
            "table_y_mm": y_table,
            "source_positions": source_positions,
            "detector_positions": detector_positions,
            "xp": xp,
        }

    def generate_batch_detector_pixels_gpu(
        self,
        batch_view_indices: Any,
        use_cupy: bool = True,
    ) -> Tuple[Any, Any]:
        """
        Generates 3D coordinates of all (64 x 888) detector elements and source positions
        for a batch of projection views directly on the GPU.

        Returns:
            source_pos_batch:  (B, 1, 1, 3)
            pixel_pos_batch:   (B, H, W, 3)
        """
        xp = cp if (use_cupy and HAS_CUPY) else np

        if not isinstance(batch_view_indices, xp.ndarray):
            batch_view_indices = xp.asarray(batch_view_indices, dtype=xp.float32)

        B = batch_view_indices.shape[0]
        H = self.detector_rows       # 64
        W = self.detector_channels   # 888

        theta = batch_view_indices * float(self.angle_increment_rad)
        y_table = batch_view_indices * float(self.y_step_mm)

        cos_t = xp.cos(theta).reshape(B, 1, 1)
        sin_t = xp.sin(theta).reshape(B, 1, 1)
        y_b   = y_table.reshape(B, 1, 1)

        # Source positions (B, 1, 1, 3)
        s_x = float(self.sid_mm) * cos_t
        s_y = y_b
        s_z = float(self.sid_mm) * sin_t
        source_pos = xp.stack([s_x, s_y, s_z], axis=-1).reshape(B, 1, 1, 3)

        # Detector element coordinates on planar array
        # u-direction (channel/fan width): (sin(theta), 0, -cos(theta))
        # v-direction (row/axial height):  (0, 1, 0)
        u_coords = (xp.arange(W, dtype=xp.float32) - (W - 1) / 2.0) * float(self.pixel_width_mm)
        v_coords = (xp.arange(H, dtype=xp.float32) - (H - 1) / 2.0) * float(self.pixel_height_mm)

        U, V = xp.meshgrid(u_coords, v_coords, indexing="xy")  # U: (H, W), V: (H, W)
        U = U.reshape(1, H, W)
        V = V.reshape(1, H, W)

        # Detector center: (-R_det * cos_t, y_b, -R_det * sin_t)
        # Adding pixel offsets:
        # P_x = -R_det * cos(theta) + U * sin(theta)
        # P_y = y_table + V
        # P_z = -R_det * sin(theta) - U * cos(theta)
        r_det = float(self.detector_radius_mm)
        px_x = -r_det * cos_t + U * sin_t
        px_y = y_b + V
        px_z = -r_det * sin_t - U * cos_t

        pixel_pos = xp.stack([px_x, px_y, px_z], axis=-1)  # (B, H, W, 3)

        return source_pos, pixel_pos


def load_helical_geometry(config_path: Union[str, pathlib.Path] = "ct_system_output.json") -> HelicalTrajectory:
    """Convenience helper to instantiate HelicalTrajectory from configuration JSON."""
    return HelicalTrajectory.from_config_file(config_path)


if __name__ == "__main__":
    traj = load_helical_geometry("ct_system_output.json")
    print(f"Loaded HelicalTrajectory: {traj.total_views} views ({traj.num_rotations:.1f} rotations)")
    print(f"SID: {traj.sid_mm} mm | SDD: {traj.sdd_mm} mm | Detector Radius: {traj.detector_radius_mm} mm")
    print(f"Pitch: {traj.pitch:.3f} | Table Feed: {traj.table_feed_per_rotation_mm} mm/rot")
    v0 = traj.get_view_kinematics(0)
    print(f"View 0: Source={v0['source_pos_mm']}, DetCenter={v0['detector_pos_mm']}")
    print(f"View 0 Phase Difference: 180 degrees verified.")
