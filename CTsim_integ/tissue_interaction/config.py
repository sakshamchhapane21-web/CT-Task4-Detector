"""Load the master CT configuration and expose per-view geometry."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np

DEFAULT_CONFIG_PATH = Path(__file__).with_name("ct_system_output.json")


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> dict:
    with Path(path).open("r", encoding="utf-8") as file:
        config = json.load(file)
    if "system_config" not in config or "trajectory" not in config:
        raise ValueError("Configuration must contain system_config and trajectory")
    if not config["trajectory"].get("views"):
        raise ValueError("Configuration contains no trajectory views")
    if "detector_config" not in config:
        raise ValueError("Configuration must contain detector_config; no hardcoded detector settings are used")
    return config


def get_detector_config(config: dict) -> dict:
    """Return and validate detector settings directly from JSON."""
    detector = config.get("detector_config")
    if not isinstance(detector, dict):
        raise ValueError("Missing detector_config in CT JSON")
    required = (
        "detector_rows", "detector_channels", "detector_element_width_mm",
        "detector_element_height_mm", "detector_width_mm", "detector_height_mm",
        "detector_type", "detector_material", "collimation_mm",
    )
    missing = [key for key in required if key not in detector]
    if missing:
        raise ValueError(f"detector_config missing required keys: {missing}")
    result = dict(detector)
    for key in ("detector_rows", "detector_channels"):
        result[key] = int(result[key])
        if result[key] <= 0:
            raise ValueError(f"{key} must be positive")
    for key in ("detector_element_width_mm", "detector_element_height_mm", "detector_width_mm", "detector_height_mm"):
        result[key] = float(result[key])
        if result[key] <= 0:
            raise ValueError(f"{key} must be positive")
    return result


def get_total_views(config: dict) -> int:
    views = config["trajectory"]["views"]
    return min(int(config["system_config"].get("total_views", len(views))), len(views))


def get_view(config: dict, view_index: int) -> dict:
    views = config["trajectory"]["views"]
    total = get_total_views(config)
    if not 0 <= int(view_index) < total:
        raise IndexError(f"view_index must be between 0 and {total - 1}")
    view = views[int(view_index)]
    source = np.array([view["source"]["x_mm"], view["source"]["y_mm"], view["source"]["z_mm"]], dtype=float)
    detector = np.array([view["detector"]["x_mm"], view["detector"]["y_mm"], view["detector"]["z_mm"]], dtype=float)
    if not np.all(np.isfinite(source)) or not np.all(np.isfinite(detector)):
        raise ValueError(f"Non-finite source/detector geometry for view {view_index}")
    if np.linalg.norm(detector - source) <= 0:
        raise ValueError(f"Source and detector coincide for view {view_index}")
    return {
        "view_index": int(view.get("view_index", view_index)),
        "angle_deg": float(view["angle_deg"]),
        "table_y_mm": float(view["table_y_mm"]),
        "source_position_mm": source,
        "detector_position_mm": detector,
    }
