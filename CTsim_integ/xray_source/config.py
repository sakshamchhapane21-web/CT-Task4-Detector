"""Source settings loaded from the master CT system JSON."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "ti_new_13_10_2026" / "ct_system_output.json"

# Defaults are only for optional source-physics settings. Geometry and detector
# dimensions are loaded from the JSON; they are not independently hardcoded here.
DEFAULTS = {
    "al_filtration_mm": 4.0,
    "focal_spot_width_mm": 0.4,
    "focal_spot_height_mm": 0.5,
    "cone_angle_deg": 12.0,
    "bowtie_min_thickness_mm": 0.5,
    "bowtie_max_thickness_mm": 8.0,
    "enable_heel_effect": True,
    "anode_angle_deg": 7.0,
    "enable_poisson_on_weights": False,
    "e_min_kev": 10.0,
    "e_step_kev": 1.0,
}


def load_source_config(json_path: str | Path = DEFAULT_CONFIG_PATH, detector_config: dict | None = None):
    """Load source parameters and derive N_SAMPLES from JSON detector dimensions."""
    with Path(json_path).open("r", encoding="utf-8") as stream:
        root = json.load(stream)

    system = root.get("system_config", {})
    source = root.get("source_config", {})
    detector = detector_config if detector_config is not None else root.get("detector_config")
    if not isinstance(detector, dict):
        raise ValueError("ct_system_output.json must contain a detector_config object")

    rows = int(detector["detector_rows"])
    channels = int(detector["detector_channels"])
    if rows <= 0 or channels <= 0:
        raise ValueError("Detector rows and channels must be positive")
    detector_pixel_count = rows * channels
    configured_count = int(source.get("n_samples", detector_pixel_count))
    if configured_count != detector_pixel_count:
        raise ValueError(
            f"source_config.n_samples ({configured_count}) must equal detector pixel count "
            f"({rows} x {channels} = {detector_pixel_count})"
        )

    focal = (
        float(source.get("focal_spot_width_mm", DEFAULTS["focal_spot_width_mm"])),
        float(source.get("focal_spot_height_mm", DEFAULTS["focal_spot_height_mm"])),
    )
    return SimpleNamespace(
        N_SAMPLES=detector_pixel_count,
        DETECTOR_ROWS=rows,
        DETECTOR_CHANNELS=channels,
        KVP=float(system["kvp"]),
        MA=float(system["tube_current_ma"]),
        EXPOSURE_TIME=float(system["exposure_time_s"]),
        FOCAL_SPOT_SIZE=focal,
        **{key.upper(): source.get(key, value) for key, value in DEFAULTS.items()},
    )
