import numpy as np

from cylinder_voxel import create_cylinder_voxel
from tissue_interaction import traverse_ray

_, material = create_cylinder_voxel(
    radius_mm=25.0,
    height_mm=50.0,
    voxel_size_mm=1.0,
    material_id=1,
)

rays = {
    "Center": ([-100, 0, 0], [1, 0, 0]),
    "Offset Z=15": ([-100, 0, 15], [1, 0, 0]),
    "Outside Z=26": ([-100, 0, 26], [1, 0, 0]),
    "Offset Z=15, Y=10": ([-100, 10, 15], [1, 0, 0]),
}

for name, (origin, direction) in rays.items():
    lengths = traverse_ray(
        np.array(origin, dtype=float),
        np.array(direction, dtype=float),
        material,
        1.0,
    )

    tissue_length = lengths.get(1, 0.0)

    print(f"{name}:")
    print(f"  Tissue path: {tissue_length:.2f} mm")
    print(f"  All material paths: {lengths}")