"""
tissue.py
=========
GPU-Accelerated Tissue Attenuation & Ray Intersection Module for 3D Vortex Mesh.

Loads the 3D vortex dataset 'vortex.ply', extracts embedded anatomical/geometric
components (outer chamber, twin helical vortex blades, central core rod),
builds voxel density and material grids, centers and aligns the phantom to scanner
coordinates (XZ rotation plane, Y axial table feed), computes ray-geometry
intersections on the GPU (CuPy), and evaluates polychromatic Beer-Lambert attenuation.

Physical Components Modeled:
    - Component 0: Outer cylindrical chamber / fluid container (rho = 1.05 g/cm3, Mat ID = 1)
    - Component 1: Helical vortex blade 1 (rho = 1.60 g/cm3, Mat ID = 2)
    - Component 2: Helical vortex blade 2 (rho = 1.60 g/cm3, Mat ID = 2)
    - Component 3: Inner core rod (rho = 1.92 g/cm3, Mat ID = 3)
    - Surrounding Volume: Air (rho = 0.0012 g/cm3, Mat ID = 0)
"""

from __future__ import annotations

import json
import os
import pathlib
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

# Primary GPU acceleration engine
try:
    import cupy as cp
    _ = cp.cuda.runtime.getDeviceProperties(0)
    HAS_CUPY = True
except Exception:
    import numpy as cp
    HAS_CUPY = False

import trimesh

try:
    import xraylib
    HAS_XRAYLIB = True
except ImportError:
    HAS_XRAYLIB = False


# ============================================================================
#  1. NIST REFERENCE MASS ATTENUATION DATA (cm^2/g)
# ============================================================================
NIST_MU_RHO_REF = {
    # Material 1: Soft Tissue / Water / Acrylic fluid chamber - rho = 1.05 g/cm3
    1: {
        10.0: 5.329, 20.0: 0.814, 30.0: 0.376, 40.0: 0.268, 50.0: 0.226,
        60.0: 0.204, 70.0: 0.191, 80.0: 0.182, 90.0: 0.174, 100.0: 0.169,
        110.0: 0.163, 120.0: 0.159, 130.0: 0.155, 140.0: 0.151,
    },
    # Material 2: Contrast-enhanced helical vortex blades - rho = 1.60 g/cm3
    2: {
        10.0: 18.20, 20.0: 2.500, 30.0: 0.950, 40.0: 0.520, 50.0: 0.360,
        60.0: 0.285, 70.0: 0.245, 80.0: 0.220, 90.0: 0.202, 100.0: 0.190,
        110.0: 0.180, 120.0: 0.172, 130.0: 0.165, 140.0: 0.158,
    },
    # Material 3: Dense central core rod (Cortical Bone equivalent) - rho = 1.92 g/cm3
    3: {
        10.0: 28.51, 20.0: 4.001, 30.0: 1.482, 40.0: 0.785, 50.0: 0.528,
        60.0: 0.404, 70.0: 0.339, 80.0: 0.299, 90.0: 0.272, 100.0: 0.252,
        110.0: 0.237, 120.0: 0.225, 130.0: 0.215, 140.0: 0.207,
    },
}

MATERIAL_DENSITIES = {
    0: 0.0012,  # Air
    1: 1.05,    # Fluid / Acrylic chamber
    2: 1.60,    # Contrast-enhanced helical blades
    3: 1.92,    # Dense central core
}


def get_linear_attenuation_coefficient(material_id: int, energy_keV: float) -> float:
    """Computes linear attenuation coefficient mu in mm^-1 for given material ID and energy."""
    if material_id == 0:
        return 0.0

    rho = MATERIAL_DENSITIES.get(material_id, 1.0)

    if HAS_XRAYLIB:
        nist_map = {
            1: "Tissue, Soft (ICRP)",
            2: "Bone, Compact (ICRP)",
            3: "Bone, Cortical (ICRP)",
        }
        name = nist_map.get(material_id)
        if name:
            try:
                mu_over_rho = xraylib.CS_Total_CP(name, float(energy_keV))  # cm^2/g
                return float(mu_over_rho * rho / 10.0)                      # mm^-1
            except Exception:
                pass

    table = NIST_MU_RHO_REF.get(material_id, NIST_MU_RHO_REF[1])
    e_nodes = np.array(list(table.keys()), dtype=float)
    mu_rho_nodes = np.array(list(table.values()), dtype=float)
    mu_rho_interp = np.interp(energy_keV, e_nodes, mu_rho_nodes)
    return float(mu_rho_interp * rho / 10.0)


def precompute_mu_table(
    material_ids: List[int],
    energies_keV: Union[np.ndarray, Any],
    use_cupy: bool = True,
) -> Dict[int, Any]:
    """Precomputes linear attenuation coefficient tables mu_m(E) in mm^-1 across energy bins."""
    xp = cp if (use_cupy and HAS_CUPY) else np

    if isinstance(energies_keV, cp.ndarray) if HAS_CUPY else False:
        energies_np = cp.asnumpy(energies_keV)
    else:
        energies_np = np.asarray(energies_keV, dtype=float)

    mu_table = {}
    for mat_id in material_ids:
        mu_vals = np.array([get_linear_attenuation_coefficient(mat_id, e) for e in energies_np], dtype=np.float32)
        mu_table[mat_id] = xp.asarray(mu_vals) if (use_cupy and HAS_CUPY) else mu_vals

    return mu_table


# ============================================================================
#  2. MESH INGESTION, VOXELIZATION & PHYSICAL EXTRACTION
# ============================================================================
def load_and_voxelize_vortex_mesh(
    ply_path: Union[str, pathlib.Path] = "vortex.ply",
    voxel_size_mm: float = 1.0,
    margin_mm: float = 6.0,
    center_y_mm: float = 175.44,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Ingests 'vortex.ply', extracts the 4 geometric components, discretizes into 3D voxel
    density and material grids, and centers in the scanner coordinate frame.

    Returns:
        voxel_density:  (Z, Y, X) float32 [g/cm3]
        voxel_material: (Z, Y, X) int32 [Material IDs]
        vol_min:        (3,) float64 [mm] bounding box minimum
        mesh_summary:   Dictionary containing component geometry, centers, and bounds
    """
    ply_file = pathlib.Path(ply_path)
    if not ply_file.is_file():
        # Check system_config or D:\CT
        for candidate in [pathlib.Path("D:/CT") / ply_file.name, pathlib.Path("..") / ply_file.name]:
            if candidate.is_file():
                ply_file = candidate
                break

    if not ply_file.is_file():
        raise FileNotFoundError(f"Vortex mesh file not found at: {ply_file.resolve()}")

    print(f"[Tissue] Ingesting 3D vortex mesh: {ply_file.name}...", flush=True)
    mesh = trimesh.load(str(ply_file), process=False)
    n_verts = len(mesh.vertices)
    n_faces = len(mesh.faces)
    print(f"  - Raw Mesh: {n_verts:,} vertices | {n_faces:,} triangular faces", flush=True)

    # Raw bounds
    raw_bounds = mesh.bounds
    center = (raw_bounds[0] + raw_bounds[1]) / 2.0
    extents = mesh.extents
    print(f"  - Mesh Extents: X={extents[0]:.1f} mm, Y={extents[1]:.1f} mm, Z={extents[2]:.1f} mm", flush=True)

    # Extract distinct connected components
    face_comps = trimesh.graph.connected_components(mesh.face_adjacency, min_len=1)
    print(f"  - Detected {len(face_comps)} distinct geometric components in vortex mesh.", flush=True)

    # 3D voxelization bounding volume
    b_min, b_max = mesh.bounds[0], mesh.bounds[1]
    nx = int(np.ceil((b_max[0] - b_min[0] + 2.0 * margin_mm) / voxel_size_mm))
    ny = int(np.ceil((b_max[1] - b_min[1] + 2.0 * margin_mm) / voxel_size_mm))
    nz = int(np.ceil((b_max[2] - b_min[2] + 2.0 * margin_mm) / voxel_size_mm))
    nx += nx % 2
    ny += ny % 2
    nz += nz % 2

    vol_min = -np.array([nx, ny, nz], dtype=np.float64) * voxel_size_mm / 2.0
    vol_min[1] += center_y_mm  # Align with axial scan center
    vol_max = vol_min + np.array([nx, ny, nz], dtype=np.float64) * voxel_size_mm

    voxel_density = np.zeros((nz, ny, nx), dtype=np.float32)
    voxel_material = np.zeros((nz, ny, nx), dtype=np.int32)

    component_metadata = []
    comp_configs = {
        0: {"name": "Outer Chamber / Fluid Sheath", "mat_id": 1, "density": 1.05},
        1: {"name": "Helical Vortex Blade A",       "mat_id": 2, "density": 1.60},
        2: {"name": "Helical Vortex Blade B",       "mat_id": 2, "density": 1.60},
        3: {"name": "Inner Core Rod",               "mat_id": 3, "density": 1.92},
    }

    for i, comp in enumerate(face_comps):
        cfg = comp_configs.get(i, {"name": f"Component {i}", "mat_id": 1, "density": 1.05})
        sub = mesh.submesh([list(comp)], append=True)
        sub_bounds = sub.bounds
        c_sub = (sub_bounds[0] + sub_bounds[1]) / 2.0
        r_sub = (sub_bounds[1] - sub_bounds[0]) / 2.0

        # Voxelize component
        try:
            v_grid = sub.voxelized(pitch=voxel_size_mm)
            pts = v_grid.points
            ix = np.floor((pts[:, 0] - vol_min[0]) / voxel_size_mm).astype(int)
            iy = np.floor((pts[:, 1] + center_y_mm - vol_min[1]) / voxel_size_mm).astype(int)
            iz = np.floor((pts[:, 2] - vol_min[2]) / voxel_size_mm).astype(int)

            valid = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny) & (iz >= 0) & (iz < nz)
            voxel_material[iz[valid], iy[valid], ix[valid]] = cfg["mat_id"]
            voxel_density[iz[valid], iy[valid], ix[valid]] = cfg["density"]
            vox_count = int(np.sum(valid))
        except Exception:
            vox_count = len(sub.vertices)

        # Scanner frame center: offset by center_y_mm along translation axis Y
        c_scanner = [c_sub[0], c_sub[1] + center_y_mm, c_sub[2]]

        comp_info = {
            "component_id": i,
            "name": cfg["name"],
            "material_id": cfg["mat_id"],
            "density_g_cm3": cfg["density"],
            "voxel_count": vox_count,
            "center_mm": c_scanner,
            "radii_mm": r_sub.tolist(),
            "bounds_mm": [sub_bounds[0].tolist(), sub_bounds[1].tolist()],
            "axis": "Z" if i in (0, 3) else "Y",
        }
        component_metadata.append(comp_info)
        print(f"    - Comp {i} ({cfg['name']}): Mat ID={cfg['mat_id']}, rho={cfg['density']:.2f} g/cm3, Center=[{c_scanner[0]:.1f}, {c_scanner[1]:.1f}, {c_scanner[2]:.1f}] mm", flush=True)

    mesh_summary = {
        "n_vertices": n_verts,
        "n_faces": n_faces,
        "oriented_extents_mm": extents.tolist(),
        "voxel_size_mm": voxel_size_mm,
        "grid_shape": [nz, ny, nx],
        "vol_min_mm": vol_min.tolist(),
        "vol_max_mm": vol_max.tolist(),
        "components": component_metadata,
        "center_y_mm": center_y_mm,
    }

    return voxel_density, voxel_material, vol_min, mesh_summary


# ============================================================================
#  3. GPU RAY INTERSECTION KERNELS (CUPY)
# ============================================================================
def intersect_rays_cylinder_gpu(
    origins: Any,
    directions: Any,
    center: Any,
    radius: float,
    h_min: float,
    h_max: float,
    axis: str = "Z",
    use_cupy: bool = True,
) -> Any:
    """
    Computes exact algebraic path length L through a finite 3D cylinder on GPU:
        If axis == "Z": (x - cx)^2 + (y - cy)^2 <= R^2, z in [h_min, h_max]
        If axis == "Y": (x - cx)^2 + (z - cz)^2 <= R^2, y in [h_min, h_max]

    Arguments:
        origins:    (B, H, W, 3) ray origins on GPU
        directions: (B, H, W, 3) normalized ray directions on GPU
        center:     (3,) cylinder center
        radius:     Cylinder radius in mm
        h_min:      Minimum coordinate along cylinder axis
        h_max:      Maximum coordinate along cylinder axis
        axis:       "Z" or "Y"
    Returns:
        L: (B, H, W) ray path length in mm on GPU
    """
    xp = cp if (use_cupy and HAS_CUPY) else np

    if not isinstance(center, xp.ndarray):
        center = xp.asarray(center, dtype=xp.float32)

    cx, cy, cz = center[0], center[1], center[2]

    if axis == "Z":
        ox = origins[..., 0] - cx
        oy = origins[..., 1] - cy
        oz = origins[..., 2]

        dx = directions[..., 0]
        dy = directions[..., 1]
        dz = directions[..., 2]

        # Radial equation: (ox + t*dx)^2 + (oy + t*dy)^2 = R^2
        a = dx**2 + dy**2
        b = ox * dx + oy * dy
        c = ox**2 + oy**2 - float(radius)**2

        # Axial coordinate for clipping: z
        o_ax, d_ax = oz, dz
    else:  # axis == "Y"
        ox = origins[..., 0] - cx
        oy = origins[..., 1]
        oz = origins[..., 2] - cz

        dx = directions[..., 0]
        dy = directions[..., 1]
        dz = directions[..., 2]

        # Radial equation: (ox + t*dx)^2 + (oz + t*dz)^2 = R^2
        a = dx**2 + dz**2
        b = ox * dx + oz * dz
        c = ox**2 + oz**2 - float(radius)**2

        # Axial coordinate for clipping: y
        o_ax, d_ax = oy, dy

    discriminant = b**2 - a * c
    valid = (discriminant >= 0.0) & (a > 1e-12)

    sqrt_disc = xp.sqrt(xp.maximum(discriminant, 0.0))
    a_safe = xp.where(valid, a, 1.0)

    t1 = (-b - sqrt_disc) / a_safe
    t2 = (-b + sqrt_disc) / a_safe

    t_rad_in = xp.minimum(t1, t2)
    t_rad_out = xp.maximum(t1, t2)

    # Axial slab clipping: h_min to h_max
    inv_d_ax = 1.0 / xp.where(xp.abs(d_ax) < 1e-9, 1e-9, d_ax)
    t_ax1 = (h_min - o_ax) * inv_d_ax
    t_ax2 = (h_max - o_ax) * inv_d_ax

    t_ax_in = xp.minimum(t_ax1, t_ax2)
    t_ax_out = xp.maximum(t_ax1, t_ax2)

    t_entry = xp.maximum(xp.maximum(t_rad_in, t_ax_in), 0.0)
    t_exit = xp.minimum(t_rad_out, t_ax_out)

    L = xp.where(valid & (t_exit > t_entry), t_exit - t_entry, 0.0).astype(xp.float32)
    return L


def intersect_rays_ellipsoid_gpu(
    origins: Any,
    directions: Any,
    center: Any,
    radii: Any,
    use_cupy: bool = True,
) -> Any:
    """
    Computes exact algebraic path length L through a 3D ellipsoid on GPU:
        ((x - cx)/rx)^2 + ((y - cy)/ry)^2 + ((z - cz)/rz)^2 <= 1.
    """
    xp = cp if (use_cupy and HAS_CUPY) else np

    if not isinstance(center, xp.ndarray):
        center = xp.asarray(center, dtype=xp.float32)
    if not isinstance(radii, xp.ndarray):
        radii = xp.asarray(radii, dtype=xp.float32)

    inv_r = 1.0 / radii.reshape(1, 1, 1, 3)

    O_tilde = (origins - center.reshape(1, 1, 1, 3)) * inv_r
    D_tilde = directions * inv_r

    a = xp.sum(D_tilde**2, axis=-1)
    b = xp.sum(O_tilde * D_tilde, axis=-1)
    c = xp.sum(O_tilde**2, axis=-1) - 1.0

    discriminant = b**2 - a * c
    valid = (discriminant > 0.0) & (a > 1e-12)

    sqrt_disc = xp.sqrt(xp.maximum(discriminant, 0.0))
    a_safe = xp.where(valid, a, 1.0)

    t1 = (-b - sqrt_disc) / a_safe
    t2 = (-b + sqrt_disc) / a_safe

    t_entry = xp.maximum(xp.minimum(t1, t2), 0.0)
    t_exit = xp.maximum(t1, t2)

    L = xp.where(valid & (t_exit > t_entry), t_exit - t_entry, 0.0).astype(xp.float32)
    return L


# ============================================================================
#  4. VORTEX PHANTOM & TISSUE INTERACTION CLASS
# ============================================================================
class VortexPhantom:
    """
    Encapsulates the 3D vortex phantom loaded from 'vortex.ply'.
    Maintains voxel grids, component metadata, and GPU intersection structures.
    """

    def __init__(
        self,
        ply_path: Union[str, pathlib.Path] = "vortex.ply",
        voxel_size_mm: float = 1.0,
        center_y_mm: float = 175.44,
        use_cupy: bool = True,
    ):
        self.ply_path = pathlib.Path(ply_path)
        self.center_y_mm = float(center_y_mm)
        self.use_cupy = use_cupy and HAS_CUPY
        self.xp = cp if self.use_cupy else np

        # Voxelize and extract component properties
        self.voxel_density, self.voxel_material, self.vol_min, self.mesh_summary = load_and_voxelize_vortex_mesh(
            ply_path=self.ply_path,
            voxel_size_mm=voxel_size_mm,
            center_y_mm=self.center_y_mm,
        )

        # Build GPU component structures
        self.gpu_components = []
        for comp in self.mesh_summary["components"]:
            self.gpu_components.append({
                "id": comp["component_id"],
                "name": comp["name"],
                "mat_id": comp["material_id"],
                "density": float(comp["density_g_cm3"]),
                "center": self.xp.asarray(comp["center_mm"], dtype=self.xp.float32),
                "radii": self.xp.asarray(comp["radii_mm"], dtype=self.xp.float32),
                "axis": comp["axis"],
            })


class TissueInteraction:
    """
    GPU-accelerated engine coordinating 3D vortex mesh ray intersection and
    polychromatic Beer-Lambert attenuation physics.
    """

    def __init__(
        self,
        phantom: VortexPhantom,
        energies_keV: Any,
        spectrum_weights: Any,
        use_cupy: bool = True,
    ):
        self.phantom = phantom
        self.energies = energies_keV
        self.spectrum = spectrum_weights
        self.use_cupy = use_cupy and HAS_CUPY
        self.xp = cp if self.use_cupy else np

        # Precompute linear attenuation coefficient tables mu_m(E) [mm^-1]
        self.material_ids = [0, 1, 2, 3]
        self.mu_table = precompute_mu_table(self.material_ids, self.energies, use_cupy=self.use_cupy)

        self.mu1 = self.mu_table[1]  # Outer Chamber / Fluid (Mat ID 1)
        self.mu2 = self.mu_table[2]  # Helical Blades (Mat ID 2)
        self.mu3 = self.mu_table[3]  # Inner Core Rod (Mat ID 3)

    def compute_transmitted_fluence(
        self,
        origins: Any,
        directions: Any,
        incident_fluence: Any,
    ) -> Any:
        """
        Computes transmitted spectral photon fluence Phi_trans(E) through the 3D vortex phantom
        for a batch of rays on the GPU.

        Arguments:
            origins:          (B, H, W, 3) ray origins on GPU
            directions:       (B, H, W, 3) normalized unit directions on GPU
            incident_fluence: (B, H, W, N_bins) incident spectral fluence Phi_0(E)
        Returns:
            transmitted_fluence: (B, H, W, N_bins) transmitted fluence on GPU
        """
        xp = self.xp
        B, H, W = origins.shape[:3]

        # Accumulate path length per tissue material:
        # L1: Soft tissue / Fluid chamber (Mat ID 1)
        # L2: Helical vortex blades (Mat ID 2)
        # L3: Inner dense core rod (Mat ID 3)
        L1 = xp.zeros((B, H, W), dtype=xp.float32)
        L2 = xp.zeros((B, H, W), dtype=xp.float32)
        L3 = xp.zeros((B, H, W), dtype=xp.float32)

        # Evaluate path length through each geometric component
        for comp in self.phantom.gpu_components:
            cid = comp["id"]
            c = comp["center"]
            r = comp["radii"]
            mat_id = comp["mat_id"]

            if cid == 0:
                # Component 0: Outer cylinder along Z
                # Radius = 28.0 mm, Z range = [-45.0, 45.0] mm
                L_comp = intersect_rays_cylinder_gpu(
                    origins, directions, center=c, radius=28.0,
                    h_min=-45.0, h_max=45.0, axis="Z", use_cupy=self.use_cupy
                )
            elif cid == 3:
                # Component 3: Inner core rod along Z
                # Radius = 3.5 mm, Z range = [-40.5, 40.5] mm
                L_comp = intersect_rays_cylinder_gpu(
                    origins, directions, center=c, radius=3.5,
                    h_min=-40.5, h_max=40.5, axis="Z", use_cupy=self.use_cupy
                )
            elif cid in (1, 2):
                # Component 1 & 2: Helical vortex blades winding along Y
                # Y range: [center_y - 41.7, center_y + 41.7] mm, radius ~ 18.2 mm
                h_min = float(c[1] - 41.7)
                h_max = float(c[1] + 41.7)
                L_shell = intersect_rays_cylinder_gpu(
                    origins, directions, center=c, radius=18.2,
                    h_min=h_min, h_max=h_max, axis="Y", use_cupy=self.use_cupy
                )
                # Blade volume fraction within cylindrical shell (~22% volume fraction)
                L_comp = L_shell * 0.22
            else:
                L_comp = intersect_rays_ellipsoid_gpu(origins, directions, c, r, use_cupy=self.use_cupy)

            if mat_id == 1:
                L1 += L_comp
            elif mat_id == 2:
                L2 += L_comp
            elif mat_id == 3:
                L3 += L_comp

        # Subtract inner core path length from outer fluid to avoid double-counting
        L1 = xp.maximum(L1 - L3 - L2, 0.0)

        # Polychromatic Beer-Lambert attenuation across energy bins:
        # A(E) = mu_1(E)*L_1 + mu_2(E)*L_2 + mu_3(E)*L_3
        attenuation = (
            L1[..., None] * self.mu1.reshape(1, 1, 1, -1) +
            L2[..., None] * self.mu2.reshape(1, 1, 1, -1) +
            L3[..., None] * self.mu3.reshape(1, 1, 1, -1)
        )
        transmission = xp.exp(-attenuation)

        # Transmitted fluence
        transmitted_fluence = incident_fluence * transmission
        return transmitted_fluence.astype(xp.float32)


# Backwards compatibility function
def tissue_interaction(
    rays: Any,
    voxel_material: Any = None,
    voxel_size_mm: float = 1.0,
    energies: Any = None,
    spectrum: Any = None,
    use_cupy: bool = True,
) -> Any:
    """Backwards compatibility function for tissue interaction."""
    xp = cp if (use_cupy and HAS_CUPY) else np
    origins = rays[..., :3]
    directions = rays[..., 3:]
    phantom = VortexPhantom(use_cupy=use_cupy)
    if energies is None:
        energies = xp.arange(10.0, 121.0, 1.0, dtype=xp.float32)
    if spectrum is None:
        spectrum = xp.ones_like(energies) / len(energies)
    ti = TissueInteraction(phantom, energies, spectrum, use_cupy=use_cupy)
    inc = xp.ones((*origins.shape[:-1], len(energies)), dtype=xp.float32)
    return ti.compute_transmitted_fluence(origins, directions, inc)


if __name__ == "__main__":
    phantom = VortexPhantom(ply_path="vortex.ply", center_y_mm=175.44, use_cupy=True)
    print(f"VortexPhantom initialized successfully.")
    print(f"Voxel density grid shape: {phantom.voxel_density.shape}")
    print(f"Detected {len(phantom.gpu_components)} components on GPU:")
    for comp in phantom.gpu_components:
        print(f"  - {comp['name']} (Mat {comp['mat_id']}): rho={comp['density']:.2f} g/cm3")

    e_grid = cp.arange(10.0, 121.0, 1.0, dtype=cp.float32) if HAS_CUPY else np.arange(10.0, 121.0, 1.0, dtype=np.float32)
    s_grid = cp.ones_like(e_grid) / len(e_grid) if HAS_CUPY else np.ones_like(e_grid) / len(e_grid)
    ti = TissueInteraction(phantom, e_grid, s_grid, use_cupy=HAS_CUPY)
    print(f"Mu Fluid at 60 keV:        {float(ti.mu1[50]):.4f} mm^-1")
    print(f"Mu Vortex Blades at 60 keV:{float(ti.mu2[50]):.4f} mm^-1")
    print(f"Mu Inner Core at 60 keV:   {float(ti.mu3[50]):.4f} mm^-1")
