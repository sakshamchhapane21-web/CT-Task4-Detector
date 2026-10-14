"""
tissue.py
=========
GPU-Accelerated Tissue Attenuation & Ray Intersection Module for 3D Anatomical Chest Mesh.

Loads the specific surface mesh 'chest_organs_attenuation.ply', extracts embedded physical
properties (density_g_cm3, mass_atten_cm2_g, linear_atten_cm_inv), centers and aligns the
mesh to scanner coordinates (XZ rotation plane, Y axial table feed), computes ray-geometry
intersections on the GPU (CuPy), and evaluates polychromatic Beer-Lambert attenuation.

Anatomical Organs Modeled:
    - Organ 0: Left Lung (rho = 0.26 g/cm3, ICRP Lung, Mat ID = 3)
    - Organ 1: Right Lung (rho = 0.26 g/cm3, ICRP Lung, Mat ID = 3)
    - Organ 2: Heart / Soft Tissue (rho = 1.06 g/cm3, ICRP Soft Tissue, Mat ID = 1)
    - Organ 3: Spine / Cortical Bone (rho = 1.92 g/cm3, ICRP Cortical Bone, Mat ID = 2)
    - Surrounding Volume: Air (rho = 0.0012 g/cm3, Mat ID = 0)
"""

from __future__ import annotations

import json
import os
import pathlib
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
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
    # Material 1: Soft Tissue / Heart (ICRP) - rho = 1.06 g/cm3
    1: {
        10.0: 5.329, 20.0: 0.814, 30.0: 0.376, 40.0: 0.268, 50.0: 0.226,
        60.0: 0.204, 70.0: 0.191, 80.0: 0.182, 90.0: 0.174, 100.0: 0.169,
        110.0: 0.163, 120.0: 0.159, 130.0: 0.155, 140.0: 0.151,
    },
    # Material 2: Cortical Bone / Spine (ICRP) - rho = 1.92 g/cm3
    2: {
        10.0: 28.51, 20.0: 4.001, 30.0: 1.482, 40.0: 0.785, 50.0: 0.528,
        60.0: 0.404, 70.0: 0.339, 80.0: 0.299, 90.0: 0.272, 100.0: 0.252,
        110.0: 0.237, 120.0: 0.225, 130.0: 0.215, 140.0: 0.207,
    },
    # Material 3: Lung Tissue (ICRP) - rho = 0.26 g/cm3
    3: {
        10.0: 5.250, 20.0: 0.805, 30.0: 0.370, 40.0: 0.265, 50.0: 0.224,
        60.0: 0.202, 70.0: 0.189, 80.0: 0.180, 90.0: 0.173, 100.0: 0.167,
        110.0: 0.162, 120.0: 0.157, 130.0: 0.153, 140.0: 0.150,
    },
}

MATERIAL_DENSITIES = {
    0: 0.0012,  # Air
    1: 1.06,    # Soft tissue / heart
    2: 1.92,    # Cortical bone / spine
    3: 0.26,    # Lung tissue
}


def get_linear_attenuation_coefficient(material_id: int, energy_keV: float) -> float:
    """Computes linear attenuation coefficient mu in mm^-1 for given material ID and energy."""
    if material_id == 0:
        return 0.0

    rho = MATERIAL_DENSITIES.get(material_id, 1.0)

    if HAS_XRAYLIB:
        nist_map = {
            1: "Tissue, Soft (ICRP)",
            2: "Bone, Cortical (ICRP)",
            3: "Lung (ICRP)",
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
def load_and_voxelize_chest_mesh(
    ply_path: Union[str, pathlib.Path] = "chest_organs_attenuation.ply",
    voxel_size_mm: float = 1.0,
    margin_mm: float = 6.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Ingests 'chest_organs_attenuation.ply', extracts embedded voxel/attenuation attributes,
    centers at scanner isocenter (0,0,0), aligns axes (Z -> Y axial feed), and discretizes
    into 3D voxel density and material grids.

    Returns:
        voxel_density:  (Z, Y, X) float32 [g/cm3]
        voxel_material: (Z, Y, X) int32 [Material IDs]
        vol_min:        (3,) float64 [mm] bounding box minimum
        mesh_summary:   Dictionary containing organ geometry, centers, and bounds
    """
    ply_file = pathlib.Path(ply_path)
    if not ply_file.is_file():
        raise FileNotFoundError(f"Mesh file not found at: {ply_file.resolve()}")

    print(f"[Tissue] Ingesting 3D anatomical chest mesh: {ply_file.name}...")
    mesh = trimesh.load(str(ply_file), process=False)
    n_verts = len(mesh.vertices)
    n_faces = len(mesh.faces)
    print(f"  - Raw Mesh: {n_verts:,} vertices | {n_faces:,} triangular faces")

    # 1. Centering at scanner isocenter (0, 0, 0) mm
    raw_bounds = mesh.bounds
    center = (raw_bounds[0] + raw_bounds[1]) / 2.0
    mesh.vertices -= center

    # 2. Re-orient to scanner frame:
    # Original mesh: X=Lateral, Y=Anterior-Posterior, Z=Longitudinal (cranial-caudal).
    # Scanner frame: XZ=In-plane gantry rotation, Y=Table axial feed.
    # Apply +90-deg rotation around X so Z maps to Y (axial), and -Y maps to Z.
    R = trimesh.transformations.rotation_matrix(np.pi / 2.0, [1.0, 0.0, 0.0])
    mesh.apply_transform(R)
    extents = mesh.extents
    print(f"  - Oriented Extents in Scanner Frame: X={extents[0]:.1f} mm (Lateral), Y={extents[1]:.1f} mm (Axial Feed), Z={extents[2]:.1f} mm (AP)")

    # 3. Extract embedded physical attributes
    dens_raw = None
    if hasattr(mesh, "metadata") and "_ply_raw" in mesh.metadata:
        vdata = mesh.metadata["_ply_raw"]["vertex"]["data"]
        if "density_g_cm3" in vdata:
            dens_raw = np.asarray(vdata["density_g_cm3"]).flatten()
            print(f"  - Extracted embedded 'density_g_cm3' property across {len(dens_raw)} vertices.")
        if "mass_atten_cm2_g" in vdata:
            mass_att = np.asarray(vdata["mass_atten_cm2_g"]).flatten()
            print(f"  - Extracted embedded 'mass_atten_cm2_g' (range: [{mass_att.min():.4f}, {mass_att.max():.4f}] cm^2/g).")
        if "linear_atten_cm_inv" in vdata:
            lin_att = np.asarray(vdata["linear_atten_cm_inv"]).flatten()
            print(f"  - Extracted embedded 'linear_atten_cm_inv' (range: [{lin_att.min():.4f}, {lin_att.max():.4f}] cm^-1).")

    # 4. Extract distinct anatomical organ components
    face_comps = trimesh.graph.connected_components(mesh.face_adjacency, min_len=1)
    print(f"  - Detected {len(face_comps)} distinct closed anatomical organ components.")

    # 5. Construct 3D voxelization bounding volume
    b_min, b_max = mesh.bounds[0], mesh.bounds[1]
    nx = int(np.ceil((b_max[0] - b_min[0] + 2.0 * margin_mm) / voxel_size_mm))
    ny = int(np.ceil((b_max[1] - b_min[1] + 2.0 * margin_mm) / voxel_size_mm))
    nz = int(np.ceil((b_max[2] - b_min[2] + 2.0 * margin_mm) / voxel_size_mm))
    nx += nx % 2
    ny += ny % 2
    nz += nz % 2

    vol_min = -np.array([nx, ny, nz], dtype=np.float64) * voxel_size_mm / 2.0
    vol_max = np.array([nx, ny, nz], dtype=np.float64) * voxel_size_mm / 2.0

    voxel_density = np.zeros((nz, ny, nx), dtype=np.float32)
    voxel_material = np.zeros((nz, ny, nx), dtype=np.int32)

    organ_metadata = []
    # Density to material lookup
    mat_lookup = [
        (0.26, 3, 0.26, "Lung Tissue (ICRP)"),
        (1.06, 1, 1.06, "Soft Tissue / Heart (ICRP)"),
        (1.92, 2, 1.92, "Cortical Bone / Spine (ICRP)"),
    ]

    for i, comp in enumerate(face_comps):
        sub = mesh.submesh([list(comp)], append=True)
        face_list = list(comp)
        v_idx = mesh.faces[face_list[0]][0]

        rho_sample = float(dens_raw[v_idx]) if dens_raw is not None else 1.0

        mat_id, rho, organ_name = 1, 1.0, "Standard Soft Tissue"
        for target_rho, m_id, r_val, name in mat_lookup:
            if abs(rho_sample - target_rho) < 0.25:
                mat_id, rho, organ_name = m_id, r_val, name
                break

        # Discrete voxel filling
        v_grid = sub.voxelized(pitch=voxel_size_mm).fill()
        pts = v_grid.points

        ix = np.floor((pts[:, 0] - vol_min[0]) / voxel_size_mm).astype(int)
        iy = np.floor((pts[:, 1] - vol_min[1]) / voxel_size_mm).astype(int)
        iz = np.floor((pts[:, 2] - vol_min[2]) / voxel_size_mm).astype(int)

        valid = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny) & (iz >= 0) & (iz < nz)
        voxel_material[iz[valid], iy[valid], ix[valid]] = mat_id
        voxel_density[iz[valid], iy[valid], ix[valid]] = rho

        sub_bounds = sub.bounds
        c_sub = (sub_bounds[0] + sub_bounds[1]) / 2.0
        r_sub = (sub_bounds[1] - sub_bounds[0]) / 2.0

        organ_info = {
            "component_id": i,
            "name": organ_name,
            "material_id": mat_id,
            "density_g_cm3": rho,
            "voxel_count": int(np.sum(valid)),
            "center_mm": c_sub.tolist(),
            "radii_mm": r_sub.tolist(),
            "bounds_mm": [sub_bounds[0].tolist(), sub_bounds[1].tolist()],
        }
        organ_metadata.append(organ_info)
        print(f"    - Organ {i} ({organ_name}): Mat ID={mat_id}, rho={rho:.2f} g/cm3, voxels={organ_info['voxel_count']:,}, Center=[{c_sub[0]:.1f}, {c_sub[1]:.1f}, {c_sub[2]:.1f}] mm")

    mesh_summary = {
        "n_vertices": n_verts,
        "n_faces": n_faces,
        "oriented_extents_mm": extents.tolist(),
        "voxel_size_mm": voxel_size_mm,
        "grid_shape": [nz, ny, nx],
        "vol_min_mm": vol_min.tolist(),
        "vol_max_mm": vol_max.tolist(),
        "organs": organ_metadata,
    }

    return voxel_density, voxel_material, vol_min, mesh_summary


# ============================================================================
#  3. GPU RAY-ORGAN INTERSECTION KERNEL (CUPY)
# ============================================================================
def intersect_rays_ellipsoid_gpu(
    origins: Any,
    directions: Any,
    center: Any,
    radii: Any,
    use_cupy: bool = True,
) -> Any:
    """
    Computes exact algebraic path length L through a 3D anatomical organ ellipsoid on GPU:
        ((x - cx)/rx)^2 + ((y - cy)/ry)^2 + ((z - cz)/rz)^2 <= 1.

    Arguments:
        origins:    (B, H, W, 3) ray origin coordinates on GPU
        directions: (B, H, W, 3) normalized ray unit directions on GPU
        center:     (3,) ellipsoid center [cx, cy, cz]
        radii:      (3,) semi-axes [rx, ry, rz]
    Returns:
        L:          (B, H, W) ray path length in mm on GPU
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
    t_exit  = xp.maximum(t1, t2)

    L = xp.where(valid & (t_exit > t_entry), t_exit - t_entry, 0.0).astype(xp.float32)
    return L


# ============================================================================
#  4. CHEST PHANTOM & TISSUE INTERACTION CLASS
# ============================================================================
class ChestPhantom:
    """
    Encapsulates the 3D anatomical chest phantom loaded from 'chest_organs_attenuation.ply'.
    Maintains voxel grids, anatomical organ metadata, and GPU ellipsoid parameters.
    """

    def __init__(
        self,
        ply_path: Union[str, pathlib.Path] = "chest_organs_attenuation.ply",
        voxel_size_mm: float = 1.0,
        use_cupy: bool = True,
    ):
        self.ply_path = pathlib.Path(ply_path)
        self.use_cupy = use_cupy and HAS_CUPY
        self.xp = cp if self.use_cupy else np

        # Voxelize and extract organ properties
        self.voxel_density, self.voxel_material, self.vol_min, self.mesh_summary = load_and_voxelize_chest_mesh(
            ply_path=self.ply_path,
            voxel_size_mm=voxel_size_mm,
        )

        # Build GPU organ structures
        self.gpu_organs = []
        for org in self.mesh_summary["organs"]:
            self.gpu_organs.append({
                "name": org["name"],
                "mat_id": org["material_id"],
                "center": self.xp.asarray(org["center_mm"], dtype=self.xp.float32),
                "radii": self.xp.asarray(org["radii_mm"], dtype=self.xp.float32),
                "density": float(org["density_g_cm3"]),
            })


class TissueInteraction:
    """
    GPU-accelerated engine coordinating 3D chest mesh ray intersection and
    polychromatic Beer-Lambert attenuation physics.
    """

    def __init__(
        self,
        phantom: ChestPhantom,
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

        self.mu1 = self.mu_table[1]  # Soft tissue / Heart
        self.mu2 = self.mu_table[2]  # Cortical bone / Spine
        self.mu3 = self.mu_table[3]  # Lung tissue (bilateral lungs)

    def compute_transmitted_fluence(
        self,
        origins: Any,
        directions: Any,
        incident_fluence: Any,
    ) -> Any:
        """
        Computes transmitted spectral photon fluence Phi_trans(E) through the 3D chest organs
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
        # L1: Soft tissue (Heart)
        # L2: Cortical bone (Spine)
        # L3: Lung tissue (Left + Right lungs)
        L1 = xp.zeros((B, H, W), dtype=xp.float32)
        L2 = xp.zeros((B, H, W), dtype=xp.float32)
        L3 = xp.zeros((B, H, W), dtype=xp.float32)

        for org in self.phantom.gpu_organs:
            L_org = intersect_rays_ellipsoid_gpu(
                origins, directions, org["center"], org["radii"], use_cupy=self.use_cupy
            )
            if org["mat_id"] == 1:
                L1 += L_org
            elif org["mat_id"] == 2:
                L2 += L_org
            elif org["mat_id"] == 3:
                L3 += L_org

        # Polychromatic Beer-Lambert attenuation across energy bins
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
    xp = cp if (use_cupy and HAS_CUPY) else np
    origins = rays[..., :3]
    directions = rays[..., 3:]
    phantom = ChestPhantom(use_cupy=use_cupy)
    ti = TissueInteraction(phantom, energies, spectrum, use_cupy=use_cupy)
    inc = xp.ones((*origins.shape[:-1], len(energies)), dtype=xp.float32)
    return ti.compute_transmitted_fluence(origins, directions, inc)


if __name__ == "__main__":
    phantom = ChestPhantom(ply_path="chest_organs_attenuation.ply", use_cupy=True)
    print(f"ChestPhantom initialized successfully.")
    print(f"Extracted voxel density grid shape: {phantom.voxel_density.shape}")
    print(f"Detected {len(phantom.gpu_organs)} organs on GPU:")
    for o in phantom.gpu_organs:
        print(f"  - {o['name']} (Mat {o['mat_id']}): rho={o['density']} g/cm3")

    e_grid = cp.arange(10.0, 121.0, 1.0, dtype=cp.float32)
    s_grid = cp.ones_like(e_grid) / len(e_grid)
    ti = TissueInteraction(phantom, e_grid, s_grid, use_cupy=True)
    print(f"Mu Lung at 60 keV:        {float(ti.mu3[50]):.4f} mm^-1")
    print(f"Mu Soft Tissue at 60 keV: {float(ti.mu1[50]):.4f} mm^-1")
    print(f"Mu Cortical Bone at 60 keV: {float(ti.mu2[50]):.4f} mm^-1")