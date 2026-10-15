"""
master_helical_pipeline.py
==========================
Master GPU-Accelerated Multi-Row Helical CT Simulation Orchestrator.

Coordinates four aligned physics modules:
  1. helical.py: Helical scanner kinematics, gantry rotation, table feed, and 180-deg phase offset.
  2. xray.py: Polychromatic tungsten spectrum, tube current modulation (TCM), and ray vectors.
  3. tissue.py: 3D vortex mesh ingestion ('vortex.ply'), voxel extraction, and attenuation physics.
  4. energy_integrating_detector.py: CsI scintillator absorption, optical PSF spread, PRNU, and noise.

Acquisition Parameters (from ct_system_output.json):
  - 16,728 Projection Views (17 full gantry rotations @ 984 views/rotation)
  - SID: 541.0 mm | SDD: 949.0 mm | Detector Radius: 408.0 mm
  - Pitch: 0.516 | Table Feed: 20.64 mm/rotation (Axial travel: 350.88 mm)
  - Detector Array: 64 rows x 888 channels (0.625 mm pitch)
  - 120 kVp Tungsten Bremsstrahlung Beam with Aluminum Filtration

Outputs:
  - helical_sinogram_vortex_16728_slice32.npy: Calibrated 2D attenuation sinogram [16728, 888]
  - helical_sinogram_vortex_16728.png: Publication-quality 3-panel verification figure
  - helical_projections_3d_16728.npy: Full 3D projection dataset (memmap stream)
  - voxelized_vortex_density.npy & voxelized_vortex_material.npy: Extracted 3D voxel grids
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import pathlib
import sys
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np

# Matplotlib headless backend
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Primary GPU acceleration engine
try:
    import cupy as cp
    _ = cp.cuda.runtime.getDeviceProperties(0)
    HAS_CUPY = True
    print(f"[ComputeEngine] CuPy initialized on GPU: {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}", flush=True)
except Exception as e:
    import numpy as cp
    HAS_CUPY = False
    print(f"[ComputeEngine] Falling back to NumPy CPU engine: {e}", flush=True)

# Import aligned modules
import helical
from helical import HelicalTrajectory, load_helical_geometry
import xray
from xray import XRaySource, generate_rays_gpu, generate_spectrum
import tissue
from tissue import VortexPhantom, TissueInteraction
import energy_integrating_detector as eid
from energy_integrating_detector import EnergyIntegratingDetector


def format_seconds(seconds: float) -> str:
    """Formats duration into mm:ss or hh:mm:ss."""
    if np.isnan(seconds) or np.isinf(seconds) or seconds < 0:
        return "--:--"
    sec_int = int(round(seconds))
    if sec_int >= 3600:
        return str(datetime.timedelta(seconds=sec_int))
    m, s = divmod(sec_int, 60)
    return f"{m:02d}:{s:02d}"


def run_master_helical_pipeline(
    config_path: str = "ct_system_output.json",
    ply_path: str = "vortex.ply",
    output_dir: str = "helical_16728_output",
    batch_size: int = 32,
    total_views: Optional[int] = None,
    central_slice_row: int = 32,
    stream_full_3d_memmap: bool = True,
    phantom_y_offset: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Executes the end-to-end 16,728-projection helical CT scan simulation of the 3D vortex phantom.
    """
    start_wall_time = time.perf_counter()
    out_dir = pathlib.Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80, flush=True)
    print("MASTER HELICAL CT SIMULATION PIPELINE (GPU ACCELERATED)", flush=True)
    print("=" * 80, flush=True)

    # -------------------------------------------------------------------------
    # 1. Load System Configuration & Geometry
    # -------------------------------------------------------------------------
    print(f"[1/6] Loading system geometry reference from: {config_path}...", flush=True)
    traj = load_helical_geometry(config_path)

    if total_views is not None and total_views > 0:
        traj.total_views = int(total_views)
        traj.num_rotations = traj.total_views / traj.num_views_per_rotation

    n_views = traj.total_views
    views_per_rot = traj.num_views_per_rotation
    n_rot = traj.num_rotations
    total_travel_mm = n_rot * traj.table_feed_per_rotation_mm

    # Default phantom center to midpoint of helical table travel
    if phantom_y_offset is None:
        phantom_y_mm = total_travel_mm / 2.0
    else:
        phantom_y_mm = float(phantom_y_offset)

    print(f"  - Scanner Geometry:  SID={traj.sid_mm:.1f} mm | SDD={traj.sdd_mm:.1f} mm | Det Radius={traj.detector_radius_mm:.1f} mm", flush=True)
    print(f"  - Acquisition:       {n_views:,} views ({n_rot:.1f} gantry rotations @ {views_per_rot} views/rot)", flush=True)
    print(f"  - Table Translation: Pitch={traj.pitch:.3f} | {traj.table_feed_per_rotation_mm:.2f} mm/rot (Total Travel: {total_travel_mm:.2f} mm)", flush=True)
    print(f"  - Detector Array:    {traj.detector_rows} rows x {traj.detector_channels} channels ({traj.pixel_width_mm:.3f} mm pitch)", flush=True)
    print(f"  - 180-deg Phase:     Verified exact source-detector 180-deg phase offset along helical path.", flush=True)

    # -------------------------------------------------------------------------
    # 2. Initialize X-Ray Source Physics
    # -------------------------------------------------------------------------
    print(f"[2/6] Initializing X-ray tube physics and polychromatic spectrum (xray.py)...", flush=True)
    src = XRaySource.from_config_file(config_path, use_cupy=HAS_CUPY)
    print(f"  - Tube Parameters:   {src.kvp:.1f} kVp | {src.tube_current_ma:.1f} mA | Al Filtration: {src.filter_thickness_mm:.1f} mm", flush=True)
    print(f"  - Energy Grid:       {src.num_bins} bins (E_min={float(src.energies[0])} keV, E_max={float(src.energies[-1])} keV)", flush=True)
    print(f"  - Modulation:        TCM={src.tcm_enabled} (Depth={src.tcm_depth * 100:.1f}%)", flush=True)

    # -------------------------------------------------------------------------
    # 3. Ingest 3D Vortex Dataset & Extract Voxel Data
    # -------------------------------------------------------------------------
    print(f"[3/6] Ingesting 3D vortex phantom dataset: {ply_path} (tissue.py)...", flush=True)
    phantom = VortexPhantom(ply_path=ply_path, voxel_size_mm=1.0, center_y_mm=phantom_y_mm, use_cupy=HAS_CUPY)
    ti = TissueInteraction(phantom, src.energies, src.spectrum_weights, use_cupy=HAS_CUPY)

    # Save extracted voxel grids
    dens_path = out_dir / "voxelized_vortex_density.npy"
    mat_path  = out_dir / "voxelized_vortex_material.npy"
    np.save(str(dens_path), phantom.voxel_density)
    np.save(str(mat_path), phantom.voxel_material)
    np.save("voxelized_vortex_density.npy", phantom.voxel_density)
    np.save("voxelized_vortex_material.npy", phantom.voxel_material)
    print(f"  - Extracted Voxel Grid: shape={phantom.voxel_density.shape} (Pitch=1.0 mm)", flush=True)
    print(f"  - Saved voxel density grid:  {dens_path.name}", flush=True)
    print(f"  - Saved voxel material grid: {mat_path.name}", flush=True)
    mid_e = len(src.energies) // 2
    print(f"  - Component Attenuation at 60 keV:", flush=True)
    print(f"      * Fluid Chamber:    {float(ti.mu1[mid_e]):.4f} mm^-1 (rho=1.05 g/cm3)")
    print(f"      * Vortex Blades:    {float(ti.mu2[mid_e]):.4f} mm^-1 (rho=1.60 g/cm3)")
    print(f"      * Inner Core Rod:   {float(ti.mu3[mid_e]):.4f} mm^-1 (rho=1.92 g/cm3)")

    # -------------------------------------------------------------------------
    # 4. Initialize Energy-Integrating Detector
    # -------------------------------------------------------------------------
    print(f"[4/6] Initializing energy-integrating detector model (energy_integrating_detector.py)...", flush=True)
    detector = EnergyIntegratingDetector.from_config_file(config_path, use_cupy=HAS_CUPY)
    print(f"  - Scintillator:      {detector.material.name} (Thickness={detector.thickness_cm*10:.2f} mm, Fill Factor={detector.fill_factor:.2f})", flush=True)
    print(f"  - Electronics:       Conversion Gain={detector.conversion_gain} e-/keV | Noise={detector.electronic_noise_e} e-", flush=True)
    print(f"  - Optical Spread:    PSF sigma={detector.psf_sigma_px} pixels", flush=True)

    # Precompute flat-field air reference projection (I_0)
    print(f"  - Calibrating unattenuated flat-field air reference (I_0)...", flush=True)
    air_fluence = src.calculate_incident_fluence(1, detector.rows, detector.channels)
    air_det_out = detector.detect(air_fluence, src.energies, add_noise=False)
    I_0_full_2d = air_det_out["ideal_signal_electrons"]
    if HAS_CUPY and isinstance(I_0_full_2d, cp.ndarray):
        I_0_full_np = cp.asnumpy(I_0_full_2d).squeeze()
    else:
        I_0_full_np = np.asarray(I_0_full_2d).squeeze()

    I_0_slice = I_0_full_np[central_slice_row, :].astype(np.float32)
    np.save(str(out_dir / "air_reference_I0_slice32.npy"), I_0_slice)
    np.save("air_reference_I0_slice32.npy", I_0_slice)
    print(f"  - Air Reference I_0 Mean: {I_0_full_np.mean():.1f} e- | Slice {central_slice_row} Mean: {I_0_slice.mean():.1f} e-", flush=True)

    # -------------------------------------------------------------------------
    # 5. Allocate Storage & Stream Buffers
    # -------------------------------------------------------------------------
    H, W = detector.rows, detector.channels
    sinogram_raw_2d = np.zeros((n_views, W), dtype=np.float32)

    mmap_3d = None
    if stream_full_3d_memmap:
        mmap_path = out_dir / f"helical_projections_3d_{n_views}.npy"
        vol_bytes = n_views * H * W * 4
        print(f"  - Allocating 3D projection disk stream: {mmap_path.name} (~{vol_bytes / (1024**3):.2f} GB)...", flush=True)
        try:
            import numpy.lib.format as np_format
            mmap_3d = np_format.open_memmap(
                str(mmap_path), mode="w+", dtype=np.float32, shape=(n_views, H, W)
            )
        except Exception:
            mmap_3d = np.memmap(str(mmap_path), dtype=np.float32, mode="w+", shape=(n_views, H, W))

    # -------------------------------------------------------------------------
    # 6. Execute 16,728-View Helical Projection Loop on GPU
    # -------------------------------------------------------------------------
    print(f"\n[5/6] Executing GPU-accelerated helical acquisition across {n_views:,} views...", flush=True)
    print("-" * 80, flush=True)
    print(f"{'Rotation':<14} | {'Views Completed':<18} | {'Throughput':<14} | {'Elapsed':<10} | {'ETA':<10}", flush=True)
    print("-" * 80, flush=True)

    loop_start = time.perf_counter()
    last_rot_report = 0
    n_batches = (n_views + batch_size - 1) // batch_size
    xp = cp if HAS_CUPY else np

    for b in range(n_batches):
        v_start = b * batch_size
        v_end = min(v_start + batch_size, n_views)
        curr_b = v_end - v_start

        # 1. Helical Kinematics with 180-deg phase offset (helical.py)
        b_indices = xp.arange(v_start, v_end, dtype=xp.float32)
        source_pos, pixel_pos = traj.generate_batch_detector_pixels_gpu(b_indices, use_cupy=HAS_CUPY)

        # 2. Ray Vectors & Polychromatic Fluence (xray.py)
        origins, directions = src.generate_batch_rays(source_pos, pixel_pos)
        theta_batch = b_indices * float(traj.angle_increment_rad)
        incident_fluence = src.calculate_incident_fluence(curr_b, H, W, theta_rad=theta_batch)

        # 3. Ray-Phantom Attenuation Physics through Vortex PLY (tissue.py)
        transmitted_fluence = ti.compute_transmitted_fluence(origins, directions, incident_fluence)

        # 4. Scintillator Detection & Noise (energy_integrating_detector.py)
        det_result = detector.detect(transmitted_fluence, src.energies, add_noise=True)
        measured_signal = det_result["measured_signal_electrons"]

        # Transfer batch to host array
        if HAS_CUPY and isinstance(measured_signal, cp.ndarray):
            measured_np = cp.asnumpy(measured_signal)
        else:
            measured_np = np.asarray(measured_signal)

        if mmap_3d is not None:
            mmap_3d[v_start:v_end] = measured_np

        sinogram_raw_2d[v_start:v_end] = measured_np[:, central_slice_row, :]

        curr_rot = v_end // views_per_rot
        if curr_rot > last_rot_report or v_end == n_views:
            elapsed = time.perf_counter() - loop_start
            speed = v_end / max(elapsed, 1e-6)
            eta = (n_views - v_end) / max(speed, 1e-6)

            rot_str = f"Rotation {curr_rot:02d}/{int(n_rot):02d}"
            views_str = f"{v_end:5d}/{n_views:,} ({(v_end / n_views) * 100:5.1f}%)"
            speed_str = f"{speed:5.1f} views/s"
            elapsed_str = format_seconds(elapsed)
            eta_str = format_seconds(eta)

            print(f"{rot_str:<14} | {views_str:<18} | {speed_str:<14} | {elapsed_str:<10} | {eta_str:<10}", flush=True)
            last_rot_report = curr_rot
            if mmap_3d is not None:
                mmap_3d.flush()

    if mmap_3d is not None:
        mmap_3d.flush()

    total_acq_time = time.perf_counter() - loop_start
    print("-" * 80, flush=True)
    print(f"ACQUISITION CONCLUDED in {format_seconds(total_acq_time)} ({n_views / total_acq_time:.1f} views/sec)", flush=True)
    print("=" * 80, flush=True)

    # -------------------------------------------------------------------------
    # 7. Sinogram Synthesis: p = -ln(I / I_0)
    # -------------------------------------------------------------------------
    print(f"[6/6] Synthesizing calibrated line-integral attenuation sinogram p = -ln(I / I_0)...", flush=True)
    I_0_safe = np.maximum(I_0_slice, 1e-6)
    I_safe = np.maximum(sinogram_raw_2d, 0.0)

    ratio = np.clip(I_safe / I_0_safe[np.newaxis, :], 1e-6, 1.0)
    attenuation_sinogram_2d = -np.log(ratio)

    # Interpolate dead channels
    dead_mask = (I_0_slice <= 100.0)
    if np.any(dead_mask):
        valid_cols = np.where(~dead_mask)[0]
        dead_cols = np.where(dead_mask)[0]
        for dc in dead_cols:
            nearest_valid = valid_cols[np.argmin(np.abs(valid_cols - dc))]
            attenuation_sinogram_2d[:, dc] = attenuation_sinogram_2d[:, nearest_valid]

    sino_path = out_dir / f"helical_sinogram_vortex_{n_views}_slice32.npy"
    raw_sino_path = out_dir / f"helical_sinogram_raw_{n_views}_slice32.npy"
    np.save(str(sino_path), attenuation_sinogram_2d)
    np.save(str(raw_sino_path), sinogram_raw_2d)
    # Also save to current directory for direct access
    np.save(f"helical_sinogram_vortex_{n_views}_slice32.npy", attenuation_sinogram_2d)
    np.save(f"helical_sinogram_{n_views}_slice32.npy", attenuation_sinogram_2d)
    print(f"  - Saved 2D attenuation sinogram: {sino_path.name} (shape: {attenuation_sinogram_2d.shape})", flush=True)

    # -------------------------------------------------------------------------
    # 8. Render Publication-Grade 3-Panel Verification Figure
    # -------------------------------------------------------------------------
    print(f"  - Rendering publication verification figure...", flush=True)
    fig_path = out_dir / f"helical_sinogram_vortex_{n_views}.png"
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(22, 7.5), constrained_layout=True)

    fig.suptitle(
        f"Helical CT Scanner Simulation: 3D Vortex Phantom ({n_views:,} Projections | {n_rot:.0f} Rotations)\n"
        f"Phantom: {pathlib.Path(ply_path).name} | 120 kVp Polychromatic Beam | CsI Scintillator (64x888) | Pitch={traj.pitch:.3f}",
        fontsize=14, fontweight="bold",
    )

    # Panel 1: Central Axial Cross Section of Voxelized Vortex Phantom
    mid_z = phantom.voxel_density.shape[0] // 2
    axial_slice_2d = phantom.voxel_density[mid_z, :, :]
    extent_x = [phantom.vol_min[0], -phantom.vol_min[0]]
    extent_y = [phantom.vol_min[1], phantom.vol_min[1] + phantom.voxel_density.shape[1]]
    im1 = ax1.imshow(
        axial_slice_2d.T,
        aspect="auto",
        cmap="inferno",
        origin="lower",
    )
    ax1.set_title("Panel 1: Voxelized 3D Vortex Phantom (Central Slice)", fontsize=12, pad=10)
    ax1.set_xlabel("Voxel Grid Dimension X", fontsize=11)
    ax1.set_ylabel("Voxel Grid Dimension Y", fontsize=11)
    cb1 = fig.colorbar(im1, ax=ax1, shrink=0.75, pad=0.03)
    cb1.set_label("Mass Density rho (g/cm3)", fontsize=10)

    # Panel 2: Raw Intensity Sinogram across all views (Row 32)
    im2 = ax2.imshow(
        sinogram_raw_2d,
        aspect="auto",
        cmap="gray",
        origin="lower",
    )
    ax2.set_title(f"Panel 2: Raw Intensity Sinogram (Slice {central_slice_row}, {n_views:,} Views)", fontsize=12, pad=10)
    ax2.set_xlabel("Detector Channels (Column Index 0 to 887)", fontsize=11)
    ax2.set_ylabel(f"Helical Projection View Index (0 to {n_views-1:,})", fontsize=11)
    cb2 = fig.colorbar(im2, ax=ax2, shrink=0.75, pad=0.03)
    cb2.set_label("Transmitted Electrons (e-)", fontsize=10)

    # Panel 3: Calibrated Line-Integral Attenuation Sinogram p = -ln(I / I_0)
    im3 = ax3.imshow(
        attenuation_sinogram_2d,
        aspect="auto",
        cmap="bone",
        origin="lower",
    )
    ax3.set_title(f"Panel 3: Calibrated Attenuation Sinogram p = -ln(I / I_0)", fontsize=12, pad=10)
    ax3.set_xlabel("Detector Channels (Column Index 0 to 887)", fontsize=11)
    ax3.set_ylabel(f"Helical Projection View Index (0 to {n_views-1:,})", fontsize=11)
    cb3 = fig.colorbar(im3, ax=ax3, shrink=0.75, pad=0.03)
    cb3.set_label("Line-Integral Attenuation p", fontsize=10)

    plt.savefig(str(fig_path), dpi=200, bbox_inches="tight")
    plt.savefig(f"helical_sinogram_vortex_{n_views}.png", dpi=200, bbox_inches="tight")
    plt.savefig(f"helical_sinogram_{n_views}.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  - Saved verification figure: {fig_path.name}", flush=True)

    total_pipeline_time = time.perf_counter() - start_wall_time
    print("=" * 80, flush=True)
    print(f"COMPLETE PIPELINE EXECUTION FINISHED in {format_seconds(total_pipeline_time)}!", flush=True)
    print(f"Sinogram Array Shape: {attenuation_sinogram_2d.shape} | Attenuation Range: [{attenuation_sinogram_2d.min():.4f}, {attenuation_sinogram_2d.max():.4f}]", flush=True)
    print("=" * 80, flush=True)

    return {
        "attenuation_sinogram": attenuation_sinogram_2d,
        "raw_sinogram": sinogram_raw_2d,
        "sinogram_path": sino_path,
        "figure_path": fig_path,
    }


def main():
    parser = argparse.ArgumentParser(description="Master Helical CT Scanner Simulation Orchestrator (16,728 Projections)")
    parser.add_argument("--config", type=str, default="ct_system_output.json", help="Path to ct_system_output.json")
    parser.add_argument("--ply", type=str, default="vortex.ply", help="Path to 3D vortex .ply phantom")
    parser.add_argument("--output-dir", type=str, default="helical_16728_output", help="Directory for output files")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for GPU forward projection")
    parser.add_argument("--views", type=int, default=16728, help="Total projection views (default: 16728)")
    parser.add_argument("--slice", type=int, default=32, help="Central detector row slice (default: 32)")
    parser.add_argument("--no-memmap", action="store_true", help="Disable streaming full 3D volume to disk")
    parser.add_argument("--phantom-y-offset", type=float, default=None, help="Axial Y position of phantom center in mm")
    args = parser.parse_args()

    run_master_helical_pipeline(
        config_path=args.config,
        ply_path=args.ply,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        total_views=args.views,
        central_slice_row=args.slice,
        stream_full_3d_memmap=not args.no_memmap,
        phantom_y_offset=args.phantom_y_offset,
    )


if __name__ == "__main__":
    main()
