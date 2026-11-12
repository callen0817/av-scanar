#!/usr/bin/env python3
"""
AV Structure CLI — Virtual Scan Stations, Structured Scans & 360° Panoramas (Milestone M15)

Standalone & batch post-processing processor for Artificial Vision:
  - Processes existing `scanHUB OS` / `AV Scan` datasets (`pointcloud_fused_rgb.ply`, `trajectory.csv`,
    `flash_anchors.json`, and `keyframes/` if available)
  - Also supports third-party point clouds (`.ply`, `.pcd`, `.xyz`) without trajectories or keyframes
  - Generates per-station shared-center outputs:
      1. Structured 3D spherical scan (`rows × cols`, foreground occlusion culling, `cartesianInvalidState` for missing rays)
      2. Aligned 360° equirectangular panorama (`panorama_rgb.jpg`, `depth_map_mm.png`, `coverage_mask.png`)
      3. ASTM E2807-11 Structured `.e57` pairing each `data3D` scan with its `images2D` `sphericalRepresentation`
  - Strictly preserves unobserved/missing coverage as missing (zero hallucination or AI fabrication).

Usage:
  python3 scripts/generate_virtual_stations.py \
      --dataset-dir projects/TestProject_2756/datasets/Corridor_Scan_A \
      --spacing 3.5 --pano-width 2048 --pano-height 1024 --validate-e57
"""

import argparse
import csv
import json
import math
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
_GUI_DIR = os.path.join(_REPO_ROOT, "src", "scanhub_gui")
if _GUI_DIR not in sys.path:
    sys.path.insert(0, _GUI_DIR)

from av_structure_engine import (
    PanoramaProvenance,
    StationGenerationConfig,
    StationPlacementMode,
    VirtualStationGenerator,
    VirtualStationSelector,
    export_av_structure_bundle,
    load_keyframes_bundle,
)
from e57_structured_writer import read_and_validate_e57


def load_point_cloud_xyz_rgb(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fast vectorized loader for PLY (ASCII & binary_little_endian), PCD (ASCII & binary), and XYZ files.
    Returns:
      xyz: (N, 3) float64
      rgb: (N, 3) uint8
      has_true_rgb: (N,) bool (False where points carry default uncolored grey [175,175,175] or [70,75,85])
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".ply":
        with open(path, "rb") as f:
            fmt = "ascii"
            vertex_count = 0
            props: List[Tuple[str, str]] = []
            in_vertex = False
            while True:
                line = f.readline()
                if not line:
                    raise ValueError(f"Unexpected EOF in PLY header: {path}")
                s = line.decode("ascii", errors="ignore").strip()
                if s.startswith("format "):
                    fmt = s.split()[1]
                elif s.startswith("element "):
                    parts = s.split()
                    in_vertex = len(parts) >= 3 and parts[1] == "vertex"
                    if in_vertex:
                        vertex_count = int(parts[2])
                elif s.startswith("property ") and in_vertex:
                    parts = s.split()
                    if len(parts) >= 3 and parts[1] != "list":
                        props.append((parts[1], parts[2]))
                elif s == "end_header":
                    break

            data_offset = f.tell()

        if vertex_count == 0:
            return np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.uint8), np.zeros(0, dtype=bool)

        prop_names = [p[1] for p in props]
        ncols = len(props)

        if fmt == "ascii":
            with open(path, "rb") as f:
                f.seek(data_offset)
                raw_vals = np.fromfile(f, dtype=np.float64, sep=" ")
            actual_rows = len(raw_vals) // ncols
            arr = raw_vals[: actual_rows * ncols].reshape(actual_rows, ncols)
            ix = prop_names.index("x") if "x" in prop_names else 0
            iy = prop_names.index("y") if "y" in prop_names else 1
            iz = prop_names.index("z") if "z" in prop_names else 2
            xyz = arr[:, [ix, iy, iz]].astype(np.float64)

            r_idx = next((prop_names.index(k) for k in ("red", "r") if k in prop_names), -1)
            g_idx = next((prop_names.index(k) for k in ("green", "g") if k in prop_names), -1)
            b_idx = next((prop_names.index(k) for k in ("blue", "b") if k in prop_names), -1)
            if r_idx >= 0 and g_idx >= 0 and b_idx >= 0:
                rgb = np.clip(arr[:, [r_idx, g_idx, b_idx]], 0, 255).astype(np.uint8)
                is_default_grey = (
                    ((rgb[:, 0] == 175) & (rgb[:, 1] == 175) & (rgb[:, 2] == 175))
                    | ((rgb[:, 0] == 70) & (rgb[:, 1] == 75) & (rgb[:, 2] == 85))
                    | ((rgb[:, 0] == 64) & (rgb[:, 1] == 68) & (rgb[:, 2] == 78))
                )
                has_true_rgb = ~is_default_grey
            else:
                rgb = np.full((len(xyz), 3), 175, dtype=np.uint8)
                has_true_rgb = np.zeros(len(xyz), dtype=bool)
            return xyz, rgb, has_true_rgb

        elif fmt == "binary_little_endian":
            type_map = {
                "float": "<f4", "float32": "<f4",
                "double": "<f8", "float64": "<f8",
                "uchar": "u1", "uint8": "u1",
                "char": "i1", "int8": "i1",
                "ushort": "<u2", "uint16": "<u2",
                "short": "<i2", "int16": "<i2",
                "uint": "<u4", "uint32": "<u4",
                "int": "<i4", "int32": "<i4",
            }
            dt = np.dtype([(name, type_map[t]) for t, name in props])
            with open(path, "rb") as f:
                f.seek(data_offset)
                rec = np.fromfile(f, dtype=dt, count=vertex_count)
            xyz = np.column_stack([rec["x"], rec["y"], rec["z"]]).astype(np.float64)
            r_name = "red" if "red" in rec.dtype.names else ("r" if "r" in rec.dtype.names else None)
            g_name = "green" if "green" in rec.dtype.names else ("g" if "g" in rec.dtype.names else None)
            b_name = "blue" if "blue" in rec.dtype.names else ("b" if "b" in rec.dtype.names else None)
            if r_name and g_name and b_name:
                rgb = np.column_stack([rec[r_name], rec[g_name], rec[b_name]]).astype(np.uint8)
                is_default_grey = (
                    ((rgb[:, 0] == 175) & (rgb[:, 1] == 175) & (rgb[:, 2] == 175))
                    | ((rgb[:, 0] == 70) & (rgb[:, 1] == 75) & (rgb[:, 2] == 85))
                )
                has_true_rgb = ~is_default_grey
            else:
                rgb = np.full((len(xyz), 3), 175, dtype=np.uint8)
                has_true_rgb = np.zeros(len(xyz), dtype=bool)
            return xyz, rgb, has_true_rgb
        else:
            raise ValueError(f"Unsupported PLY format '{fmt}' in {path}")

    elif ext == ".xyz":
        raw = np.loadtxt(path, dtype=np.float64)
        if raw.ndim == 1:
            raw = raw.reshape(1, -1)
        xyz = raw[:, :3].astype(np.float64)
        if raw.shape[1] >= 6:
            rgb = np.clip(raw[:, 3:6], 0, 255).astype(np.uint8)
            has_true_rgb = np.ones(len(xyz), dtype=bool)
        else:
            rgb = np.full((len(xyz), 3), 175, dtype=np.uint8)
            has_true_rgb = np.zeros(len(xyz), dtype=bool)
        return xyz, rgb, has_true_rgb

    else:
        raise ValueError(f"Unsupported point cloud extension '{ext}' for {path}")


def load_trajectory_csv(csv_path: str) -> List[Tuple]:
    if not os.path.exists(csv_path):
        return []
    recs: List[Tuple] = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                t = float(row.get("timestamp_sec", 0.0))
                x = float(row.get("x_m", 0.0))
                y = float(row.get("y_m", 0.0))
                z = float(row.get("z_m", 0.0))
                qx = float(row.get("qx", 0.0))
                qy = float(row.get("qy", 0.0))
                qz = float(row.get("qz", 0.0))
                qw = float(row.get("qw", 1.0))
                yaw = float(row.get("yaw_rad", 0.0))
                recs.append((t, x, y, z, qx, qy, qz, qw, yaw))
            except Exception:
                continue
    return recs


def parse_mode(mode_str: str) -> StationPlacementMode:
    s = mode_str.strip().lower()
    if s in ("manual", "manual_only", "flash"):
        return StationPlacementMode.MANUAL_ONLY
    if s in ("auto", "auto_only", "trajectory"):
        return StationPlacementMode.AUTO_ONLY
    return StationPlacementMode.HYBRID


def main() -> int:
    parser = argparse.ArgumentParser(
        description="AV Structure CLI — Generate Virtual Scan Stations, 360° Panoramas & Structured E57"
    )
    parser.add_argument("--dataset-dir", type=str, default=None, help="Path to captured dataset directory")
    parser.add_argument("--cloud", type=str, default=None, help="Explicit path to .ply or .xyz point cloud")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory (defaults to dataset-dir)")
    parser.add_argument(
        "--mode",
        type=str,
        default="hybrid",
        choices=["hybrid", "manual_only", "auto_only"],
        help="Virtual station placement mode",
    )
    parser.add_argument("--spacing", type=float, default=3.5, help="Automatic trajectory station spacing (m)")
    parser.add_argument("--pano-width", type=int, default=2048, help="360° Panorama width (px)")
    parser.add_argument("--pano-height", type=int, default=1024, help="360° Panorama height (px)")
    parser.add_argument("--scan-rows", type=int, default=256, help="Structured spherical scan elevation rows")
    parser.add_argument("--scan-cols", type=int, default=1024, help="Structured spherical scan azimuth cols")
    parser.add_argument(
        "--station",
        action="append",
        default=[],
        help="Add explicit station 'x,y,z' or 'x,y,z,yaw_deg' (can be specified multiple times)",
    )
    parser.add_argument("--no-gap-fill", action="store_true", help="Disable gated <8cm 1px pinhole fill")
    parser.add_argument("--validate-e57", action="store_true", help="Run full CRC-32C + XML round-trip check on E57")
    parser.add_argument("--project-name", type=str, default="AV ScanAR Project", help="Project name for AV Spatial Bundle v1.1")
    parser.add_argument("--scan-date", type=str, default="", help="Scan date YYYY-MM-DD for 4D historical continuity")
    parser.add_argument("--scan-id", type=str, default="", help="Unique scan label (defaults to output directory name)")
    parser.add_argument("--reference-cloud", type=str, default=None, help="Optional baseline .ply/.xyz/.bin cloud for 4D ICP alignment")
    parser.add_argument("--avproj-dir", type=str, default=None, help="Optional AV Showcase project root to create/append project.avproj")
    args = parser.parse_args()

    ds_dir = os.path.abspath(args.dataset_dir) if args.dataset_dir else None
    cloud_path = os.path.abspath(args.cloud) if args.cloud else None

    if cloud_path is None and ds_dir is not None:
        for candidate in (
            "pointcloud_cslam_merged.ply",
            "pointcloud_fused_rgb.ply",
            "pointcloud_stereo.ply",
            "pointcloud_airy.ply",
        ):
            p = os.path.join(ds_dir, candidate)
            if os.path.exists(p):
                cloud_path = p
                break

    if cloud_path is None or not os.path.exists(cloud_path):
        print("[ERROR] Could not locate input point cloud. Specify --dataset-dir or --cloud.", file=sys.stderr)
        return 1

    out_dir = os.path.abspath(args.output_dir) if args.output_dir else (ds_dir or os.path.dirname(cloud_path))
    os.makedirs(out_dir, exist_ok=True)

    t0 = time.perf_counter()
    print(f"[AV STRUCTURE] Loading point cloud: {cloud_path}")
    xyz, rgb, has_true_rgb = load_point_cloud_xyz_rgb(cloud_path)
    load_ms = (time.perf_counter() - t0) * 1000.0
    true_rgb_pct = (float(np.count_nonzero(has_true_rgb)) * 100.0 / max(1, len(xyz)))
    print(f"  -> Loaded {len(xyz):,} points in {load_ms:.1f} ms (True RGB points: {true_rgb_pct:.1f}%)")

    traj_recs: List[Tuple] = []
    flash_anchors: List[Dict] = []
    keyframes = []

    if ds_dir is not None:
        traj_recs = load_trajectory_csv(os.path.join(ds_dir, "trajectory.csv"))
        fa_path = os.path.join(ds_dir, "flash_anchors.json")
        if os.path.exists(fa_path):
            try:
                with open(fa_path, "r", encoding="utf-8") as ff:
                    loaded_fa = json.load(ff)
                    if isinstance(loaded_fa, list):
                        flash_anchors.extend(loaded_fa)
            except Exception:
                pass
        keyframes = load_keyframes_bundle(ds_dir)

    for idx, st_spec in enumerate(args.station):
        parts = [float(v.strip()) for v in st_spec.split(",")]
        if len(parts) >= 3:
            yaw_rad = math.radians(parts[3]) if len(parts) >= 4 else 0.0
            flash_anchors.append(
                {
                    "label": f"cli{idx:02d}",
                    "x": parts[0],
                    "y": parts[1],
                    "z": parts[2],
                    "yaw": yaw_rad,
                    "source_mode": "cli_manual",
                }
            )

    print(
        f"  -> Inputs: Trajectory poses={len(traj_recs)}, Manual/Flash anchors={len(flash_anchors)}, "
        f"Calibrated RGB Keyframes={len(keyframes)}"
    )

    cfg = StationGenerationConfig(
        placement_mode=parse_mode(args.mode),
        auto_spacing_m=float(args.spacing),
        scan_rows=int(args.scan_rows),
        scan_cols=int(args.scan_cols),
        pano_width=int(args.pano_width),
        pano_height=int(args.pano_height),
        enable_small_gap_fill=not args.no_gap_fill,
    )

    stations = VirtualStationSelector.select_stations(
        cfg,
        flash_anchors=flash_anchors,
        trajectory_records=traj_recs,
        keyframes=keyframes,
        points_xyz=xyz,
    )
    print(f"[AV STRUCTURE] Selected {len(stations)} Virtual Scan Station(s) (mode={cfg.placement_mode.value}):")

    gen = VirtualStationGenerator(cfg)
    results = []
    for st in stations:
        res = gen.synthesize_station(
            st,
            points_xyz=xyz,
            points_rgb=rgb,
            points_has_true_rgb=has_true_rgb,
            keyframes=keyframes,
        )
        results.append(res)
        prov = res.provenance_mask
        unobs_pct = float(np.count_nonzero(prov == int(PanoramaProvenance.UNOBSERVED))) * 100.0 / max(1, prov.size)
        print(
            f"  * {st.station_id:<22} pos=({st.position_xyz[0]:+6.2f}, {st.position_xyz[1]:+6.2f}, {st.position_xyz[2]:+5.2f})m | "
            f"ScanCov={res.scan_coverage_pct:5.1f}% ({res.valid_scan_points:,} rays) | "
            f"PanoRGB={res.rgb_coverage_pct:5.1f}% | Unobs(Preserved)={unobs_pct:5.1f}% | {res.synthesis_time_ms:.0f}ms"
        )

    ref_xyz = None
    if args.reference_cloud and os.path.exists(args.reference_cloud):
        if args.reference_cloud.lower().endswith(".bin"):
            from av_structure_engine import read_point_cloud_lod0_bin
            ref_xyz, _ = read_point_cloud_lod0_bin(args.reference_cloud)
        else:
            ref_xyz, _, _ = load_point_cloud_xyz_rgb(args.reference_cloud)

    bundle = export_av_structure_bundle(
        save_dir=out_dir,
        results=results,
        export_e57=True,
        e57_filename="structured_stations.e57",
        jpeg_quality=cfg.jpeg_quality,
        points_xyz=xyz,
        points_rgb=rgb,
        project_name=args.project_name,
        scan_date=args.scan_date,
        scan_id=args.scan_id,
        reference_cloud_xyz=ref_xyz,
        av_showcase_project_dir=os.path.abspath(args.avproj_dir) if args.avproj_dir else None,
    )
    e57_path = bundle.get("structured_e57_path") or (
        os.path.join(out_dir, bundle["structured_e57_file"]) if bundle.get("structured_e57_file") else ""
    )
    print(f"[AV STRUCTURE] Exported AV Spatial Bundle v1.1 ({bundle.get('floor_count', 1)} floor(s)) -> {out_dir}")
    if bundle.get("av_walkthrough_bundle_path"):
        print(f"  -> Walkthrough Bundle: {bundle['av_walkthrough_bundle_path']}")
    if bundle.get("av_showcase_avproj_path"):
        print(f"  -> AV Showcase Project: {bundle['av_showcase_avproj_path']}")
    if e57_path and os.path.exists(e57_path):
        print(f"  -> Structured E57: {e57_path} ({os.path.getsize(e57_path) / (1024 * 1024):.2f} MB)")

    if args.validate_e57 and e57_path:
        v = read_and_validate_e57(e57_path)
        if not v.valid:
            print(f"[ERROR] E57 validation failed: {v.error_message}", file=sys.stderr)
            return 2
        print(
            f"  -> ASTM E2807-11 Validation PASSED: {v.crc_verified_pages}/{v.total_pages} CRC-32C pages verified, "
            f"data3D={v.station_count}, images2D={v.image_count}, GUIDs matched={v.associated_guids_matched}"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
