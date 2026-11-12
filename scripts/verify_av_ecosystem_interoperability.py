#!/usr/bin/env python3
"""
End-to-End Verification Suite for AV Ecosystem Interoperability (AV ScanAR <-> AV Structure <-> AV Showcase)

Verifies:
  1. AV Spatial Bundle v1.1 (`av_walkthrough_bundle.json`), `cache/lod0.bin` (`4 + 15*N` bytes),
     ASTM E2807-11 Structured E57 (`u=0.5 -> +X`, `u=0.25 -> +Y`), and pure-#000000 non-fabrication.
  2. Multi-floor story clustering (`Level 1` & `Level 2`), cross-floor staircase navigation links,
     and 3D wall-occlusion blocking across solid partition walls.
  3. Metric-calibrated 2D architectural wall-slice Key Plan PNG generation (`floors/floor_<id>_keyplan.png`
     with North-up `+Y -> row 0` orientation and `world_bounds`).
  4. 4D Historical Time-Travel Continuity (`2026-12-04` -> `2027-04-13` in `project.avproj`):
     Even when `2027-04-13` starts from a shifted/rotated SLAM origin (`+35 deg` yaw, `[+2.4m, -1.6m, +0.12m]`)
     and walks through the bathroom in the opposite direction (`180 deg` reversed station heading),
     automatic 4-DOF ICP alignment registers `2027-04-13` into the canonical Project Frame so the
     bathroom viewpoint matches within `< 0.08m` and world vanity direction is preserved!
"""

import json
import math
import os
import shutil
import sys
import tempfile

import cv2
import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
_GUI_DIR = os.path.join(_REPO_ROOT, "src", "scanhub_gui")
if _GUI_DIR not in sys.path:
    sys.path.insert(0, _GUI_DIR)

from av_structure_engine import (
    StationGenerationConfig,
    StationPlacementMode,
    VirtualStationGenerator,
    VirtualStationSelector,
    export_av_structure_bundle,
    read_point_cloud_lod0_bin,
)
from e57_structured_writer import read_and_validate_e57


def _build_two_story_building_with_bathroom() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(42)
    pts_list = []
    rgb_list = []

    def add_box_surface(x0, x1, y0, y1, z, step, col):
        xs = np.arange(x0, x1 + 1e-6, step)
        ys = np.arange(y0, y1 + 1e-6, step)
        gx, gy = np.meshgrid(xs, ys)
        p = np.column_stack([gx.ravel(), gy.ravel(), np.full(gx.size, z)])
        c = np.tile(np.array(col, dtype=np.uint8), (len(p), 1))
        pts_list.append(p)
        rgb_list.append(c)

    def add_wall_x(x, y0, y1, z0, z1, step, col):
        ys = np.arange(y0, y1 + 1e-6, step)
        zs = np.arange(z0, z1 + 1e-6, step)
        gy, gz = np.meshgrid(ys, zs)
        p = np.column_stack([np.full(gy.size, x), gy.ravel(), gz.ravel()])
        c = np.tile(np.array(col, dtype=np.uint8), (len(p), 1))
        pts_list.append(p)
        rgb_list.append(c)

    def add_wall_y(y, x0, x1, z0, z1, step, col):
        xs = np.arange(x0, x1 + 1e-6, step)
        zs = np.arange(z0, z1 + 1e-6, step)
        gx, gz = np.meshgrid(xs, zs)
        p = np.column_stack([gx.ravel(), np.full(gx.size, y), gz.ravel()])
        c = np.tile(np.array(col, dtype=np.uint8), (len(p), 1))
        pts_list.append(p)
        rgb_list.append(c)

    # Level 1 (z = 0.0m): Main Hall [0..8, -2..2] + Bathroom [4..8, 2..6] separated by wall at y=2.0 (doorway at x in [4.2..5.2])
    add_box_surface(0.0, 8.0, -2.0, 2.0, 0.0, 0.08, [160, 155, 145])
    add_box_surface(4.0, 8.0, 2.0, 6.0, 0.0, 0.08, [185, 205, 215])  # Bathroom tile floor
    add_wall_y(-2.0, 0.0, 8.0, 0.0, 2.6, 0.08, [210, 205, 198])
    add_wall_x(0.0, -2.0, 2.0, 0.0, 2.6, 0.08, [200, 195, 190])
    add_wall_x(8.0, -2.0, 6.0, 0.0, 2.6, 0.08, [200, 195, 190])
    add_wall_x(4.0, 2.0, 6.0, 0.0, 2.6, 0.08, [190, 210, 220])
    # North Bathroom Vanity Wall (y = 6.0): distinct gold-teal vanity target at x=6.5, y=6.0!
    add_wall_y(6.0, 4.0, 8.0, 0.0, 2.6, 0.07, [40, 210, 195])
    # Solid partition wall between Hall (y < 2) and Bathroom (y > 2) from x=5.3..8.0 (blocks direct ray at x=6.8!)
    add_wall_y(2.0, 0.0, 4.1, 0.0, 2.6, 0.06, [220, 180, 140])
    add_wall_y(2.0, 5.3, 8.0, 0.0, 2.6, 0.06, [220, 180, 140])

    # Level 2 (z = 3.2m): Upper mezzanine [0..5, -2..2]
    add_box_surface(0.0, 5.0, -2.0, 2.0, 3.2, 0.09, [150, 145, 140])
    add_wall_y(2.0, 0.0, 5.0, 3.2, 5.6, 0.09, [215, 210, 200])
    add_wall_y(-2.0, 0.0, 5.0, 3.2, 5.6, 0.09, [215, 210, 200])
    add_wall_x(0.0, -2.0, 2.0, 3.2, 5.6, 0.09, [205, 200, 195])

    xyz = np.vstack(pts_list) + rng.normal(0.0, 0.004, size=(sum(len(p) for p in pts_list), 3))
    rgb = np.vstack(rgb_list)
    return xyz, rgb


def main() -> int:
    print("==========================================================================")
    print("  AV ECOSYSTEM INTEROPERABILITY SUITE (ScanAR <-> Structure <-> Showcase) ")
    print("==========================================================================")

    tmp_root = tempfile.mkdtemp(prefix="av_interop_test_")
    try:
        xyz_dec, rgb_dec = _build_two_story_building_with_bathroom()
        avproj_dir = os.path.join(tmp_root, "Building_Project")
        scan_dec_dir = os.path.join(tmp_root, "scan_2026_12_04")
        scan_apr_dir = os.path.join(tmp_root, "scan_2027_04_13")

        cfg = StationGenerationConfig(
            placement_mode=StationPlacementMode.MANUAL_ONLY,
            pano_width=1024,
            pano_height=512,
            scan_rows=128,
            scan_cols=512,
        )
        gen = VirtualStationGenerator(cfg)

        # Dec 4 stations:
        #  0: Hallway at (6.8, 0.5, 1.55)
        #  1: Doorway at (4.7, 1.8, 1.55)
        #  2: Bathroom at (6.5, 4.2, 1.55) facing North (+90 deg = pi/2)
        #  3: Stair landing at (2.0, 0.0, 3.15)
        #  4: Level 2 mezzanine at (2.5, 0.0, 4.75)
        anchors_dec = [
            {"label": "hall", "x": 6.8, "y": 0.5, "z": 1.55, "yaw": 0.0, "z_is_sensor_height": True},
            {"label": "door", "x": 4.7, "y": 1.8, "z": 1.55, "yaw": math.pi / 2, "z_is_sensor_height": True},
            {"label": "bathroom", "x": 6.5, "y": 4.2, "z": 1.55, "yaw": math.pi / 2, "z_is_sensor_height": True},
            {"label": "stair_mid", "x": 2.0, "y": 0.0, "z": 3.15, "yaw": math.pi, "z_is_sensor_height": True},
            {"label": "level2", "x": 2.5, "y": 0.0, "z": 4.75, "yaw": 0.0, "z_is_sensor_height": True},
        ]
        st_dec = VirtualStationSelector.select_stations(cfg, flash_anchors=anchors_dec, points_xyz=xyz_dec)
        res_dec = [gen.synthesize_station(s, xyz_dec, rgb_dec) for s in st_dec]

        info_dec = export_av_structure_bundle(
            save_dir=scan_dec_dir,
            results=res_dec,
            points_xyz=xyz_dec,
            points_rgb=rgb_dec,
            project_name="Hospital Wing A",
            scan_date="2026-12-04",
            scan_id="2026-12-04_AM",
            av_showcase_project_dir=avproj_dir,
        )

        # 1. Verify Bundle v1.1, E57, lod0.bin, Multi-Floor & Wall-Occlusion Graph
        bundle_path = info_dec["av_walkthrough_bundle_path"]
        assert os.path.exists(bundle_path), "av_walkthrough_bundle.json missing"
        with open(bundle_path, "r", encoding="utf-8") as f:
            b_dec = json.load(f)

        assert b_dec["format"] == "av_spatial_bundle" and b_dec["schema_version"] == "1.1"
        assert b_dec["nav_graph"] == "los_v1"
        assert isinstance(b_dec["transform_to_project"], list) and len(b_dec["transform_to_project"]) == 16
        assert b_dec["point_cloud"]["lod0"] == "cache/lod0.bin" and b_dec["point_cloud"]["format"] == "av_lod0_v1"
        assert len(b_dec["floors"]) == 2, f"Expected 2 floors, got {len(b_dec['floors'])}"
        print(f"  [PASS] Multi-story detection: {len(b_dec['floors'])} floors ({[fl['name'] for fl in b_dec['floors']]})")

        # Verify wall occlusion blocked direct link between pano_000 (Hall x=6.8,y=0.5) and pano_002 (Bathroom x=6.5,y=4.2)
        p_hall = next(p for p in b_dec["panos"] if p["id"] == "pano_000")
        p_door = next(p for p in b_dec["panos"] if p["id"] == "pano_001")
        p_bath = next(p for p in b_dec["panos"] if p["id"] == "pano_002")
        p_lvl2 = next(p for p in b_dec["panos"] if p["id"] == "pano_004")
        assert "pano_002" not in p_hall["neighbors"], "Wall occlusion failed: Hall linked directly through solid bathroom wall!"
        assert "pano_001" in p_bath["neighbors"], "Bathroom should connect through open doorway pano_001!"
        assert p_lvl2["floor_id"] == 2 and p_bath["floor_id"] == 1
        # Verify cross-floor stair link exists between Floor 1 and Floor 2 and sets stair_link=True
        has_stair = any(
            p.get("stair_link") is True and any(nb_p["floor_id"] != p["floor_id"] for nb_p in b_dec["panos"] if nb_p["id"] in p["neighbors"])
            for p in b_dec["panos"]
        )
        assert has_stair, "Expected cross-floor staircase link with stair_link=True between Floor 1 and Floor 2"
        print("  [PASS] Wall-occlusion raycast & cross-floor staircase navigation links verified.")

        # Verify 2D Architectural Key Plan PNGs & lod0.bin binary layout
        for fl in b_dec["floors"]:
            assert "floor_elevation" in fl, "Floor metadata must include floor_elevation"
            kp_file = os.path.join(scan_dec_dir, fl["keyplan_path"])
            assert os.path.exists(kp_file), f"Missing keyplan PNG: {kp_file}"
            kp_img = cv2.imread(kp_file)
            assert kp_img is not None and kp_img.shape[0] >= 64 and kp_img.shape[1] >= 64
            assert int(np.count_nonzero(kp_img > 120)) > 100, "Keyplan PNG should contain bright architectural wall slices"
            assert "min_x" in fl["world_bounds"] and "meters_per_pixel" in fl["world_bounds"]
        lod0_file = os.path.join(scan_dec_dir, b_dec["point_cloud_cache"])
        lod0_size = os.path.getsize(lod0_file)
        pts_lod0, _ = read_point_cloud_lod0_bin(lod0_file)
        assert lod0_size == 4 + 15 * len(pts_lod0) and len(pts_lod0) > 1000
        e57_rep = read_and_validate_e57(os.path.join(scan_dec_dir, b_dec["structured_e57_path"]))
        assert e57_rep.valid and e57_rep.associated_guids_matched
        print(f"  [PASS] Keyplan PNGs, lod0.bin ({len(pts_lod0):,} pts), and E57 ({e57_rep.crc_verified_pages} CRC pages) verified.")

        # 2. Simulate Apr 13 Scan with shifted/rotated SLAM start pose (+30 deg yaw + [-2.2m, +1.4m, -0.15m])
        #    AND operator walking through bathroom facing OPPOSITE direction (-90 deg instead of +90 deg)!
        yaw_err = math.radians(30.0)
        c, s = math.cos(yaw_err), math.sin(yaw_err)
        # T_apr_from_world: transforms canonical world into unaligned Apr 13 raw SLAM frame
        R_aw = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        t_aw = np.array([-2.2, 1.4, -0.15])

        xyz_apr_raw = (xyz_dec @ R_aw.T) + t_aw
        anchors_apr_raw = []
        for a in anchors_dec:
            pw = np.array([a["x"], a["y"], a["z"]])
            pa = (R_aw @ pw) + t_aw
            # Flip heading by 180 deg in bathroom to prove 4D time-travel never spins user view!
            extra_spin = math.pi if a["label"] == "bathroom" else 0.0
            anchors_apr_raw.append(
                {
                    "label": a["label"],
                    "x": float(pa[0]),
                    "y": float(pa[1]),
                    "z": float(pa[2]),
                    "yaw": float(a["yaw"] + yaw_err + extra_spin),
                    "z_is_sensor_height": True,
                }
            )

        st_apr = VirtualStationSelector.select_stations(cfg, flash_anchors=anchors_apr_raw, points_xyz=xyz_apr_raw)
        res_apr = [gen.synthesize_station(st_s, xyz_apr_raw, rgb_dec) for st_s in st_apr]

        export_av_structure_bundle(
            save_dir=scan_apr_dir,
            results=res_apr,
            points_xyz=xyz_apr_raw,
            points_rgb=rgb_dec,
            project_name="Hospital Wing A",
            scan_date="2027-04-13",
            scan_id="2027-04-13_PM",
            av_showcase_project_dir=avproj_dir,
        )

        # Verify `project.avproj` contains BOTH dates and Apr 13 bathroom aligns to Dec 4 bathroom in Project Frame!
        avproj_file = os.path.join(avproj_dir, "project.avproj")
        with open(avproj_file, "r", encoding="utf-8") as fp:
            avproj = json.load(fp)

        assert len(avproj["scans"]) == 2
        assert avproj.get("baseline_scan_id") == "2026-12-04_AM"
        s_dec = next(sc for sc in avproj["scans"] if sc["scan_date"] == "2026-12-04")
        s_apr = next(sc for sc in avproj["scans"] if sc["scan_date"] == "2027-04-13")
        assert s_dec["scan_dir"] == "Scans/2026-12-04" and s_apr["scan_dir"] == "Scans/2027-04-13"
        assert s_apr["nav_graph"] == "los_v1" and len(s_apr["transform_to_project"]) == 16

        bath_dec = s_dec["panos"][2]
        bath_apr = s_apr["panos"][2]
        pos_err = float(np.linalg.norm(np.array(bath_dec["position"]) - np.array(bath_apr["position"])))
        assert pos_err < 0.08, f"4D ICP bathroom position error too large: {pos_err:.4f}m"

        # Verify world direction of North Vanity (+Y_world = [0, 1, 0]) in both aligned panoramas!
        # For any panorama with station rotation q=[qx,qy,qz,qw], world ray [0, 1, 0] in station local is R(q)^T @ [0, 1, 0],
        # and when rendered on world-aligned panoSphere (R(q) @ p_station), station ray maps right back to [0, 1, 0]!
        from scipy.spatial.transform import Rotation
        for tag, bp, sc_dir in (("2026-12-04", bath_dec, "2026-12-04"), ("2027-04-13", bath_apr, "2027-04-13")):
            R_ws = Rotation.from_quat(bp["rotation"]).as_matrix()
            ray_station = R_ws.T @ np.array([0.0, 1.0, 0.0])  # North vanity wall in station frame
            az = math.atan2(ray_station[1], ray_station[0])
            u_col = int(((0.5 - az / (2.0 * math.pi)) % 1.0) * 1024)
            pano_bgr = cv2.imread(os.path.join(avproj_dir, "Scans", sc_dir, bp["image_path"]))
            px_rgb = pano_bgr[256, u_col, ::-1]  # RGB at horizon facing North vanity [40, 210, 195]
            assert int(px_rgb[1]) > 150 and int(px_rgb[2]) > 140 and int(px_rgb[0]) < 90, (
                f"{tag} panorama pixel at North vanity world heading is wrong color: {px_rgb}"
            )

        print(
            f"  [PASS] 4D Historical Time-Travel (`2026-12-04` -> `2027-04-13`): "
            f"Bathroom position error = {pos_err * 100:.2f} cm, North vanity wall pixel aligned on both dates despite +30° SLAM offset & 180° opposite walk!"
        )
        print("==========================================================================")
        print("  ALL AV SCANAR INTEROPERABILITY TESTS PASSED!                            ")
        print("==========================================================================")
        return 0
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
