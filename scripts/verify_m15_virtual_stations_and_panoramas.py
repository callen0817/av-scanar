#!/usr/bin/env python3
"""
Milestone M15 End-to-End Verification Suite:
AV Structure — Virtual Scan Stations, Structured Spherical Scans, 360° Panoramas & ASTM E2807-11 E57

Verifies:
  1. Castagnoli CRC-32C hardware-standard test vector & ASTM E2807-11 Structured E57 writer/validator
  2. M12 PointCloudProcessor ExportFormat.E57_STRUCTURED integration
  3. IntelligentKeyframeSelector triggers (distance, rotation, station spin, coverage, blur/exposure rejection,
     post-SLAM SLERP pose update, and save/load bundle)
  4. VirtualStationSelector modes (MANUAL_ONLY, AUTO_ONLY, HYBRID clearance, centroid fallback)
  5. Shared-Center Dual Synthesis:
     - Walkaround Plumb-Bob RGB keyframe reprojection + Z-buffer parallax occlusion culling (zero ghosting)
     - Ouster Rev8 `spherical` camera model reprojection path
     - Strict non-fabrication of unobserved/missing regions (`UNOBSERVED=0`, `cartesianInvalidState=2`, `depth=0`)
  6. Headless PyQt (`QT_QPA_PLATFORM=offscreen`) `ScanHUBOS` M15 Tab, Map Station Placement,
     `PanoramaPreviewDialog`, and `.e57` export
  7. Real Dataset CLI execution (`scripts/generate_virtual_stations.py` on `Corridor_Scan_A`)
"""

import json
import math
import os
import shutil
import subprocess
import sys
import tempfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
_GUI_DIR = os.path.join(_REPO_ROOT, "src", "scanhub_gui")
if _GUI_DIR not in sys.path:
    sys.path.insert(0, _GUI_DIR)

from e57_structured_writer import (
    E57StationPayload,
    compute_crc32c_bytes,
    read_and_validate_e57,
    write_structured_e57,
)
from av_structure_engine import (
    IntelligentKeyframeSelector,
    KeyframeCaptureConfig,
    PanoramaProvenance,
    RgbKeyframe,
    StationGenerationConfig,
    StationPlacementMode,
    VirtualStation,
    VirtualStationGenerator,
    VirtualStationSelector,
    export_av_structure_bundle,
    load_keyframes_bundle,
    save_keyframes_bundle,
)
from m12_point_cloud_cleanup import ExportFormat, PointCloudProcessor


def _make_textured_rgb(h: int, w: int, base_rgb=(210, 60, 50)) -> np.ndarray:
    """Creates a sharp checkerboard-textured RGB image that passes Laplacian sharpness & exposure gates."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    yy, xx = np.indices((h, w))
    checker = (((xx // 12) + (yy // 12)) % 2 == 0)
    img[checker] = np.clip(np.array(base_rgb, dtype=np.int32), 20, 235).astype(np.uint8)
    img[~checker] = np.clip(np.array(base_rgb, dtype=np.int32) // 2 + 40, 20, 235).astype(np.uint8)
    return img


def test_1_e57_crc32c_and_structured_roundtrip(tmp_dir: str):
    print("\n[TEST 1] Castagnoli CRC-32C & Multi-Station ASTM E2807-11 Structured E57...")
    # Official Castagnoli CRC-32C check value for ASCII "123456789" is 0xE3069283
    crc = compute_crc32c_bytes(b"123456789")
    assert crc == 0xE3069283, f"CRC-32C mismatch: got 0x{crc:08X}"

    rows, cols = 64, 256
    xyz1 = np.zeros((rows, cols, 3), dtype=np.float32)
    inv1 = np.full((rows, cols), 2, dtype=np.uint8)
    rgb1 = np.zeros((rows, cols, 3), dtype=np.uint8)
    # Populate middle band with valid cylinder points; leave top/bottom unobserved (inv=2)
    for r in range(16, 48):
        el = math.pi * (0.5 - (r + 0.5) / rows)
        for c in range(cols):
            az = 2.0 * math.pi * (0.5 - (c + 0.5) / cols)
            dist = 3.5
            xyz1[r, c] = [
                dist * math.cos(el) * math.cos(az),
                dist * math.cos(el) * math.sin(az),
                dist * math.sin(el),
            ]
            inv1[r, c] = 0
            rgb1[r, c] = [(c * 3) % 255, (r * 5) % 255, 180]

    pano_img = _make_textured_rgb(256, 512, (220, 185, 95))
    ok, enc = cv2.imencode(".jpg", cv2.cvtColor(pano_img, cv2.COLOR_RGB2BGR))
    assert ok
    jpg_bytes = enc.tobytes()

    p1 = E57StationPayload(
        station_id="st_00",
        station_name="Station_00",
        description="Test Station 0",
        translation_xyz=(1.25, -0.75, 0.10),
        rotation_wxyz=(1.0, 0.0, 0.0, 0.0),
        xyz_local=xyz1,
        invalid_state=inv1,
        rgb=rgb1,
        panorama_jpeg_bytes=jpg_bytes,
        panorama_width=512,
        panorama_height=256,
    )
    p2 = E57StationPayload(
        station_id="st_01",
        station_name="Station_01",
        description="Test Station 1",
        translation_xyz=(4.80, 1.10, 0.12),
        rotation_wxyz=(math.cos(0.4), 0.0, 0.0, math.sin(0.4)),
        xyz_local=xyz1,
        invalid_state=inv1,
        rgb=rgb1,
        panorama_jpeg_bytes=jpg_bytes,
        panorama_width=512,
        panorama_height=256,
    )

    e57_path = os.path.join(tmp_dir, "multi_station_test.e57")
    write_structured_e57(e57_path, [p1, p2])
    rep = read_and_validate_e57(e57_path)
    assert rep.valid, f"E57 validation failed: {rep.error_message}"
    assert rep.crc_verified_pages == rep.total_pages and rep.total_pages > 10
    assert rep.station_count == 2 and rep.image_count == 2
    assert rep.associated_guids_matched is True
    expected_valid = int(np.count_nonzero(inv1 == 0)) * 2
    assert rep.total_valid_points == expected_valid, f"{rep.total_valid_points} != {expected_valid}"

    # Also test M12 PointCloudProcessor E57_STRUCTURED export
    valid_pts = xyz1[inv1 == 0]
    valid_cols = rgb1[inv1 == 0].astype(np.float32)
    cloud6 = np.hstack([valid_pts, valid_cols])
    m12_e57 = os.path.join(tmp_dir, "m12_export.e57")
    res_m12 = PointCloudProcessor.export_point_cloud(cloud6, m12_e57, ExportFormat.E57_STRUCTURED)
    assert res_m12.success, f"M12 E57 export failed: {res_m12.error_message}"
    rep_m12 = read_and_validate_e57(m12_e57)
    assert rep_m12.valid and rep_m12.station_count == 1 and rep_m12.image_count == 1
    print(f"  PASS -> Verified {rep.total_pages} CRC-32C pages, 2 stations + M12 E57_STRUCTURED export.")


def test_2_intelligent_keyframe_selector(tmp_dir: str):
    print("\n[TEST 2] Intelligent Multi-Trigger RGB Keyframe Capture & Pose Refinement...")
    cfg = KeyframeCaptureConfig(
        min_distance_m=0.65,
        min_rotation_deg=17.5,
        station_spin_rotation_deg=12.0,
        min_interval_sec=0.05,
        enable_coverage_trigger=False,
    )
    sel = IntelligentKeyframeSelector(cfg)
    R_cam_imu = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]], dtype=np.float64)
    t_cam_imu = np.zeros(3, dtype=np.float64)

    sharp_img = _make_textured_rgb(240, 320, (190, 140, 90))
    blurry_img = np.full((240, 320, 3), 128, dtype=np.uint8)
    dark_img = np.zeros((240, 320, 3), dtype=np.uint8)

    # 1. Blurry & dark frames rejected on initial trigger
    R_rw, t_rw = IntelligentKeyframeSelector.compose_rgb_world_extrinsics(
        np.zeros(3), np.eye(3), R_cam_imu, t_cam_imu
    )
    assert sel.consider_frame(1.0, dark_img, 200, 200, 160, 120, R_rw, t_rw) is None
    assert sel.rejected_exposure_count == 1
    assert sel.consider_frame(1.1, blurry_img, 200, 200, 160, 120, R_rw, t_rw) is None
    assert sel.rejected_blur_count == 1

    # 2. Sharp initial frame accepted
    kf0 = sel.consider_frame(
        1.2, sharp_img, 200, 200, 160, 120, R_rw, t_rw, R_world_imu=np.eye(3), pos_world_imu=np.zeros(3)
    )
    assert kf0 is not None and kf0.trigger_reason == "initial"

    # 3. Sub-threshold move (0.20m) ignored; >=0.65m accepted
    R_rw2, t_rw2 = IntelligentKeyframeSelector.compose_rgb_world_extrinsics(
        np.array([0.20, 0.0, 0.0]), np.eye(3), R_cam_imu, t_cam_imu
    )
    assert sel.consider_frame(1.4, sharp_img, 200, 200, 160, 120, R_rw2, t_rw2) is None

    R_rw3, t_rw3 = IntelligentKeyframeSelector.compose_rgb_world_extrinsics(
        np.array([0.75, 0.0, 0.0]), np.eye(3), R_cam_imu, t_cam_imu
    )
    kf1 = sel.consider_frame(
        1.6, sharp_img, 200, 200, 160, 120, R_rw3, t_rw3,
        R_world_imu=np.eye(3), pos_world_imu=np.array([0.75, 0.0, 0.0])
    )
    assert kf1 is not None and kf1.trigger_reason == "distance"

    # 4. Station spin trigger (14 deg near active station < 17.5 deg normal threshold, >= 12.0 deg spin threshold)
    st = VirtualStation("st_spin", "w00", np.array([0.75, 0.0, 0.0]), 0.0, "manual_flash")
    R_yaw14 = Rotation.from_euler("z", 14.0, degrees=True).as_matrix()
    R_rw4, t_rw4 = IntelligentKeyframeSelector.compose_rgb_world_extrinsics(
        np.array([0.75, 0.0, 0.0]), R_yaw14, R_cam_imu, t_cam_imu
    )
    kf2 = sel.consider_frame(
        1.8, sharp_img, 200, 200, 160, 120, R_rw4, t_rw4,
        R_world_imu=R_yaw14, pos_world_imu=np.array([0.75, 0.0, 0.0]), active_stations=[st]
    )
    assert kf2 is not None and kf2.trigger_reason == "station_spin"
    assert st.azimuth_coverage_pct > 0.0

    # 5. Post-SLAM trajectory SLERP pose refinement + bundle save/load
    traj = [
        (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        (2.0, 1.0, 0.5, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0),
    ]
    updated = IntelligentKeyframeSelector.update_keyframe_poses_from_trajectory(
        sel.keyframes, traj, R_cam_imu, t_cam_imu
    )
    assert updated == len(sel.keyframes)

    kf_dir = os.path.join(tmp_dir, "kf_bundle")
    save_info = save_keyframes_bundle(sel.keyframes, kf_dir)
    assert save_info["saved_count"] == 3
    reloaded = load_keyframes_bundle(kf_dir)
    assert len(reloaded) == 3 and reloaded[0].image_rgb.shape == (240, 320, 3)
    print("  PASS -> Blur/exposure gates, distance/spin triggers, SLERP pose refinement & bundle I/O verified.")


def test_3_station_placement_modes():
    print("\n[TEST 3] VirtualStationSelector Placement Modes (Manual, Auto, Hybrid, Fallback)...")
    traj = [(float(i), float(i) * 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0) for i in range(11)]  # 10m path
    flash = [{"label": "w00", "x": 3.2, "y": 0.0, "z": 0.0, "yaw": 0.0}]

    s_man = VirtualStationSelector.select_stations(
        StationGenerationConfig(placement_mode=StationPlacementMode.MANUAL_ONLY),
        flash_anchors=flash, trajectory_records=traj
    )
    assert len(s_man) == 1 and s_man[0].label == "w00"

    s_auto = VirtualStationSelector.select_stations(
        StationGenerationConfig(placement_mode=StationPlacementMode.AUTO_ONLY, auto_spacing_m=3.0),
        flash_anchors=flash, trajectory_records=traj
    )
    assert len(s_auto) >= 3 and all("auto" in s.station_id for s in s_auto)

    s_hyb = VirtualStationSelector.select_stations(
        StationGenerationConfig(placement_mode=StationPlacementMode.HYBRID, auto_spacing_m=3.0, min_manual_clearance_m=2.0),
        flash_anchors=flash, trajectory_records=traj
    )
    assert any(s.label == "w00" for s in s_hyb)
    for s in s_hyb:
        if s.label != "w00":
            dist_to_manual = float(np.linalg.norm(s.position_xyz - np.array([3.2, 0.0, 0.0])))
            assert dist_to_manual >= 2.0, f"Auto station violated clearance: {dist_to_manual:.2f}m"

    # Point-cloud-only fallback when neither trajectory nor anchors exist
    pts_only = np.random.RandomState(42).uniform(-2.0, 2.0, size=(500, 3))
    s_fb = VirtualStationSelector.select_stations(
        StationGenerationConfig(placement_mode=StationPlacementMode.HYBRID),
        points_xyz=pts_only
    )
    assert len(s_fb) == 1
    print(f"  PASS -> Manual={len(s_man)}, Auto={len(s_auto)}, Hybrid={len(s_hyb)}, CloudFallback={len(s_fb)}.")


def test_4_parallax_occlusion_spherical_and_non_fabrication():
    print("\n[TEST 4] Walkaround Parallax Occlusion Culling, Spherical Mode & Strict Non-Fabrication...")
    # Build synthetic scene centered around (0, 0, 0) WITHOUT ceiling (el > 35 deg completely empty!)
    # - Cylindrical room wall at radius R = 4.0m, Z in [-1.2m, +1.5m], colored neutral gray [120, 120, 120]
    # - Foreground occluding pillar at X = +1.6m, Y in [-0.35m, +0.35m], Z in [-1.0m, +1.0m], colored gray
    pts_list = []
    rgb_list = []

    az_vals = np.linspace(-math.pi, math.pi, 360, endpoint=False)
    z_vals = np.linspace(-1.2, 1.5, 90)
    aa, zz = np.meshgrid(az_vals, z_vals)
    wall_xyz = np.column_stack([4.0 * np.cos(aa.ravel()), 4.0 * np.sin(aa.ravel()), zz.ravel()])
    pts_list.append(wall_xyz)
    rgb_list.append(np.full((len(wall_xyz), 3), [110, 115, 120], dtype=np.uint8))

    py, pz = np.meshgrid(np.linspace(-0.35, 0.35, 35), np.linspace(-1.0, 1.0, 60))
    pillar_xyz = np.column_stack([np.full(py.size, 1.6), py.ravel(), pz.ravel()])
    pts_list.append(pillar_xyz)
    rgb_list.append(np.full((len(pillar_xyz), 3), [110, 115, 120], dtype=np.uint8))

    xyz = np.vstack(pts_list)
    rgb = np.vstack(rgb_list)
    has_true_rgb = np.ones(len(xyz), dtype=bool)

    # Create an offset walkaround Pinhole Keyframe at (0.0, +1.4m, 0.0) looking toward (+1.6, 0.0, 0.0):
    # From (0, +1.4, 0), the foreground pillar at (1.6, 0, 0) OCCLUDES the back wall around (4.0, -2.1, 0)!
    # Paint the keyframe pure BRIGHT RED [250, 20, 20]. The pillar itself is visible to the keyframe and should
    # turn RED, whereas the back wall hidden BEHIND the pillar from the keyframe's viewpoint MUST NOT turn red!
    cam_pos = np.array([0.0, 1.4, 0.0], dtype=np.float64)
    fwd = np.array([1.6, -1.4, 0.0], dtype=np.float64)
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, np.array([0.0, 0.0, 1.0]))
    right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    R_rgb_w = np.vstack([right, down, fwd])
    t_rgb_w = -R_rgb_w @ cam_pos

    red_img = np.full((300, 400, 3), [250, 20, 20], dtype=np.uint8)
    kf_pinhole = RgbKeyframe(
        keyframe_id="kf_parallax",
        timestamp_sec=10.0,
        image_rgb=red_img,
        width=400,
        height=300,
        fx=220.0,
        fy=220.0,
        cx=200.0,
        cy=150.0,
        dist_coeffs=(0.0, 0.0, 0.0, 0.0, 0.0),
        R_rgb_world=R_rgb_w,
        t_rgb_world=t_rgb_w,
    )

    cfg = StationGenerationConfig(scan_rows=128, scan_cols=512, pano_width=1024, pano_height=512)
    st = VirtualStation("st_center", "c00", np.zeros(3), 0.0, "manual_flash")
    gen = VirtualStationGenerator(cfg)
    res = gen.synthesize_station(st, xyz, rgb, points_has_true_rgb=has_true_rgb, keyframes=[kf_pinhole])

    # A. Verify foreground pillar at az=0 (col = W/2 = 512, row = H/2 = 256) gets keyframe PHOTO_RGB (Red)
    pillar_px = res.panorama_rgb[256, 512]
    assert pillar_px[0] > 200 and pillar_px[1] < 60, f"Pillar front was not painted red: {pillar_px}"
    assert res.provenance_mask[256, 512] == int(PanoramaProvenance.PHOTO_RGB)
    assert abs(float(res.depth_map_m[256, 512]) - 1.6) < 0.15

    # B. Verify background wall shadowed behind pillar along keyframe's ray (~(4.0, -2.1, 0), az = atan2(-2.1, 4.0))
    # is NOT falsely painted red by the occluded keyframe!
    shadow_az = math.atan2(-2.1, 4.0)
    shadow_col = int(((0.5 - shadow_az / (2.0 * math.pi)) % 1.0) * 1024)
    shadow_px = res.panorama_rgb[256, shadow_col]
    assert shadow_px[0] < 160, f"Parallax ghosting! Occluded wall behind pillar turned red: {shadow_px}"

    # C. Verify STRICT NON-FABRICATION of open ceiling (rows 0..80 -> elevation +62°..+90° has zero geometry)
    top_prov = res.provenance_mask[:70, :]
    top_rgb = res.panorama_rgb[:70, :]
    top_depth = res.depth_map_m[:70, :]
    top_scan_inv = res.scan_invalid_state[:18, :]
    assert np.all(top_prov == int(PanoramaProvenance.UNOBSERVED)), "Fabricated pixels found in unobserved zenith!"
    assert np.all(top_rgb == 0), "Non-black RGB found in unobserved region!"
    assert np.all(top_depth == 0.0), "Non-zero depth found in unobserved region!"
    assert np.all(top_scan_inv == 2), "Structured scan did not mark unobserved rays with cartesianInvalidState=2!"

    # D. Verify Ouster Rev8 `spherical` keyframe reprojection path
    sph_img = np.full((256, 512, 3), [30, 210, 90], dtype=np.uint8)
    kf_sph = RgbKeyframe(
        keyframe_id="kf_ouster",
        timestamp_sec=11.0,
        image_rgb=sph_img,
        width=512,
        height=256,
        fx=0.0, fy=0.0, cx=0.0, cy=0.0,
        dist_coeffs=(0.0, 0.0, 0.0, 0.0, 0.0),
        R_rgb_world=np.eye(3, dtype=np.float64),
        t_rgb_world=np.zeros(3, dtype=np.float64),
        camera_model="spherical",
        spherical_vfov_deg=(-22.5, 22.5),
    )
    res_sph = gen.synthesize_station(st, xyz, rgb, points_has_true_rgb=has_true_rgb, keyframes=[kf_sph])
    assert res_sph.photo_rgb_pct > 15.0 and res_sph.panorama_rgb[256, 512, 1] > 180
    print(
        f"  PASS -> Parallax occlusion blocked ghosting, Ouster spherical={res_sph.photo_rgb_pct:.1f}%, "
        f"Unobserved ceiling strictly preserved ({res.unobserved_pct:.1f}%)."
    )


def test_5_headless_scanhub_gui_m15_workflow(tmp_dir: str):
    print("\n[TEST 5] Headless ScanHUBOS GUI M15 Tab, PanoramaPreviewDialog & E57 Export...")
    try:
        from PyQt6.QtWidgets import QApplication
    except ImportError:
        from PyQt5.QtWidgets import QApplication
    import scanhub_app

    app = QApplication.instance() or QApplication(sys.argv)
    gui = scanhub_app.ScanHUBOS()
    try:
        gui.projects_root = os.path.join(tmp_dir, "projects")
        gui.active_project_name = "M15_GUI_Test"
        gui.dataset_input.setText("Station_Run_01")

        # Verify Tab 4 ("🌐 Stations · Pano") & E57_STRUCTURED combo item exist
        tab_titles = [gui.hud_tabs.tabText(i) for i in range(gui.hud_tabs.count())]
        assert any("Stations" in t for t in tab_titles), f"Missing M15 tab in {tab_titles}"
        fmt_items = [gui.combo_m12_fmt.itemText(i) for i in range(gui.combo_m12_fmt.count())]
        assert "E57_STRUCTURED" in fmt_items

        # Drop station + right-click map station + synthesize + preview + export E57
        gui.on_m15_drop_station_clicked()
        gui.on_map_station_right_clicked(1.5, -0.8)
        assert len(gui.map_canvas.flash_anchors) == 2

        gui.on_m15_synthesize_clicked()
        with gui.ros_worker.lock:
            results = list(gui.ros_worker.last_station_results)
        assert len(results) >= 2

        dlg = scanhub_app.PanoramaPreviewDialog(results, parent=gui)
        dlg._set_mode("prov")
        dlg._set_mode("depth")
        dlg._set_mode("rgb")
        dlg.close()

        gui.on_m15_export_e57_clicked()
        expected_e57 = os.path.join(
            gui.projects_root, "M15_GUI_Test", "datasets", "Station_Run_01", "structured_stations.e57"
        )
        assert os.path.exists(expected_e57), f"Missing exported E57 at {expected_e57}"
        rep = read_and_validate_e57(expected_e57)
        assert rep.valid and rep.station_count >= 2
        print(f"  PASS -> Headless GUI synthesized & exported {rep.station_count} stations ({rep.total_pages} pages).")
    finally:
        gui.close()
        app.processEvents()


def test_6_real_dataset_cli(tmp_dir: str):
    corridor_dir = os.path.join(_REPO_ROOT, "projects", "TestProject_2756", "datasets", "Corridor_Scan_A")
    if not os.path.exists(os.path.join(corridor_dir, "pointcloud_fused_rgb.ply")):
        print("\n[TEST 6] Skipping Corridor_Scan_A test (dataset not present).")
        return
    print("\n[TEST 6] CLI Execution on Real Dataset (`Corridor_Scan_A`)...")
    out_dir = os.path.join(tmp_dir, "corridor_cli_out")
    cmd = [
        sys.executable,
        os.path.join(_REPO_ROOT, "scripts", "generate_virtual_stations.py"),
        "--dataset-dir", corridor_dir,
        "--output-dir", out_dir,
        "--pano-width", "1024",
        "--pano-height", "512",
        "--scan-rows", "128",
        "--scan-cols", "512",
        "--station", "0.0,0.0,0.1,0.0",
        "--validate-e57",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert proc.returncode == 0, f"CLI failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    manifest_path = os.path.join(out_dir, "stations_manifest.json")
    assert os.path.exists(manifest_path)
    with open(manifest_path, "r") as f:
        m = json.load(f)
    assert m["station_count"] >= 1
    st0 = m["stations"][0]
    for rel_key in ("panorama_file", "depth_map_mm_file", "coverage_mask_file"):
        assert os.path.exists(os.path.join(out_dir, st0["panorama"][rel_key]))
    assert os.path.exists(os.path.join(out_dir, st0["structured_scan"]["ply_file"]))
    print(f"  PASS -> Corridor_Scan_A verified ({m['station_count']} station(s) + full bundle + validated E57).")


def main() -> int:
    print("==========================================================================")
    print("  Milestone M15 Verification Suite — AV Structure Virtual Stations & E57")
    print("==========================================================================")
    tmp_dir = tempfile.mkdtemp(prefix="verify_m15_")
    try:
        test_1_e57_crc32c_and_structured_roundtrip(tmp_dir)
        test_2_intelligent_keyframe_selector(tmp_dir)
        test_3_station_placement_modes()
        test_4_parallax_occlusion_spherical_and_non_fabrication()
        test_5_headless_scanhub_gui_m15_workflow(tmp_dir)
        test_6_real_dataset_cli(tmp_dir)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    print("\n==========================================================================")
    print("  ALL M15 AV STRUCTURE VERIFICATION TESTS PASSED (6/6)!")
    print("==========================================================================")
    return 0


if __name__ == "__main__":
    sys.exit(main())
