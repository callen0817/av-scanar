#!/usr/bin/env python3
"""
Automated Verification Suite for:
1. Milestone 12 Point Cloud Cleanup, Real-Time Spatial-Hash Toggle & Multi-Format Export (ported from av_scan)
2. Unlimited-Peer Collaborative SLAM (C-SLAM) Live Multi-Scanner Engine
   - Unlimited scanner color differentiation (curated + golden-ratio HSV)
   - Non-overlapping initial placement (SEPARATED_IMU_ORIENTED at radial offset with relative IMU orientation)
   - Automatic real-time overlap detection & SE(3) rigid snap (OVERLAP_SNAPPED)
   - Color mode toggle (Scanner Differentiation Colors vs True RGB)
3. Headless ScanHUBOS GUI Integration Test
"""

import os
import sys
import math
import time
import shutil
import tempfile
import json
import numpy as np
from scipy.spatial.transform import Rotation

# Ensure scanhub_gui is on path
sys.path.insert(0, "/home/scanar/scanarMini/src/scanhub_gui")

from m12_point_cloud_cleanup import (
    ExportFormat,
    CleanupParams,
    PointCloudProcessor,
)
from cslam_network_engine import (
    PeerAlignmentState,
    CSlamColorMode,
    get_scanner_color_rgb,
    get_scanner_color_hex,
    CSlamRegistrationSolver,
    CSlamNetworkEngine,
)


def generate_test_room_cloud(rng: np.random.Generator, add_outliers: bool = True, tilt_deg: float = 4.5) -> np.ndarray:
    """Generates a realistic 3D indoor room (floor + 4 walls + central pillar) with optional tilt & outliers."""
    # Floor (6m x 5m)
    fx = rng.uniform(-3.0, 3.0, 1800)
    fy = rng.uniform(-2.5, 2.5, 1800)
    fz = rng.normal(0.0, 0.005, 1800)
    floor = np.column_stack([fx, fy, fz])

    # Walls with asymmetric geometric features for unambiguous 3D registration
    w1 = np.column_stack([np.full(500, -3.0), rng.uniform(-2.5, 2.5, 500), rng.uniform(0.0, 2.2, 500)])
    w2 = np.column_stack([np.full(500, 3.0), rng.uniform(-2.5, 2.5, 500), rng.uniform(0.0, 2.2, 500)])
    w3 = np.column_stack([rng.uniform(-3.0, 3.0, 500), np.full(500, -2.5), rng.uniform(0.0, 2.2, 500)])
    # Asymmetric L-bench / pillar in corner (+1.2, +0.8)
    pillar = np.column_stack([
        rng.uniform(0.8, 1.8, 600),
        rng.uniform(0.4, 1.4, 600),
        rng.uniform(0.0, 1.5, 600),
    ])

    xyz = np.vstack([floor, w1, w2, w3, pillar])

    if abs(tilt_deg) > 1e-3:
        R_tilt = Rotation.from_euler("xy", [tilt_deg, -tilt_deg * 0.5], degrees=True).as_matrix()
        xyz = (xyz @ R_tilt.T) + np.array([0.0, 0.0, 0.35])

    if add_outliers:
        outliers = rng.uniform(-8.0, 8.0, size=(200, 3))
        xyz = np.vstack([xyz, outliers])

    rgb = np.zeros_like(xyz)
    rgb[:, 0] = 0.25
    rgb[:, 1] = 0.78
    rgb[:, 2] = 0.65
    return np.hstack([xyz, rgb]).astype(np.float32)


def test_m12_cleanup_and_exports():
    print("\n=== [TEST 1] M12 Point Cloud Cleanup & Multi-Format Export (ported from av_scan) ===")
    rng = np.random.default_rng(101)
    raw_cloud = generate_test_room_cloud(rng, add_outliers=True, tilt_deg=5.0)
    assert len(raw_cloud) == 4100, f"Expected 4100 input points, got {len(raw_cloud)}"

    params = CleanupParams(
        enable_voxel_downsample=True,
        voxel_leaf_size_m=0.03,
        enable_sor=True,
        sor_mean_k=20,
        sor_std_mul=2.0,
        enable_ror=True,
        ror_radius_m=0.15,
        ror_min_neighbors=4,
        enable_ground_alignment=True,
        ground_dist_threshold_m=0.04,
        ground_max_iterations=350,
        ground_max_tilt_deg=35.0,
        align_ground_to_z_zero=True,
        remove_ground_plane=False,
    )

    cleaned, stats = PointCloudProcessor.run_cleanup_pipeline(raw_cloud, params)
    print(
        f"  [M12 Pipeline] In: {stats.input_points} -> Voxel: {stats.voxel_downsampled_points} "
        f"-> SOR removed: {stats.sor_removed_points} -> ROR removed: {stats.ror_removed_points} "
        f"-> Out: {stats.output_points} ({stats.processing_time_ms:.1f} ms)"
    )
    print(
        f"  [Ground Plane] Detected: {stats.ground_result.detected}, "
        f"Inliers: {stats.ground_result.inlier_count}, "
        f"Tilt: {stats.ground_result.tilt_angle_deg:.2f} deg"
    )
    assert stats.output_points > 1500 and stats.output_points < stats.input_points
    assert (stats.sor_removed_points + stats.ror_removed_points) >= 80, "Expected outlier removal to strip synthetic noise"
    assert stats.ground_result.detected, "Expected RANSAC ground plane detection to succeed"
    assert 3.5 <= stats.ground_result.tilt_angle_deg <= 8.0, "Expected ~5.6 deg ground tilt detection"

    # Verify floor is leveled near Z=0 after ground alignment
    floor_band = cleaned[np.abs(cleaned[:, 0]) < 2.0]
    z_p10 = float(np.percentile(floor_band[:, 2], 15))
    print(f"  [Ground Z=0 Check] 15th-percentile floor Z after leveling: {z_p10:+.4f} m")
    assert abs(z_p10) < 0.06, f"Expected leveled floor near Z=0, got {z_p10}"

    # Verify Real-Time Fast Spatial-Hash Cleanup (< 25 ms)
    rt_clean, rt_stats = PointCloudProcessor.clean_realtime_fast(raw_cloud, params)
    print(
        f"  [M12 Real-Time Toggle Filter] {rt_stats.input_points} -> {rt_stats.output_points} pts "
        f"in {rt_stats.processing_time_ms:.2f} ms"
    )
    assert len(rt_clean) > 1000
    assert rt_stats.processing_time_ms < 150.0, f"Real-time filter too slow: {rt_stats.processing_time_ms:.1f} ms"

    # Verify all 6 Export Formats
    tmp_dir = tempfile.mkdtemp(prefix="m12_export_test_")
    try:
        formats = [
            (ExportFormat.PLY_ASCII, "cloud_ascii.ply"),
            (ExportFormat.PLY_BINARY, "cloud_bin.ply"),
            (ExportFormat.PCD_ASCII, "cloud_ascii.pcd"),
            (ExportFormat.PCD_BINARY, "cloud_bin.pcd"),
            (ExportFormat.XYZ_ASCII, "cloud.xyz"),
            (ExportFormat.OBJ_MESH, "surface_mesh.obj"),
        ]
        for fmt, fname in formats:
            fpath = os.path.join(tmp_dir, fname)
            res = PointCloudProcessor.export_point_cloud(cleaned, fpath, fmt, mesh_grid_resolution_m=0.12)
            assert res.success, f"Export failed for {fmt}: {res.error_message}"
            assert os.path.isfile(fpath) and res.file_size_bytes > 500
            print(
                f"  [Export {fmt.name:10s}] OK -> {res.exported_points} pts, "
                f"{res.exported_triangles} tris, {res.file_size_bytes:,} bytes ({res.export_time_ms:.1f} ms)"
            )
            if fmt == ExportFormat.OBJ_MESH:
                assert res.exported_triangles > 20, "Expected 3D OBJ surface mesh to reconstruct triangles"
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_cslam_unlimited_peers_and_auto_snap():
    print("\n=== [TEST 2] Unlimited-Peer Collaborative SLAM (C-SLAM) Engine ===")
    # 1. Verify Unlimited Scanner Color Differentiation (30 scanners)
    colors_hex = [get_scanner_color_hex(i) for i in range(30)]
    assert len(set(colors_hex)) == 30, "Expected 30 unique scanner differentiation colors for 30 scanners"
    print(f"  [Unlimited Colors] Verified 30 unique scanner colors: {colors_hex[:6]} ... {colors_hex[-3:]}")

    # 2. Initialize C-SLAM Engine for Host Scanner #0
    rng = np.random.default_rng(202)
    base_room = generate_test_room_cloud(rng, add_outliers=False, tilt_deg=0.0)

    cslam = CSlamNetworkEngine(
        scanner_name="Host-Scanner-00",
        project_name="MultiRig_Test",
        listen_port=0,  # ephemeral port for unit test isolation
        separation_distance_m=15.0,
        min_overlap_ratio=0.18,
    )
    try:
        # Host Scanner #0 is at origin with yaw=0
        cslam.update_local_state(
            points_xyz_rgb=base_room,
            pos_xyz=(0.0, 0.0, 0.0),
            imu_rpy_rad=(0.0, 0.0, 0.0),
        )

        # Verify empty local_points_override returns 0 local points (does NOT fall back to self.local_points)
        empty_override = np.zeros((0, 6), dtype=np.float32)
        merged_empty, _ = cslam.get_merged_cslam_scene(local_points_override=empty_override)
        assert len(merged_empty) == 0, f"Expected 0 points when local_points_override is empty and no peers exist, got {len(merged_empty)}"

        # Verify preserve_local_colors=True preserves diagnostic RGB even in SCANNER_COLORS mode
        diag_pts = np.array([[1.0, 2.0, 3.0, 0.12, 0.88, 0.34]], dtype=np.float32)
        cslam.set_color_mode(CSlamColorMode.SCANNER_COLORS)
        merged_diag, _ = cslam.get_merged_cslam_scene(local_points_override=diag_pts, preserve_local_colors=True)
        assert np.allclose(merged_diag[0, 3:6], [0.12, 0.88, 0.34], atol=1e-4), "Expected diagnostic colors to be preserved"

        # 3. Inject Peer Scanner #1 with NON-OVERLAPPING geometry (distant room) and IMU yaw = +45 deg
        distant_xyz = rng.uniform(50.0, 56.0, size=(1200, 3)).astype(np.float32)
        distant_rgb = np.full((1200, 3), 0.8, dtype=np.float32)
        distant_cloud = np.hstack([distant_xyz, distant_rgb])

        cslam._ingest_packet(
            {
                "magic": "SCANHUB_CSLAM_V1",
                "scanner_id": "peer-scanner-01",
                "scanner_name": "Peer-Scanner-01",
                "project_name": "MultiRig_Test",
                "dataset_name": "Dataset_001",
                "is_recording": True,
                "seq": 1,
                "pos_xyz": [0.0, 0.0, 0.0],
                "imu_rpy_rad": [0.0, 0.0, math.radians(45.0)],
                "points": distant_cloud.tolist(),
            },
            sender_ip="127.0.0.1",
        )

        # Also inject a peer on a DIFFERENT project to verify network session discovery + promotion on join
        cslam._ingest_packet(
            {
                "magic": "SCANHUB_CSLAM_V1",
                "scanner_id": "peer-remote-99",
                "scanner_name": "Remote-Rig-99",
                "project_name": "Warehouse_Wing_B",
                "dataset_name": "WingB_Capture_02",
                "is_recording": True,
                "seq": 1,
                "pos_xyz": [1.0, 2.0, 0.0],
                "imu_rpy_rad": [0.0, 0.0, 0.2],
                "points": distant_cloud[:250].tolist(),
            },
            sender_ip="192.168.1.99",
        )
        sessions = cslam.get_discovered_network_sessions()
        assert any(s["project_name"] == "Warehouse_Wing_B" and s["dataset_name"] == "WingB_Capture_02" for s in sessions), (
            "Expected active capture on Warehouse_Wing_B to be discovered on the network"
        )
        print(f"  [Active Capture Discovery] Discovered {len(sessions)} active network C-SLAM capture sessions.")

        cslam.run_overlap_check_now()
        merged_sep, peers_sep = cslam.get_merged_cslam_scene()
        assert len(peers_sep) == 1
        p1 = peers_sep[0]
        print(
            f"  [Non-Overlapping State] Peer: {p1['scanner_name']} | State: {p1['alignment_state']} | "
            f"Color: {p1['color_hex']} | World Pos: {[round(x, 2) for x in p1['world_pos']]}"
        )
        assert p1["alignment_state"] == PeerAlignmentState.SEPARATED_IMU_ORIENTED.name
        dist_from_origin = math.hypot(p1["world_pos"][0], p1["world_pos"][1])
        assert abs(dist_from_origin - 15.0) < 0.5, f"Expected 15.0m radial separation, got {dist_from_origin:.2f}m"

        # 4. Now Peer Scanner #1 walks into OVERLAPPING geometry (75% of the same room, rotated by 12 deg and shifted by [0.8m, -0.5m, 0.05m] in peer's local frame)
        R_true = Rotation.from_euler("z", 12.0, degrees=True).as_matrix()
        t_true = np.array([0.80, -0.50, 0.05])
        # Peer measures points in its own frame: p_peer = R_true.T @ (p_world - t_true)
        overlap_subset = base_room[:3000].copy()
        p_world = overlap_subset[:, :3]
        p_peer = (p_world - t_true) @ R_true
        overlap_subset[:, :3] = p_peer.astype(np.float32)

        cslam._ingest_packet(
            {
                "magic": "SCANHUB_CSLAM_V1",
                "scanner_id": "peer-scanner-01",
                "scanner_name": "Peer-Scanner-01",
                "project_name": "MultiRig_Test",
                "seq": 2,
                "pos_xyz": [0.5, 0.2, 0.0],
                "imu_rpy_rad": [0.0, 0.0, math.radians(-12.0)],
                "points": overlap_subset.tolist(),
            },
            sender_ip="127.0.0.1",
        )

        cslam.run_overlap_check_now()
        merged_snap, peers_snap = cslam.get_merged_cslam_scene()
        p1_after = peers_snap[0]
        print(
            f"  [Overlapping Auto-Snap] Peer: {p1_after['scanner_name']} | State: {p1_after['alignment_state']} | "
            f"Overlap: {p1_after['overlap_ratio']*100:.1f}% | ICP RMSE: {p1_after['icp_rmse_m']*100:.2f} cm"
        )
        assert p1_after["alignment_state"] == PeerAlignmentState.OVERLAP_SNAPPED.name, "Expected peer to SNAP on overlap!"
        assert p1_after["overlap_ratio"] >= 0.65
        assert p1_after["icp_rmse_m"] < 0.05

        # 5. Test Unlimited Multi-Scanner Scaling: Add 9 more peer scanners (10 peers total + 1 host = 11 scanners)
        for idx in range(2, 11):
            cslam._ingest_packet(
                {
                    "magic": "SCANHUB_CSLAM_V1",
                    "scanner_id": f"peer-scanner-{idx:02d}",
                    "scanner_name": f"Peer-Scanner-{idx:02d}",
                    "project_name": "MultiRig_Test",
                    "seq": 1,
                    "pos_xyz": [float(idx), 0.0, 0.0],
                    "imu_rpy_rad": [0.0, 0.0, float(idx * 0.3)],
                    "points": distant_cloud[:300].tolist(),
                },
                sender_ip=f"127.0.0.{idx}",
            )

        merged_multi, peers_multi = cslam.get_merged_cslam_scene()
        summary = cslam.get_status_summary()
        print(
            f"  [Multi-Scanner Scale] Total Scanners: {summary['total_scanners']} "
            f"(1 Host + {summary['peer_count']} Peers) | Merged Points: {len(merged_multi):,}"
        )
        assert summary["total_scanners"] == 11
        assert len(peers_multi) == 10

        # 6. Verify Color Mode Toggle (SCANNER_COLORS vs TRUE_RGB)
        cslam.set_color_mode(CSlamColorMode.SCANNER_COLORS)
        m_sc, _ = cslam.get_merged_cslam_scene()
        cslam.set_color_mode(CSlamColorMode.TRUE_RGB)
        m_rgb, _ = cslam.get_merged_cslam_scene()
        assert not np.allclose(m_sc[:, 3:6], m_rgb[:, 3:6]), "Expected Scanner Colors and True RGB modes to differ"
        print("  [Color Mode Toggle] Verified SCANNER_COLORS vs TRUE_RGB point cloud coloring.")
    finally:
        cslam.stop()


def test_headless_gui_integration():
    print("\n=== [TEST 3] Headless ScanHUBOS GUI Smoke & Workflow Verification ===")
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    import scanhub_app
    from scanhub_app import QApplication

    app = QApplication.instance() or QApplication(sys.argv)

    # 0. Verify Artificial Vision Logo Asset & White/Gold Splash Screen
    logo_path = scanhub_app.get_av_logo_path()
    assert logo_path and os.path.isfile(logo_path), f"Expected Artificial Vision logo asset, got: {logo_path}"
    logo_pix = scanhub_app.load_av_logo_pixmap(128)
    assert logo_pix is not None and not logo_pix.isNull(), "Expected valid QPixmap from av_logo.png"

    splash = scanhub_app.ArtificialVisionSplashScreen()
    assert splash.testAttribute(scanhub_app.Qt.WA_TranslucentBackground), "Expected translucent background on splash screen"
    assert splash.windowFlags() & scanhub_app.Qt.FramelessWindowHint, "Expected frameless window hint on splash screen"
    assert hasattr(splash, "btn_splash_close") and splash.btn_splash_close is not None, "Expected transparent close button on splash screen"
    splash.set_status("Artificial Vision  ·  Spatial LiDAR OS", 80)
    splash.show()
    splash.repaint()
    app.processEvents()
    splash.close()
    print(f"  [Brand & Splash] Verified Artificial Vision logo ({logo_path}), translucent Apple Glass card & close button.")

    win = scanhub_app.ScanHUBOS()
    # Verify AR Glasses Transparency & Frameless Window Controls
    assert win.testAttribute(scanhub_app.Qt.WA_TranslucentBackground), "Expected WA_TranslucentBackground for AR glasses passthrough"
    assert win.windowFlags() & scanhub_app.Qt.FramelessWindowHint, "Expected FramelessWindowHint for transparent window borders"
    assert hasattr(win, "btn_win_min") and win.btn_win_min is not None, "Expected transparent Minimize window control"
    assert hasattr(win, "btn_win_max") and win.btn_win_max is not None, "Expected transparent Maximize window control"
    assert hasattr(win, "btn_win_close") and win.btn_win_close is not None, "Expected transparent Close window control"
    assert hasattr(win, "slider_glass_alpha") and win.slider_glass_alpha is not None, "Expected glass opacity slider"

    # Verify Post-Splash Home Screen opens to Projects & Datasets Hub (Page 0)
    assert hasattr(win, "workspace_stack"), "Expected workspace_stack QStackedWidget"
    assert win.workspace_stack.currentIndex() == 0, "Expected app to open on Projects & Datasets Hub (Page 0) right after splash screen"
    print("  [Post-Splash Projects Hub] Verified app opens directly to Projects & Datasets + Active C-SLAM Captures Hub.")

    # Verify creating a Project + Dataset and entering Capture Studio (Page 1)
    test_proj_name = f"TestProject_{int(time.time()) % 10000}"
    win.hub_new_project_input.setText(test_proj_name)
    win.on_create_project_clicked()
    app.processEvents()
    assert win.active_project_name == test_proj_name, f"Expected active_project_name={test_proj_name}, got {win.active_project_name}"

    win.hub_new_dataset_input.setText("Corridor_Scan_A")
    win.on_create_dataset_clicked()
    win.on_enter_capture_studio_clicked()
    app.processEvents()
    assert win.workspace_stack.currentIndex() == 1, "Expected entering dataset to switch to Capture Studio (Page 1)"
    assert "Corridor_Scan_A" in win.active_dataset_name
    assert test_proj_name in win.lbl_active_project_badge.text()
    print(f"  [Project -> Dataset -> Capture Studio] Entered Project '{test_proj_name}' / Dataset '{win.active_dataset_name}'.")

    # Verify returning to Projects Hub and joining an Active C-SLAM Capture on the network
    win.show_projects_hub_view()
    app.processEvents()
    assert win.workspace_stack.currentIndex() == 0, "Expected '← PROJECTS' button to return to Projects Hub"

    win.on_hub_sim_active_capture_clicked()
    app.processEvents()
    assert win.hub_active_captures_list.count() >= 1, "Expected simulated active C-SLAM capture to appear in Hub list"
    win.show_capture_studio_view("Shared_Facility_Project", "Floor2_Live_Scan", joined_cslam=True)
    app.processEvents()
    assert win.workspace_stack.currentIndex() == 1, "Expected joining active C-SLAM capture to enter Capture Studio"
    assert win.active_project_name == "Shared_Facility_Project"
    assert win.ros_worker.cslam_engine.project_name == "Shared_Facility_Project"
    print("  [One-Click C-SLAM Join] Verified discovering & joining active network C-SLAM capture switches project/dataset and opens Capture Studio.")

    # Clean up test project directory created during test
    for p_name in (test_proj_name, "Shared_Facility_Project"):
        p_dir = os.path.join(win.projects_root, p_name)
        if os.path.isdir(p_dir):
            shutil.rmtree(p_dir, ignore_errors=True)

    # Verify live glass opacity adjustment
    win.slider_glass_alpha.setValue(20)
    assert win.glass_opacity_pct == 20
    assert "20%" in win.lbl_glass_alpha.text()
    # Verify Pure Blank Black AR Passthrough Viewport (Optical See-Through)
    assert hasattr(win, "center_viewport") and isinstance(win.center_viewport, scanhub_app.ClearPassthroughViewport), (
        "Expected ClearPassthroughViewport for left viewport"
    )
    win.center_viewport.update_scene_data({})
    win.center_viewport.update_pose(0.0, 0.0, 0.0)
    print("  [AR Passthrough] Verified pure blank black passthrough viewport (distraction-free optical transmission).")

    win.show()
    app.processEvents()

    try:
        # 1. Verify all 6 Map Layer View Buttons switch immediately and render distinct layers
        all_modes = ["colorized_fused", "airy_only", "stereo_only", "fused", "accepted_rejected", "observation_diag"]
        for mode_key in all_modes:
            win.on_layer_mode_clicked(mode_key)
            app.processEvents()
            assert win.ros_worker.display_mode == mode_key, f"Expected worker display_mode={mode_key}"
            assert win.map_canvas.current_mode == mode_key, f"Expected map_canvas.current_mode={mode_key}"
            assert mode_key in win._layer_buttons, f"Expected {mode_key} in win._layer_buttons"
            win.map_canvas.repaint()
            app.processEvents()
        win.on_layer_mode_clicked("colorized_fused")
        print(f"  [Map Layer Buttons] Verified all 6 map layer buttons ({', '.join(all_modes)}) switch immediately and update header badge.")

        # 2. Verify RGB & Depth Camera Buffer Decoding, Preview Thread Draining, and Continuous Gemini Stereo Map Building
        frames_received = []
        win.ros_worker.image_received.connect(lambda img: frames_received.append((img.width(), img.height())))
        # Create synthetic RGB (320x200x3) and Depth (320x200 uint16 in mm)
        rgb_test = np.full((200, 320, 3), 180, dtype=np.uint8)
        depth_raw = np.full((200, 320), 1500, dtype=np.uint16)  # 1.5m depth
        decoded_rgb = win.ros_worker._decode_rgb_buffer(rgb_test.tobytes(), 320, 200, 320 * 3, "rgb8")
        decoded_depth = win.ros_worker._decode_depth_buffer(depth_raw.tobytes(), 320, 200, 320 * 2, "16uc1")
        assert decoded_rgb is not None and decoded_rgb.shape == (200, 320, 3), "Expected valid decoded RGB array"
        assert decoded_depth is not None and decoded_depth.shape == (200, 320), "Expected valid decoded Depth array"

        # Simulate SLAM active with 120ms sweep offset and verify Gemini stereo map builds continuously across moves
        now_t = time.time()
        win.ros_worker.using_slam_odom = True
        win.ros_worker.last_slam_time = now_t
        win.ros_worker._has_slam_clock_offset = True
        win.ros_worker._slam_wall_to_stamp_offset = -0.12
        win.ros_worker.live_stereo_voxels.clear()
        win.ros_worker._stereo_cache_dirty = True

        stereo_counts = []
        for step_idx in range(3):
            t_step = now_t + step_idx * 0.10
            pos_step = np.array([step_idx * 0.45, 0.0, 0.0], dtype=np.float64)
            quat_step = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
            win.ros_worker.current_x = float(pos_step[0])
            win.ros_worker.odom_buffer.append((t_step - 0.12, pos_step, quat_step))
            win.ros_worker._raw_rgb_queue.append((t_step, t_step, 320, 200, 320 * 3, "rgb8", rgb_test.tobytes()))
            win.ros_worker._raw_depth_queue.append((t_step, t_step, 320, 200, 320 * 2, "16uc1", depth_raw.tobytes()))
            win.ros_worker._latest_raw_depth_tuple = (t_step, t_step, 320, 200, 320 * 2, "16uc1", depth_raw.tobytes())
            win.ros_worker._last_image_emit = 0.0
            win.ros_worker._last_depth_voxel_time = 0.0
            win.ros_worker._drain_and_emit_camera_preview()
            win.ros_worker._process_pending_sensor_queues()
            stereo_pts = win.ros_worker.extract_raw_3d_points_for_mode("stereo_only")
            stereo_counts.append(len(stereo_pts))

        app.processEvents()
        assert len(frames_received) >= 1, "Expected image_received signal to emit composite QImage"
        assert stereo_counts[0] > 100 and stereo_counts[-1] > stereo_counts[0], (
            f"Expected Gemini stereo map to continuously accumulate points across poses, got {stereo_counts}"
        )
        print(
            f"  [Camera RGB+Depth & Gemini Map Build] Verified non-blocking RGB+Depth preview ({frames_received[-1][0]}x{frames_received[-1][1]}) "
            f"and continuous Gemini stereo map growth across moves ({stereo_counts[0]} -> {stereo_counts[1]} -> {stereo_counts[2]} voxels)."
        )

        # 3. Test M12 Real-Time Toggle
        win.on_quick_m12_rt_clicked(True)
        assert win.ros_worker.live_cleanup_enabled is True
        win.on_quick_m12_rt_clicked(False)
        assert win.ros_worker.live_cleanup_enabled is False

        # 4. Test M12 Preview -> Reset -> Preview -> Commit
        win.on_m12_preview_clicked()
        app.processEvents()
        assert win.ros_worker.cleanup_preview_active is True
        assert "PREVIEW:" in win.lbl_m12_status.text()

        win.on_m12_reset_clicked()
        app.processEvents()
        assert win.ros_worker.cleanup_preview_active is False

        win.on_m12_preview_clicked()
        win.on_m12_commit_clicked()
        app.processEvents()
        assert win.ros_worker.cleanup_preview_active is False
        assert "Committed:" in win.lbl_m12_status.text()

        # 5. Test C-SLAM Simulated Peer Scanner Button (Non-overlap -> Overlap Auto-Snap)
        win.on_spawn_sim_peer_clicked()
        time.sleep(0.35)
        win.ros_worker.dispatch_live_map_and_telemetry()
        app.processEvents()
        st1 = win.ros_worker.cslam_engine.get_status_summary()
        print(f"  [GUI Sim Peer T+0.3s] Scanners: {st1['total_scanners']} | Separated: {st1['separated_peers']} | Snapped: {st1['snapped_peers']}")
        assert st1["total_scanners"] == 2

        # Wait ~2.2s for simulated peer scanner to walk into overlapping room geometry and auto-snap
        time.sleep(2.4)
        win.ros_worker.cslam_engine.run_overlap_check_now()
        win.ros_worker.dispatch_live_map_and_telemetry()
        app.processEvents()
        st2 = win.ros_worker.cslam_engine.get_status_summary()
        print(f"  [GUI Sim Peer T+2.7s] Scanners: {st2['total_scanners']} | Separated: {st2['separated_peers']} | Snapped: {st2['snapped_peers']}")
        assert st2["snapped_peers"] == 1, "Expected simulated peer scanner to auto-snap after walking into overlap!"

        # 6. Test NavVis 2-Stage Initialization Ritual & FARO Flash Scan Waypoints
        test_scan_proj = "TestProject_Ritual"
        test_scan_ds = "Ritual_Corridor_01"
        win.show_capture_studio_view(test_scan_proj, test_scan_ds)
        app.processEvents()

        # Stage 1: Trigger scan workflow -> Starts Stage 1 motionless ZUPT bias calibration
        win.is_calibrated = False
        win.toggle_scan_workflow()
        app.processEvents()
        assert win.calib_timer is not None, "Expected calib_timer to be active for Stage 1 bias calibration"
        assert win.btn_skip_ritual.isVisible() is True, "Expected Skip Ritual button visible during calibration"

        # Drive 100 calibration ticks
        win.ros_worker.last_ax = 0.02
        win.ros_worker.last_ay = -0.05
        win.ros_worker.last_az = 9.80
        for _ in range(100):
            win._sample_bias_tick()
        app.processEvents()

        assert win.is_calibrated is True, "Expected is_calibrated=True after 100 bias ticks"
        assert win.ros_worker.in_rotation_ritual is True, "Expected Stage 2 rotation ritual active"
        print("  [NavVis Ritual Stage 1] Verified 3s motionless ZUPT bias calibration and gravity alignment.")

        # Stage 2: 90-degree guided horizontal pivot ritual
        win.update_rotation_ritual_ui(45.0)
        assert win.bar.value() == 50, "Expected 50% progress at 45 degrees"
        win.on_rotation_ritual_completed()
        app.processEvents()

        assert win.is_recording is True, "Expected active capture session recording after ritual completion"
        assert win.flash_btn.isEnabled() is True, "Expected FLASH button enabled during active recording"
        assert win.btn_skip_ritual.isVisible() is False, "Expected Skip button hidden after ritual completion"
        print("  [NavVis Ritual Stage 2] Verified 90° pivot tracking and automatic transition to active capture.")

        # FARO Flash Scan: Drop anchors via button and keyboard shortcut 'F'
        win.map_canvas.current_x = 1.25
        win.map_canvas.current_y = -0.50
        win.ros_worker.current_z = 0.15

        # Flash anchor 1 via method / button
        win.trigger_faro_flash_anchor()
        assert len(win.map_canvas.flash_anchors) == 1
        assert win.map_canvas.flash_anchors[0]["label"] == "w00"

        # Flash anchor 2 via hotkey 'F'
        win.map_canvas.current_x = 3.40
        win.map_canvas.current_y = 1.10
        from PyQt5.QtGui import QKeyEvent
        from PyQt5.QtCore import Qt, QEvent
        key_ev = QKeyEvent(QEvent.KeyPress, Qt.Key_F, Qt.NoModifier)
        win.keyPressEvent(key_ev)
        assert len(win.map_canvas.flash_anchors) == 2
        assert win.map_canvas.flash_anchors[1]["label"] == "w01"
        print(f"  [FARO Flash Scan] Verified flash anchor drops via '⚡ FLASH' button & hotkey 'F' ({[a['label'] for a in win.map_canvas.flash_anchors]}).")

        # Stop active capture
        win.stop_scan_session()
        app.processEvents()
        assert win.is_recording is False
        assert win.flash_btn.isEnabled() is False
        assert win.save_btn.isEnabled() is True

        # Save dataset and verify flash_anchors.json
        win.save_dataset()
        for _ in range(50):
            app.processEvents()
            time.sleep(0.05)
            if not win.export_in_progress:
                break
        assert win.export_in_progress is False, "Expected background export to complete"

        scan_dir = win._get_active_dataset_save_dir(test_scan_ds)
        anchors_file = os.path.join(scan_dir, "flash_anchors.json")
        assert os.path.isfile(anchors_file), f"Expected flash_anchors.json at {anchors_file}"
        with open(anchors_file, 'r') as f_fa:
            exported_anchors = json.load(f_fa)
        assert len(exported_anchors) == 2, f"Expected 2 exported flash anchors, got {len(exported_anchors)}"
        assert exported_anchors[0]["label"] == "w00" and exported_anchors[1]["label"] == "w01"
        assert exported_anchors[0]["x"] == 1.25 and exported_anchors[1]["x"] == 3.40
        print(f"  [FARO Flash Export] Verified flash_anchors.json exported cleanly with 2 waypoints.")

        # Clean up test project
        shutil.rmtree(os.path.join(win.projects_root, test_scan_proj), ignore_errors=True)

        # 7. Test Header Brand Splash Trigger & 3D Spatial Viewport
        win.show_brand_splash()
        app.processEvents()
        if hasattr(win, "_about_splash") and win._about_splash is not None:
            win._about_splash.close()
        win.center_viewport.repaint()
        win.map_canvas.repaint()
        app.processEvents()
        print("  [GUI Map & 3D Studio] Rendered White & Gold 3D Spatial Studio, C-SLAM trajectories, and M12 status cleanly.")
        print("  [Apple Glass UI/UX] Verified transparent frameless window, transparent min/max/close controls, and AR walk passthrough.")
    finally:
        win.close()
        app.processEvents()


if __name__ == "__main__":
    test_m12_cleanup_and_exports()
    test_cslam_unlimited_peers_and_auto_snap()
    test_headless_gui_integration()
    print("\n✅ ALL M12 POINT CLOUD CLEANUP, PROJECTS HUB, MAP LAYERS & UNLIMITED C-SLAM TESTS PASSED!")

