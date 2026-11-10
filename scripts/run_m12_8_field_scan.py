#!/usr/bin/env python3
"""
Milestone M12.8: Production MVP Acceptance Field Scan Runner

Executes a realistic, room-scale indoor field scan with live multi-modal
odometry, LiDAR point clouds, and SLERP-interpolated RGB-D integration.
Exports the standard production point cloud bundle:
  - pointcloud_fused_rgb.ply / .pcd
  - pointcloud_airy.ply / .pcd
  - pointcloud_stereo.ply / .pcd
  - trajectory.csv
  - trajectory_tum.txt
  - flash_anchors.json
  - scan_metadata.json
"""

import os
import sys
import time
import math
import json
import cv2
import argparse
import threading
import struct
import numpy as np
from collections import deque
from scipy.spatial.transform import Rotation
from scipy.spatial import cKDTree

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import PointCloud2, Image, CameraInfo
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge

# ==============================================================================
# RIGID EXTRINSICS & GEOMETRIC TRANSFORMS (ACTIVE 45-DEG PRODUCTION PROFILE)
# ==============================================================================
try:
    import yaml as _yaml
except ImportError:
    _yaml = None

# 1. IMU from LiDAR Frame (Factory DIFOP Calibration & robosense.yaml)
R_IMU_LIDAR = np.array([
    [ 0.012366, -0.999888,  0.008426],
    [-0.999919, -0.012340,  0.003184],
    [-0.003079, -0.008465, -0.999959]
], dtype=np.float64)
t_IMU_LIDAR = np.array([0.004164, 0.004315, -0.004411], dtype=np.float64)

# 2. Camera Depth Optical Frame (Left IR) from LiDAR Frame (scanarV4_45deg)
R_CAM_LIDAR = np.array([
    [ 0.015886, -0.999704,  0.018422],
    [ 0.714412, -0.001542, -0.699724],
    [ 0.699546,  0.024276,  0.714175]
], dtype=np.float64)
t_CAM_LIDAR = np.array([0.043444, -0.063060, -0.029978], dtype=np.float64)

# 5. RGB Optical Frame from Camera Depth Optical Frame (Factory EEPROM T_rgb_depth)
R_RGB_CAM_DEPTH = np.array([
    [ 0.999999, -0.001189,  0.000174],
    [ 0.001189,  0.999999, -0.000658],
    [-0.000174,  0.000658,  0.999999]
], dtype=np.float64)
t_RGB_CAM_DEPTH = np.array([0.0237306, -0.0000932, -0.0004246], dtype=np.float64)

if _yaml is not None:
    try:
        _fl_path = "/home/scanar/scanarMini/src/FAST-LIVO2/config/robosense.yaml"
        if os.path.exists(_fl_path):
            with open(_fl_path, "r") as _f:
                _fl_cfg = _yaml.safe_load(_f)
            _ext = _fl_cfg["/**"]["ros__parameters"]["extrin_calib"]
            R_IMU_LIDAR = np.array(_ext["extrinsic_R"], dtype=np.float64).reshape(3, 3)
            t_IMU_LIDAR = np.array(_ext["extrinsic_T"], dtype=np.float64)
        _pe_path = "/home/scanar/scanarMini/config/production_extrinsics.yaml"
        if os.path.exists(_pe_path):
            with open(_pe_path, "r") as _f:
                _pe_cfg = _yaml.safe_load(_f)
            _T_c0_l = np.array(_pe_cfg["active_transforms"]["T_cam0_airy_lidar"]["matrix"], dtype=np.float64)
            R_CAM_LIDAR = _T_c0_l[:3, :3]
            t_CAM_LIDAR = _T_c0_l[:3, 3]
            _T_rgb_d = np.array(_pe_cfg["factory_transforms"]["T_rgb_depth"]["matrix"], dtype=np.float64)
            R_RGB_CAM_DEPTH = _T_rgb_d[:3, :3]
            t_RGB_CAM_DEPTH = _T_rgb_d[:3, 3]
    except Exception:
        pass

# 3. Inverted: LiDAR from Camera Depth Optical Frame
R_LIDAR_CAM = R_CAM_LIDAR.T
t_LIDAR_CAM = -R_CAM_LIDAR.T @ t_CAM_LIDAR

# 4. IMU from Camera Depth Optical Frame (Direct rigid composition)
R_IMU_CAM_DEPTH = R_IMU_LIDAR @ R_LIDAR_CAM
t_IMU_CAM_DEPTH = R_IMU_LIDAR @ t_LIDAR_CAM + t_IMU_LIDAR

# 6. RGB Optical Frame from LiDAR Frame
R_RGB_LIDAR = R_RGB_CAM_DEPTH @ R_CAM_LIDAR
t_RGB_LIDAR = R_RGB_CAM_DEPTH @ t_CAM_LIDAR + t_RGB_CAM_DEPTH


def slerp_quat(q0, q1, alpha):
    """Spherical Linear Interpolation between q0 and q1 [x, y, z, w]."""
    alpha = float(np.clip(alpha, 0.0, 1.0))
    dot = np.dot(q0, q1)
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        q = (1.0 - alpha) * q0 + alpha * q1
        return q / np.linalg.norm(q)
    theta_0 = np.arccos(dot)
    sin_theta_0 = np.sin(theta_0)
    theta = theta_0 * alpha
    sin_theta = np.sin(theta)
    s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0
    return s0 * q0 + s1 * q1


def fast_voxel_downsample(xyz, rgb=None, voxel_size=0.015):
    """Ultra-fast integer hashing voxel downsampler."""
    if len(xyz) == 0:
        return xyz, rgb
    keys = np.floor(xyz / voxel_size).astype(np.int32)
    packed = (keys[:, 0].astype(np.int64) * 73856093) ^ \
             (keys[:, 1].astype(np.int64) * 19349663) ^ \
             (keys[:, 2].astype(np.int64) * 83492791)
    _, u_idx = np.unique(packed, return_index=True)
    if rgb is not None and len(rgb) == len(xyz):
        return xyz[u_idx], rgb[u_idx]
    return xyz[u_idx], None


D_RGB_DEFAULT = np.array([-0.03113378, 0.03646491, 0.00048966, -0.00030747, -0.01320058], dtype=np.float64)


def project_pts_to_rgb_plumb_bob(pts_rgb, fx, fy, cx, cy, d_coeffs=D_RGB_DEFAULT):
    """Vectorized Brown-Conrady (Plumb-Bob) camera projection for 3D points in RGB optical frame."""
    zr = pts_rgb[:, 2]
    safe_z = np.where(zr > 1e-4, zr, 1.0)
    xn = pts_rgb[:, 0] / safe_z
    yn = pts_rgb[:, 1] / safe_z
    r2 = xn * xn + yn * yn
    r4 = r2 * r2
    r6 = r4 * r2
    k1, k2, p1, p2, k3 = d_coeffs[:5]
    radial = 1.0 + k1 * r2 + k2 * r4 + k3 * r6
    xd = xn * radial + 2.0 * p1 * xn * yn + p2 * (r2 + 2.0 * xn * xn)
    yd = yn * radial + p1 * (r2 + 2.0 * yn * yn) + 2.0 * p2 * xn * yn
    u = np.round(fx * xd + cx).astype(np.int32)
    v = np.round(fy * yd + cy).astype(np.int32)
    return u, v, zr


def multi_sweep_voxel_centroids(sweeps, voxel_size=0.015):
    """
    Aggregates a list of (sec, xyz_w) LiDAR sweeps into multi-hit 1.5 cm voxel centroids.
    Filters transient dynamic objects (e.g. walking people/ghost trails) when >=15 sweeps exist.
    """
    valid_sweeps = [s[1] for s in sweeps if s[1] is not None and len(s[1]) > 0]
    if len(valid_sweeps) == 0:
        return np.empty((0, 3), dtype=np.float64)

    per_sweep_keys = []
    per_sweep_pts = []
    for xyz in valid_sweeps:
        gx = np.floor(xyz[:, 0] / voxel_size).astype(np.int64) + 500000
        gy = np.floor(xyz[:, 1] / voxel_size).astype(np.int64) + 500000
        gz = np.floor(xyz[:, 2] / voxel_size).astype(np.int64) + 500000
        packed = (gx << 42) | (gy << 21) | gz
        u_keys, u_idx = np.unique(packed, return_index=True)
        per_sweep_keys.append(u_keys)
        per_sweep_pts.append(xyz[u_idx])

    all_keys = np.concatenate(per_sweep_keys)
    all_pts = np.vstack(per_sweep_pts)

    unique_voxels, inv_idx, counts = np.unique(all_keys, return_inverse=True, return_counts=True)
    sum_x = np.bincount(inv_idx, weights=all_pts[:, 0])
    sum_y = np.bincount(inv_idx, weights=all_pts[:, 1])
    sum_z = np.bincount(inv_idx, weights=all_pts[:, 2])

    min_hits = 3 if len(valid_sweeps) >= 15 else 1
    keep = counts >= min_hits
    if not np.any(keep):
        keep = counts >= 1

    c_keep = counts[keep].astype(np.float64)
    return np.column_stack((
        sum_x[keep] / c_keep,
        sum_y[keep] / c_keep,
        sum_z[keep] / c_keep,
    ))


class M128FieldScanNode(Node):
    def __init__(self, session_id="M12_8_Field_Scan"):
        super().__init__('m12_8_field_scan_node')
        self.session_id = session_id
        self.export_dir = f"/home/scanar/scanarMini/data/{session_id}"
        os.makedirs(self.export_dir, exist_ok=True)

        self.bridge = CvBridge()
        self.lock = threading.RLock()

        # State buffers
        self.is_recording = False
        self.start_time = None
        self.stop_time = None

        self.odom_buffer = deque(maxlen=3000)
        self.trajectory_records = []
        self.airy_sweeps = []
        self.gemini_frames = []
        self.rgb_keyframes = deque(maxlen=600)

        self.latest_rgb_image = None
        self.latest_rgb_stamp = 0.0

        # Factory Default Intrinsics
        # Depth: 848x480
        self.fx_d = 411.82974
        self.fy_d = 411.82974
        self.cx_d = 423.66876
        self.cy_d = 240.99377

        # RGB: 1280x800
        self.fx_rgb = 609.0065
        self.fy_rgb = 609.0156
        self.cx_rgb = 643.8688
        self.cy_rgb = 399.3265
        self.d_rgb = D_RGB_DEFAULT.copy()

        # Counters
        self.cnt_odom = 0
        self.cnt_airy = 0
        self.cnt_depth = 0
        self.cnt_rgb = 0

        # Subscriptions
        qos_reliable = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=100)
        qos_sensor = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=100)

        self.sub_odom = self.create_subscription(Odometry, '/aft_mapped_to_init', self.odom_callback, qos_reliable)
        self.sub_registered = self.create_subscription(PointCloud2, '/cloud_registered', self.registered_callback, qos_reliable)
        self.sub_rgb = self.create_subscription(Image, '/camera/color/image_raw', self.rgb_callback, qos_sensor)
        self.sub_depth = self.create_subscription(Image, '/camera/depth/image_raw', self.depth_callback, qos_sensor)
        self.sub_depth_info = self.create_subscription(CameraInfo, '/camera/depth/camera_info', self.depth_info_callback, qos_sensor)
        self.sub_rgb_info = self.create_subscription(CameraInfo, '/camera/color/camera_info', self.rgb_info_callback, qos_sensor)

        self.get_logger().info(f"[M12.8 Field Scan Node] Initialized. Output target: {self.export_dir}")

    def depth_info_callback(self, msg: CameraInfo):
        if msg.k[0] > 10.0:
            self.fx_d = float(msg.k[0])
            self.fy_d = float(msg.k[4])
            self.cx_d = float(msg.k[2])
            self.cy_d = float(msg.k[5])

    def rgb_info_callback(self, msg: CameraInfo):
        if msg.k[0] > 10.0:
            self.fx_rgb = float(msg.k[0])
            self.fy_rgb = float(msg.k[4])
            self.cx_rgb = float(msg.k[2])
            self.cy_rgb = float(msg.k[5])
        if len(msg.d) >= 5 and any(abs(x) > 1e-9 for x in msg.d[:5]):
            self.d_rgb = np.array(msg.d[:5], dtype=np.float64)

    def odom_callback(self, msg: Odometry):
        sec = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        pos = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z], dtype=np.float64)
        quat = np.array([msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w], dtype=np.float64)

        with self.lock:
            self.cnt_odom += 1
            self.odom_buffer.append((sec, pos, quat))
            if self.is_recording:
                rot = Rotation.from_quat(quat)
                euler = rot.as_euler('zyx', degrees=False)
                self.trajectory_records.append((sec, pos[0], pos[1], pos[2], quat[0], quat[1], quat[2], quat[3], euler[0]))

    def rgb_callback(self, msg: Image):
        sec = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        with self.lock:
            self.cnt_rgb += 1
        try:
            cv_img_bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            cv_img_rgb = cv2.cvtColor(cv_img_bgr, cv2.COLOR_BGR2RGB)
            with self.lock:
                self.latest_rgb_image = cv_img_rgb
                self.latest_rgb_stamp = sec
                if self.is_recording:
                    pos_w, quat_w = self._get_interpolated_pose_internal(sec)
                    if pos_w is not None and quat_w is not None:
                        R_w_imu = Rotation.from_quat(quat_w).as_matrix()
                        self.rgb_keyframes.append((sec, cv_img_rgb, R_w_imu, pos_w))
        except Exception:
            pass

    def _get_interpolated_pose_internal(self, t_query):
        if len(self.odom_buffer) < 2:
            return None, None
        stamps = np.array([p[0] for p in self.odom_buffer])
        idx = np.searchsorted(stamps, t_query)
        if idx == 0:
            if abs(t_query - stamps[0]) <= 0.1:
                return self.odom_buffer[0][1], self.odom_buffer[0][2]
            return None, None
        if idx >= len(self.odom_buffer):
            if abs(t_query - stamps[-1]) <= 0.1:
                return self.odom_buffer[-1][1], self.odom_buffer[-1][2]
            return None, None

        t0, p0, q0 = self.odom_buffer[idx - 1]
        t1, p1, q1 = self.odom_buffer[idx]
        dt = t1 - t0
        if dt <= 1e-6:
            return p0, q0

        alpha = (t_query - t0) / dt
        p_interp = (1.0 - alpha) * p0 + alpha * p1
        q_interp = slerp_quat(q0, q1, alpha)
        return p_interp, q_interp

    def get_interpolated_pose(self, t_query):
        with self.lock:
            return self._get_interpolated_pose_internal(t_query)

    def registered_callback(self, msg: PointCloud2):
        sec = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        with self.lock:
            self.cnt_airy += 1

        if not self.is_recording:
            return

        num_points = msg.width * msg.height
        step = msg.point_step
        if num_points == 0 or step < 12:
            return

        # Vectorized zero-copy PointCloud2 unpacking (/cloud_registered is ALREADY in world frame camera_init)
        raw_data = bytes(msg.data)
        if len(raw_data) < num_points * step:
            return

        dt = np.dtype({
            'names': ['x', 'y', 'z'],
            'formats': ['<f4', '<f4', '<f4'],
            'offsets': [0, 4, 8],
            'itemsize': step
        })
        pts = np.frombuffer(raw_data, dtype=dt, count=num_points)
        x = pts['x']
        y = pts['y']
        z = pts['z']
        valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
        if not np.any(valid):
            return
        x, y, z = x[valid], y[valid], z[valid]
        d_sq = x * x + y * y + z * z
        range_mask = (d_sq > 0.04) & (d_sq < 1600.0)
        if np.any(range_mask):
            arr = np.column_stack((x[range_mask], y[range_mask], z[range_mask])).astype(np.float64)
            with self.lock:
                self.airy_sweeps.append((sec, arr))

    def depth_callback(self, msg: Image):
        sec = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        with self.lock:
            self.cnt_depth += 1

        if not self.is_recording:
            return

        try:
            depth_raw = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        except Exception:
            return

        pos_w, quat_w = self.get_interpolated_pose(sec)
        if pos_w is None or quat_w is None:
            return

        # 1. Transform: Depth Optical Frame -> World Frame
        R_w_imu = Rotation.from_quat(quat_w).as_matrix()
        R_w_cam_depth = R_w_imu @ R_IMU_CAM_DEPTH
        t_w_cam_depth = R_w_imu @ t_IMU_CAM_DEPTH + pos_w

        # 2. Subsample depth image (stride 2 for dense detail)
        sub = 2
        d_sub = depth_raw[::sub, ::sub].astype(np.float64)
        if depth_raw.dtype == np.uint16:
            d_sub *= 0.001

        mask = (d_sub >= 0.30) & (d_sub <= 4.00)
        v_idx, u_idx = np.nonzero(mask)
        if len(v_idx) == 0:
            return

        u_px = u_idx * sub
        v_px = v_idx * sub
        z_vals = d_sub[v_idx, u_idx]

        # 3. Unproject depth pixels to 3D in Camera Depth Optical Frame
        x_cam = (u_px - self.cx_d) * z_vals / self.fx_d
        y_cam = (v_px - self.cy_d) * z_vals / self.fy_d
        pts_cam = np.vstack((x_cam, y_cam, z_vals)).T

        # 4. Transform to World Frame
        pts_w = pts_cam @ R_w_cam_depth.T + t_w_cam_depth

        # 5. Project into RGB frame with Plumb-Bob lens distortion for truecolor RGB assignment
        with self.lock:
            rgb_img = self.latest_rgb_image
            rgb_stamp = self.latest_rgb_stamp

        colors = np.full((len(pts_w), 3), 175, dtype=np.uint8)

        if rgb_img is not None and abs(sec - rgb_stamp) <= 0.10:
            rgb_h, rgb_w = rgb_img.shape[:2]
            pts_rgb_frame = (pts_cam @ R_RGB_CAM_DEPTH.T) + t_RGB_CAM_DEPTH
            u_r, v_r, zr = project_pts_to_rgb_plumb_bob(
                pts_rgb_frame, self.fx_rgb, self.fy_rgb, self.cx_rgb, self.cy_rgb, self.d_rgb
            )
            in_fov = (zr > 0.2) & (u_r >= 0) & (u_r < rgb_w) & (v_r >= 0) & (v_r < rgb_h)
            if np.any(in_fov):
                colors[in_fov] = rgb_img[v_r[in_fov], u_r[in_fov]]

        with self.lock:
            self.gemini_frames.append((sec, pts_w, colors))

    def colorize_airy_cloud(self, airy_pts):
        """Projects authoritative Airy LiDAR points into recorded RGB keyframes with Plumb-Bob distortion & Z-buffer occlusion."""
        n_pts = len(airy_pts)
        if n_pts == 0:
            return np.empty((0, 3), dtype=np.uint8)

        airy_colors = np.full((n_pts, 3), 175, dtype=np.uint8)
        best_dist = np.full(n_pts, np.inf, dtype=np.float64)

        with self.lock:
            keyframes = list(self.rgb_keyframes)

        if len(keyframes) == 0:
            return airy_colors

        # Subsample up to 60 distributed keyframes across the scan for comprehensive multi-view projection
        step = max(1, len(keyframes) // 60)
        selected_kfs = keyframes[::step]

        for sec, img_rgb, R_w_imu, pos_w in selected_kfs:
            p_imu = (airy_pts - pos_w) @ R_w_imu
            p_lidar = (p_imu - t_IMU_LIDAR) @ R_IMU_LIDAR
            p_rgb = p_lidar @ R_RGB_LIDAR.T + t_RGB_LIDAR

            u, v, zr = project_pts_to_rgb_plumb_bob(
                p_rgb, self.fx_rgb, self.fy_rgb, self.cx_rgb, self.cy_rgb, self.d_rgb
            )
            rgb_h, rgb_w = img_rgb.shape[:2]
            in_fov = (zr > 0.3) & (zr < 6.0) & (u >= 0) & (u < rgb_w) & (v >= 0) & (v < rgb_h)
            fov_idx = np.nonzero(in_fov)[0]
            if len(fov_idx) == 0:
                continue

            u_f = u[fov_idx]
            v_f = v[fov_idx]
            z_f = zr[fov_idx]

            # 320x200 Z-buffer occlusion gate so foreground walls block occluded background surfaces
            u_b = np.clip(u_f // 4, 0, (rgb_w // 4) - 1)
            v_b = np.clip(v_f // 4, 0, (rgb_h // 4) - 1)
            z_buf = np.full((rgb_h // 4, rgb_w // 4), np.inf, dtype=np.float64)
            np.minimum.at(z_buf, (v_b, u_b), z_f)

            visible = z_f <= (z_buf[v_b, u_b] + 0.15)
            vis_idx = fov_idx[visible]
            if len(vis_idx) == 0:
                continue

            z_vis = z_f[visible]
            better = z_vis < best_dist[vis_idx]
            upd_idx = vis_idx[better]
            if len(upd_idx) > 0:
                best_dist[upd_idx] = z_vis[better]
                airy_colors[upd_idx] = img_rgb[v[upd_idx], u[upd_idx]]

        return airy_colors

    def run_asps_fusion(self, airy_pts, airy_rgb, gemini_pts, gemini_rgb, tau_snap=0.020, tau_conflict=0.080):
        """Three-class Adaptive Surface Patch Snapping (ASPS) Fusion."""
        if len(airy_pts) == 0:
            return gemini_pts, gemini_rgb
        if len(gemini_pts) == 0:
            return airy_pts, airy_rgb

        tree_a = cKDTree(airy_pts)
        dists, indices = tree_a.query(gemini_pts, k=1, workers=-1)

        mask_snap = dists <= tau_snap
        mask_conflict = (dists > tau_snap) & (dists <= tau_conflict)
        mask_supp = dists > tau_conflict

        # Snapped Points: adopts LiDAR geometry, preserves dense RGB color
        p_snap = gemini_pts[mask_snap]
        c_snap = gemini_rgb[mask_snap]

        # Supplemental Points: fills regions outside LiDAR coverage
        p_supp = gemini_pts[mask_supp]
        c_supp = gemini_rgb[mask_supp]

        fused_xyz = np.vstack((airy_pts, p_snap, p_supp))
        fused_rgb = np.vstack((airy_rgb, c_snap, c_supp))
        return fused_xyz, fused_rgb

    def write_ply(self, filepath, xyz, rgb=None):
        n = len(xyz)
        has_color = (rgb is not None and len(rgb) == n)
        header = f"ply\nformat ascii 1.0\nelement vertex {n}\nproperty float x\nproperty float y\nproperty float z\n"
        if has_color:
            header += "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        header += "end_header\n"

        if has_color:
            table = np.zeros(n, dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("r", "u1"), ("g", "u1"), ("b", "u1")])
            table["x"] = xyz[:, 0]
            table["y"] = xyz[:, 1]
            table["z"] = xyz[:, 2]
            table["r"] = rgb[:, 0]
            table["g"] = rgb[:, 1]
            table["b"] = rgb[:, 2]
            with open(filepath, 'wb') as f:
                f.write(header.encode('ascii'))
                np.savetxt(f, table, fmt="%.4f %.4f %.4f %d %d %d")
        else:
            table = np.zeros(n, dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])
            table["x"] = xyz[:, 0]
            table["y"] = xyz[:, 1]
            table["z"] = xyz[:, 2]
            with open(filepath, 'wb') as f:
                f.write(header.encode('ascii'))
                np.savetxt(f, table, fmt="%.4f %.4f %.4f")

    def write_pcd(self, filepath, xyz, rgb=None):
        n = len(xyz)
        has_color = (rgb is not None and len(rgb) == n)
        header = "# .PCD v0.7 - Point Cloud Data\nVERSION 0.7\n"
        if has_color:
            header += "FIELDS x y z rgb\nSIZE 4 4 4 4\nTYPE F F F U\nCOUNT 1 1 1 1\n"
        else:
            header += "FIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\n"
        header += f"WIDTH {n}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS {n}\nDATA ascii\n"

        if has_color:
            rgb_uint = (rgb[:, 0].astype(np.uint32) << 16) | (rgb[:, 1].astype(np.uint32) << 8) | rgb[:, 2].astype(np.uint32)
            table = np.zeros(n, dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("c", "u4")])
            table["x"] = xyz[:, 0]
            table["y"] = xyz[:, 1]
            table["z"] = xyz[:, 2]
            table["c"] = rgb_uint
            with open(filepath, 'wb') as f:
                f.write(header.encode('ascii'))
                np.savetxt(f, table, fmt="%.4f %.4f %.4f %u")
        else:
            table = np.zeros(n, dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])
            table["x"] = xyz[:, 0]
            table["y"] = xyz[:, 1]
            table["z"] = xyz[:, 2]
            with open(filepath, 'wb') as f:
                f.write(header.encode('ascii'))
                np.savetxt(f, table, fmt="%.4f %.4f %.4f")

    def export_production_bundle(self):
        duration = (self.stop_time - self.start_time) if (self.start_time and self.stop_time) else 0.0
        print(f"\n[EXPORT] Processing M12.8 Field Scan dataset ({duration:.2f}s)...")

        # 1. Trajectory Files
        traj_csv = os.path.join(self.export_dir, "trajectory.csv")
        traj_tum = os.path.join(self.export_dir, "trajectory_tum.txt")

        with open(traj_csv, 'w') as f_csv, open(traj_tum, 'w') as f_tum:
            f_csv.write("timestamp_sec,x_m,y_m,z_m,qx,qy,qz,qw,yaw_rad\n")
            for r in self.trajectory_records:
                sec, x, y, z, qx, qy, qz, qw, yaw = r
                f_csv.write(f"{sec:.6f},{x:.6f},{y:.6f},{z:.6f},{qx:.6f},{qy:.6f},{qz:.6f},{qw:.6f},{yaw:.6f}\n")
                f_tum.write(f"{sec:.6f} {x:.6f} {y:.6f} {z:.6f} {qx:.6f} {qy:.6f} {qz:.6f} {qw:.6f}\n")

        # 2. Multi-Sweep Voxel Centroid Filtering & Ghost Removal for Airy LiDAR Cloud
        airy_clean = multi_sweep_voxel_centroids(self.airy_sweeps, voxel_size=0.015)

        airy_rgb = self.colorize_airy_cloud(airy_clean)

        # 3. Voxelize Gemini Stereo Depth Cloud
        all_g_xyz, all_g_rgb = [], []
        for gf in self.gemini_frames:
            if gf[1] is not None and len(gf[1]) > 0:
                all_g_xyz.append(gf[1])
                all_g_rgb.append(gf[2])

        if len(all_g_xyz) > 0:
            g_raw_xyz = np.vstack(all_g_xyz)
            g_raw_rgb = np.vstack(all_g_rgb)
            gemini_clean_xyz, gemini_clean_rgb = fast_voxel_downsample(g_raw_xyz, g_raw_rgb, voxel_size=0.015)
        else:
            gemini_clean_xyz = np.empty((0, 3), dtype=np.float64)
            gemini_clean_rgb = np.empty((0, 3), dtype=np.uint8)

        # 4. Three-Class ASPS Fusion
        fused_xyz, fused_rgb = self.run_asps_fusion(airy_clean, airy_rgb, gemini_clean_xyz, gemini_clean_rgb)

        # 5. Write PLY and PCD files (Ultra-fast batch writer)
        # pointcloud_airy: Authoritative LiDAR reference geometry only (no RGB projection)
        self.write_ply(os.path.join(self.export_dir, "pointcloud_airy.ply"), airy_clean, None)
        self.write_pcd(os.path.join(self.export_dir, "pointcloud_airy.pcd"), airy_clean, None)
        # pointcloud_stereo: Dense stereo geometry with RGB color
        self.write_ply(os.path.join(self.export_dir, "pointcloud_stereo.ply"), gemini_clean_xyz, gemini_clean_rgb)
        self.write_pcd(os.path.join(self.export_dir, "pointcloud_stereo.pcd"), gemini_clean_xyz, gemini_clean_rgb)
        # pointcloud_fused_rgb: Final ASPS fused geometry with RGB color
        self.write_ply(os.path.join(self.export_dir, "pointcloud_fused_rgb.ply"), fused_xyz, fused_rgb)
        self.write_pcd(os.path.join(self.export_dir, "pointcloud_fused_rgb.pcd"), fused_xyz, fused_rgb)

        with open(os.path.join(self.export_dir, "flash_anchors.json"), 'w') as fa:
            json.dump([], fa, indent=2)

        meta = {
            "session_id": self.session_id,
            "export_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "capture_duration_seconds": float(round(duration, 2)),
            "point_counts": {
                "authoritative_airy_points": int(len(airy_clean)),
                "supplemental_gemini_points": int(len(gemini_clean_xyz)),
                "fused_total_points": int(len(fused_xyz)),
                "density_multiplier": float(round(len(fused_xyz) / max(1, len(airy_clean)), 2))
            },
            "sensor_telemetry": {
                "fastlivo_odometry_received": self.cnt_odom,
                "airy_sweeps_received": self.cnt_airy,
                "gemini_depth_frames_received": self.cnt_depth,
                "gemini_rgb_frames_received": self.cnt_rgb
            }
        }

        with open(os.path.join(self.export_dir, "scan_metadata.json"), 'w') as fm:
            json.dump(meta, fm, indent=2)

        print(f"[EXPORT SUCCESS] Production bundle exported to: {self.export_dir}")
        print(f"  - Authoritative Airy Cloud: {len(airy_clean):,} points")
        print(f"  - Supplemental Gemini Cloud: {len(gemini_clean_xyz):,} points")
        print(f"  - Final Fused Colored Cloud: {len(fused_xyz):,} points")
        return meta


def main():
    parser = argparse.ArgumentParser(description="M12.8 Field Scan Runner")
    parser.add_argument("--session-id", default="M12_8_Field_Scan_03", help="Session ID / output directory name")
    parser.add_argument("--duration", type=float, default=30.0, help="Scan duration in seconds (0 for manual stop via Ctrl+C)")
    parser.add_argument("--countdown", type=int, default=5, help="Countdown seconds before recording starts")
    args = parser.parse_args()

    rclpy.init()
    node = M128FieldScanNode(session_id=args.session_id)

    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    print("[M12.8] Waiting for active topics...")
    t_wait = time.time()
    while (node.cnt_odom < 5 or node.cnt_airy < 5 or node.cnt_depth < 5) and (time.time() - t_wait < 10.0):
        time.sleep(0.1)

    print(f"[M12.8] Topics active: Odom={node.cnt_odom}, Airy={node.cnt_airy}, Depth={node.cnt_depth}, RGB={node.cnt_rgb}")

    if args.countdown > 0:
        print(f"\n==========================================")
        print(f"M12.8 FIELD SCAN COUNTDOWN ({args.countdown}s):")
        for c in range(args.countdown, 0, -1):
            print(f"  >>> {c} ...")
            time.sleep(1.0)
        print(f"  >>> RECORDING ACTIVE — BEGIN SCANNING ROUTE! <<<")
        print(f"==========================================\n")

    node.start_time = time.time()
    node.is_recording = True

    try:
        t0 = time.time()
        while rclpy.ok():
            elapsed = time.time() - t0
            if args.duration > 0 and elapsed >= args.duration:
                break
            print(f"  [SCANNING] {elapsed:.1f}/{args.duration:.1f}s | Airy Sweeps: {len(node.airy_sweeps)} | Depth Frames: {len(node.gemini_frames)} | Odom Poses: {len(node.trajectory_records)}")
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\n[M12.8] Manual stop triggered by user.")

    node.is_recording = False
    node.stop_time = time.time()
    print("\n>>> [SCAN COMPLETE] Exporting production datasets... <<<")
    node.export_production_bundle()

    rclpy.shutdown()
    spin_thread.join(timeout=2.0)

if __name__ == '__main__':
    main()
