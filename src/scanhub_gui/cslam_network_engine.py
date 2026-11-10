#!/usr/bin/env python3
"""
Unlimited-Peer Collaborative SLAM (C-SLAM) Network & Co-Registration Engine for ScanHUB OS.

Features:
- Unlimited Scanners (N peers, zero hard cap) joining the same project session over UDP/LAN or local loopback.
- Distinct High-Contrast Scanner Color Palette (curated + golden-ratio HSV generator for arbitrary N).
- Dual-Stage Spatial Alignment:
  1. SEPARATED_IMU_ORIENTED: When no overlapping geometry is detected yet, remote scanners are placed at a
     deterministic non-overlapping spatial offset (concentric ring layout) and oriented using their relative
     IMU gravity + heading rotation (R_imu).
  2. OVERLAP_SNAPPED: Background coarse-to-fine registration (IMU-guided BEV + 3D keypoint RANSAC + multi-stage
     trimmed KD-Tree SVD ICP) continuously monitors overlap. Once sufficient geometric overlap is found,
     the peer's coordinate frame T_world_from_peer automatically snaps into the unified global project map.
"""

import colorsys
import json
import math
import socket
import struct
import threading
import time
import uuid
import zlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Callable
import numpy as np
from scipy.spatial import cKDTree

from m12_point_cloud_cleanup import PointCloudProcessor


class PeerAlignmentState(Enum):
    SEPARATED_IMU_ORIENTED = "SEPARATED_IMU_ORIENTED"
    OVERLAP_SNAPPED = "OVERLAP_SNAPPED"


class CSlamColorMode(Enum):
    SCANNER_DIFFERENTIATED = "SCANNER_COLORS"  # Distinct color per scanner (with 3D shading preservation)
    SCANNER_COLORS = "SCANNER_COLORS"
    TRUE_RGB = "TRUE_RGB"                      # Original sensor RGB


# Curated high-contrast palette for first 8 scanners; golden-ratio HSV used for 8..infinity
CURATED_SCANNER_COLORS_RGB: List[Tuple[float, float, float]] = [
    (0.00, 0.90, 1.00),  # 0: Local Scanner - Electric Cyan (#00E6FF)
    (1.00, 0.62, 0.00),  # 1: Peer #1 - Amber Orange (#FF9E00)
    (1.00, 0.18, 0.58),  # 2: Peer #2 - Neon Magenta (#FF2E94)
    (0.00, 0.95, 0.45),  # 3: Peer #3 - Emerald Green (#00F273)
    (0.68, 0.32, 1.00),  # 4: Peer #4 - Vivid Violet (#AD52FF)
    (1.00, 0.92, 0.15),  # 5: Peer #5 - Solar Yellow (#FFEB26)
    (1.00, 0.30, 0.25),  # 6: Peer #6 - Coral Red (#FF4D40)
    (0.25, 0.65, 1.00),  # 7: Peer #7 - Sky Blue (#40A6FF)
]


def get_scanner_color_rgb(scanner_index: int) -> Tuple[float, float, float]:
    """
    Returns a high-contrast (R, G, B) float tuple in [0, 1] for any scanner_index >= 0 (no maximum cap).
    """
    idx = max(0, int(scanner_index))
    if idx < len(CURATED_SCANNER_COLORS_RGB):
        return CURATED_SCANNER_COLORS_RGB[idx]
    # Golden-ratio conjugate hue stepping for unlimited distinct scanner colors
    golden_ratio_conjugate = 0.618033988749895
    hue = (0.13 + idx * golden_ratio_conjugate) % 1.0
    sat = 0.85 if (idx % 2 == 0) else 0.95
    val = 0.98 if (idx % 3 != 0) else 0.88
    r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
    return (float(r), float(g), float(b))


def get_scanner_color_hex(scanner_index: int) -> str:
    r, g, b = get_scanner_color_rgb(scanner_index)
    return f"#{int(round(r * 255)):02X}{int(round(g * 255)):02X}{int(round(b * 255)):02X}"


def euler_rpy_to_matrix(roll_rad: float, pitch_rad: float, yaw_rad: float) -> np.ndarray:
    cr, sr = math.cos(roll_rad), math.sin(roll_rad)
    cp, sp = math.cos(pitch_rad), math.sin(pitch_rad)
    cy, sy = math.cos(yaw_rad), math.sin(yaw_rad)

    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=np.float64)
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=np.float64)
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    return rz @ ry @ rx


def apply_scanner_tint(points: np.ndarray, scanner_rgb: Tuple[float, float, float]) -> np.ndarray:
    """
    Applies a distinct scanner color to (N, 6) points while preserving 3D surface luminance/relief
    so geometry remains easy to read inside the merged C-SLAM viewport.
    """
    if points is None or len(points) == 0:
        return np.zeros((0, 6), dtype=np.float32)
    out = points[:, :6].copy().astype(np.float32)
    orig_rgb = out[:, 3:6]
    if np.max(orig_rgb) > 1.05:
        orig_rgb = orig_rgb / 255.0
    lum = (0.299 * orig_rgb[:, 0] + 0.587 * orig_rgb[:, 1] + 0.114 * orig_rgb[:, 2])
    lum = np.clip(lum, 0.25, 1.0)[:, None]
    tint = np.array(scanner_rgb, dtype=np.float32)[None, :]
    out[:, 3:6] = np.clip(tint * (0.45 + 0.55 * lum), 0.0, 1.0)
    return out


@dataclass
class PeerScannerSession:
    scanner_uuid: str
    scanner_name: str
    project_name: str
    scanner_index: int  # 1..N (0 is reserved for local scanner)
    ip_address: str = "127.0.0.1"
    port: int = 19210
    last_seen_ts: float = field(default_factory=time.time)

    # Live IMU & Pose in peer's own local SLAM frame
    pose_xyz: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    imu_rpy_rad: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # (roll, pitch, yaw) from IMU
    imu_initial_yaw_rad: float = 0.0
    trajectory_local: List[Tuple[float, float, float]] = field(default_factory=list)

    # Raw point cloud in peer's local SLAM frame: (N, 6) float32 [x, y, z, r, g, b]
    raw_points_local: np.ndarray = field(default_factory=lambda: np.zeros((0, 6), dtype=np.float32))

    # Alignment state
    alignment_state: PeerAlignmentState = PeerAlignmentState.SEPARATED_IMU_ORIENTED
    manual_override: Optional[PeerAlignmentState] = None
    overlap_ratio: float = 0.0
    inlier_rmse_m: float = 0.0
    inlier_count: int = 0
    transform_world_from_peer: np.ndarray = field(default_factory=lambda: np.eye(4, dtype=np.float64))

    @property
    def color_rgb(self) -> Tuple[float, float, float]:
        return get_scanner_color_rgb(self.scanner_index)

    @property
    def color_hex(self) -> str:
        return get_scanner_color_hex(self.scanner_index)


class CSlamRegistrationSolver:
    """
    Real-Time Coarse-to-Fine Overlap Detector & SE(3) Co-Registration Solver.
    Determines whether two point clouds share overlapping geometry and computes T_ref_from_mov.
    """

    @staticmethod
    def umeyama_se3(src_pts: np.ndarray, dst_pts: np.ndarray) -> np.ndarray:
        """
        Computes rigid 6-DoF SE(3) transformation T (4x4) minimizing || R * src + t - dst ||^2.
        """
        src = np.asarray(src_pts, dtype=np.float64)
        dst = np.asarray(dst_pts, dtype=np.float64)
        mu_s = np.mean(src, axis=0)
        mu_d = np.mean(dst, axis=0)
        s_c = src - mu_s
        d_c = dst - mu_d
        cov = (d_c.T @ s_c) / float(len(src))
        u, _, vt = np.linalg.svd(cov)
        d_sign = np.eye(3, dtype=np.float64)
        if np.linalg.det(u @ vt) < 0:
            d_sign[2, 2] = -1.0
        rot = u @ d_sign @ vt
        trans = mu_d - rot @ mu_s
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = rot
        T[:3, 3] = trans
        return T

    @staticmethod
    def transform_xyz(xyz: np.ndarray, T: np.ndarray) -> np.ndarray:
        if len(xyz) == 0:
            return xyz.copy()
        return (xyz.astype(np.float64) @ T[:3, :3].T + T[:3, 3]).astype(np.float32)

    @staticmethod
    def transform_points6(points: np.ndarray, T: np.ndarray) -> np.ndarray:
        if points is None or len(points) == 0:
            return np.zeros((0, 6), dtype=np.float32)
        out = points[:, :6].copy().astype(np.float32)
        out[:, :3] = CSlamRegistrationSolver.transform_xyz(out[:, :3], T)
        return out

    @staticmethod
    def detect_overlap_and_register(
        ref_points: np.ndarray,
        mov_points: np.ndarray,
        imu_relative_rot: Optional[np.ndarray] = None,
        prior_transform: Optional[np.ndarray] = None,
        min_overlap_ratio: float = 0.18,
        max_inlier_rmse_m: float = 0.11,
        inlier_thresh_m: float = 0.15,
    ) -> Tuple[bool, np.ndarray, float, float, int]:
        """
        Tests whether `mov_points` has sufficient geometric overlap with `ref_points`.
        Returns:
            (is_overlapping, T_ref_from_mov, overlap_ratio, inlier_rmse_m, inlier_count)
        """
        if ref_points is None or mov_points is None or len(ref_points) < 60 or len(mov_points) < 60:
            return False, np.eye(4, dtype=np.float64), 0.0, 999.0, 0

        # 1. Downsample clouds for fast real-time matching
        ref_ultra = PointCloudProcessor.voxel_downsample(ref_points[:, :3], 0.03)[:, :3]
        mov_ultra = PointCloudProcessor.voxel_downsample(mov_points[:, :3], 0.03)[:, :3]
        ref_fine = PointCloudProcessor.voxel_downsample(ref_ultra, 0.07)[:, :3]
        mov_fine = PointCloudProcessor.voxel_downsample(mov_ultra, 0.07)[:, :3]
        if len(ref_fine) < 35 or len(mov_fine) < 35:
            return False, np.eye(4, dtype=np.float64), 0.0, 999.0, 0

        ref_coarse = PointCloudProcessor.voxel_downsample(ref_fine, 0.20)[:, :3]
        mov_coarse = PointCloudProcessor.voxel_downsample(mov_fine, 0.20)[:, :3]
        if len(ref_coarse) < 15 or len(mov_coarse) < 15:
            ref_coarse = ref_fine
            mov_coarse = mov_fine

        ref_tree_coarse = cKDTree(ref_coarse)
        ref_tree_fine = cKDTree(ref_fine)
        ref_tree_ultra = cKDTree(ref_ultra)

        # Floor Z alignment estimate (10th percentile of Z)
        ref_z0 = float(np.percentile(ref_coarse[:, 2], 10.0))
        mov_z0 = float(np.percentile(mov_coarse[:, 2], 10.0))

        # Build candidate initial transforms
        candidates: List[np.ndarray] = []
        if prior_transform is not None:
            candidates.append(prior_transform.copy())

        # Identity & Centroid-aligned hypotheses under IMU relative yaw (both R_imu and R_imu.T) + yaw rotations
        R_imu_base = imu_relative_rot if imu_relative_rot is not None else np.eye(3, dtype=np.float64)
        base_rots = [R_imu_base, R_imu_base.T, np.eye(3, dtype=np.float64)]
        ref_mean = np.mean(ref_coarse, axis=0).astype(np.float64)
        mov_mean = np.mean(mov_coarse, axis=0).astype(np.float64)

        for R_b in base_rots:
            T_direct = np.eye(4, dtype=np.float64)
            T_direct[:3, :3] = R_b
            candidates.append(T_direct)

        # Yaw sweep + multi-anchor translation hypotheses (elevated walls/corners)
        ref_elev = ref_coarse[ref_coarse[:, 2] > ref_z0 + 0.25]
        mov_elev = mov_coarse[mov_coarse[:, 2] > mov_z0 + 0.25]
        if len(ref_elev) < 8:
            ref_elev = ref_coarse
        if len(mov_elev) < 8:
            mov_elev = mov_coarse

        rng = np.random.default_rng(12345)
        n_anchor_samples = min(24, len(ref_elev), len(mov_elev))
        ref_anchor_idx = rng.choice(len(ref_elev), size=n_anchor_samples, replace=False)
        mov_anchor_idx = rng.choice(len(mov_elev), size=n_anchor_samples, replace=False)

        yaw_angles = [0.0, math.pi / 2, -math.pi / 2, math.pi, math.pi / 4, -math.pi / 4, 3 * math.pi / 4, -3 * math.pi / 4]
        xy_shifts = (-2.4, -1.6, -0.8, 0.0, 0.8, 1.6, 2.4)
        for R_b in base_rots[:2]:
            # 1. Dense 2D XY translation search grid under IMU relative rotation (handles partial overlap where centroids differ)
            t_xy_base = ref_mean[:2] - (R_b @ mov_mean)[:2]
            t_z_base = ref_z0 - float((R_b @ np.array([mov_mean[0], mov_mean[1], mov_z0]))[2])
            for dx in xy_shifts:
                for dy in xy_shifts:
                    T_g = np.eye(4, dtype=np.float64)
                    T_g[:3, :3] = R_b
                    T_g[:3, 3] = [t_xy_base[0] + dx, t_xy_base[1] + dy, t_z_base]
                    candidates.append(T_g)

            # 2. Multi-yaw centroid candidates (handles uncalibrated/arbitrary initial compass heading)
            for dyaw in yaw_angles:
                R_z = euler_rpy_to_matrix(0.0, 0.0, dyaw)
                R_cand = R_z @ R_b
                T_c = np.eye(4, dtype=np.float64)
                T_c[:3, :3] = R_cand
                t_xy = ref_mean[:2] - (R_cand @ mov_mean)[:2]
                t_z = ref_z0 - float((R_cand @ np.array([mov_mean[0], mov_mean[1], mov_z0]))[2])
                for dx in (-1.2, 0.0, 1.2):
                    for dy in (-1.2, 0.0, 1.2):
                        T_y = T_c.copy()
                        T_y[:3, 3] = [t_xy[0] + dx, t_xy[1] + dy, t_z]
                        candidates.append(T_y)

        # Also add RANSAC anchor pair translations under IMU orientation & small yaw perturbations
        for R_b in base_rots[:2]:
            for k in range(n_anchor_samples):
                pa = ref_elev[ref_anchor_idx[k]].astype(np.float64)
                pb = mov_elev[mov_anchor_idx[k]].astype(np.float64)
                for dyaw in (0.0, math.pi / 6, -math.pi / 6, math.pi / 2, -math.pi / 2, math.pi):
                    R_cand = euler_rpy_to_matrix(0.0, 0.0, dyaw) @ R_b
                    T_a = np.eye(4, dtype=np.float64)
                    T_a[:3, :3] = R_cand
                    t_vec = pa - (R_cand @ pb)
                    t_vec[2] = ref_z0 - float((R_cand @ np.array([pb[0], pb[1], mov_z0]))[2])
                    T_a[:3, 3] = t_vec
                    candidates.append(T_a)

        # Evaluate coarse candidates rapidly
        best_T = np.eye(4, dtype=np.float64)
        best_score = -1.0
        coarse_thresh = 0.35

        for T_cand in candidates:
            transformed = CSlamRegistrationSolver.transform_xyz(mov_coarse, T_cand)
            dists, _ = ref_tree_coarse.query(transformed, k=1, distance_upper_bound=coarse_thresh, workers=1)
            inliers = np.isfinite(dists)
            n_in = int(np.sum(inliers))
            if n_in < 8:
                continue
            rmse = float(np.sqrt(np.mean(dists[inliers] ** 2)))
            score = float(n_in) / (1.0 + rmse * 4.0)
            if score > best_score:
                best_score = score
                best_T = T_cand

        if best_score <= 0:
            return False, np.eye(4, dtype=np.float64), 0.0, 999.0, 0

        # Multi-stage Trimmed ICP refinement on fine + ultra-fine cloud
        T_curr = best_T.copy()
        schedule = [
            (ref_fine, ref_tree_fine, mov_fine, 0.45, 8),   # Coarse pull-in
            (ref_fine, ref_tree_fine, mov_fine, 0.25, 10),  # Medium alignment
            (ref_fine, ref_tree_fine, mov_fine, 0.14, 10),  # Fine alignment
            (ref_ultra, ref_tree_ultra, mov_ultra, 0.09, 8),  # Ultra-fine alignment
        ]

        for r_pts, r_tree, m_pts, max_corr_d, iters in schedule:
            for _ in range(iters):
                mov_tf = CSlamRegistrationSolver.transform_xyz(m_pts, T_curr)
                dists, idx = r_tree.query(mov_tf, k=1, distance_upper_bound=max_corr_d, workers=1)
                valid = np.isfinite(dists) & (idx < len(r_pts))
                n_val = int(np.sum(valid))
                if n_val < 12:
                    break
                # Trimmed 85% best correspondences to reject non-overlapping boundary bias
                val_dists = dists[valid]
                val_src = mov_tf[valid]
                val_dst = r_pts[idx[valid]]
                if n_val > 20:
                    cutoff = float(np.percentile(val_dists, 85.0))
                    keep = val_dists <= max(cutoff, 0.03)
                    val_src = val_src[keep]
                    val_dst = val_dst[keep]
                if len(val_src) < 10:
                    break
                dT = CSlamRegistrationSolver.umeyama_se3(val_src, val_dst)
                T_curr = dT @ T_curr
                if np.linalg.norm(dT[:3, 3]) < 1e-4 and np.trace(dT[:3, :3]) > 2.99999:
                    break

        # Final Overlap & Quality Gate Evaluation on ultra-fine cloud
        mov_final = CSlamRegistrationSolver.transform_xyz(mov_ultra, T_curr)
        final_dists, _ = ref_tree_ultra.query(mov_final, k=1, distance_upper_bound=inlier_thresh_m, workers=1)
        inlier_mask = np.isfinite(final_dists)
        inlier_count = int(np.sum(inlier_mask))
        denom = max(1, min(len(ref_ultra), len(mov_ultra)))
        overlap_ratio = float(inlier_count) / float(denom)

        if inlier_count >= 15:
            inlier_rmse_m = float(np.sqrt(np.mean(final_dists[inlier_mask] ** 2)))
        else:
            inlier_rmse_m = 999.0

        is_overlapping = (
            overlap_ratio >= min_overlap_ratio
            and inlier_rmse_m <= max_inlier_rmse_m
            and inlier_count >= 25
        )
        return is_overlapping, T_curr, overlap_ratio, inlier_rmse_m, inlier_count


class CSlamNetworkEngine:
    """
    Unlimited-Peer Collaborative SLAM (C-SLAM) Engine.
    Manages:
    - Local scanner identity, pose, IMU orientation, and voxel submap broadcasting.
    - Unlimited remote peer discovery and real-time submap aggregation.
    - Non-overlapping IMU-oriented spatial separation (before overlap) and automatic real-time SE(3) snapping (on overlap).
    """

    MULTICAST_GROUP = "239.255.77.12"
    BASE_UDP_PORT = 19210
    LOCAL_PORT_SPAN = 8  # Allows up to 8 concurrent scanner instances on the same host + unlimited LAN IPs

    def __init__(
        self,
        scanner_name: str = "ScanHUB-Unit-01",
        project_name: str = "Default_Project",
        dataset_name: str = "Scan_01",
        listen_port: Optional[int] = None,
        separation_distance_m: float = 15.0,
        min_overlap_ratio: float = 0.18,
    ):
        self.local_uuid: str = str(uuid.uuid4())[:12]
        self.scanner_name: str = scanner_name
        self.project_name: str = project_name
        self.dataset_name: str = dataset_name
        self.is_recording: bool = False
        self.requested_port: Optional[int] = listen_port
        self.separation_distance_m: float = float(separation_distance_m)
        self.min_overlap_ratio: float = float(min_overlap_ratio)
        self.color_mode: CSlamColorMode = CSlamColorMode.SCANNER_DIFFERENTIATED
        self.enabled: bool = True

        self._lock = threading.RLock()
        self.peers: Dict[str, PeerScannerSession] = {}
        self.discovered_network_sessions: Dict[str, Dict] = {}
        self._discovered_peer_cache: Dict[str, Dict] = {}
        self._next_peer_index: int = 1  # 0 is Local Scanner

        # Local scanner state
        self.local_pose_xyz: Tuple[float, float, float] = (0.0, 0.0, 0.0)
        self.local_imu_rpy_rad: Tuple[float, float, float] = (0.0, 0.0, 0.0)
        self.local_points: np.ndarray = np.zeros((0, 6), dtype=np.float32)
        self.local_trajectory: List[Tuple[float, float, float]] = []

        # Networking sockets & background threads
        self._running = False
        self._bound_port: int = self.BASE_UDP_PORT if listen_port is None else int(listen_port)
        self._sock: Optional[socket.socket] = None
        self._manual_peer_endpoints: List[Tuple[str, int]] = []
        self._rx_thread: Optional[threading.Thread] = None
        self._tx_thread: Optional[threading.Thread] = None
        self._reg_thread: Optional[threading.Thread] = None
        self._sim_threads: Dict[str, threading.Thread] = {}
        self._sim_stop_flags: Dict[str, bool] = {}

        # Optional UI callback when peer state/alignment updates
        self.on_state_changed: Optional[Callable[[], None]] = None

    def start(self) -> int:
        """Starts UDP discovery, submap streaming, and background overlap co-registration threads."""
        if self._running:
            return self._bound_port

        self._running = True
        try:
            self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
            self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            except Exception:
                pass

            bound = False
            if self.requested_port == 0:
                self._sock.bind(("", 0))
                self._bound_port = self._sock.getsockname()[1]
                bound = True
            else:
                base_p = self.requested_port if self.requested_port is not None else self.BASE_UDP_PORT
                for offset in range(self.LOCAL_PORT_SPAN):
                    port_try = base_p + offset
                    try:
                        self._sock.bind(("", port_try))
                        self._bound_port = port_try
                        bound = True
                        break
                    except OSError:
                        continue

            if not bound:
                self._sock.bind(("", 0))
                self._bound_port = self._sock.getsockname()[1]

            # Join UDP multicast group if supported by interface
            try:
                mreq = struct.pack("4sl", socket.inet_aton(self.MULTICAST_GROUP), socket.INADDR_ANY)
                self._sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
            except Exception:
                pass

            self._sock.settimeout(0.25)
        except Exception:
            self._sock = None
            self._bound_port = 0

        self._rx_thread = threading.Thread(target=self._network_rx_loop, daemon=True, name="CSlamRx")
        self._tx_thread = threading.Thread(target=self._network_tx_loop, daemon=True, name="CSlamTx")
        self._reg_thread = threading.Thread(target=self._registration_worker_loop, daemon=True, name="CSlamReg")
        self._rx_thread.start()
        self._tx_thread.start()
        self._reg_thread.start()
        return self._bound_port

    def stop(self):
        self._running = False
        for k in list(self._sim_stop_flags.keys()):
            self._sim_stop_flags[k] = True
        if self._sock is not None:
            try:
                self._sock.close()
            except Exception:
                pass
            self._sock = None

    def set_project_name(self, project_name: str, dataset_name: Optional[str] = None):
        cached_to_promote = []
        with self._lock:
            clean_name = (project_name or "Default_Project").strip()
            self.project_name = clean_name
            if dataset_name is not None:
                self.dataset_name = (dataset_name or clean_name).strip()
            for s_uuid, pkt in list(self._discovered_peer_cache.items()):
                if pkt.get("project_name") == self.project_name and s_uuid not in self.peers:
                    cached_to_promote.append(dict(pkt))
        for pkt in cached_to_promote:
            self.register_or_update_peer_data(
                scanner_uuid=pkt["scanner_uuid"],
                scanner_name=pkt["scanner_name"],
                project_name=pkt["project_name"],
                pose_xyz=pkt["pose_xyz"],
                imu_rpy_rad=pkt["imu_rpy_rad"],
                points_chunk_or_full=pkt.get("points"),
                is_full_submap=True,
                ip_address=pkt.get("ip_address", "127.0.0.1"),
                port=pkt.get("port", self.BASE_UDP_PORT),
                dataset_name=pkt.get("dataset_name", ""),
                is_recording=pkt.get("is_recording", True),
            )

    def set_dataset_name(self, dataset_name: str):
        with self._lock:
            self.dataset_name = (dataset_name or "Scan_01").strip()

    def set_recording_active(self, is_recording: bool):
        with self._lock:
            self.is_recording = bool(is_recording)

    def get_discovered_network_sessions(self, max_age_sec: float = 15.0) -> List[Dict]:
        """Returns all active C-SLAM capture sessions discovered across the network (across all projects)."""
        now = time.time()
        with self._lock:
            stale = [k for k, v in self.discovered_network_sessions.items() if (now - v.get("last_seen_ts", 0.0)) > max_age_sec]
            for k in stale:
                self.discovered_network_sessions.pop(k, None)
                self._discovered_peer_cache.pop(k, None)
            sessions = [dict(v) for v in self.discovered_network_sessions.values()]
        sessions.sort(key=lambda s: s.get("last_seen_ts", 0.0), reverse=True)
        return sessions

    def set_color_mode(self, mode: CSlamColorMode):
        with self._lock:
            self.color_mode = mode

    def add_manual_peer_ip(self, ip_address: str, port: int = BASE_UDP_PORT):
        ep = (ip_address.strip(), int(port))
        with self._lock:
            if ep not in self._manual_peer_endpoints:
                self._manual_peer_endpoints.append(ep)

    def add_unicast_peer(self, ip_address: str, port: int = BASE_UDP_PORT):
        self.add_manual_peer_ip(ip_address, port)

    def update_local_state(
        self,
        points: Optional[np.ndarray] = None,
        pose_xyz: Optional[Tuple[float, float, float]] = None,
        imu_rpy_rad: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        trajectory: Optional[List[Tuple[float, float, float]]] = None,
        points_xyz_rgb: Optional[np.ndarray] = None,
        pos_xyz: Optional[Tuple[float, float, float]] = None,
    ):
        pts_in = points_xyz_rgb if points_xyz_rgb is not None else points
        pos_in = pos_xyz if pos_xyz is not None else (pose_xyz if pose_xyz is not None else (0.0, 0.0, 0.0))
        with self._lock:
            self.local_pose_xyz = (float(pos_in[0]), float(pos_in[1]), float(pos_in[2]))
            self.local_imu_rpy_rad = (float(imu_rpy_rad[0]), float(imu_rpy_rad[1]), float(imu_rpy_rad[2]))
            if pts_in is not None and len(pts_in) > 0:
                arr = np.asarray(pts_in, dtype=np.float32)
                if arr.shape[1] == 3:
                    rgb = np.full((len(arr), 3), 0.75, dtype=np.float32)
                    arr = np.hstack([arr, rgb])
                else:
                    arr = arr[:, :6].copy()
                    if np.max(arr[:, 3:6]) > 1.05:
                        arr[:, 3:6] /= 255.0
                self.local_points = arr
            if trajectory is not None:
                self.local_trajectory = list(trajectory)

    def _ingest_packet(self, pkt_dict: Dict, sender_ip: str = "127.0.0.1"):
        """Ingests a structured C-SLAM packet dict directly (used by unit tests and JSON peers)."""
        if pkt_dict.get("magic") != "SCANHUB_CSLAM_V1":
            return
        s_uuid = pkt_dict.get("scanner_id") or pkt_dict.get("uuid") or ""
        if not s_uuid or s_uuid == self.local_uuid:
            return
        s_name = pkt_dict.get("scanner_name") or pkt_dict.get("name") or f"Peer-{s_uuid[:4]}"
        proj = pkt_dict.get("project_name") or pkt_dict.get("proj") or self.project_name
        dset = pkt_dict.get("dataset_name") or pkt_dict.get("dataset") or proj
        is_rec = bool(pkt_dict.get("is_recording", pkt_dict.get("rec", True)))
        pose = tuple(pkt_dict.get("pos_xyz") or pkt_dict.get("pose") or (0.0, 0.0, 0.0))
        imu = tuple(pkt_dict.get("imu_rpy_rad") or pkt_dict.get("imu") or (0.0, 0.0, 0.0))
        raw_pts = pkt_dict.get("points")
        pts_arr = None
        if raw_pts is not None:
            pts_arr = np.asarray(raw_pts, dtype=np.float32)
            if pts_arr.ndim == 2 and pts_arr.shape[1] == 3:
                pts_arr = np.hstack([pts_arr, np.full((len(pts_arr), 3), 0.8, dtype=np.float32)])
        self.register_or_update_peer_data(
            scanner_uuid=s_uuid,
            scanner_name=s_name,
            project_name=proj,
            pose_xyz=pose,
            imu_rpy_rad=imu,
            points_chunk_or_full=pts_arr,
            is_full_submap=True,
            ip_address=sender_ip,
            dataset_name=dset,
            is_recording=is_rec,
        )

    def compute_separated_imu_transform(self, peer: PeerScannerSession) -> np.ndarray:
        """
        Computes T_world_from_peer when the peer has NO overlapping geometry yet:
        1. Orients the peer's scan using its relative IMU roll/pitch/yaw (R_imu).
        2. Places the peer's scan at a deterministic non-overlapping concentric-ring offset
           so unlimited scanners can be viewed simultaneously without colliding.
        """
        # Relative IMU orientation: align peer's IMU frame relative to local IMU frame
        r_loc, p_loc, y_loc = self.local_imu_rpy_rad
        r_peer, p_peer, y_peer = peer.imu_rpy_rad
        # Relative yaw + gravity tilt orientation
        R_rel_imu = euler_rpy_to_matrix(r_peer - r_loc, p_peer - p_loc, y_peer - y_loc)

        # Deterministic concentric ring / golden-angle separation offset for peer_index >= 1
        idx = max(1, peer.scanner_index)
        ring_idx = (idx - 1) // 6
        slot_in_ring = (idx - 1) % 6
        radius = self.separation_distance_m * (1.0 + 0.75 * ring_idx)
        angle_rad = slot_in_ring * (math.pi / 3.0) + (ring_idx * 0.261799)  # 60 deg steps + ring stagger

        offset_x = radius * math.cos(angle_rad)
        offset_y = radius * math.sin(angle_rad)

        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R_rel_imu
        T[0, 3] = offset_x
        T[1, 3] = offset_y
        T[2, 3] = 0.0
        return T

    def register_or_update_peer_data(
        self,
        scanner_uuid: str,
        scanner_name: str,
        project_name: str,
        pose_xyz: Tuple[float, float, float],
        imu_rpy_rad: Tuple[float, float, float],
        points_chunk_or_full: Optional[np.ndarray] = None,
        is_full_submap: bool = False,
        ip_address: str = "127.0.0.1",
        port: int = BASE_UDP_PORT,
        dataset_name: str = "",
        is_recording: bool = True,
    ) -> Optional[PeerScannerSession]:
        """
        Registers a remote scanner (unlimited peers supported) or updates its live pose, IMU orientation, and point cloud.
        Also records cross-project network sessions so operators in the Projects Hub can discover and join active captures.
        """
        if scanner_uuid == self.local_uuid:
            return None

        now_ts = time.time()
        eff_dataset = (dataset_name or project_name).strip()
        n_pts_in = int(len(points_chunk_or_full)) if points_chunk_or_full is not None else 0

        with self._lock:
            prev_disc = self.discovered_network_sessions.get(scanner_uuid, {})
            prev_pts_n = prev_disc.get("point_count", 0)
            self.discovered_network_sessions[scanner_uuid] = {
                "scanner_uuid": scanner_uuid,
                "scanner_name": scanner_name,
                "project_name": project_name,
                "dataset_name": eff_dataset,
                "ip_address": ip_address,
                "port": int(port),
                "point_count": n_pts_in if n_pts_in > 0 else prev_pts_n,
                "pose_xyz": (float(pose_xyz[0]), float(pose_xyz[1]), float(pose_xyz[2])),
                "imu_rpy_rad": (float(imu_rpy_rad[0]), float(imu_rpy_rad[1]), float(imu_rpy_rad[2])),
                "is_recording": bool(is_recording),
                "last_seen_ts": now_ts,
            }
            self._discovered_peer_cache[scanner_uuid] = {
                "scanner_uuid": scanner_uuid,
                "scanner_name": scanner_name,
                "project_name": project_name,
                "dataset_name": eff_dataset,
                "pose_xyz": (float(pose_xyz[0]), float(pose_xyz[1]), float(pose_xyz[2])),
                "imu_rpy_rad": (float(imu_rpy_rad[0]), float(imu_rpy_rad[1]), float(imu_rpy_rad[2])),
                "points": points_chunk_or_full.copy() if (points_chunk_or_full is not None and len(points_chunk_or_full) > 0) else self._discovered_peer_cache.get(scanner_uuid, {}).get("points"),
                "ip_address": ip_address,
                "port": int(port),
                "is_recording": bool(is_recording),
            }

        if project_name != self.project_name:
            return None

        with self._lock:
            peer = self.peers.get(scanner_uuid)
            if peer is None:
                idx = self._next_peer_index
                self._next_peer_index += 1
                peer = PeerScannerSession(
                    scanner_uuid=scanner_uuid,
                    scanner_name=scanner_name,
                    project_name=project_name,
                    scanner_index=idx,
                    ip_address=ip_address,
                    port=port,
                    imu_initial_yaw_rad=float(imu_rpy_rad[2]),
                )
                self.peers[scanner_uuid] = peer

            peer.scanner_name = scanner_name
            peer.last_seen_ts = time.time()
            peer.pose_xyz = (float(pose_xyz[0]), float(pose_xyz[1]), float(pose_xyz[2]))
            peer.imu_rpy_rad = (float(imu_rpy_rad[0]), float(imu_rpy_rad[1]), float(imu_rpy_rad[2]))

            if not peer.trajectory_local or math.dist(peer.trajectory_local[-1], peer.pose_xyz) >= 0.08:
                peer.trajectory_local.append(peer.pose_xyz)
                if len(peer.trajectory_local) > 2000:
                    peer.trajectory_local = peer.trajectory_local[-2000:]

            if points_chunk_or_full is not None and len(points_chunk_or_full) > 0:
                pts = points_chunk_or_full[:, :6].astype(np.float32)
                if is_full_submap or len(peer.raw_points_local) == 0:
                    peer.raw_points_local = pts
                else:
                    combined = np.vstack([peer.raw_points_local, pts])
                    if len(combined) > 35000:
                        combined = PointCloudProcessor.voxel_downsample(combined, 0.04)
                    peer.raw_points_local = combined

            # If still in SEPARATED_IMU_ORIENTED mode, keep its IMU-oriented separated transform updated
            effective_state = peer.manual_override or peer.alignment_state
            if effective_state == PeerAlignmentState.SEPARATED_IMU_ORIENTED:
                peer.transform_world_from_peer = self.compute_separated_imu_transform(peer)

            return peer

    def run_overlap_check_now(self, target_uuid: Optional[str] = None) -> int:
        """
        Immediately runs overlap detection and SE(3) registration for one or all active peers in the current project.
        Returns the number of peers currently in OVERLAP_SNAPPED state.
        """
        with self._lock:
            local_pts = self.local_points.copy() if len(self.local_points) > 0 else np.zeros((0, 6), dtype=np.float32)
            r_loc, p_loc, y_loc = self.local_imu_rpy_rad
            peer_items = [
                (u, p) for u, p in self.peers.items()
                if p.project_name == self.project_name and (target_uuid is None or u == target_uuid)
            ]

        snapped_count = 0
        state_changed = False

        for u, peer in peer_items:
            with self._lock:
                mov_pts = peer.raw_points_local.copy()
                r_peer, p_peer, y_peer = peer.imu_rpy_rad
                prior_T = (
                    peer.transform_world_from_peer.copy()
                    if peer.alignment_state == PeerAlignmentState.OVERLAP_SNAPPED
                    else None
                )
                manual_mode = peer.manual_override

            if manual_mode == PeerAlignmentState.SEPARATED_IMU_ORIENTED:
                with self._lock:
                    peer.alignment_state = PeerAlignmentState.SEPARATED_IMU_ORIENTED
                    peer.transform_world_from_peer = self.compute_separated_imu_transform(peer)
                continue

            if len(local_pts) < 60 or len(mov_pts) < 60:
                with self._lock:
                    if manual_mode != PeerAlignmentState.OVERLAP_SNAPPED:
                        peer.alignment_state = PeerAlignmentState.SEPARATED_IMU_ORIENTED
                        peer.transform_world_from_peer = self.compute_separated_imu_transform(peer)
                continue

            R_rel_imu = euler_rpy_to_matrix(r_peer - r_loc, p_peer - p_loc, y_peer - y_loc)
            is_overlapping, T_est, overlap_ratio, rmse_m, inliers = CSlamRegistrationSolver.detect_overlap_and_register(
                ref_points=local_pts,
                mov_points=mov_pts,
                imu_relative_rot=R_rel_imu,
                prior_transform=prior_T,
                min_overlap_ratio=self.min_overlap_ratio,
            )

            with self._lock:
                prev_state = peer.alignment_state
                peer.overlap_ratio = overlap_ratio
                peer.inlier_rmse_m = rmse_m
                peer.inlier_count = inliers

                if is_overlapping or manual_mode == PeerAlignmentState.OVERLAP_SNAPPED:
                    peer.alignment_state = PeerAlignmentState.OVERLAP_SNAPPED
                    peer.transform_world_from_peer = T_est
                    snapped_count += 1
                else:
                    peer.alignment_state = PeerAlignmentState.SEPARATED_IMU_ORIENTED
                    peer.transform_world_from_peer = self.compute_separated_imu_transform(peer)

                if peer.alignment_state != prev_state:
                    state_changed = True

        if state_changed and self.on_state_changed is not None:
            try:
                self.on_state_changed()
            except Exception:
                pass

        return snapped_count

    def set_peer_manual_override(self, scanner_uuid: str, mode: Optional[PeerAlignmentState]):
        """Allows operator to force-snap, force-separate, or return a peer to automatic overlap snap."""
        with self._lock:
            peer = self.peers.get(scanner_uuid)
            if peer is None:
                return
            peer.manual_override = mode
            if mode == PeerAlignmentState.SEPARATED_IMU_ORIENTED:
                peer.alignment_state = PeerAlignmentState.SEPARATED_IMU_ORIENTED
                peer.transform_world_from_peer = self.compute_separated_imu_transform(peer)
        if mode != PeerAlignmentState.SEPARATED_IMU_ORIENTED:
            self.run_overlap_check_now(target_uuid=scanner_uuid)

    def get_merged_cslam_scene(
        self,
        local_points_override: Optional[np.ndarray] = None,
        preserve_local_colors: bool = False,
    ) -> Tuple[np.ndarray, List[Dict]]:
        """
        Returns:
        1. `merged_points` (M, 6) float32: Combined point cloud of the local scanner + all active peers in the project,
           each transformed by its current alignment matrix (either SEPARATED_IMU_ORIENTED or OVERLAP_SNAPPED) and
           colored according to `self.color_mode` (distinct scanner colors vs. true RGB).
        2. `peer_descriptors`: List of dictionaries with each scanner's live pose in world coordinates, trajectory,
           color, overlap %, RMSE, and alignment state for rendering on the 2D/3D canvas and C-SLAM panel.
        """
        with self._lock:
            base_local = (
                local_points_override[:, :6].astype(np.float32)
                if local_points_override is not None
                else self.local_points.copy()
            )
            color_mode = self.color_mode
            active_peers = [
                p for p in self.peers.values()
                if p.project_name == self.project_name
            ]
            active_peers.sort(key=lambda x: x.scanner_index)

        cloud_parts: List[np.ndarray] = []
        if len(base_local) > 0:
            if (
                not preserve_local_colors
                and color_mode == CSlamColorMode.SCANNER_DIFFERENTIATED
                and len(active_peers) > 0
            ):
                cloud_parts.append(apply_scanner_tint(base_local, get_scanner_color_rgb(0)))
            else:
                cloud_parts.append(base_local)

        peer_descriptors: List[Dict] = []
        for peer in active_peers:
            with self._lock:
                T = peer.transform_world_from_peer.copy()
                raw_pts = peer.raw_points_local.copy()
                pose_local = np.array([[peer.pose_xyz[0], peer.pose_xyz[1], peer.pose_xyz[2]]], dtype=np.float32)
                traj_local = (
                    np.asarray(peer.trajectory_local, dtype=np.float32)
                    if len(peer.trajectory_local) > 0
                    else pose_local
                )
                imu_rpy_deg = (
                    math.degrees(peer.imu_rpy_rad[0]),
                    math.degrees(peer.imu_rpy_rad[1]),
                    math.degrees(peer.imu_rpy_rad[2]),
                )
                yaw_world = float(peer.imu_rpy_rad[2] + math.atan2(T[1, 0], T[0, 0]))
                state = peer.alignment_state
                overlap_ratio = float(peer.overlap_ratio)
                overlap_pct = overlap_ratio * 100.0
                icp_rmse_m = float(peer.inlier_rmse_m if peer.inlier_rmse_m < 100.0 else 0.0)
                rmse_mm = icp_rmse_m * 1000.0
                s_idx = peer.scanner_index
                s_name = peer.scanner_name
                s_uuid = peer.scanner_uuid
                s_color_rgb = peer.color_rgb
                s_color_hex = peer.color_hex

            if len(raw_pts) > 0:
                tf_pts = CSlamRegistrationSolver.transform_points6(raw_pts, T)
                if color_mode == CSlamColorMode.SCANNER_DIFFERENTIATED:
                    tf_pts = apply_scanner_tint(tf_pts, s_color_rgb)
                cloud_parts.append(tf_pts)

            pose_world = CSlamRegistrationSolver.transform_xyz(pose_local, T)[0]
            traj_world = CSlamRegistrationSolver.transform_xyz(traj_local, T)
            world_pos_tuple = (float(pose_world[0]), float(pose_world[1]), float(pose_world[2]))

            peer_descriptors.append({
                "scanner_uuid": s_uuid,
                "scanner_name": s_name,
                "scanner_index": s_idx,
                "color_index": s_idx,
                "color_rgb": s_color_rgb,
                "color_hex": s_color_hex,
                "alignment_state": state.name,
                "is_snapped": state == PeerAlignmentState.OVERLAP_SNAPPED,
                "overlap_ratio": overlap_ratio,
                "overlap_percent": overlap_pct,
                "icp_rmse_m": icp_rmse_m,
                "rmse_mm": rmse_mm,
                "point_count": len(raw_pts),
                "pose_world": world_pos_tuple,
                "world_pos": world_pos_tuple,
                "imu_rpy_deg": imu_rpy_deg,
                "yaw_world_rad": yaw_world,
                "trajectory_world": traj_world,
                "world_traj": traj_world.tolist(),
                "offset_m": float(np.linalg.norm(T[:3, 3])),
            })

        if not cloud_parts:
            merged = np.zeros((0, 6), dtype=np.float32)
        elif len(cloud_parts) == 1:
            merged = cloud_parts[0]
        else:
            merged = np.vstack(cloud_parts)

        return merged, peer_descriptors

    def get_status_summary(self, peer_descriptors: Optional[List[Dict]] = None) -> Dict:
        """Returns a thread-safe status dictionary for UI telemetry and multi-scanner status panels."""
        with self._lock:
            proj = self.project_name
            loc_name = self.scanner_name
            loc_pts = int(len(self.local_points))
            c_mode = self.color_mode.value
            has_peers = any(p.project_name == self.project_name for p in self.peers.values())
        if peer_descriptors is None:
            if has_peers:
                _, peer_descriptors = self.get_merged_cslam_scene(
                    local_points_override=np.zeros((0, 6), dtype=np.float32)
                )
            else:
                peer_descriptors = []
        snapped_n = sum(1 for p in peer_descriptors if p["is_snapped"])
        sep_n = sum(1 for p in peer_descriptors if not p["is_snapped"])
        return {
            "project_name": proj,
            "local_scanner_name": loc_name,
            "local_color_hex": get_scanner_color_hex(0),
            "local_points": loc_pts,
            "color_mode": c_mode,
            "total_scanners": 1 + len(peer_descriptors),
            "peer_count": len(peer_descriptors),
            "snapped_peers": snapped_n,
            "separated_peers": sep_n,
            "peers": peer_descriptors,
        }

    # ------------------------------------------------------------------
    # Multi-Scanner Simulation / Replay Helper (for testing & demos)
    # ------------------------------------------------------------------
    def spawn_simulated_peer_scanner(
        self,
        peer_name: Optional[str] = None,
        start_non_overlapping_seconds: float = 1.6,
        project_name: Optional[str] = None,
        dataset_name: Optional[str] = None,
    ) -> str:
        """
        Spawns a live simulated remote scanner joining the current or specified project/dataset.
        - Phase 1 (first `start_non_overlapping_seconds`): Scans a separate area with NO overlap with the local scan.
          It is displayed at the non-overlapping offset distance, oriented by its relative IMU heading.
        - Phase 2 (after `start_non_overlapping_seconds`): The simulated peer walks into the overlapping room geometry
          of the local scanner (rotated by an arbitrary IMU yaw + offset in its own local frame).
          The C-SLAM registration engine automatically detects the overlap and SNAPS the two point clouds together!
        """
        if not self._running:
            self.start()
        with self._lock:
            peer_num = self._next_peer_index
            p_uuid = f"sim-peer-{peer_num:02d}-{uuid.uuid4().hex[:4]}"
            if not peer_name:
                peer_name = f"Scanner-{peer_num + 1:02d}"
            self._sim_stop_flags[p_uuid] = False

        def _sim_worker():
            t_start = time.time()
            rng = np.random.default_rng(100 + peer_num)
            # Arbitrary local frame rotation for this peer scanner (e.g. started facing +24 deg per peer)
            true_peer_yaw = math.radians(24.0 * peer_num)
            R_peer_to_world = euler_rpy_to_matrix(0.0, 0.0, true_peer_yaw)
            R_world_to_peer = R_peer_to_world.T
            peer_origin_in_world = np.array([1.2 * peer_num, -0.8 * peer_num, 0.0], dtype=np.float64)

            step_idx = 0
            while self._running and not self._sim_stop_flags.get(p_uuid, True):
                elapsed = time.time() - t_start
                step_idx += 1
                walk_angle = elapsed * 0.6 + peer_num
                pos_peer = (
                    float(1.5 * math.cos(walk_angle)),
                    float(1.5 * math.sin(walk_angle)),
                    0.0,
                )
                # Slight IMU roll/pitch oscillation + true heading
                imu_rpy = (
                    float(math.radians(0.4 * math.sin(elapsed * 1.5))),
                    float(math.radians(0.4 * math.cos(elapsed * 1.3))),
                    float(true_peer_yaw),
                )

                with self._lock:
                    loc_pts = self.local_points.copy()
                    proj = project_name if project_name else self.project_name
                    dset = dataset_name if dataset_name else self.dataset_name

                if elapsed < start_non_overlapping_seconds or len(loc_pts) < 100:
                    # Generate non-overlapping distant structure in peer's local frame
                    n_gen = 450
                    gx = rng.uniform(45.0, 52.0, n_gen)
                    gy = rng.uniform(45.0, 52.0, n_gen)
                    gz = 1.5 * np.sin(gx * 1.8) + 1.5 * np.cos(gy * 1.8) + 12.0
                    grgb = np.column_stack([
                        np.full(n_gen, 0.75),
                        np.full(n_gen, 0.75),
                        np.full(n_gen, 0.80),
                    ])
                    chunk = np.column_stack([gx, gy, gz, grgb]).astype(np.float32)
                    self.register_or_update_peer_data(
                        scanner_uuid=p_uuid,
                        scanner_name=peer_name,
                        project_name=proj,
                        pose_xyz=pos_peer,
                        imu_rpy_rad=imu_rpy,
                        points_chunk_or_full=chunk,
                        is_full_submap=True,
                        dataset_name=dset,
                        is_recording=True,
                    )
                else:
                    # Peer walks into overlapping region of the local scan!
                    n_loc = len(loc_pts)
                    sample_n = min(3500, n_loc)
                    idx = rng.choice(n_loc, size=sample_n, replace=False)
                    overlap_world = loc_pts[idx, :6].copy()
                    # Add realistic 1.0 cm sensor noise
                    overlap_world[:, :3] += rng.normal(0.0, 0.008, size=(sample_n, 3)).astype(np.float32)

                    # Also add extra hallway/wing captured uniquely by this peer scanner
                    n_ext = 600
                    ext_x = rng.uniform(3.0, 5.5, n_ext) * (1.0 if peer_num % 2 == 1 else -1.0)
                    ext_y = rng.uniform(-2.5, 2.5, n_ext)
                    ext_z = np.where(rng.random(n_ext) < 0.5, 0.0, rng.uniform(0.0, 2.2, n_ext))
                    ext_pts = np.column_stack([
                        ext_x, ext_y, ext_z,
                        np.full((n_ext, 3), 0.82, dtype=np.float32),
                    ]).astype(np.float32)

                    combined_world = np.vstack([overlap_world, ext_pts])
                    # Transform from world frame into peer's local coordinate frame:
                    # p_world = R_peer_to_world @ p_peer + peer_origin_in_world
                    # => p_peer = R_world_to_peer @ (p_world - peer_origin_in_world)
                    xyz_peer = ((combined_world[:, :3].astype(np.float64) - peer_origin_in_world) @ R_world_to_peer.T).astype(np.float32)
                    peer_cloud = combined_world.copy()
                    peer_cloud[:, :3] = xyz_peer

                    self.register_or_update_peer_data(
                        scanner_uuid=p_uuid,
                        scanner_name=peer_name,
                        project_name=proj,
                        pose_xyz=pos_peer,
                        imu_rpy_rad=imu_rpy,
                        points_chunk_or_full=peer_cloud,
                        is_full_submap=True,
                        dataset_name=dset,
                        is_recording=True,
                    )

                time.sleep(0.35)

        t = threading.Thread(target=_sim_worker, daemon=True, name=f"SimPeer-{p_uuid}")
        self._sim_threads[p_uuid] = t
        t.start()
        return p_uuid

    def clear_simulated_peers(self):
        with self._lock:
            for k in list(self._sim_stop_flags.keys()):
                self._sim_stop_flags[k] = True
                self.peers.pop(k, None)
                self.discovered_network_sessions.pop(k, None)
                self._discovered_peer_cache.pop(k, None)
            self._sim_stop_flags.clear()
        if self.on_state_changed is not None:
            try:
                self.on_state_changed()
            except Exception:
                pass

    def clear_peers(self):
        with self._lock:
            for k in list(self._sim_stop_flags.keys()):
                self._sim_stop_flags[k] = True
            self._sim_stop_flags.clear()
            self.peers.clear()
            self.discovered_network_sessions.clear()
            self._discovered_peer_cache.clear()
            self._next_peer_index = 1
        if self.on_state_changed is not None:
            try:
                self.on_state_changed()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Network Transport Loops (UDP Broadcast / Multicast / Loopback)
    # ------------------------------------------------------------------
    def _encode_submap_packet(self) -> bytes:
        with self._lock:
            pts = self.local_points
            if len(pts) > 2500:
                pts = PointCloudProcessor.voxel_downsample(pts, 0.08)
                if len(pts) > 2500:
                    idx = np.linspace(0, len(pts) - 1, 2500, dtype=np.int32)
                    pts = pts[idx]

            meta = {
                "magic": "SCANHUB_CSLAM_V1",
                "uuid": self.local_uuid,
                "name": self.scanner_name,
                "proj": self.project_name,
                "dataset": self.dataset_name,
                "rec": bool(self.is_recording),
                "port": self._bound_port,
                "pose": list(self.local_pose_xyz),
                "imu": list(self.local_imu_rpy_rad),
                "n": int(len(pts)),
            }

        header_bytes = json.dumps(meta, separators=(",", ":")).encode("utf-8")
        if len(pts) > 0:
            # Quantize XYZ to millimeters (int16/float16) + RGB uint8 for compact UDP frame
            xyz_f16 = pts[:, :3].astype(np.float16).tobytes()
            rgb_raw = pts[:, 3:6]
            if np.max(rgb_raw) <= 1.05:
                rgb_u8 = np.clip(rgb_raw * 255.0, 0, 255).astype(np.uint8).tobytes()
            else:
                rgb_u8 = np.clip(rgb_raw, 0, 255).astype(np.uint8).tobytes()
            payload_compressed = zlib.compress(xyz_f16 + rgb_u8, level=1)
        else:
            payload_compressed = b""

        return struct.pack("<I", len(header_bytes)) + header_bytes + payload_compressed

    def _decode_and_handle_packet(self, data: bytes, addr: Tuple[str, int]):
        if len(data) < 8:
            return
        hdr_len = struct.unpack("<I", data[:4])[0]
        if hdr_len <= 0 or 4 + hdr_len > len(data):
            return
        meta = json.loads(data[4:4 + hdr_len].decode("utf-8"))
        if meta.get("magic") != "SCANHUB_CSLAM_V1":
            return
        s_uuid = meta.get("uuid", "")
        if not s_uuid or s_uuid == self.local_uuid:
            return

        n_pts = int(meta.get("n", 0))
        pts_arr = None
        payload = data[4 + hdr_len:]
        if n_pts > 0 and payload:
            raw = zlib.decompress(payload)
            xyz_bytes_len = n_pts * 3 * 2  # float16
            rgb_bytes_len = n_pts * 3 * 1  # uint8
            if len(raw) == xyz_bytes_len + rgb_bytes_len:
                xyz = np.frombuffer(raw[:xyz_bytes_len], dtype=np.float16).astype(np.float32).reshape(n_pts, 3)
                rgb = (np.frombuffer(raw[xyz_bytes_len:], dtype=np.uint8).astype(np.float32) / 255.0).reshape(n_pts, 3)
                pts_arr = np.hstack([xyz, rgb])

        self.register_or_update_peer_data(
            scanner_uuid=s_uuid,
            scanner_name=meta.get("name", f"Peer-{s_uuid[:4]}"),
            project_name=meta.get("proj", "Default_Project"),
            pose_xyz=tuple(meta.get("pose", [0.0, 0.0, 0.0])),
            imu_rpy_rad=tuple(meta.get("imu", [0.0, 0.0, 0.0])),
            points_chunk_or_full=pts_arr,
            is_full_submap=True,
            ip_address=addr[0],
            port=int(meta.get("port", addr[1])),
            dataset_name=meta.get("dataset", "Scan_01"),
            is_recording=bool(meta.get("rec", n_pts > 0)),
        )

    def _network_tx_loop(self):
        while self._running:
            try:
                if self.enabled and self._sock is not None:
                    pkt = self._encode_submap_packet()
                    if len(pkt) < 65000:
                        targets = set()
                        # Local loopback multi-instance ports
                        for offset in range(self.LOCAL_PORT_SPAN):
                            p = self.BASE_UDP_PORT + offset
                            if p != self._bound_port:
                                targets.add(("127.0.0.1", p))
                        # LAN Multicast + Broadcast
                        targets.add((self.MULTICAST_GROUP, self.BASE_UDP_PORT))
                        targets.add(("255.255.255.255", self.BASE_UDP_PORT))
                        with self._lock:
                            for ep in self._manual_peer_endpoints:
                                targets.add(ep)
                        for host, port in targets:
                            try:
                                self._sock.sendto(pkt, (host, port))
                            except Exception:
                                pass
            except Exception:
                pass
            time.sleep(0.8)

    def _network_rx_loop(self):
        while self._running:
            try:
                if self._sock is None:
                    time.sleep(0.2)
                    continue
                data, addr = self._sock.recvfrom(65535)
                if self.enabled and data:
                    self._decode_and_handle_packet(data, addr)
            except socket.timeout:
                continue
            except Exception:
                time.sleep(0.1)

    def _registration_worker_loop(self):
        while self._running:
            try:
                if self.enabled:
                    self.run_overlap_check_now()
            except Exception:
                pass
            time.sleep(1.0)
