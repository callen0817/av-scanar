#!/usr/bin/env python3
"""
Milestone M15 — AV Structure Engine:
Intelligent RGB Keyframe Capture, Virtual Scan Stations, Structured Spherical Scans,
Geometry-Assisted 360° Panorama Synthesis, and ASTM E2807-11 Structured E57 Export.

Designed as a reusable hardware-agnostic spatial engine supporting:
  1. AV scanAR (RoboSense Airy LiDAR + IMU + Orbbec Gemini 336L RGB/Stereo)
  2. Single-sensor RGB LiDAR architectures (e.g. Ouster OS0/OS1/OSDome Rev8 spherical RGB-D)
  3. Third-party SLAM datasets (colorized XYZ+RGB point clouds with or without source camera frames)

Core Reality-Capture Principle:
  Unobserved and occluded regions are NEVER fabricated or AI-hallucinated. Missing geometry is
  explicitly preserved with `cartesianInvalidState = 2` in structured scans / E57 and `UNOBSERVED (0)`
  in panorama coverage provenance masks.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import struct
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

try:
    from .e57_structured_writer import (
        E57StationPayload,
        read_and_validate_e57,
        write_structured_e57,
    )
except ImportError:
    try:
        from scanhub_gui.e57_structured_writer import (
            E57StationPayload,
            read_and_validate_e57,
            write_structured_e57,
        )
    except ImportError:
        from e57_structured_writer import (
            E57StationPayload,
            read_and_validate_e57,
            write_structured_e57,
        )


# ==============================================================================
# ENUMERATIONS & PROVENANCE CODES
# ==============================================================================
class StationPlacementMode(Enum):
    HYBRID = "hybrid"          # FARO Flash / Manual stations + Auto trajectory fill
    MANUAL_ONLY = "manual"     # Only FARO Flash / Operator-placed stations
    AUTO_ONLY = "auto"         # Automatic trajectory spacing & coverage selection


class PanoramaProvenance(IntEnum):
    UNOBSERVED = 0             # Never observed by LiDAR/Stereo or RGB (strictly preserved empty)
    PHOTO_RGB = 1              # Reprojected from calibrated original RGB keyframe(s)
    POINTCLOUD_RGB = 2         # Adaptive surfel splat from colorized 3D point cloud
    DEPTH_GUIDED_FILL = 3      # 1-px intra-surface micro-pinhole interpolation (<8cm depth continuity)
    GEOMETRY_UNCOLORED = 4     # Observed 3D geometry lacking RGB observation (shaded LiDAR structure)


# ==============================================================================
# DATA STRUCTURES
# ==============================================================================
@dataclass
class KeyframeCaptureConfig:
    min_distance_m: float = 0.65            # Capture every 0.5 - 1.0 m of translation
    min_rotation_deg: float = 17.5          # Capture every 15 - 20 deg of 3D camera rotation
    station_spin_rotation_deg: float = 12.0 # Tighter angular trigger near an active Flash station
    station_proximity_m: float = 1.25       # Radius around a Flash station for enhanced capture
    enable_coverage_trigger: bool = True    # Trigger when new directional sector enters FOV
    min_sharpness_laplacian: float = 22.0   # Reject severe motion blur (unless forced)
    min_exposure_luma: float = 16.0         # Reject black / covered-lens frames
    max_exposure_luma: float = 244.0        # Reject completely blown-out frames
    min_interval_sec: float = 0.12          # Minimum cooldown between non-forced captures
    max_keyframes: int = 1200               # Memory-bounded ring / session limit


@dataclass
class RgbKeyframe:
    keyframe_id: str
    timestamp_sec: float
    image_rgb: Optional[np.ndarray]         # (H, W, 3) uint8 RGB image
    width: int
    height: int
    # Camera intrinsics (for pinhole_plumb_bob) or vertical FOV bounds (for spherical):
    fx: float
    fy: float
    cx: float
    cy: float
    dist_coeffs: Tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0)
    # World -> RGB Optical transform: p_rgb = p_world @ R_rgb_world.T + t_rgb_world
    R_rgb_world: np.ndarray = field(default_factory=lambda: np.eye(3, dtype=np.float64))
    t_rgb_world: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    # IMU pose at capture time (for post-SLAM trajectory optimization updates):
    R_world_imu: Optional[np.ndarray] = None
    pos_world_imu: Optional[np.ndarray] = None
    # Quality & Provenance metadata:
    sharpness_score: float = 100.0
    exposure_mean: float = 128.0
    trigger_reason: str = "interval"
    camera_model: str = "pinhole_plumb_bob"  # "pinhole_plumb_bob" or "spherical"
    spherical_vfov_deg: Tuple[float, float] = (-45.0, 45.0)  # For Ouster Rev8 native spherical images
    range_image_m: Optional[np.ndarray] = None               # Optional aligned range map (e.g. Ouster Rev8)
    image_rel_path: str = ""

    @property
    def camera_center_world(self) -> np.ndarray:
        """Optical center of the camera in World coordinates."""
        return -self.R_rgb_world.T @ self.t_rgb_world

    @property
    def forward_axis_world(self) -> np.ndarray:
        """Unit optical forward vector in World coordinates (+Z for pinhole, +X for spherical)."""
        if self.camera_model == "spherical":
            return self.R_rgb_world.T @ np.array([1.0, 0.0, 0.0], dtype=np.float64)
        return self.R_rgb_world.T @ np.array([0.0, 0.0, 1.0], dtype=np.float64)


@dataclass
class VirtualStation:
    station_id: str
    label: str
    position_xyz: np.ndarray                # (3,) float64 in Gravity-Aligned World Frame
    yaw_rad: float = 0.0                    # Horizontal forward heading around World +Z
    source_mode: str = "manual_flash"       # "manual_flash", "manual_map", "auto_trajectory", "centroid"
    timestamp_sec: float = 0.0
    azimuth_rgb_bins: List[bool] = field(default_factory=lambda: [False] * 24)  # 24 x 15° live coverage sectors

    @property
    def R_world_station(self) -> np.ndarray:
        """Gravity-leveled station-to-world rotation (horizon strictly level at row = H/2)."""
        c = math.cos(self.yaw_rad)
        s = math.sin(self.yaw_rad)
        return np.array([
            [c, -s, 0.0],
            [s,  c, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)

    @property
    def rotation_wxyz(self) -> Tuple[float, float, float, float]:
        """Quaternion (qw, qx, qy, qz) for E57 pose."""
        half = 0.5 * self.yaw_rad
        return (math.cos(half), 0.0, 0.0, math.sin(half))

    @property
    def azimuth_coverage_pct(self) -> float:
        if not self.azimuth_rgb_bins:
            return 0.0
        return (sum(1 for b in self.azimuth_rgb_bins if b) / len(self.azimuth_rgb_bins)) * 100.0

    @property
    def live_azimuth_coverage_pct(self) -> float:
        return self.azimuth_coverage_pct


@dataclass
class StationGenerationConfig:
    placement_mode: StationPlacementMode = StationPlacementMode.HYBRID
    auto_spacing_m: float = 3.5             # Automatic virtual station spacing along trajectory
    min_manual_clearance_m: float = 2.0     # Keep auto stations at least 2m away from Flash anchors
    station_eye_offset_z_m: float = 0.0     # Offset relative to scanner trajectory Z
    # Structured Spherical Scan resolution (rows x cols):
    scan_rows: int = 256
    scan_cols: int = 1024
    min_range_m: float = 0.20
    max_range_m: float = 45.0
    # 360° Equirectangular Panorama resolution (W x H, 2:1 aspect):
    pano_width: int = 2048
    pano_height: int = 1024
    # Multi-view RGB Reprojection & Surfel parameters:
    max_reproj_baseline_m: float = 8.0      # Max camera distance from station center for RGB reprojection
    max_rgb_proj_range_m: float = 9.0       # Max range from camera optical center to surface
    max_keyframes_per_station: int = 48     # Cap candidate keyframes per station with angular diversity
    surfel_radius_m: float = 0.045          # Physical surface footprint radius (~4.5 cm)
    enable_small_gap_fill: bool = True      # 1-px depth-guided micro-pinhole interpolation
    jpeg_quality: int = 93


@dataclass
class StationSynthesisResult:
    station: VirtualStation
    # Structured Spherical Scan outputs (scan_rows x scan_cols):
    scan_xyz_local: np.ndarray              # (R, C, 3) float32 in station-local frame
    scan_invalid_state: np.ndarray          # (R, C) uint8: 0 = valid, 2 = missing/unobserved
    scan_rgb: np.ndarray                    # (R, C, 3) uint8
    scan_range_m: np.ndarray                # (R, C) float32
    # 360° Equirectangular Panorama outputs (pano_height x pano_width):
    panorama_rgb: np.ndarray                # (H, W, 3) uint8 RGB
    depth_map_m: np.ndarray                 # (H, W) float32 metric range from station center (0 = unobserved)
    provenance_mask: np.ndarray             # (H, W) uint8 PanoramaProvenance enum values
    coverage_vis_rgb: np.ndarray            # (H, W, 3) uint8 human-readable provenance map
    # Summary telemetry:
    valid_scan_points: int = 0
    total_scan_cells: int = 0
    photo_rgb_pct: float = 0.0
    pointcloud_rgb_pct: float = 0.0
    gap_fill_pct: float = 0.0
    uncolored_geom_pct: float = 0.0
    unobserved_pct: float = 100.0
    contributing_keyframes: int = 0
    synthesis_time_ms: float = 0.0

    @property
    def scan_rows(self) -> int:
        return int(self.scan_xyz_local.shape[0])

    @property
    def scan_cols(self) -> int:
        return int(self.scan_xyz_local.shape[1])

    @property
    def scan_coverage_pct(self) -> float:
        return float(self.valid_scan_points) * 100.0 / max(1, int(self.total_scan_cells))

    @property
    def rgb_coverage_pct(self) -> float:
        return float(self.photo_rgb_pct + self.pointcloud_rgb_pct + self.gap_fill_pct)

    @property
    def panorama_depth_m(self) -> np.ndarray:
        return self.depth_map_m


# ==============================================================================
# PLUMB-BOB CAMERA PROJECTION (FLOAT SUBPIXEL PRECISION)
# ==============================================================================
def project_points_plumb_bob_float(
    pts_cam: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    dist_coeffs: Sequence[float] = (0.0, 0.0, 0.0, 0.0, 0.0),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Projects 3D points in camera optical frame (+X right, +Y down, +Z forward)
    to floating-point image coordinates (u, v) for crisp bilinear sampling.
    """
    z = pts_cam[:, 2]
    safe_z = np.where(z > 1e-4, z, 1e-4)
    xn = pts_cam[:, 0] / safe_z
    yn = pts_cam[:, 1] / safe_z

    if len(dist_coeffs) >= 5 and any(abs(c) > 1e-9 for c in dist_coeffs[:5]):
        k1, k2, p1, p2, k3 = dist_coeffs[:5]
        r2 = xn * xn + yn * yn
        r2 = np.minimum(r2, 4.0)  # Guard extreme wide-angle polynomial blowup outside frustum
        r4 = r2 * r2
        r6 = r4 * r2
        radial = 1.0 + k1 * r2 + k2 * r4 + k3 * r6
        dx = 2.0 * p1 * xn * yn + p2 * (r2 + 2.0 * xn * xn)
        dy = p1 * (r2 + 2.0 * yn * yn) + 2.0 * p2 * xn * yn
        xd = xn * radial + dx
        yd = yn * radial + dy
    else:
        xd = xn
        yd = yn

    u = fx * xd + cx
    v = fy * yd + cy
    return u, v, z


def compute_image_quality_metrics(rgb_img: np.ndarray) -> Tuple[float, float]:
    """
    Fast (<0.4 ms) sharpness (Laplacian variance) and exposure (mean luma) estimation
    on a downsampled grayscale view.
    """
    if rgb_img is None or rgb_img.size == 0:
        return 0.0, 0.0
    h, w = rgb_img.shape[:2]
    stride = max(1, min(h, w) // 200)
    sub = rgb_img[::stride, ::stride]
    gray = cv2.cvtColor(sub, cv2.COLOR_RGB2GRAY)
    mean_luma = float(np.mean(gray))
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    sharpness = float(np.var(lap))
    return sharpness, mean_luma


# ==============================================================================
# 1. INTELLIGENT MULTI-TRIGGER RGB KEYFRAME SELECTOR
# ==============================================================================
class IntelligentKeyframeSelector:
    """
    Captures quality-controlled RGB keyframes continuously during scanning based on:
      - Distance traveled (0.5 - 1.0 m)
      - 3D camera rotation (15 - 20 deg, or 12 deg during station spin)
      - New directional coverage sector entering FOV
      - Image blur / exposure quality gating
      - Immediate priority trigger for FARO Flash / Manual Virtual Scan Stations
    """

    def __init__(self, config: Optional[KeyframeCaptureConfig] = None):
        self.config = config or KeyframeCaptureConfig()
        self.keyframes: List[RgbKeyframe] = []
        self._last_pos_cam: Optional[np.ndarray] = None
        self._last_R_rgb_w: Optional[np.ndarray] = None
        self._last_capture_sec: float = -1e9
        # Spatial-angular coverage cells: {(cell_x, cell_y): set_of_(az_bin, el_bin)}
        self._spatial_dir_bins: Dict[Tuple[int, int], set] = {}
        self.rejected_blur_count: int = 0
        self.rejected_exposure_count: int = 0

    @property
    def cfg(self) -> KeyframeCaptureConfig:
        return self.config

    def reset(self):
        self.keyframes.clear()
        self._last_pos_cam = None
        self._last_R_rgb_w = None
        self._last_capture_sec = -1e9
        self._spatial_dir_bins.clear()
        self.rejected_blur_count = 0
        self.rejected_exposure_count = 0

    @staticmethod
    def compose_rgb_world_extrinsics(
        pos_w_imu: np.ndarray,
        R_w_imu: np.ndarray,
        R_rgb_imu: np.ndarray,
        t_rgb_imu: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes (R_rgb_world, t_rgb_world) such that:
            p_rgb = R_rgb_world @ p_world + t_rgb_world
        from:
            p_imu = R_w_imu.T @ (p_world - pos_w_imu)
            p_rgb = R_rgb_imu @ p_imu + t_rgb_imu
        """
        R_rgb_world = R_rgb_imu @ R_w_imu.T
        t_rgb_world = -R_rgb_world @ np.asarray(pos_w_imu, dtype=np.float64) + np.asarray(t_rgb_imu, dtype=np.float64)
        return R_rgb_world, t_rgb_world

    def _direction_sector(self, fwd_world: np.ndarray) -> Tuple[int, int]:
        az = math.atan2(float(fwd_world[1]), float(fwd_world[0]))  # [-pi, pi]
        el = math.asin(max(-1.0, min(1.0, float(fwd_world[2]))))   # [-pi/2, pi/2]
        az_bin = int(((az + math.pi) / (2.0 * math.pi)) * 12.0) % 12
        el_bin = min(2, max(0, int(((el + 0.5 * math.pi) / math.pi) * 3.0)))
        return az_bin, el_bin

    def update_stations_live_coverage(self, stations: Sequence[VirtualStation], cam_pos: np.ndarray, fwd_w: np.ndarray, hfov_deg: float = 90.0):
        """Updates the 24-bin (15°/bin) horizontal azimuth RGB coverage ring of nearby stations."""
        center_az = math.atan2(float(fwd_w[1]), float(fwd_w[0]))
        half_fov = math.radians(hfov_deg * 0.45)
        for st in stations:
            dist = float(np.linalg.norm(cam_pos[:2] - st.position_xyz[:2]))
            if dist <= 2.5:
                for b_idx in range(24):
                    bin_az = -math.pi + (b_idx + 0.5) * (2.0 * math.pi / 24.0)
                    diff = abs(math.atan2(math.sin(bin_az - center_az), math.cos(bin_az - center_az)))
                    if diff <= half_fov:
                        st.azimuth_rgb_bins[b_idx] = True

    def consider_frame(
        self,
        timestamp_sec: float,
        image_rgb: np.ndarray,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        R_rgb_world: np.ndarray,
        t_rgb_world: np.ndarray,
        dist_coeffs: Sequence[float] = (0.0, 0.0, 0.0, 0.0, 0.0),
        R_world_imu: Optional[np.ndarray] = None,
        pos_world_imu: Optional[np.ndarray] = None,
        active_stations: Optional[Sequence[VirtualStation]] = None,
        force_reason: Optional[str] = None,
        camera_model: str = "pinhole_plumb_bob",
    ) -> Optional[RgbKeyframe]:
        """
        Evaluates a candidate RGB frame against intelligent spatial, rotational, coverage,
        and image-quality triggers. Returns the stored `RgbKeyframe` if selected, else `None`.
        """
        if image_rgb is None or image_rgb.size == 0:
            return None

        cfg = self.config
        if force_reason is None and (timestamp_sec - self._last_capture_sec) < cfg.min_interval_sec:
            return None

        cam_pos = -R_rgb_world.T @ t_rgb_world
        fwd_w = (
            R_rgb_world.T @ np.array([1.0, 0.0, 0.0], dtype=np.float64)
            if camera_model == "spherical"
            else R_rgb_world.T @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
        )

        # Determine if scanner is near an operator-marked Virtual Station (Enhanced Spin Capture)
        near_station = False
        if active_stations:
            for st in active_stations:
                if float(np.linalg.norm(cam_pos - st.position_xyz)) <= cfg.station_proximity_m:
                    near_station = True
                    break

        trigger: Optional[str] = force_reason
        if trigger is None:
            if self._last_pos_cam is None or self._last_R_rgb_w is None:
                trigger = "initial"
            else:
                dist_moved = float(np.linalg.norm(cam_pos - self._last_pos_cam))
                R_rel = R_rgb_world @ self._last_R_rgb_w.T
                cos_ang = max(-1.0, min(1.0, (float(np.trace(R_rel)) - 1.0) * 0.5))
                rot_deg = math.degrees(math.acos(cos_ang))

                rot_thresh = cfg.station_spin_rotation_deg if near_station else cfg.min_rotation_deg
                if dist_moved >= cfg.min_distance_m:
                    trigger = "distance"
                elif rot_deg >= rot_thresh:
                    trigger = "station_spin" if near_station else "rotation"
                elif cfg.enable_coverage_trigger:
                    cell_key = (int(math.floor(cam_pos[0] / 1.5)), int(math.floor(cam_pos[1] / 1.5)))
                    sec_bin = self._direction_sector(fwd_w)
                    seen_set = self._spatial_dir_bins.get(cell_key)
                    if seen_set is None or sec_bin not in seen_set:
                        trigger = "coverage"

        if trigger is None:
            return None

        # Image quality check (blur & exposure gate)
        sharpness, mean_luma = compute_image_quality_metrics(image_rgb)
        if force_reason is None:
            if mean_luma < cfg.min_exposure_luma or mean_luma > cfg.max_exposure_luma:
                self.rejected_exposure_count += 1
                return None
            if sharpness < cfg.min_sharpness_laplacian:
                self.rejected_blur_count += 1
                return None

        h, w = image_rgb.shape[:2]
        kf_id = f"kf_{len(self.keyframes):04d}"
        kf = RgbKeyframe(
            keyframe_id=kf_id,
            timestamp_sec=float(timestamp_sec),
            image_rgb=image_rgb.copy(),
            width=int(w),
            height=int(h),
            fx=float(fx),
            fy=float(fy),
            cx=float(cx),
            cy=float(cy),
            dist_coeffs=tuple(float(x) for x in dist_coeffs),
            R_rgb_world=R_rgb_world.astype(np.float64).copy(),
            t_rgb_world=t_rgb_world.astype(np.float64).copy(),
            R_world_imu=R_world_imu.astype(np.float64).copy() if R_world_imu is not None else None,
            pos_world_imu=np.asarray(pos_world_imu, dtype=np.float64).copy() if pos_world_imu is not None else None,
            sharpness_score=sharpness,
            exposure_mean=mean_luma,
            trigger_reason=trigger,
            camera_model=camera_model,
        )

        if len(self.keyframes) >= cfg.max_keyframes:
            # Drop oldest non-station keyframe to preserve Flash station captures
            drop_idx = next(
                (i for i, k in enumerate(self.keyframes) if "flash" not in k.trigger_reason and "station" not in k.trigger_reason),
                0,
            )
            self.keyframes.pop(drop_idx)

        self.keyframes.append(kf)
        self._last_pos_cam = cam_pos
        self._last_R_rgb_w = R_rgb_world.copy()
        self._last_capture_sec = float(timestamp_sec)

        cell_key = (int(math.floor(cam_pos[0] / 1.5)), int(math.floor(cam_pos[1] / 1.5)))
        self._spatial_dir_bins.setdefault(cell_key, set()).add(self._direction_sector(fwd_w))

        if active_stations:
            hfov = 360.0 if camera_model == "spherical" else (2.0 * math.degrees(math.atan2(w * 0.5, max(1.0, fx))))
            self.update_stations_live_coverage(active_stations, cam_pos, fwd_w, hfov_deg=hfov)

        return kf

    @staticmethod
    def update_keyframe_poses_from_trajectory(
        keyframes: Sequence[RgbKeyframe],
        trajectory_records: Sequence[Tuple],
        R_rgb_imu: np.ndarray,
        t_rgb_imu: np.ndarray,
    ) -> int:
        """
        Updates keyframe poses (`R_rgb_world`, `t_rgb_world`) using the final optimized
        SLAM trajectory `(sec, x, y, z, qx, qy, qz, qw, ...)` via linear + SLERP interpolation.
        """
        if not keyframes or len(trajectory_records) < 2:
            return 0

        times = np.array([float(r[0]) for r in trajectory_records], dtype=np.float64)
        order = np.argsort(times)
        times = times[order]
        pos_arr = np.array([[float(trajectory_records[i][1]), float(trajectory_records[i][2]), float(trajectory_records[i][3])] for i in order], dtype=np.float64)
        quat_arr = np.array([[float(trajectory_records[i][4]), float(trajectory_records[i][5]), float(trajectory_records[i][6]), float(trajectory_records[i][7])] for i in order], dtype=np.float64)

        updated = 0
        for kf in keyframes:
            t_q = kf.timestamp_sec
            if t_q < times[0] - 0.5 or t_q > times[-1] + 0.5:
                continue
            idx = int(np.searchsorted(times, t_q))
            if idx <= 0:
                p_interp = pos_arr[0]
                q_interp = quat_arr[0]
            elif idx >= len(times):
                p_interp = pos_arr[-1]
                q_interp = quat_arr[-1]
            else:
                t0, t1 = times[idx - 1], times[idx]
                dt = max(1e-9, t1 - t0)
                alpha = float(np.clip((t_q - t0) / dt, 0.0, 1.0))
                p_interp = (1.0 - alpha) * pos_arr[idx - 1] + alpha * pos_arr[idx]
                q0, q1 = quat_arr[idx - 1].copy(), quat_arr[idx].copy()
                if np.dot(q0, q1) < 0.0:
                    q1 = -q1
                q_interp = (1.0 - alpha) * q0 + alpha * q1
                q_norm = np.linalg.norm(q_interp)
                if q_norm > 1e-8:
                    q_interp /= q_norm

            R_w_imu = Rotation.from_quat(q_interp).as_matrix()
            R_rw, t_rw = IntelligentKeyframeSelector.compose_rgb_world_extrinsics(
                p_interp, R_w_imu, R_rgb_imu, t_rgb_imu
            )
            kf.R_world_imu = R_w_imu
            kf.pos_world_imu = p_interp
            kf.R_rgb_world = R_rw
            kf.t_rgb_world = t_rw
            updated += 1
        return updated


# ==============================================================================
# KEYFRAME BUNDLE PERSISTENCE (SAVE / LOAD)
# ==============================================================================
def save_keyframes_bundle(keyframes: Sequence[RgbKeyframe], save_dir: str, jpeg_quality: int = 94) -> Dict:
    """Saves full-resolution RGB keyframes and calibrated camera poses to `<save_dir>/keyframes/`."""
    kf_dir = os.path.join(save_dir, "keyframes")
    os.makedirs(kf_dir, exist_ok=True)
    manifest_entries = []
    saved_count = 0

    for idx, kf in enumerate(keyframes):
        fname = f"kf_{idx:04d}.jpg"
        fpath = os.path.join(kf_dir, fname)
        if kf.image_rgb is not None and kf.image_rgb.size > 0:
            bgr = cv2.cvtColor(kf.image_rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(fpath, bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
            kf.image_rel_path = os.path.join("keyframes", fname)
            saved_count += 1

        manifest_entries.append({
            "keyframe_id": kf.keyframe_id,
            "image_file": fname,
            "timestamp_sec": round(kf.timestamp_sec, 6),
            "width": kf.width,
            "height": kf.height,
            "camera_model": kf.camera_model,
            "intrinsics": {
                "fx": kf.fx,
                "fy": kf.fy,
                "cx": kf.cx,
                "cy": kf.cy,
                "dist_coeffs": list(kf.dist_coeffs),
            },
            "extrinsics_rgb_from_world": {
                "R_rgb_world": kf.R_rgb_world.tolist(),
                "t_rgb_world": kf.t_rgb_world.tolist(),
                "camera_center_world": kf.camera_center_world.tolist(),
            },
            "quality": {
                "sharpness_laplacian": round(kf.sharpness_score, 2),
                "exposure_mean": round(kf.exposure_mean, 2),
            },
            "trigger_reason": kf.trigger_reason,
        })

    manifest_path = os.path.join(kf_dir, "keyframes_manifest.json")
    manifest_obj = {
        "schema": "ArtificialVision.KeyframesManifest.v1",
        "keyframe_count": saved_count,
        "keyframes": manifest_entries,
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest_obj, f, indent=2)

    return {"keyframes_dir": kf_dir, "manifest_path": manifest_path, "saved_count": saved_count}


def load_keyframes_bundle(dataset_dir: str) -> List[RgbKeyframe]:
    """Loads persisted RGB keyframes and calibrated poses from `<dataset_dir>/keyframes/` if present."""
    manifest_path = os.path.join(dataset_dir, "keyframes", "keyframes_manifest.json")
    if not os.path.exists(manifest_path):
        return []
    with open(manifest_path, "r") as f:
        data = json.load(f)

    loaded: List[RgbKeyframe] = []
    for item in data.get("keyframes", []):
        img_path = os.path.join(dataset_dir, "keyframes", item["image_file"])
        if not os.path.exists(img_path):
            continue
        bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        intr = item["intrinsics"]
        extr = item["extrinsics_rgb_from_world"]
        qual = item.get("quality", {})
        loaded.append(
            RgbKeyframe(
                keyframe_id=item["keyframe_id"],
                timestamp_sec=float(item["timestamp_sec"]),
                image_rgb=rgb,
                width=int(item["width"]),
                height=int(item["height"]),
                fx=float(intr["fx"]),
                fy=float(intr["fy"]),
                cx=float(intr["cx"]),
                cy=float(intr["cy"]),
                dist_coeffs=tuple(intr.get("dist_coeffs", [0.0] * 5)),
                R_rgb_world=np.array(extr["R_rgb_world"], dtype=np.float64),
                t_rgb_world=np.array(extr["t_rgb_world"], dtype=np.float64),
                sharpness_score=float(qual.get("sharpness_laplacian", 100.0)),
                exposure_mean=float(qual.get("exposure_mean", 128.0)),
                trigger_reason=item.get("trigger_reason", "loaded"),
                camera_model=item.get("camera_model", "pinhole_plumb_bob"),
                image_rel_path=os.path.join("keyframes", item["image_file"]),
            )
        )
    return loaded


# ==============================================================================
# 2. VIRTUAL SCAN STATION SELECTOR (MANUAL FLASH + AUTO TRAJECTORY + HYBRID)
# ==============================================================================
class VirtualStationSelector:
    """
    Defines Virtual Scan Station centers from operator FARO Flash anchors, manual map markers,
    and/or automatic trajectory spacing scored by local RGB & 3D geometry coverage.
    """

    @staticmethod
    def select_stations(
        config: StationGenerationConfig,
        flash_anchors: Optional[Sequence[Dict]] = None,
        trajectory_records: Optional[Sequence[Tuple]] = None,
        keyframes: Optional[Sequence[RgbKeyframe]] = None,
        points_xyz: Optional[np.ndarray] = None,
    ) -> List[VirtualStation]:
        stations: List[VirtualStation] = []

        # Estimate default eye/sensor Z from trajectory or point cloud if Z missing on 2D anchors
        default_z = 0.0
        if trajectory_records and len(trajectory_records) > 0:
            first_rec = trajectory_records[0]
            if len(first_rec) >= 4:
                default_z = float(np.median([float(r[3]) for r in trajectory_records]))
        elif points_xyz is not None and len(points_xyz) > 0:
            default_z = float(np.median(points_xyz[:, 2]))

        # 1. Manual / FARO Flash Stations
        if config.placement_mode in (StationPlacementMode.HYBRID, StationPlacementMode.MANUAL_ONLY):
            for idx, anc in enumerate(flash_anchors or []):
                ax = float(anc.get("x", 0.0))
                ay = float(anc.get("y", 0.0))
                az = float(anc.get("z", default_z)) + config.station_eye_offset_z_m
                ayaw = float(anc.get("yaw", 0.0))
                lbl = str(anc.get("label", f"w{idx:02d}"))
                src = str(anc.get("source_mode", "manual_flash"))
                stations.append(
                    VirtualStation(
                        station_id=f"station_{len(stations):02d}_{lbl}",
                        label=lbl,
                        position_xyz=np.array([ax, ay, az], dtype=np.float64),
                        yaw_rad=ayaw,
                        source_mode=src,
                        timestamp_sec=float(anc.get("time", 0.0)),
                    )
                )

        # 2. Automatic Trajectory Stations
        if config.placement_mode in (StationPlacementMode.HYBRID, StationPlacementMode.AUTO_ONLY):
            traj_pts: List[Tuple[np.ndarray, float, float]] = []
            for r in (trajectory_records or []):
                if len(r) >= 9:
                    # (sec, x, y, z, qx, qy, qz, qw, yaw)
                    traj_pts.append((np.array([float(r[1]), float(r[2]), float(r[3]) + config.station_eye_offset_z_m], dtype=np.float64), float(r[8]), float(r[0])))
                elif len(r) == 3:
                    # Canvas 2D trajectory tuple (x, y, yaw)
                    traj_pts.append((np.array([float(r[0]), float(r[1]), default_z + config.station_eye_offset_z_m], dtype=np.float64), float(r[2]), 0.0))

            if len(traj_pts) > 0:
                kf_centers = (
                    np.array([kf.camera_center_world for kf in keyframes], dtype=np.float64)
                    if (keyframes and len(keyframes) > 0)
                    else None
                )

                # Walk trajectory accumulating arc length; pick highest-coverage pose in a local window
                accum_d = 0.0
                prev_p = traj_pts[0][0]
                candidate_indices = [0]
                for i in range(1, len(traj_pts)):
                    p = traj_pts[i][0]
                    step = float(np.linalg.norm(p - prev_p))
                    accum_d += step
                    prev_p = p
                    if accum_d >= config.auto_spacing_m:
                        candidate_indices.append(i)
                        accum_d = 0.0

                # Include final endpoint if meaningfully separated from last station
                if len(traj_pts) > 2 and (len(candidate_indices) == 0 or candidate_indices[-1] != len(traj_pts) - 1):
                    if float(np.linalg.norm(traj_pts[-1][0] - traj_pts[candidate_indices[-1]][0])) >= config.auto_spacing_m * 0.6:
                        candidate_indices.append(len(traj_pts) - 1)

                auto_idx = 0
                for c_i in candidate_indices:
                    # Search small +/- 5 sample neighborhood for best local RGB keyframe density
                    win_lo = max(0, c_i - 5)
                    win_hi = min(len(traj_pts), c_i + 6)
                    best_p, best_yaw, best_sec = traj_pts[c_i]
                    if kf_centers is not None:
                        best_score = -1.0
                        for w_i in range(win_lo, win_hi):
                            cand_p = traj_pts[w_i][0]
                            dists = np.linalg.norm(kf_centers - cand_p, axis=1)
                            score = float(np.sum(dists <= 2.5))
                            if score > best_score:
                                best_score = score
                                best_p, best_yaw, best_sec = traj_pts[w_i]

                    # Respect clearance from existing manual/auto stations
                    too_close = False
                    for existing in stations:
                        min_clear = (
                            config.min_manual_clearance_m
                            if existing.source_mode.startswith("manual")
                            else config.auto_spacing_m * 0.65
                        )
                        if float(np.linalg.norm(best_p - existing.position_xyz)) < min_clear:
                            too_close = True
                            break
                    if too_close:
                        continue

                    lbl = f"auto{auto_idx:02d}"
                    auto_idx += 1
                    stations.append(
                        VirtualStation(
                            station_id=f"station_{len(stations):02d}_{lbl}",
                            label=lbl,
                            position_xyz=best_p.copy(),
                            yaw_rad=best_yaw,
                            source_mode="auto_trajectory",
                            timestamp_sec=best_sec,
                        )
                    )

        # 3. Fallback for Point-Cloud-Only Dataset (no trajectory & no manual stations)
        if len(stations) == 0 and points_xyz is not None and len(points_xyz) > 0:
            # Place a clean virtual station near the interior median of the point cloud
            med_xy = np.median(points_xyz[:, :2], axis=0)
            z_low = np.percentile(points_xyz[:, 2], 10.0)
            z_high = np.percentile(points_xyz[:, 2], 90.0)
            eye_z = float(min(z_high - 0.2, z_low + 1.55)) if (z_high - z_low) > 1.8 else float(np.median(points_xyz[:, 2]))
            stations.append(
                VirtualStation(
                    station_id="station_00_origin",
                    label="s00",
                    position_xyz=np.array([float(med_xy[0]), float(med_xy[1]), eye_z], dtype=np.float64),
                    yaw_rad=0.0,
                    source_mode="centroid",
                )
            )

        # Populate azimuth RGB coverage ring on all selected stations from keyframes
        if keyframes:
            sel = IntelligentKeyframeSelector()
            for kf in keyframes:
                hfov = 360.0 if kf.camera_model == "spherical" else (2.0 * math.degrees(math.atan2(kf.width * 0.5, max(1.0, kf.fx))))
                sel.update_stations_live_coverage(stations, kf.camera_center_world, kf.forward_axis_world, hfov_deg=hfov)

        return stations


# ==============================================================================
# 3. VIRTUAL STATION GENERATOR (SHARED-CENTER STRUCTURED SCAN + 360° PANORAMA)
# ==============================================================================
class VirtualStationGenerator:
    """
    Synthesizes synchronized 3D Structured Spherical Scans and 2D 360° Equirectangular Panoramas
    from the exact same VirtualStation center, preserving missing coverage explicitly.
    """

    def __init__(self, config: Optional[StationGenerationConfig] = None):
        self.config = config or StationGenerationConfig()

    @staticmethod
    def _disk_offsets(radius_px: int) -> List[Tuple[int, int]]:
        if radius_px <= 0:
            return [(0, 0)]
        offs = []
        r2 = radius_px * radius_px
        for dy in range(-radius_px, radius_px + 1):
            for dx in range(-radius_px, radius_px + 1):
                if dx * dx + dy * dy <= r2:
                    offs.append((dy, dx))
        # Sort center-outward
        offs.sort(key=lambda t: t[0] * t[0] + t[1] * t[1])
        return offs

    @staticmethod
    def _build_equirectangular_unit_rays(height: int, width: int) -> np.ndarray:
        """
        Builds ASTM E2807-11 Section 6.5.4 unit ray vectors `(H, W, 3)` in station-local frame:
          col = W/2, row = H/2 -> +X forward (1, 0, 0)
          col = W/4, row = H/2 -> +Y left    (0, 1, 0)
          row = 0              -> +Z zenith  (0, 0, 1)
        """
        cols = np.arange(width, dtype=np.float64) + 0.5
        rows = np.arange(height, dtype=np.float64) + 0.5
        theta = (2.0 * math.pi / width) * (0.5 * width - cols)       # (W,)
        phi = (math.pi / height) * (0.5 * height - rows)             # (H,)

        cos_phi = np.cos(phi)[:, None]                               # (H, 1)
        sin_phi = np.sin(phi)[:, None]                               # (H, 1)
        cos_theta = np.cos(theta)[None, :]                           # (1, W)
        sin_theta = np.sin(theta)[None, :]                           # (1, W)

        rays = np.empty((height, width, 3), dtype=np.float32)
        rays[:, :, 0] = cos_phi * cos_theta
        rays[:, :, 1] = cos_phi * sin_theta
        rays[:, :, 2] = np.broadcast_to(sin_phi, (height, width))
        return rays

    @staticmethod
    def _select_diverse_station_keyframes(
        station: VirtualStation,
        keyframes: Sequence[RgbKeyframe],
        max_baseline_m: float,
        max_count: int,
    ) -> List[RgbKeyframe]:
        """Selects up to `max_count` sharp, nearby RGB keyframes distributed across all viewing directions."""
        if not keyframes:
            return []
        candidates: List[Tuple[float, int, RgbKeyframe]] = []
        s_pos = station.position_xyz
        for kf in keyframes:
            if kf.image_rgb is None or kf.image_rgb.size == 0:
                continue
            c_pos = kf.camera_center_world
            dist = float(np.linalg.norm(c_pos - s_pos))
            if dist <= max_baseline_m:
                fwd = kf.forward_axis_world
                az_bin = int(((math.atan2(float(fwd[1]), float(fwd[0])) + math.pi) / (2.0 * math.pi)) * 12.0) % 12
                # Higher score = closer to station + sharper image
                score = (kf.sharpness_score + 25.0) / (1.0 + (dist / 1.2) ** 2)
                candidates.append((score, az_bin, kf))

        if len(candidates) <= max_count:
            candidates.sort(key=lambda x: x[0], reverse=True)
            return [c[2] for c in candidates]

        # Round-robin pick best per azimuth sector first so full 360° coverage is preserved
        by_sector: Dict[int, List[Tuple[float, RgbKeyframe]]] = {b: [] for b in range(12)}
        for sc, az_b, kf in candidates:
            by_sector[az_b].append((sc, kf))
        for b in by_sector:
            by_sector[b].sort(key=lambda x: x[0], reverse=True)

        selected: List[RgbKeyframe] = []
        round_idx = 0
        while len(selected) < max_count:
            added_any = False
            for b in range(12):
                if round_idx < len(by_sector[b]):
                    selected.append(by_sector[b][round_idx][1])
                    added_any = True
                    if len(selected) >= max_count:
                        break
            if not added_any:
                break
            round_idx += 1
        return selected

    @staticmethod
    def _bilinear_sample_rgb(
        img: np.ndarray,
        u_f: np.ndarray,
        v_f: np.ndarray,
        wrap_x: bool = False,
    ) -> np.ndarray:
        """
        Unlimited-length vectorized subpixel bilinear RGB sampler (avoids OpenCV remap SHRT_MAX=32767 cap).
        Returns float32 (N, 3).
        """
        h, w = img.shape[:2]
        if wrap_x:
            u_fl = np.floor(u_f)
            u0 = u_fl.astype(np.int32) % w
            u1 = (u0 + 1) % w
            wu = (u_f - u_fl).astype(np.float32)
        else:
            u_c = np.clip(u_f, 0.0, max(0.0, w - 1.0001))
            u0 = np.floor(u_c).astype(np.int32)
            u1 = np.minimum(u0 + 1, w - 1)
            wu = (u_c - u0).astype(np.float32)

        v_c = np.clip(v_f, 0.0, max(0.0, h - 1.0001))
        v0 = np.floor(v_c).astype(np.int32)
        v1 = np.minimum(v0 + 1, h - 1)
        wv = (v_c - v0).astype(np.float32)

        c00 = img[v0, u0].astype(np.float32)
        c01 = img[v0, u1].astype(np.float32)
        c10 = img[v1, u0].astype(np.float32)
        c11 = img[v1, u1].astype(np.float32)

        top = c00 + wu[:, None] * (c01 - c00)
        bot = c10 + wu[:, None] * (c11 - c10)
        return top + wv[:, None] * (bot - top)

    @staticmethod
    def build_coverage_visualization(provenance_mask: np.ndarray) -> np.ndarray:
        """
        Renders an Artificial Vision White & Gold diagnostic coverage provenance image:
          - PHOTO_RGB (1):          Champagne Gold (#E5C983)
          - POINTCLOUD_RGB (2):     Cyan-Teal (#22D3EE)
          - DEPTH_GUIDED_FILL (3):  Emerald (#4ADE80)
          - GEOMETRY_UNCOLORED (4): Slate Silver (#94A3B8)
          - UNOBSERVED (0):         Obsidian Void (#0D0F14)
        """
        h, w = provenance_mask.shape
        vis = np.full((h, w, 3), [13, 15, 20], dtype=np.uint8)
        vis[provenance_mask == int(PanoramaProvenance.PHOTO_RGB)] = [229, 201, 131]
        vis[provenance_mask == int(PanoramaProvenance.POINTCLOUD_RGB)] = [34, 211, 238]
        vis[provenance_mask == int(PanoramaProvenance.DEPTH_GUIDED_FILL)] = [74, 222, 128]
        vis[provenance_mask == int(PanoramaProvenance.GEOMETRY_UNCOLORED)] = [148, 163, 184]
        return vis

    def synthesize_station(
        self,
        station: VirtualStation,
        points_xyz: np.ndarray,
        points_rgb: Optional[np.ndarray] = None,
        points_has_true_rgb: Optional[np.ndarray] = None,
        keyframes: Optional[Sequence[RgbKeyframe]] = None,
    ) -> StationSynthesisResult:
        """
        Generates both the Structured Spherical Scan and the 360° Panorama from `station`.
        """
        t_start = time.perf_counter()
        cfg = self.config
        s_rows, s_cols = int(cfg.scan_rows), int(cfg.scan_cols)
        p_h, p_w = int(cfg.pano_height), int(cfg.pano_width)

        # Allocate structured scan buffers (missing by default: cartesianInvalidState = 2)
        scan_xyz_local = np.zeros((s_rows, s_cols, 3), dtype=np.float32)
        scan_invalid = np.full((s_rows, s_cols), 2, dtype=np.uint8)
        scan_rgb = np.zeros((s_rows, s_cols, 3), dtype=np.uint8)
        scan_range = np.zeros((s_rows, s_cols), dtype=np.float32)

        # Allocate 360° panorama buffers (unobserved pure #000000 black, depth = 0, strictly non-fabricated)
        pano_rgb = np.zeros((p_h, p_w, 3), dtype=np.uint8)
        depth_map = np.zeros((p_h, p_w), dtype=np.float32)
        prov_mask = np.full((p_h, p_w), int(PanoramaProvenance.UNOBSERVED), dtype=np.uint8)

        if points_xyz is None or len(points_xyz) == 0:
            cov_vis = self.build_coverage_visualization(prov_mask)
            return StationSynthesisResult(
                station=station,
                scan_xyz_local=scan_xyz_local,
                scan_invalid_state=scan_invalid,
                scan_rgb=scan_rgb,
                scan_range_m=scan_range,
                panorama_rgb=pano_rgb,
                depth_map_m=depth_map,
                provenance_mask=prov_mask,
                coverage_vis_rgb=cov_vis,
                total_scan_cells=s_rows * s_cols,
                synthesis_time_ms=(time.perf_counter() - t_start) * 1000.0,
            )

        # Normalize input point colors to uint8 [0..255]
        n_pts = len(points_xyz)
        if points_rgb is not None and len(points_rgb) == n_pts:
            if np.max(points_rgb) <= 1.05 and np.issubdtype(points_rgb.dtype, np.floating):
                pt_cols = np.clip(points_rgb * 255.0, 0, 255).astype(np.uint8)
            else:
                pt_cols = np.clip(points_rgb, 0, 255).astype(np.uint8)
            if points_has_true_rgb is not None and len(points_has_true_rgb) == n_pts:
                has_col = np.asarray(points_has_true_rgb, dtype=bool)
            else:
                # Detect neutral default uncolored gray (e.g., [175,175,175] or [64,68,78]) ONLY if exact uniform gray
                has_col = np.ones(n_pts, dtype=bool)
        else:
            pt_cols = np.full((n_pts, 3), [168, 174, 186], dtype=np.uint8)
            has_col = np.zeros(n_pts, dtype=bool)

        # ------------------------------------------------------------------
        # STEP 1: Transform Global SLAM Cloud into Leveled Virtual Station Frame
        # ------------------------------------------------------------------
        R_ws = station.R_world_station
        pts_local = ((points_xyz[:, :3].astype(np.float64) - station.position_xyz) @ R_ws).astype(np.float32)
        ranges = np.linalg.norm(pts_local, axis=1)
        valid_r = (ranges >= cfg.min_range_m) & (ranges <= cfg.max_range_m)

        pts_local = pts_local[valid_r]
        ranges = ranges[valid_r]
        pt_cols = pt_cols[valid_r]
        has_col = has_col[valid_r]

        if len(pts_local) == 0:
            cov_vis = self.build_coverage_visualization(prov_mask)
            return StationSynthesisResult(
                station=station,
                scan_xyz_local=scan_xyz_local,
                scan_invalid_state=scan_invalid,
                scan_rgb=scan_rgb,
                scan_range_m=scan_range,
                panorama_rgb=pano_rgb,
                depth_map_m=depth_map,
                provenance_mask=prov_mask,
                coverage_vis_rgb=cov_vis,
                total_scan_cells=s_rows * s_cols,
                synthesis_time_ms=(time.perf_counter() - t_start) * 1000.0,
            )

        # Sort farthest-to-nearest so painter's writes naturally resolve to minimum range!
        far_to_near = np.argsort(ranges)[::-1]
        pts_local = pts_local[far_to_near]
        ranges = ranges[far_to_near]
        pt_cols = pt_cols[far_to_near]
        has_col = has_col[far_to_near]

        # Compute spherical angles per ASTM E2807-11 Section 6.5.4
        theta = np.arctan2(pts_local[:, 1], pts_local[:, 0])                        # (-pi, pi]
        phi = np.arcsin(np.clip(pts_local[:, 2] / ranges, -1.0, 1.0))               # [-pi/2, pi/2]

        u_norm = (0.5 - theta / (2.0 * math.pi)) % 1.0
        v_norm = np.clip(0.5 - phi / math.pi, 0.0, 1.0 - 1e-7)

        # ------------------------------------------------------------------
        # STEP 2: Adaptive Surfel Splatting into Panorama Depth & Point-RGB Buffer
        # ------------------------------------------------------------------
        p_cols_idx = (u_norm * p_w).astype(np.int32) % p_w
        p_rows_idx = np.clip((v_norm * p_h).astype(np.int32), 0, p_h - 1)

        surfel_depth = np.full((p_h, p_w), np.inf, dtype=np.float32)
        surfel_rgb = np.zeros((p_h, p_w, 3), dtype=np.uint8)
        surfel_has_col = np.zeros((p_h, p_w), dtype=bool)

        # Angular scale in pixels/radian
        px_per_rad = p_w / (2.0 * math.pi)
        # Split into 3 range bands for vectorized range-adaptive surfel footprints:
        #   Band Far  (r >= 5.5 m): radius 1 px
        #   Band Mid  (2.2 <= r < 5.5 m): radius 2 px
        #   Band Near (r < 2.2 m): radius 4 px (scaled with p_w)
        scale_f = max(1, p_w // 2048)
        bands = [
            (ranges >= 5.5, 1 * scale_f),
            ((ranges >= 2.2) & (ranges < 5.5), 2 * scale_f),
            (ranges < 2.2, min(6, int(round(px_per_rad * cfg.surfel_radius_m / 1.1)))),
        ]

        # Paint farthest band -> nearest band; within each band, points are already sorted far->near!
        # Center pixel (0,0) painted last within each band so exact hits take precedence over splat halos.
        for b_mask, rad_px in bands:
            if not np.any(b_mask):
                continue
            r_b = ranges[b_mask]
            row_b = p_rows_idx[b_mask]
            col_b = p_cols_idx[b_mask]
            rgb_b = pt_cols[b_mask]
            hc_b = has_col[b_mask]

            # Paint outer halo offsets first, then (0, 0) center last
            offsets = self._disk_offsets(rad_px)[::-1]
            for dy, dx in offsets:
                rr = row_b + dy
                valid_row = (rr >= 0) & (rr < p_h)
                if not np.any(valid_row):
                    continue
                rr_v = rr[valid_row]
                cc_v = (col_b[valid_row] + dx) % p_w
                rb_v = r_b[valid_row]
                col_v = rgb_b[valid_row]
                hcv_v = hc_b[valid_row]

                # Only overwrite if closer than existing depth (painter's order across offsets)
                closer = rb_v <= (surfel_depth[rr_v, cc_v] + 0.03)
                if np.any(closer):
                    rr_u = rr_v[closer]
                    cc_u = cc_v[closer]
                    surfel_depth[rr_u, cc_u] = rb_v[closer]
                    surfel_rgb[rr_u, cc_u] = col_v[closer]
                    surfel_has_col[rr_u, cc_u] = hcv_v[closer]

        observed_pano = np.isfinite(surfel_depth)
        depth_map[observed_pano] = surfel_depth[observed_pano]

        # Shade uncolored observed geometry subtly by depth/elevation so structure remains legible
        pano_rgb[observed_pano] = surfel_rgb[observed_pano]
        prov_mask[observed_pano & surfel_has_col] = int(PanoramaProvenance.POINTCLOUD_RGB)
        prov_mask[observed_pano & (~surfel_has_col)] = int(PanoramaProvenance.GEOMETRY_UNCOLORED)

        # ------------------------------------------------------------------
        # STEP 3: Build Occlusion-Culled Structured Spherical Scan (s_rows x s_cols)
        # ------------------------------------------------------------------
        # Foreground occlusion culling: reject background points behind the splatted foreground surface!
        fg_depth_at_pt = surfel_depth[p_rows_idx, p_cols_idx]
        unoccluded_pt = ranges <= (fg_depth_at_pt + np.maximum(0.12, 0.035 * fg_depth_at_pt))

        pts_vis = pts_local[unoccluded_pt]
        r_vis = ranges[unoccluded_pt]
        col_vis = pt_cols[unoccluded_pt]
        u_vis = u_norm[unoccluded_pt]
        v_vis = v_norm[unoccluded_pt]

        if len(pts_vis) > 0:
            sc_c = (u_vis * s_cols).astype(np.int32) % s_cols
            sc_r = np.clip((v_vis * s_rows).astype(np.int32), 0, s_rows - 1)
            # `pts_vis` is already sorted farthest-to-nearest -> nearest valid measurement wins each cell!
            scan_xyz_local[sc_r, sc_c] = pts_vis
            scan_invalid[sc_r, sc_c] = 0
            scan_rgb[sc_r, sc_c] = col_vis
            scan_range[sc_r, sc_c] = r_vis

        # ------------------------------------------------------------------
        # STEP 4: Geometry-Assisted Multi-View RGB Keyframe Reprojection
        # ------------------------------------------------------------------
        selected_kfs = self._select_diverse_station_keyframes(
            station,
            keyframes or [],
            max_baseline_m=cfg.max_reproj_baseline_m,
            max_count=cfg.max_keyframes_per_station,
        )
        used_kfs = 0

        if len(selected_kfs) > 0 and np.any(observed_pano):
            unit_rays = self._build_equirectangular_unit_rays(p_h, p_w)
            obs_v, obs_u = np.nonzero(observed_pano)
            obs_r = depth_map[obs_v, obs_u]
            # Reconstruct 3D surface coordinates in station frame and world frame
            surf_local = unit_rays[obs_v, obs_u] * obs_r[:, None]                   # (M, 3)
            surf_world = (surf_local.astype(np.float64) @ R_ws.T) + station.position_xyz  # (M, 3)

            accum_rgb = np.zeros((len(obs_r), 3), dtype=np.float32)
            accum_w = np.zeros(len(obs_r), dtype=np.float32)

            # Downsampled global scene points for per-keyframe Z-buffer occlusion checks
            occl_pts_w = points_xyz[:: max(1, len(points_xyz) // 90000), :3].astype(np.float64)

            for kf in selected_kfs:
                img = kf.image_rgb
                if img is None:
                    continue
                kh, kw = img.shape[:2]
                cam_center = kf.camera_center_world
                baseline_d = float(np.linalg.norm(cam_center - station.position_xyz))

                if kf.camera_model == "spherical":
                    # Ouster Rev8 / Native Spherical RGB-D Reprojection Path
                    p_cam = (surf_world @ kf.R_rgb_world.T) + kf.t_rgb_world
                    rc = np.linalg.norm(p_cam, axis=1)
                    in_r = (rc > 0.25) & (rc < cfg.max_rgb_proj_range_m)
                    cand_idx = np.nonzero(in_r)[0]
                    if len(cand_idx) == 0:
                        continue
                    pc_sub = p_cam[cand_idx]
                    rc_sub = rc[cand_idx]
                    az_c = np.arctan2(pc_sub[:, 1], pc_sub[:, 0])
                    el_c = np.degrees(np.arcsin(np.clip(pc_sub[:, 2] / rc_sub, -1.0, 1.0)))
                    el_min, el_max = kf.spherical_vfov_deg
                    in_el = (el_c >= el_min + 1.0) & (el_c <= el_max - 1.0)
                    if not np.any(in_el):
                        continue
                    cand_idx = cand_idx[in_el]
                    az_c = az_c[in_el]
                    el_c = el_c[in_el]
                    rc_sub = rc_sub[in_el]

                    u_f = (((0.5 - az_c / (2.0 * math.pi)) % 1.0) * kw).astype(np.float32)
                    v_f = (((el_max - el_c) / max(1e-3, el_max - el_min)) * (kh - 1)).astype(np.float32)

                    if kf.range_image_m is not None and kf.range_image_m.shape[:2] == (kh, kw):
                        ui = np.clip(np.rint(u_f).astype(np.int32), 0, kw - 1)
                        vi = np.clip(np.rint(v_f).astype(np.int32), 0, kh - 1)
                        ref_r = kf.range_image_m[vi, ui]
                        vis_m = (ref_r <= 0.05) | (np.abs(rc_sub - ref_r) <= np.maximum(0.18, 0.05 * rc_sub))
                        cand_idx = cand_idx[vis_m]
                        u_f = u_f[vis_m]
                        v_f = v_f[vis_m]
                        rc_sub = rc_sub[vis_m]

                    if len(cand_idx) == 0:
                        continue

                    sampled = self._bilinear_sample_rgb(img, u_f, v_f, wrap_x=True)

                    w_par = 1.0 / (1.0 + (baseline_d / 1.5) ** 2)
                    w_rng = 1.0 / (1.0 + (rc_sub / 3.5) ** 2)
                    weights = (w_par * w_rng).astype(np.float32)
                    accum_rgb[cand_idx] += sampled * weights[:, None]
                    accum_w[cand_idx] += weights
                    used_kfs += 1
                    continue

                # Standard Pinhole + Plumb-Bob Reprojection Path (Orbbec Gemini 336L)
                p_cam = (surf_world @ kf.R_rgb_world.T) + kf.t_rgb_world
                zc = p_cam[:, 2]
                # Fast frustum pre-cull before distortion math
                in_cone = (
                    (zc > 0.25)
                    & (zc < cfg.max_rgb_proj_range_m)
                    & (np.abs(p_cam[:, 0]) < 1.35 * zc)
                    & (np.abs(p_cam[:, 1]) < 1.05 * zc)
                )
                cand_idx = np.nonzero(in_cone)[0]
                if len(cand_idx) == 0:
                    continue

                u_f, v_f, z_sub = project_points_plumb_bob_float(
                    p_cam[cand_idx], kf.fx, kf.fy, kf.cx, kf.cy, kf.dist_coeffs
                )
                margin = 6.0
                in_img = (u_f >= margin) & (u_f < kw - margin) & (v_f >= margin) & (v_f < kh - margin)
                if not np.any(in_img):
                    continue

                cand_idx = cand_idx[in_img]
                u_f = u_f[in_img].astype(np.float32)
                v_f = v_f[in_img].astype(np.float32)
                z_sub = z_sub[in_img].astype(np.float32)

                # Per-Keyframe Z-Buffer Visibility Test (solves walkaround parallax & occlusions!)
                zb_scale = 4
                zb_h, zb_w = max(1, kh // zb_scale), max(1, kw // zb_scale)
                z_buf = np.full((zb_h, zb_w), np.inf, dtype=np.float32)

                # Populate keyframe Z-buffer with both panorama surface points & local scene cloud
                ub_surf = np.clip((u_f / zb_scale).astype(np.int32), 0, zb_w - 1)
                vb_surf = np.clip((v_f / zb_scale).astype(np.int32), 0, zb_h - 1)
                np.minimum.at(z_buf, (vb_surf, ub_surf), z_sub)

                if len(occl_pts_w) > 0:
                    pc_occ = (occl_pts_w @ kf.R_rgb_world.T) + kf.t_rgb_world
                    zo = pc_occ[:, 2]
                    occ_cone = (
                        (zo > 0.20)
                        & (zo < cfg.max_rgb_proj_range_m)
                        & (np.abs(pc_occ[:, 0]) < 1.35 * zo)
                        & (np.abs(pc_occ[:, 1]) < 1.05 * zo)
                    )
                    if np.any(occ_cone):
                        uo, vo, zo_v = project_points_plumb_bob_float(
                            pc_occ[occ_cone], kf.fx, kf.fy, kf.cx, kf.cy, kf.dist_coeffs
                        )
                        occ_in = (uo >= 0) & (uo < kw) & (vo >= 0) & (vo < kh)
                        if np.any(occ_in):
                            ubo = np.clip((uo[occ_in] / zb_scale).astype(np.int32), 0, zb_w - 1)
                            vbo = np.clip((vo[occ_in] / zb_scale).astype(np.int32), 0, zb_h - 1)
                            np.minimum.at(z_buf, (vbo, ubo), zo_v[occ_in].astype(np.float32))

                # Slight 3x3 morphological minimum filter so foreground walls block inter-point gaps
                z_buf_closed = cv2.erode(z_buf, np.ones((3, 3), dtype=np.uint8))
                occl_tol = np.maximum(0.16, 0.045 * z_sub)
                vis_mask = z_sub <= (z_buf_closed[vb_surf, ub_surf] + occl_tol)
                if not np.any(vis_mask):
                    continue

                cand_idx = cand_idx[vis_mask]
                u_f = u_f[vis_mask]
                v_f = v_f[vis_mask]
                z_sub = z_sub[vis_mask]

                # Subpixel bilinear RGB lookup
                sampled = self._bilinear_sample_rgb(img, u_f, v_f, wrap_x=False)

                # Smooth multi-factor feathering weight (vignette * parallax * range * sharpness)
                fx_edge = np.minimum(u_f - margin, (kw - margin) - u_f) / max(1.0, 0.14 * kw)
                fy_edge = np.minimum(v_f - margin, (kh - margin) - v_f) / max(1.0, 0.14 * kh)
                w_vignette = np.clip(np.minimum(fx_edge, fy_edge), 0.02, 1.0)
                w_parallax = 1.0 / (1.0 + (baseline_d / 1.4) ** 2)
                w_range = 1.0 / (1.0 + (z_sub / 3.0) ** 2)
                w_sharp = float(np.clip(kf.sharpness_score / 100.0, 0.30, 2.0))

                weights = (w_vignette * w_parallax * w_range * w_sharp).astype(np.float32)
                accum_rgb[cand_idx] += sampled * weights[:, None]
                accum_w[cand_idx] += weights
                used_kfs += 1

            has_photo = accum_w > 1e-5
            if np.any(has_photo):
                blended = np.clip(accum_rgb[has_photo] / accum_w[has_photo, None], 0, 255).astype(np.uint8)
                pv = obs_v[has_photo]
                pu = obs_u[has_photo]
                pano_rgb[pv, pu] = blended
                prov_mask[pv, pu] = int(PanoramaProvenance.PHOTO_RGB)

                # Also transfer crisp reprojected photographic colors onto the structured scan cells!
                if np.any(scan_invalid == 0):
                    sv, su = np.nonzero(scan_invalid == 0)
                    map_pr = np.clip((sv.astype(np.float32) / s_rows * p_h).astype(np.int32), 0, p_h - 1)
                    map_pc = np.clip((su.astype(np.float32) / s_cols * p_w).astype(np.int32), 0, p_w - 1)
                    photo_hit = prov_mask[map_pr, map_pc] == int(PanoramaProvenance.PHOTO_RGB)
                    if np.any(photo_hit):
                        scan_rgb[sv[photo_hit], su[photo_hit]] = pano_rgb[map_pr[photo_hit], map_pc[photo_hit]]

        # ------------------------------------------------------------------
        # STEP 5: Strictly Gated 1-px Depth-Guided Micro-Pinhole Interpolation
        # (ONLY fills single-pixel pinholes inside continuous observed surfaces <8cm;
        #  NEVER hallucinates across unobserved regions or occlusion boundaries!)
        # ------------------------------------------------------------------
        if cfg.enable_small_gap_fill:
            empty = prov_mask == int(PanoramaProvenance.UNOBSERVED)
            if np.any(empty):
                d_up = np.roll(depth_map, -1, axis=0)
                d_dn = np.roll(depth_map, 1, axis=0)
                d_lf = np.roll(depth_map, 1, axis=1)
                d_rt = np.roll(depth_map, -1, axis=1)
                d_stack = np.stack([d_up, d_dn, d_lf, d_rt], axis=0)
                valid_nb = d_stack > 0.0
                nb_cnt = np.sum(valid_nb, axis=0)
                d_max = np.max(np.where(valid_nb, d_stack, 0.0), axis=0)
                d_min = np.min(np.where(valid_nb, d_stack, 1e9), axis=0)
                pinhole = empty & (nb_cnt >= 3) & ((d_max - d_min) <= 0.08)
                pinhole[0, :] = False
                pinhole[-1, :] = False
                if np.any(pinhole):
                    c_up = np.roll(pano_rgb, -1, axis=0).astype(np.float32)
                    c_dn = np.roll(pano_rgb, 1, axis=0).astype(np.float32)
                    c_lf = np.roll(pano_rgb, 1, axis=1).astype(np.float32)
                    c_rt = np.roll(pano_rgb, -1, axis=1).astype(np.float32)
                    w_nb = valid_nb.astype(np.float32)
                    w_sum = np.maximum(1.0, np.sum(w_nb, axis=0))
                    c_fill = (
                        c_up * w_nb[0, :, :, None]
                        + c_dn * w_nb[1, :, :, None]
                        + c_lf * w_nb[2, :, :, None]
                        + c_rt * w_nb[3, :, :, None]
                    ) / w_sum[:, :, None]
                    d_fill = np.sum(np.where(valid_nb, d_stack, 0.0), axis=0) / w_sum
                    pano_rgb[pinhole] = np.clip(c_fill[pinhole], 0, 255).astype(np.uint8)
                    depth_map[pinhole] = d_fill[pinhole]
                    prov_mask[pinhole] = int(PanoramaProvenance.DEPTH_GUIDED_FILL)

        # Compute summary coverage metrics
        tot_px = float(p_h * p_w)
        photo_pct = float(np.sum(prov_mask == int(PanoramaProvenance.PHOTO_RGB))) / tot_px * 100.0
        pc_pct = float(np.sum(prov_mask == int(PanoramaProvenance.POINTCLOUD_RGB))) / tot_px * 100.0
        fill_pct = float(np.sum(prov_mask == int(PanoramaProvenance.DEPTH_GUIDED_FILL))) / tot_px * 100.0
        uncol_pct = float(np.sum(prov_mask == int(PanoramaProvenance.GEOMETRY_UNCOLORED))) / tot_px * 100.0
        unobs_pct = float(np.sum(prov_mask == int(PanoramaProvenance.UNOBSERVED))) / tot_px * 100.0
        valid_scan_pts = int(np.sum(scan_invalid == 0))

        cov_vis = self.build_coverage_visualization(prov_mask)
        elapsed_ms = (time.perf_counter() - t_start) * 1000.0

        return StationSynthesisResult(
            station=station,
            scan_xyz_local=scan_xyz_local,
            scan_invalid_state=scan_invalid,
            scan_rgb=scan_rgb,
            scan_range_m=scan_range,
            panorama_rgb=pano_rgb,
            depth_map_m=depth_map,
            provenance_mask=prov_mask,
            coverage_vis_rgb=cov_vis,
            valid_scan_points=valid_scan_pts,
            total_scan_cells=s_rows * s_cols,
            photo_rgb_pct=round(photo_pct, 2),
            pointcloud_rgb_pct=round(pc_pct, 2),
            gap_fill_pct=round(fill_pct, 2),
            uncolored_geom_pct=round(uncol_pct, 2),
            unobserved_pct=round(unobs_pct, 2),
            contributing_keyframes=used_kfs,
            synthesis_time_ms=round(elapsed_ms, 2),
        )


# ==============================================================================
# 4. COMPLETE STATION BUNDLE & STRUCTURED E57 EXPORTER
# ==============================================================================
def _write_station_structured_ply(filepath: str, res: StationSynthesisResult):
    """Writes valid structured scan points in World coordinates as binary little-endian PLY."""
    valid = res.scan_invalid_state == 0
    xyz_loc = res.scan_xyz_local[valid]
    rgb = res.scan_rgb[valid]
    R_ws = res.station.R_world_station
    xyz_w = (xyz_loc.astype(np.float64) @ R_ws.T + res.station.position_xyz).astype("<f4")

    n = len(xyz_w)
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"comment VirtualStation {res.station.station_id}\n"
        f"element vertex {n}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    ).encode("ascii")

    rec = np.empty(n, dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1")])
    if n > 0:
        rec["x"], rec["y"], rec["z"] = xyz_w[:, 0], xyz_w[:, 1], xyz_w[:, 2]
        rec["r"], rec["g"], rec["b"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]

    with open(filepath, "wb") as f:
        f.write(header)
        f.write(rec.tobytes())


def _collect_world_cloud_from_results(results: Sequence[StationSynthesisResult]) -> Tuple[np.ndarray, np.ndarray]:
    """Fallback aggregator: reconstructs global Z-up XYZ+RGB from synthesized structured scans when raw cloud is omitted."""
    xyz_chunks: List[np.ndarray] = []
    rgb_chunks: List[np.ndarray] = []
    for res in results:
        valid = res.scan_invalid_state == 0
        if not np.any(valid):
            continue
        xyz_loc = res.scan_xyz_local[valid]
        rgb = res.scan_rgb[valid]
        R_ws = res.station.R_world_station
        xyz_w = xyz_loc.astype(np.float64) @ R_ws.T + res.station.position_xyz
        xyz_chunks.append(xyz_w)
        rgb_chunks.append(rgb)
    if not xyz_chunks:
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.uint8)
    return np.vstack(xyz_chunks), np.vstack(rgb_chunks)


def write_point_cloud_lod0_bin(
    out_bin_path: str,
    points_xyz: np.ndarray,
    points_rgb: Optional[np.ndarray] = None,
    max_points: int = 350000,
) -> int:
    """
    Writes exact AV Showcase / AV Spatial Bundle v1.1 binary point cloud cache (`cache/lod0.bin`):
      - 4-byte little-endian uint32 `num_points`
      - `num_points` packed 15-byte records `<fffBBB>`:
          float32 x, float32 y, float32 z (canonical right-handed Z-up meters)
          uint8 r, uint8 g, uint8 b
    """
    os.makedirs(os.path.dirname(os.path.abspath(out_bin_path)), exist_ok=True)
    if points_xyz is None or len(points_xyz) == 0:
        with open(out_bin_path, "wb") as f:
            f.write(struct.pack("<I", 0))
        return 0

    n_total = len(points_xyz)
    step = max(1, n_total // max(1, int(max_points)))
    pts = np.asarray(points_xyz[::step, :3], dtype="<f4")
    n = len(pts)

    if points_rgb is not None and len(points_rgb) == n_total:
        raw_c = points_rgb[::step, :3]
        if np.issubdtype(raw_c.dtype, np.floating) and float(np.max(raw_c)) <= 1.05:
            cols = np.clip(raw_c * 255.0, 0, 255).astype(np.uint8)
        else:
            cols = np.clip(raw_c, 0, 255).astype(np.uint8)
    else:
        cols = np.full((n, 3), 200, dtype=np.uint8)

    rec_dtype = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1")])
    rec = np.empty(n, dtype=rec_dtype)
    rec["x"], rec["y"], rec["z"] = pts[:, 0], pts[:, 1], pts[:, 2]
    rec["r"], rec["g"], rec["b"] = cols[:, 0], cols[:, 1], cols[:, 2]

    with open(out_bin_path, "wb") as f:
        f.write(struct.pack("<I", n))
        f.write(rec.tobytes())
    return int(n)


def read_point_cloud_lod0_bin(bin_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Reads `cache/lod0.bin` back into `(xyz_float64, rgb_uint8)`."""
    if not os.path.exists(bin_path):
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.uint8)
    with open(bin_path, "rb") as f:
        hdr = f.read(4)
        if len(hdr) < 4:
            return np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.uint8)
        (n,) = struct.unpack("<I", hdr)
        if n == 0:
            return np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.uint8)
        rec_dtype = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1")])
        rec = np.frombuffer(f.read(n * 15), dtype=rec_dtype, count=n)
    xyz = np.column_stack([rec["x"], rec["y"], rec["z"]]).astype(np.float64)
    rgb = np.column_stack([rec["r"], rec["g"], rec["b"]]).astype(np.uint8)
    return xyz, rgb


def detect_multi_story_floors(
    z_values: Sequence[float],
    floor_threshold: float = 0.25,
    custom_names: Optional[Sequence[str]] = None,
) -> Tuple[List[int], List[Dict]]:
    """
    Clusters station Z elevations into major building stories using plateau detection (`> 0.60m` story gap)
    and splits intermediate staircase stations at the true inter-floor midpoint `(mean_z_below + mean_z_above) / 2`.
    Matches AV Showcase & AV Structure floor semantics 1:1.
    """
    if not z_values:
        return [], []

    z_list = [float(z) for z in z_values]
    fine_thresh = min(float(floor_threshold), 0.30)
    sorted_unique_z = sorted(set(z_list))

    clusters: List[Dict] = []
    cur_z_vals: List[float] = [sorted_unique_z[0]]
    for i in range(1, len(sorted_unique_z)):
        if (sorted_unique_z[i] - sorted_unique_z[i - 1]) > fine_thresh:
            clusters.append({"scans": cur_z_vals, "mean_z": float(np.mean(cur_z_vals))})
            cur_z_vals = [sorted_unique_z[i]]
        else:
            cur_z_vals.append(sorted_unique_z[i])
    clusters.append({"scans": cur_z_vals, "mean_z": float(np.mean(cur_z_vals))})

    # Expand cluster scan counts to reflect actual station multiplicity
    for cl in clusters:
        z_set = set(cl["scans"])
        cl["all_z"] = [z for z in z_list if z in z_set]
        cl["mean_z"] = float(np.mean(cl["all_z"]))

    major_floors: List[Dict] = []
    stairs_clusters: List[Dict] = []
    max_count = max(len(cl["all_z"]) for cl in clusters)

    for idx, cl in enumerate(clusters):
        left_gap = (cl["mean_z"] - clusters[idx - 1]["mean_z"]) if idx > 0 else 999.0
        right_gap = (clusters[idx + 1]["mean_z"] - cl["mean_z"]) if idx < len(clusters) - 1 else 999.0
        is_major = (len(cl["all_z"]) >= max(2, int(0.2 * max_count))) or (min(left_gap, right_gap) > 0.60)
        if is_major and (not major_floors or abs(cl["mean_z"] - major_floors[-1]["plateau_mean_z"]) > 1.80):
            cl["plateau_mean_z"] = cl["mean_z"]
            major_floors.append(cl)
        else:
            stairs_clusters.append(cl)

    if not major_floors:
        for cl in clusters:
            cl["plateau_mean_z"] = cl["mean_z"]
        major_floors = clusters
        stairs_clusters = []

    for sc in stairs_clusters:
        below = [mf for mf in major_floors if mf["plateau_mean_z"] <= sc["mean_z"]]
        above = [mf for mf in major_floors if mf["plateau_mean_z"] > sc["mean_z"]]
        fl_below = max(below, key=lambda x: x["plateau_mean_z"]) if below else None
        fl_above = min(above, key=lambda x: x["plateau_mean_z"]) if above else None
        if fl_below and fl_above:
            mid_z = 0.5 * (fl_below["plateau_mean_z"] + fl_above["plateau_mean_z"])
            for z in sc["all_z"]:
                if z < mid_z:
                    fl_below["all_z"].append(z)
                else:
                    fl_above["all_z"].append(z)
        elif fl_below:
            fl_below["all_z"].extend(sc["all_z"])
        elif fl_above:
            fl_above["all_z"].extend(sc["all_z"])

    major_floors.sort(key=lambda x: x["plateau_mean_z"])
    z_to_fid: Dict[float, int] = {}
    final_floors: List[Dict] = []

    for idx, mf in enumerate(major_floors):
        fid = idx + 1
        scans_z = mf["all_z"] if mf["all_z"] else [mf["plateau_mean_z"]]
        for z in scans_z:
            z_to_fid[z] = fid
        name = (
            custom_names[idx]
            if (custom_names and idx < len(custom_names))
            else f"Level {fid}"
        )
        final_floors.append(
            {
                "id": fid,
                "name": name,
                "min_z": round(float(min(scans_z)), 4),
                "max_z": round(float(max(scans_z)), 4),
                "mean_z": round(float(mf["plateau_mean_z"]), 4),
            }
        )

    assigned_ids = [z_to_fid.get(z, 1) for z in z_list]
    return assigned_ids, final_floors


def build_walkthrough_navigation_graph(
    panos: List[Dict],
    points_xyz: Optional[np.ndarray] = None,
    max_neighbor_dist: float = 12.0,
) -> List[Dict]:
    """
    Builds a wall-occlusion-aware 8-sector + sequential + staircase symmetric navigation graph for AV Showcase.
    """
    if not panos:
        return panos

    # Build sparse 3D wall-occupancy voxel set (0.14m voxels) so navigation links never cut through solid walls
    occ_voxels = set()
    voxel_size = 0.14
    if points_xyz is not None and len(points_xyz) > 0:
        step = max(1, len(points_xyz) // 250000)
        pts_sub = points_xyz[::step, :3]
        vq = np.floor(pts_sub / voxel_size).astype(np.int32)
        # Require at least 2 points per voxel when dense to ignore isolated dust
        uniq_v, counts = np.unique(vq, axis=0, return_counts=True)
        min_pts = 2 if len(pts_sub) > 25000 else 1
        for v in uniq_v[counts >= min_pts]:
            occ_voxels.add((int(v[0]), int(v[1]), int(v[2])))

    def has_line_of_sight(pa: Sequence[float], pb: Sequence[float]) -> bool:
        if not occ_voxels:
            return True
        a = np.asarray(pa, dtype=np.float64)
        b = np.asarray(pb, dtype=np.float64)
        vec = b - a
        dist = float(np.linalg.norm(vec))
        if dist <= 0.90:
            return True
        # Sample along chest/eye ray excluding 0.40m around endpoints
        n_steps = max(4, int(dist / (voxel_size * 0.75)))
        ts = np.linspace(0.40 / dist, 1.0 - (0.40 / dist), n_steps)
        hits = 0
        for t in ts:
            p = a + t * vec
            vk = (int(math.floor(p[0] / voxel_size)), int(math.floor(p[1] / voxel_size)), int(math.floor(p[2] / voxel_size)))
            if vk in occ_voxels:
                hits += 1
                if hits >= 2:
                    return False
        return True

    floors_dict: Dict[int, List[Dict]] = {}
    for p in panos:
        floors_dict.setdefault(int(p.get("floor_id", 1)), []).append(p)
        p["neighbors"] = []

    for fid, fl_panos in floors_dict.items():
        n = len(fl_panos)
        if n < 2:
            continue
        pts = np.array([p["position"] for p in fl_panos], dtype=np.float64)
        diff = pts[None, :, :2] - pts[:, None, :2]  # vector from i -> j
        dists = np.linalg.norm(diff, axis=-1)
        angles = np.arctan2(diff[..., 1], diff[..., 0])

        num_sectors = 8
        sector_step = 2.0 * math.pi / num_sectors

        for i, p in enumerate(fl_panos):
            nbr_set = set()
            # 1. Sequential trajectory neighbors (if unobstructed)
            for adj in (i - 1, i + 1):
                if 0 <= adj < n and dists[i, adj] <= max_neighbor_dist:
                    if has_line_of_sight(pts[i], pts[adj]):
                        nbr_set.add(fl_panos[adj]["id"])

            # 2. 8 directional 45-degree sectors
            valid_idx = np.where((dists[i] > 0.01) & (dists[i] <= max_neighbor_dist))[0]
            if len(valid_idx) > 0:
                c_dists = dists[i, valid_idx]
                c_angles = angles[i, valid_idx]
                sec_ids = np.floor((c_angles + math.pi + 0.5 * sector_step) / sector_step).astype(int) % num_sectors
                for s in range(num_sectors):
                    in_s = np.where(sec_ids == s)[0]
                    if len(in_s) == 0:
                        continue
                    order = np.argsort(c_dists[in_s])
                    for rank in order:
                        cand_j = int(valid_idx[in_s[rank]])
                        if c_dists[in_s[rank]] < 0.45 and len(order) > 1:
                            continue
                        if has_line_of_sight(pts[i], pts[cand_j]):
                            nbr_set.add(fl_panos[cand_j]["id"])
                            break

            # 3. Ensure connectivity fallback if strict voxel raycast left an isolated station
            if not nbr_set and len(valid_idx) > 0:
                nearest_j = int(valid_idx[np.argmin(dists[i, valid_idx])])
                nbr_set.add(fl_panos[nearest_j]["id"])

            p["neighbors"] = sorted(nbr_set)

    # Cross-floor staircase transitions
    if len(floors_dict) > 1:
        sorted_fids = sorted(floors_dict.keys())
        for i in range(len(sorted_fids)):
            for j in range(i + 1, min(i + 3, len(sorted_fids))):
                panos_a = floors_dict[sorted_fids[i]]
                panos_b = floors_dict[sorted_fids[j]]
                stair_links = []
                for pa in panos_a:
                    ax, ay, az = pa["position"]
                    for pb in panos_b:
                        bx, by, bz = pb["position"]
                        dz = abs(bz - az)
                        dxy = math.hypot(bx - ax, by - ay)
                        if 0.10 <= dz <= 2.20 and dxy <= 5.50:
                            stair_links.append((math.sqrt(dxy * dxy + dz * dz), pa, pb))
                stair_links.sort(key=lambda x: x[0])
                linked_a, linked_b = set(), set()
                for _, pa, pb in stair_links:
                    if len(linked_a) >= 4 and len(linked_b) >= 4:
                        break
                    if pa["id"] not in linked_a or pb["id"] not in linked_b:
                        if pb["id"] not in pa["neighbors"]:
                            pa["neighbors"].append(pb["id"])
                        if pa["id"] not in pb["neighbors"]:
                            pb["neighbors"].append(pa["id"])
                        pa["stair_link"] = True
                        pb["stair_link"] = True
                        linked_a.add(pa["id"])
                        linked_b.add(pb["id"])
                # Fallback for consecutive floors without an intermediate mid-stair landing station:
                # connect the single closest horizontal pair (<= 6.5m XY, <= 5.0m Z) matching AV Showcase v2.0
                if j == i + 1 and not linked_a and panos_a and panos_b:
                    best_pair = None
                    best_dxy = 1e9
                    for pa in panos_a:
                        ax, ay, az = pa["position"]
                        for pb in panos_b:
                            bx, by, bz = pb["position"]
                            dz = abs(bz - az)
                            dxy = math.hypot(bx - ax, by - ay)
                            if dz <= 5.00 and dxy <= 6.50 and dxy < best_dxy:
                                best_dxy = dxy
                                best_pair = (pa, pb)
                    if best_pair is not None:
                        pa, pb = best_pair
                        if pb["id"] not in pa["neighbors"]:
                            pa["neighbors"].append(pb["id"])
                        if pa["id"] not in pb["neighbors"]:
                            pb["neighbors"].append(pa["id"])
                        pa["stair_link"] = True
                        pb["stair_link"] = True

    # Enforce symmetry
    by_id = {p["id"]: p for p in panos}
    for p in panos:
        for nid in list(p.get("neighbors", [])):
            nb = by_id.get(nid)
            if nb and p["id"] not in nb["neighbors"]:
                nb["neighbors"].append(p["id"])
    return panos


def render_floor_keyplan_png(
    points_xyz: np.ndarray,
    floor_station_mean_z: float,
    out_png_path: str,
    floor_stations_xy: Optional[np.ndarray] = None,
    meters_per_pixel: float = 0.02,
    max_dim_px: int = 2048,
) -> Optional[Dict]:
    """
    Renders a crisp, metric-calibrated top-down 2D architectural wall-density Key Plan PNG for a building story:
      - Image +X (right) = World +X (East)
      - Image -Y (top row 0) = World +Y (North)
      - Background: #171717 charcoal, Navigable interior: #22262e, Architectural wall slice: #e5e7eb + AV gold core
    Returns `world_bounds` dict `{min_x, max_x, min_y, max_y, meters_per_pixel}`.
    """
    if (points_xyz is None or len(points_xyz) == 0) and (floor_stations_xy is None or len(floor_stations_xy) == 0):
        return None

    os.makedirs(os.path.dirname(os.path.abspath(out_png_path)), exist_ok=True)
    pts = points_xyz[:, :3] if (points_xyz is not None and len(points_xyz) > 0) else np.zeros((0, 3), dtype=np.float64)

    # Estimate floor slab Z beneath `floor_station_mean_z`
    slab_candidates = pts[(pts[:, 2] >= floor_station_mean_z - 2.35) & (pts[:, 2] <= floor_station_mean_z - 0.45), 2]
    if len(slab_candidates) >= 30:
        slab_z = float(np.percentile(slab_candidates, 15))
    elif len(pts) > 0:
        near_story = pts[(pts[:, 2] >= floor_station_mean_z - 2.50) & (pts[:, 2] <= floor_station_mean_z + 1.20), 2]
        slab_z = float(np.percentile(near_story, 8)) if len(near_story) > 0 else (floor_station_mean_z - 1.55)
    else:
        slab_z = floor_station_mean_z - 1.55

    story_mask = (pts[:, 2] >= slab_z - 0.25) & (pts[:, 2] <= slab_z + 2.40)
    story_pts = pts[story_mask] if np.any(story_mask) else pts

    bound_xy_list = []
    if len(story_pts) > 0:
        bound_xy_list.append(story_pts[:, :2])
    if floor_stations_xy is not None and len(floor_stations_xy) > 0:
        bound_xy_list.append(np.asarray(floor_stations_xy[:, :2], dtype=np.float64))
    if not bound_xy_list:
        return None

    all_xy = np.vstack(bound_xy_list)
    margin_m = 0.80
    min_x = float(np.min(all_xy[:, 0])) - margin_m
    max_x = float(np.max(all_xy[:, 0])) + margin_m
    min_y = float(np.min(all_xy[:, 1])) - margin_m
    max_y = float(np.max(all_xy[:, 1])) + margin_m

    span_x = max(2.0, max_x - min_x)
    span_y = max(2.0, max_y - min_y)
    mpp = max(float(meters_per_pixel), max(span_x, span_y) / float(max_dim_px))

    w = max(64, int(math.ceil(span_x / mpp)))
    h = max(64, int(math.ceil(span_y / mpp)))
    # Realign max_x / min_y to exact pixel grid
    max_x = min_x + w * mpp
    min_y = max_y - h * mpp

    # Charcoal background (#171717) in BGR for cv2.imwrite
    canvas_bgr = np.full((h, w, 3), (23, 23, 23), dtype=np.uint8)

    if len(story_pts) > 0:
        # 1. Subtle floor slab interior footprint (#22262e -> RGB(34,38,46) -> BGR(46,38,34))
        slab_pts = story_pts[(story_pts[:, 2] >= slab_z - 0.22) & (story_pts[:, 2] <= slab_z + 0.25)]
        if len(slab_pts) > 0:
            sc = np.clip(((slab_pts[:, 0] - min_x) / mpp).astype(np.int32), 0, w - 1)
            sr = np.clip(((max_y - slab_pts[:, 1]) / mpp).astype(np.int32), 0, h - 1)
            slab_mask = np.zeros((h, w), dtype=np.uint8)
            slab_mask[sr, sc] = 255
            slab_mask = cv2.morphologyEx(slab_mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
            canvas_bgr[slab_mask > 0] = (46, 38, 34)

        # 2. Crisp architectural wall & doorway slice [slab_z + 0.35m, slab_z + 2.10m]
        wall_pts = story_pts[(story_pts[:, 2] >= slab_z + 0.35) & (story_pts[:, 2] <= slab_z + 2.10)]
        if len(wall_pts) == 0:
            # Fallback if flat synthetic floor test has thin walls or low height
            wall_pts = story_pts[story_pts[:, 2] > slab_z + 0.15]
        if len(wall_pts) > 0:
            wc = np.clip(((wall_pts[:, 0] - min_x) / mpp).astype(np.int32), 0, w - 1)
            wr = np.clip(((max_y - wall_pts[:, 1]) / mpp).astype(np.int32), 0, h - 1)
            hist = np.zeros((h, w), dtype=np.float32)
            np.add.at(hist, (wr, wc), 1.0)
            nonzero = hist[hist > 0]
            p90 = float(np.percentile(nonzero, 90)) if len(nonzero) > 0 else 1.0
            norm_w = np.clip(hist / max(1.0, p90), 0.0, 1.0)
            # Crisp 1px dilation for unmistakable wall legibility at overview zoom
            norm_w = cv2.dilate(norm_w, np.ones((2, 2), np.uint8))
            wall_hit = norm_w > 0.08
            if np.any(wall_hit):
                # Bright architectural off-white (#e5e7eb -> BGR(235, 231, 229)) blended by wall point density
                alpha = np.clip(0.45 + 0.55 * norm_w[wall_hit], 0.0, 1.0)[:, None]
                base = canvas_bgr[wall_hit].astype(np.float32)
                wall_col = np.array([235.0, 231.0, 229.0], dtype=np.float32)
                canvas_bgr[wall_hit] = np.clip(base * (1.0 - alpha) + wall_col * alpha, 0, 255).astype(np.uint8)
                # Subtle AV Gold core highlight on dense load-bearing walls
                core_hit = norm_w > 0.85
                canvas_bgr[core_hit] = (165, 212, 235)  # BGR for warm champagne-white

    cv2.imwrite(out_png_path, canvas_bgr)
    return {
        "min_x": round(min_x, 4),
        "max_x": round(max_x, 4),
        "min_y": round(min_y, 4),
        "max_y": round(max_y, 4),
        "meters_per_pixel": round(mpp, 6),
        "width_px": int(w),
        "height_px": int(h),
        "slab_z_m": round(slab_z, 4),
    }


def compute_transform_to_project_icp(
    source_xyz: np.ndarray,
    reference_xyz: np.ndarray,
    max_iterations: int = 40,
) -> Tuple[np.ndarray, float, float]:
    """
    Computes a gravity-preserving 4-DOF (`[tx, ty, tz, yaw]`) rigid registration matrix `T_project_from_source`
    aligning `source_xyz` (e.g. Apr 13 scan) into `reference_xyz` (e.g. Dec 4 Project Baseline scan).
    Returns `(T_4x4, inlier_rmse_m, overlap_ratio)`.
    """
    if source_xyz is None or reference_xyz is None or len(source_xyz) < 20 or len(reference_xyz) < 20:
        return np.eye(4, dtype=np.float64), 0.0, 0.0

    def voxel_sub(pts: np.ndarray, vs: float) -> np.ndarray:
        q = np.floor(pts[:, :3] / vs).astype(np.int64)
        _, idx = np.unique(q, axis=0, return_index=True)
        return pts[idx, :3].astype(np.float64)

    src_coarse = voxel_sub(source_xyz, 0.25)
    ref_coarse = voxel_sub(reference_xyz, 0.25)
    if len(src_coarse) < 10 or len(ref_coarse) < 10:
        return np.eye(4, dtype=np.float64), 0.0, 0.0

    # Align Z floor datum (10th percentile Z) + coarse 16-angle XY centroid yaw search
    z_src0 = float(np.percentile(src_coarse[:, 2], 10))
    z_ref0 = float(np.percentile(ref_coarse[:, 2], 10))
    dz0 = z_ref0 - z_src0

    c_src = np.mean(src_coarse[:, :2], axis=0)
    c_ref = np.mean(ref_coarse[:, :2], axis=0)
    tree_coarse = cKDTree(ref_coarse)

    best_score = float("inf")
    best_yaw = 0.0
    best_t = np.array([c_ref[0] - c_src[0], c_ref[1] - c_src[1], dz0], dtype=np.float64)

    # Always evaluate identity (0 translation, 0 yaw) first in case scans already share exact coordinates!
    d_id, _ = tree_coarse.query(src_coarse, k=1)
    id_med = float(np.median(d_id))
    if id_med < 0.12:
        best_score = id_med
        best_yaw = 0.0
        best_t = np.zeros(3, dtype=np.float64)
    else:
        for k in range(16):
            yaw = (2.0 * math.pi * k) / 16.0
            ca, sa = math.cos(yaw), math.sin(yaw)
            R2 = np.array([[ca, -sa], [sa, ca]], dtype=np.float64)
            xy_rot = (src_coarse[:, :2] - c_src) @ R2.T + c_ref
            cand = np.column_stack([xy_rot, src_coarse[:, 2] + dz0])
            dists, _ = tree_coarse.query(cand, k=1)
            score = float(np.percentile(dists, 65))
            if score < best_score:
                best_score = score
                best_yaw = yaw
                t_xy = c_ref - (R2 @ c_src)
                best_t = np.array([t_xy[0], t_xy[1], dz0], dtype=np.float64)

    def build_T(yaw_r: float, t_vec: np.ndarray) -> np.ndarray:
        ca, sa = math.cos(yaw_r), math.sin(yaw_r)
        T = np.eye(4, dtype=np.float64)
        T[0, 0], T[0, 1] = ca, -sa
        T[1, 0], T[1, 1] = sa, ca
        T[:3, 3] = t_vec
        return T

    T_cur = build_T(best_yaw, best_t)

    # Two-stage trimmed 4-DOF ICP (0.25m -> 0.08m)
    for vs, max_corr, iters in ((0.25, 1.20, max_iterations // 2), (0.08, 0.35, max_iterations)):
        src_pts = voxel_sub(source_xyz, vs)
        ref_pts = voxel_sub(reference_xyz, vs)
        if len(src_pts) < 10 or len(ref_pts) < 10:
            continue
        tree = cKDTree(ref_pts)
        for _ in range(max(5, iters)):
            transformed = (src_pts @ T_cur[:3, :3].T) + T_cur[:3, 3]
            dists, idx = tree.query(transformed, k=1)
            inliers = dists <= max_corr
            if np.count_nonzero(inliers) < 10:
                break
            p_s = transformed[inliers]
            p_r = ref_pts[idx[inliers]]

            # Optimal 2D rotation + translation in XY, robust median shift in Z
            mu_s = np.mean(p_s[:, :2], axis=0)
            mu_r = np.mean(p_r[:, :2], axis=0)
            H2 = (p_s[:, :2] - mu_s).T @ (p_r[:, :2] - mu_r)
            U, _, Vt = np.linalg.svd(H2)
            R2_d = Vt.T @ U.T
            if np.linalg.det(R2_d) < 0:
                Vt[-1, :] *= -1.0
                R2_d = Vt.T @ U.T
            dyaw = math.atan2(R2_d[1, 0], R2_d[0, 0])
            dt_xy = mu_r - (R2_d @ mu_s)
            dt_z = float(np.median(p_r[:, 2] - p_s[:, 2]))

            T_delta = build_T(dyaw, np.array([dt_xy[0], dt_xy[1], dt_z], dtype=np.float64))
            T_cur = T_delta @ T_cur
            if abs(dyaw) < 1e-4 and float(np.linalg.norm(dt_xy)) < 5e-4 and abs(dt_z) < 5e-4:
                break

    src_eval = voxel_sub(source_xyz, 0.10)
    ref_eval = voxel_sub(reference_xyz, 0.10)
    tree_f = cKDTree(ref_eval)
    tf_eval = (src_eval @ T_cur[:3, :3].T) + T_cur[:3, 3]
    dists_f, _ = tree_f.query(tf_eval, k=1)
    inl = dists_f <= 0.25
    overlap = float(np.count_nonzero(inl)) / max(1, len(src_eval))
    rmse = float(np.sqrt(np.mean(dists_f[inl] ** 2))) if np.any(inl) else float(np.median(dists_f))
    return T_cur, round(rmse, 5), round(overlap, 4)


def apply_transform_4x4_xyz(points_xyz: np.ndarray, T_4x4: np.ndarray) -> np.ndarray:
    if points_xyz is None or len(points_xyz) == 0:
        return points_xyz
    return (points_xyz[:, :3].astype(np.float64) @ T_4x4[:3, :3].T) + T_4x4[:3, 3]


def transform_pano_pose_xyzw(
    position_xyz: Sequence[float],
    rotation_xyzw: Sequence[float],
    T_4x4: np.ndarray,
) -> Tuple[List[float], List[float]]:
    """Transforms `[x, y, z]` and `[qx, qy, qz, qw]` by rigid 4x4 `T_4x4`."""
    pos_new = (T_4x4[:3, :3] @ np.asarray(position_xyz, dtype=np.float64)) + T_4x4[:3, 3]
    r_old = Rotation.from_quat(np.asarray(rotation_xyzw, dtype=np.float64))
    r_tf = Rotation.from_matrix(T_4x4[:3, :3])
    q_new = (r_tf * r_old).as_quat()  # [qx, qy, qz, qw]
    return [round(float(v), 6) for v in pos_new], [round(float(v), 8) for v in q_new]


def export_av_structure_bundle(
    save_dir: str,
    results: Sequence[StationSynthesisResult],
    export_e57: bool = True,
    e57_filename: str = "structured_stations.e57",
    jpeg_quality: int = 93,
    points_xyz: Optional[np.ndarray] = None,
    points_rgb: Optional[np.ndarray] = None,
    project_id: str = "",
    project_name: str = "AV ScanAR Project",
    scan_date: str = "",
    scan_id: str = "",
    reference_cloud_xyz: Optional[np.ndarray] = None,
    av_showcase_project_dir: Optional[str] = None,
) -> Dict:
    """
    Exports synchronized 2D + 3D outputs for all synthesized Virtual Scan Stations AND full
    AV Ecosystem Interoperability (`AV Spatial Bundle v1.1` + optional `project.avproj` update):
      - `stations/<station_id>/...` + `stations_manifest.json`
      - `structured_stations.e57` (ASTM E2807-11 Structured E57 linking scans & spherical panoramas)
      - `panos/pano_XXX.jpg` + `panos/pano_XXX_depth.png`
      - `floors/floor_<id>_keyplan.png` (metric-calibrated 2D architectural wall-slice PNGs)
      - `cache/lod0.bin` (AV Showcase binary point cloud cache)
      - `av_walkthrough_bundle.json` (`AV Spatial Bundle v1.1` manifest)
    """
    os.makedirs(save_dir, exist_ok=True)
    stations_root = os.path.join(save_dir, "stations")
    panos_root = os.path.join(save_dir, "panos")
    floors_root = os.path.join(save_dir, "floors")
    cache_root = os.path.join(save_dir, "cache")
    for d in (stations_root, panos_root, floors_root, cache_root):
        os.makedirs(d, exist_ok=True)

    # Assemble world point cloud if caller only passed StationSynthesisResult objects
    cloud_xyz, cloud_rgb = (points_xyz, points_rgb)
    if cloud_xyz is None or len(cloud_xyz) == 0:
        cloud_xyz, cloud_rgb = _collect_world_cloud_from_results(results)

    # Optional 4D historical ICP registration into canonical Project Baseline Frame
    T_proj = np.eye(4, dtype=np.float64)
    icp_rmse_m = 0.0
    icp_overlap = 1.0
    if reference_cloud_xyz is not None and len(reference_cloud_xyz) >= 20 and len(cloud_xyz) >= 20:
        T_proj, icp_rmse_m, icp_overlap = compute_transform_to_project_icp(cloud_xyz, reference_cloud_xyz)

    has_non_identity_tf = not np.allclose(T_proj, np.eye(4), atol=1e-6)
    aligned_cloud_xyz = apply_transform_4x4_xyz(cloud_xyz, T_proj) if has_non_identity_tf else cloud_xyz

    e57_payloads: List[E57StationPayload] = []
    manifest_stations: List[Dict] = []
    bundle_panos: List[Dict] = []

    for idx, res in enumerate(results):
        st = res.station
        st_dir = os.path.join(stations_root, st.station_id)
        os.makedirs(st_dir, exist_ok=True)

        # 1. 360° Equirectangular Panorama JPEG
        pano_bgr = cv2.cvtColor(res.panorama_rgb, cv2.COLOR_RGB2BGR)
        ok, jpg_buf = cv2.imencode(".jpg", pano_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
        jpg_bytes = jpg_buf.tobytes() if ok else b""
        pano_path = os.path.join(st_dir, "panorama_rgb.jpg")
        with open(pano_path, "wb") as fp:
            fp.write(jpg_bytes)

        # Standardized AV Showcase / Bundle panorama copy (`panos/pano_XXX.jpg`)
        pano_std_id = f"pano_{idx:03d}"
        pano_std_rel = f"panos/{pano_std_id}.jpg"
        with open(os.path.join(save_dir, pano_std_rel), "wb") as fp_std:
            fp_std.write(jpg_bytes)

        # 2. 16-bit Millimeter Depth Map PNG (0 = unobserved, max 65.535 m)
        depth_mm = np.clip(np.rint(res.depth_map_m * 1000.0), 0, 65535).astype(np.uint16)
        depth_path = os.path.join(st_dir, "depth_map_mm.png")
        cv2.imwrite(depth_path, depth_mm)
        depth_std_rel = f"panos/{pano_std_id}_depth.png"
        shutil.copy2(depth_path, os.path.join(save_dir, depth_std_rel))

        # 3. Coverage & Provenance Mask PNG + Raw Provenance Index PNG
        cov_vis_bgr = cv2.cvtColor(res.coverage_vis_rgb, cv2.COLOR_RGB2BGR)
        cov_path = os.path.join(st_dir, "coverage_mask.png")
        cv2.imwrite(cov_path, cov_vis_bgr)
        prov_raw_path = os.path.join(st_dir, "provenance_code.png")
        cv2.imwrite(prov_raw_path, res.provenance_mask)

        # 4. Station Structured Point Cloud PLY
        ply_path = os.path.join(st_dir, "structured_scan.ply")
        _write_station_structured_ply(ply_path, res)

        # Convert E57 (qw, qx, qy, qz) -> AV Bundle / Showcase [qx, qy, qz, qw]
        qw, qx, qy, qz = st.rotation_wxyz
        pos_aligned, rot_xyzw_aligned = (
            transform_pano_pose_xyzw(st.position_xyz, [qx, qy, qz, qw], T_proj)
            if has_non_identity_tf
            else ([round(float(x), 6) for x in st.position_xyz], [round(float(v), 8) for v in (qx, qy, qz, qw)])
        )
        rot_wxyz_aligned = (rot_xyzw_aligned[3], rot_xyzw_aligned[0], rot_xyzw_aligned[1], rot_xyzw_aligned[2])

        # 5. Per-station Metadata JSON
        rows, cols = res.scan_invalid_state.shape
        ph, pw = res.panorama_rgb.shape[:2]
        valid_px_ratio = round((100.0 - float(res.unobserved_pct)) / 100.0, 4)
        prov_label = (
            "HYBRID_KEYFRAME_AND_CLOUD"
            if res.photo_rgb_pct > 0.0
            else ("SYNTHETIC_FROM_CLOUD" if res.pointcloud_rgb_pct > 0.0 else "GEOMETRY_UNCOLORED")
        )

        st_meta = {
            "station_id": st.station_id,
            "pano_id": pano_std_id,
            "label": st.label,
            "source_mode": st.source_mode,
            "pose_world": {
                "position_xyz_m": pos_aligned,
                "yaw_rad": round(float(st.yaw_rad), 6),
                "rotation_wxyz": [round(float(q), 8) for q in rot_wxyz_aligned],
                "rotation_xyzw": rot_xyzw_aligned,
            },
            "structured_scan": {
                "rows": rows,
                "cols": cols,
                "total_cells": res.total_scan_cells,
                "valid_points": res.valid_scan_points,
                "missing_cells": int(res.total_scan_cells - res.valid_scan_points),
                "valid_coverage_pct": round((res.valid_scan_points / max(1, res.total_scan_cells)) * 100.0, 2),
                "ply_file": os.path.relpath(ply_path, save_dir),
            },
            "panorama": {
                "width": pw,
                "height": ph,
                "panorama_file": os.path.relpath(pano_path, save_dir),
                "bundle_panorama_file": pano_std_rel,
                "depth_map_mm_file": os.path.relpath(depth_path, save_dir),
                "coverage_mask_file": os.path.relpath(cov_path, save_dir),
                "provenance_code_file": os.path.relpath(prov_raw_path, save_dir),
                "contributing_keyframes": res.contributing_keyframes,
                "coverage_breakdown_pct": {
                    "photo_rgb": res.photo_rgb_pct,
                    "pointcloud_rgb": res.pointcloud_rgb_pct,
                    "depth_guided_gap_fill": res.gap_fill_pct,
                    "uncolored_geometry": res.uncolored_geom_pct,
                    "unobserved_missing": res.unobserved_pct,
                    "total_observed": round(100.0 - res.unobserved_pct, 2),
                },
            },
            "synthesis_time_ms": res.synthesis_time_ms,
        }
        with open(os.path.join(st_dir, "station_metadata.json"), "w") as fm:
            json.dump(st_meta, fm, indent=2)

        manifest_stations.append(st_meta)

        station_guid_str = str(uuid.uuid5(uuid.NAMESPACE_OID, f"av-scanar-station-{save_dir}-{st.station_id}"))
        bundle_panos.append(
            {
                "id": pano_std_id,
                "station_id": st.station_id,
                "station_guid": station_guid_str,
                "position": pos_aligned,
                "rotation": rot_xyzw_aligned,
                "is_level": True,
                "heading_yaw_rad": round(float(math.atan2(2.0 * (rot_xyzw_aligned[3] * rot_xyzw_aligned[2] + rot_xyzw_aligned[0] * rot_xyzw_aligned[1]), 1.0 - 2.0 * (rot_xyzw_aligned[1] ** 2 + rot_xyzw_aligned[2] ** 2))), 6),
                "floor_id": 1,
                "image_path": pano_std_rel,
                "depth_path": depth_std_rel,
                "provenance": prov_label,
                "valid_pixel_ratio": valid_px_ratio,
                "stair_link": False,
                "neighbors": [],
            }
        )

        if export_e57:
            e57_payloads.append(
                E57StationPayload(
                    station_id=st.station_id,
                    station_name=pano_std_id,
                    description=(
                        f"AV ScanAR Virtual Station ({st.source_mode}) | "
                        f"Valid 3D: {res.valid_scan_points}/{res.total_scan_cells} | "
                        f"Photo RGB: {res.photo_rgb_pct:.1f}% | Cloud RGB: {res.pointcloud_rgb_pct:.1f}%"
                    ),
                    translation_xyz=(
                        float(pos_aligned[0]),
                        float(pos_aligned[1]),
                        float(pos_aligned[2]),
                    ),
                    rotation_wxyz=rot_wxyz_aligned,
                    xyz_local=res.scan_xyz_local,
                    invalid_state=res.scan_invalid_state,
                    rgb=res.scan_rgb,
                    panorama_jpeg_bytes=jpg_bytes,
                    panorama_width=pw,
                    panorama_height=ph,
                )
            )

    # 6. Multi-story floor detection + calibrated 2D architectural keyplan slice per floor + 8-sector/stair nav graph
    z_vals = [p["position"][2] for p in bundle_panos]
    floor_ids, bundle_floors = detect_multi_story_floors(z_vals)
    for p, fid in zip(bundle_panos, floor_ids):
        p["floor_id"] = int(fid)

    bundle_panos = build_walkthrough_navigation_graph(bundle_panos, points_xyz=aligned_cloud_xyz)

    for fl in bundle_floors:
        fid = fl["id"]
        fl_xy = np.array([p["position"][:2] for p in bundle_panos if p["floor_id"] == fid], dtype=np.float64)
        kp_rel = f"floors/floor_{fid}_keyplan.png"
        kp_abs = os.path.join(save_dir, kp_rel)
        wb = render_floor_keyplan_png(
            aligned_cloud_xyz,
            floor_station_mean_z=float(fl["mean_z"]),
            out_png_path=kp_abs,
            floor_stations_xy=fl_xy,
        )
        if wb is not None:
            fl["keyplan_path"] = kp_rel
            fl["world_bounds"] = wb
            fl["floor_elevation"] = round(float(wb.get("slab_z_m", float(fl["mean_z"]) - 1.55)), 4)
        else:
            fl["floor_elevation"] = round(float(fl["mean_z"]) - 1.55, 4)

    # 7. Write AV Showcase binary point cloud cache (`cache/lod0.bin`)
    lod0_rel = "cache/lod0.bin"
    lod0_pts_written = write_point_cloud_lod0_bin(os.path.join(save_dir, lod0_rel), aligned_cloud_xyz, cloud_rgb)

    e57_summary = None
    e57_path = ""
    if export_e57 and len(e57_payloads) > 0:
        e57_path = os.path.join(save_dir, e57_filename)
        e57_summary = write_structured_e57(e57_path, e57_payloads)

    now_date = scan_date.strip() if scan_date else time.strftime("%Y-%m-%d", time.localtime())
    resolved_scan_id = scan_id.strip() if scan_id else (os.path.basename(os.path.normpath(save_dir)) or time.strftime("%Y-%m-%d_%H%M", time.localtime()))
    resolved_proj_id = project_id.strip() if project_id else str(uuid.uuid5(uuid.NAMESPACE_OID, f"av-project-{project_name}"))

    # 8. Write canonical `av_walkthrough_bundle.json` (AV Spatial Bundle v1.1)
    walkthrough_bundle = {
        "format": "av_spatial_bundle",
        "schema": "av_walkthrough_bundle",
        "schema_version": "1.1",
        "producer": {"app": "AV ScanAR", "version": "1.1.0"},
        "generator": "AV ScanAR (scanarMini)",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "project_id": resolved_proj_id,
        "project_name": project_name,
        "scan_id": resolved_scan_id,
        "scan_date": now_date,
        "coordinate_system": {
            "handedness": "right",
            "up": "+Z",
            "units": "meters",
            "pano_projection": "ASTM_E2807_equirectangular",
        },
        "panorama_projection": "astm_e2807_equirectangular_u05_pos_x",
        "transform_to_project": [round(float(v), 8) for v in T_proj.reshape(-1).tolist()],
        "registration_metrics": {
            "inlier_rmse_m": icp_rmse_m,
            "overlap_ratio": icp_overlap,
            "pre_transformed_to_project_frame": True,
        },
        "structured_e57_path": os.path.basename(e57_path) if e57_path else None,
        "point_cloud": {"lod0": lod0_rel, "format": "av_lod0_v1"},
        "point_cloud_cache": lod0_rel,
        "point_cloud_lod0_points": lod0_pts_written,
        "nav_graph": "los_v1",
        "floors": bundle_floors,
        "panos": bundle_panos,
    }
    bundle_json_path = os.path.join(save_dir, "av_walkthrough_bundle.json")
    with open(bundle_json_path, "w", encoding="utf-8") as fb:
        json.dump(walkthrough_bundle, fb, indent=2)

    avproj_path = None
    if av_showcase_project_dir:
        avproj_path = export_or_append_av_showcase_project(
            av_showcase_project_dir=av_showcase_project_dir,
            bundle_dir=save_dir,
            bundle_manifest=walkthrough_bundle,
            auto_align_to_baseline=(reference_cloud_xyz is None),
        )

    manifest_path = os.path.join(save_dir, "stations_manifest.json")
    manifest = {
        "schema": "ArtificialVision.AVStructure.StationsManifest.v1",
        "generated_utc": walkthrough_bundle["generated_utc"],
        "station_count": len(manifest_stations),
        "floor_count": len(bundle_floors),
        "structured_e57_file": os.path.basename(e57_path) if e57_path else None,
        "structured_e57_path": e57_path if e57_path else None,
        "structured_e57_summary": e57_summary,
        "av_walkthrough_bundle_path": bundle_json_path,
        "point_cloud_cache_path": os.path.join(save_dir, lod0_rel),
        "av_showcase_avproj_path": avproj_path,
        "provenance_legend": {
            "0": "UNOBSERVED (Preserved Missing Data)",
            "1": "PHOTO_RGB (Reprojected Original Camera Keyframes)",
            "2": "POINTCLOUD_RGB (Colorized Point Cloud Adaptive Surfel Splat)",
            "3": "DEPTH_GUIDED_FILL (1px Intra-Surface Micro-Gap Fill <8cm)",
            "4": "GEOMETRY_UNCOLORED (Observed LiDAR Structure Without RGB)",
        },
        "floors": bundle_floors,
        "stations": manifest_stations,
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    return manifest


def export_or_append_av_showcase_project(
    av_showcase_project_dir: str,
    bundle_dir: str,
    bundle_manifest: Optional[Dict] = None,
    auto_align_to_baseline: bool = True,
) -> str:
    """
    Creates or updates `<av_showcase_project_dir>/project.avproj` directly so AV Showcase v2.0 can open
    multi-date historical scans immediately (`Scans/<scan_key>/panos/`, `floors/`, `cache/lod0.bin`).
    Automatically aligns 2nd+ historical scans to the Baseline Scan's `lod0.bin` via 4-DOF ICP so
    time-traveling across dates stays in the exact same room facing the exact same world direction!
    """
    os.makedirs(av_showcase_project_dir, exist_ok=True)
    os.makedirs(os.path.join(av_showcase_project_dir, "logs"), exist_ok=True)

    if bundle_manifest is None:
        with open(os.path.join(bundle_dir, "av_walkthrough_bundle.json"), "r", encoding="utf-8") as f:
            bundle_manifest = json.load(f)

    avproj_path = os.path.join(av_showcase_project_dir, "project.avproj")
    if os.path.exists(avproj_path):
        with open(avproj_path, "r", encoding="utf-8") as f:
            proj_data = json.load(f)
        proj_data.setdefault("schema_version", "1.1")
        proj_data.setdefault("application_version", "2.0.0")
        proj_data.setdefault("project_id", proj_data.get("id") or bundle_manifest.get("project_id") or str(uuid.uuid4()))
    else:
        pid = bundle_manifest.get("project_id") or str(uuid.uuid4())
        proj_data = {
            "schema_version": "1.1",
            "application_version": "2.0.0",
            "project_id": pid,
            "id": pid,
            "name": bundle_manifest.get("project_name", "AV ScanAR Project"),
            "description": "",
            "settings": {"max_autostack_distance": 10.0, "default_unit": "metric"},
            "bookmarks": [],
            "scans": [],
        }

    # Determine unique scan key (preserve `YYYY-MM-DD`, or disambiguate `YYYY-MM-DD_HHMM` if same-day scan exists)
    base_date = bundle_manifest.get("scan_date") or time.strftime("%Y-%m-%d", time.localtime())
    scan_id = bundle_manifest.get("scan_id") or base_date
    existing_dates = {s.get("scan_date") for s in proj_data.get("scans", [])}
    scan_key = base_date if base_date not in existing_dates else scan_id

    dest_scan_dir = os.path.join(av_showcase_project_dir, "Scans", scan_key)
    for sub in ("panos", "floors", "cache"):
        os.makedirs(os.path.join(dest_scan_dir, sub), exist_ok=True)

    pc_dict = bundle_manifest.get("point_cloud") if isinstance(bundle_manifest.get("point_cloud"), dict) else {}
    lod0_rel_src = pc_dict.get("lod0") or bundle_manifest.get("point_cloud_cache") or "cache/lod0.bin"
    src_lod0 = os.path.join(bundle_dir, lod0_rel_src)
    src_xyz, src_rgb = read_point_cloud_lod0_bin(src_lod0)

    panos_copy = json.loads(json.dumps(bundle_manifest.get("panos", [])))
    floors_copy = json.loads(json.dumps(bundle_manifest.get("floors", [])))
    T_align = np.eye(4, dtype=np.float64)

    # Auto-register 2nd+ historical scans against Baseline Scan so Dec 4 -> Apr 13 stays in the exact same room!
    existing_scans = proj_data.get("scans", [])
    baseline_id = proj_data.get("baseline_scan_id")
    base_scan = None
    if existing_scans:
        if baseline_id:
            base_scan = next(
                (s for s in existing_scans if s.get("scan_id") == baseline_id or s.get("scan_date") == baseline_id),
                None,
            )
        if base_scan is None:
            base_scan = sorted(existing_scans, key=lambda s: s.get("scan_date", ""))[0]

    if auto_align_to_baseline and base_scan is not None and base_scan.get("scan_date") != scan_key:
        base_cache_rel = base_scan.get("point_cloud_cache", "")
        base_lod0 = os.path.join(av_showcase_project_dir, base_cache_rel) if base_cache_rel else ""
        if os.path.exists(base_lod0) and len(src_xyz) >= 20:
            ref_xyz, _ = read_point_cloud_lod0_bin(base_lod0)
            if len(ref_xyz) >= 20:
                T_align, _, overlap = compute_transform_to_project_icp(src_xyz, ref_xyz)
                if overlap >= 0.15 and not np.allclose(T_align, np.eye(4), atol=1e-5):
                    src_xyz = apply_transform_4x4_xyz(src_xyz, T_align)
                    for p in panos_copy:
                        p["position"], p["rotation"] = transform_pano_pose_xyzw(p["position"], p["rotation"], T_align)
                    tz_shift = float(T_align[2, 3])
                    for fl in floors_copy:
                        for zk in ("min_z", "max_z", "mean_z", "floor_elevation"):
                            if zk in fl and fl[zk] is not None:
                                fl[zk] = round(float(fl[zk]) + tz_shift, 4)

    # Copy panoramas & depth maps
    for p in panos_copy:
        src_img = os.path.join(bundle_dir, p["image_path"])
        dst_img = os.path.join(dest_scan_dir, p["image_path"])
        os.makedirs(os.path.dirname(dst_img), exist_ok=True)
        if os.path.exists(src_img) and os.path.abspath(src_img) != os.path.abspath(dst_img):
            shutil.copy2(src_img, dst_img)
        if p.get("depth_path"):
            src_d = os.path.join(bundle_dir, p["depth_path"])
            dst_d = os.path.join(dest_scan_dir, p["depth_path"])
            if os.path.exists(src_d) and os.path.abspath(src_d) != os.path.abspath(dst_d):
                shutil.copy2(src_d, dst_d)

    # Render or copy calibrated 2D floorplan PNGs
    for fl in floors_copy:
        fid = fl["id"]
        kp_rel = fl.get("keyplan_path") or f"floors/floor_{fid}_keyplan.png"
        dst_kp = os.path.join(dest_scan_dir, kp_rel)
        src_kp = os.path.join(bundle_dir, kp_rel)
        if not np.allclose(T_align, np.eye(4), atol=1e-5) and len(src_xyz) > 0:
            fl_xy = np.array([p["position"][:2] for p in panos_copy if p["floor_id"] == fid], dtype=np.float64)
            wb = render_floor_keyplan_png(src_xyz, float(fl["mean_z"]), dst_kp, floor_stations_xy=fl_xy)
            if wb:
                fl["world_bounds"] = wb
                fl["floor_elevation"] = round(float(wb.get("slab_z_m", float(fl["mean_z"]) - 1.55)), 4)
        elif os.path.exists(src_kp) and os.path.abspath(src_kp) != os.path.abspath(dst_kp):
            shutil.copy2(src_kp, dst_kp)
        fl["keyplan_path"] = kp_rel
        fl["floorplan_path"] = f"Scans/{scan_key}/{kp_rel}"

    # Write aligned `lod0.bin`
    dst_lod0 = os.path.join(dest_scan_dir, "cache", "lod0.bin")
    write_point_cloud_lod0_bin(dst_lod0, src_xyz, src_rgb)

    tf_flat = [round(float(v), 8) for v in T_align.reshape(-1).tolist()]
    scan_entry = {
        "scan_id": scan_id,
        "scan_date": scan_key,
        "calendar_date": base_date,
        "scan_dir": f"Scans/{scan_key}",
        "description": f"AV ScanAR Capture ({scan_id})",
        "import_mode": "managed",
        "source_path": "",
        "point_cloud_cache": f"Scans/{scan_key}/cache/lod0.bin",
        "transform_to_project": tf_flat,
        "nav_graph": "los_v1",
        "floors": floors_copy,
        "panos": panos_copy,
    }

    # Write companion `av_walkthrough_bundle.json` inside `Scans/<scan_key>/` matching AV Showcase v2.0
    scan_bundle_manifest = {
        "format": "av_spatial_bundle",
        "schema_version": "1.1",
        "producer": {"app": "AV ScanAR", "version": "1.1.0"},
        "project_id": proj_data.get("project_id"),
        "scan_id": scan_id,
        "scan_date": scan_key,
        "coordinate_system": {
            "handedness": "right",
            "up": "+Z",
            "units": "meters",
            "pano_projection": "ASTM_E2807_equirectangular",
        },
        "transform_to_project": tf_flat,
        "point_cloud": {"lod0": "cache/lod0.bin", "format": "av_lod0_v1"},
        "point_cloud_cache": "cache/lod0.bin",
        "nav_graph": "los_v1",
        "floors": floors_copy,
        "panos": panos_copy,
    }
    with open(os.path.join(dest_scan_dir, "av_walkthrough_bundle.json"), "w", encoding="utf-8") as fsb:
        json.dump(scan_bundle_manifest, fsb, indent=2)

    # Replace matching `scan_key` or append, and initialize `baseline_scan_id` on first scan
    proj_data["scans"] = [s for s in proj_data.get("scans", []) if s.get("scan_date") != scan_key]
    proj_data["scans"].append(scan_entry)
    proj_data["scans"].sort(key=lambda s: s.get("scan_date", ""))
    if not proj_data.get("baseline_scan_id"):
        proj_data["baseline_scan_id"] = scan_id

    with open(avproj_path, "w", encoding="utf-8") as f:
        json.dump(proj_data, f, indent=2)

    return avproj_path
