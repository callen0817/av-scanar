#!/usr/bin/env python3
"""
scanHUB Reality Capture OS v2.0 (Milestone M12.9 — GUI Observability & Multi-Modal Diagnostics)

Primary Desktop & Field UI Interface for scanarMini Rig.
Layout:
- Left / Center Viewport (60%): Clear Passthrough Viewport (#000000 pure black for AR Glasses).
- Right Peripheral HUD (40%):
  1. Isolated Dual Camera Streams (Clean RGB Left | Pure Depth Heatmap Right).
  2. Multi-Layer Point Cloud Map (Airy, Gemini, Fused, Colorized, Accept/Reject, Coverage Diag).
  3. Multi-Factor Gemini Gating & Range Sliders (0.3–4.0m, Surface Agreement).
  4. Telemetry & Structural Diagnostics (RGB Coverage %, Dual %, Gemini Stats, LiDAR Floor/Wall/Ceiling %).
  5. Telemetry Console Log.
"""

import faulthandler
faulthandler.enable()

import sys
import os
import time
import math
import json
import re
import cv2
import struct
import subprocess
import threading
import numpy as np
from datetime import datetime
from collections import deque
from typing import Optional, Dict, List, Tuple, Any
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

# Default ROS 2 domain and FastDDS Shared Memory RMW (bypasses 212KB UDP rmem_max for 3MB RGB & PointCloud2)
if "ROS_DOMAIN_ID" not in os.environ:
    os.environ["ROS_DOMAIN_ID"] = "0"
os.environ["RMW_IMPLEMENTATION"] = "rmw_fastrtps_cpp"

# PyQt5 / PyQt6 imports
try:
    from PyQt6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, 
                                 QPushButton, QLabel, QLineEdit, QComboBox, QTextEdit, 
                                 QStatusBar, QProgressBar, QMessageBox, QInputDialog,
                                 QFrame, QSplitter, QDialog, QSlider, QCheckBox, QFormLayout,
                                 QButtonGroup, QRadioButton, QGroupBox, QTabWidget,
                                 QDoubleSpinBox, QSpinBox, QSizeGrip, QStackedWidget,
                                 QListWidget, QListWidgetItem)
    from PyQt6.QtCore import QProcess, QTimer, Qt, QThread, pyqtSignal, QPointF, QPoint, QRect, QRectF
    from PyQt6.QtGui import (QFont, QPalette, QColor, QPainter, QPen, QBrush, QPolygonF,
                             QPolygon, QImage, QPixmap, QIcon, QLinearGradient, QRadialGradient, QPainterPath)
except ImportError:
    from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, 
                                 QPushButton, QLabel, QLineEdit, QComboBox, QTextEdit, 
                                 QStatusBar, QProgressBar, QMessageBox, QInputDialog,
                                 QFrame, QSplitter, QDialog, QSlider, QCheckBox, QFormLayout,
                                 QButtonGroup, QRadioButton, QGroupBox, QTabWidget,
                                 QDoubleSpinBox, QSpinBox, QSizeGrip, QStackedWidget,
                                 QListWidget, QListWidgetItem)
    from PyQt5.QtCore import QProcess, QTimer, Qt, QThread, pyqtSignal, QPointF, QPoint, QRect, QRectF
    from PyQt5.QtGui import (QFont, QPalette, QColor, QPainter, QPen, QBrush, QPolygonF,
                             QPolygon, QImage, QPixmap, QIcon, QLinearGradient, QRadialGradient, QPainterPath)


# ROS 2 Python imports (gracefully optional for standalone M12 / C-SLAM mode)
try:
    import rclpy
    from rclpy.node import Node
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.qos import qos_profile_sensor_data, QoSProfile, ReliabilityPolicy, HistoryPolicy
    from sensor_msgs.msg import Image, PointCloud2, Imu, NavSatFix, CameraInfo
    from nav_msgs.msg import Odometry
    from std_msgs.msg import String
    from cv_bridge import CvBridge
    HAS_ROS2 = True
except ImportError:
    rclpy = None
    SingleThreadedExecutor = None
    CvBridge = None
    HAS_ROS2 = False

# Ensure local gui directory is on sys.path for direct module imports
_GUI_DIR = os.path.dirname(os.path.abspath(__file__))
if _GUI_DIR not in sys.path:
    sys.path.insert(0, _GUI_DIR)

from m12_point_cloud_cleanup import (
    PointCloudProcessor,
    CleanupParams,
    ProcessingStats,
    ExportFormat,
    ExportResult,
)
from cslam_network_engine import (
    CSlamNetworkEngine,
    CSlamColorMode,
    PeerAlignmentState,
    get_scanner_color_hex,
    get_scanner_color_rgb,
)

# Import local scanHUB engine modules
try:
    from storage_monitor import StorageMonitor
    from sensor_monitor import SensorMonitor
    from diagnostic_recorder import DiagnosticRecorder
except ImportError:
    try:
        from .storage_monitor import StorageMonitor
        from .sensor_monitor import SensorMonitor
        from .diagnostic_recorder import DiagnosticRecorder
    except ImportError:
        StorageMonitor = None
        SensorMonitor = None
        DiagnosticRecorder = None


# ==============================================================================
# RIGID EXTRINSICS & GEOMETRIC TRANSFORMS (AUTHORITATIVE PRODUCTION PROFILE)
# ==============================================================================
try:
    import yaml as _yaml
except ImportError:
    _yaml = None

# Default Fallbacks:
# 1. IMU from LiDAR Frame (Factory DIFOP Calibration & robosense.yaml)
R_IMU_LIDAR = np.array([
    [ 0.012366, -0.999888,  0.008426],
    [-0.999919, -0.012340,  0.003184],
    [-0.003079, -0.008465, -0.999959]
], dtype=np.float64)
t_IMU_LIDAR = np.array([0.004164, 0.004315, -0.004411], dtype=np.float64)

# 2. Camera Depth Optical Frame (Left IR) from IMU Frame (production_extrinsics.yaml active_transforms.T_cam0_airy_imu)
# p_cam0 = T_cam0_airy_imu * p_imu
R_CAM_IMU = np.array([
    [ 0.999944, -0.003490, -0.010008],
    [ 0.004480, -0.716562,  0.697509],
    [-0.009605, -0.697515, -0.716506]
], dtype=np.float64)
t_CAM_IMU = np.array([0.039251, -0.056910, -0.030089], dtype=np.float64)

# Inverse: IMU from Camera Depth Optical Frame
# p_imu = R_CAM_IMU.T @ (p_cam0 - t_CAM_IMU) = R_CAM_IMU.T @ p_cam0 - R_CAM_IMU.T @ t_CAM_IMU
R_IMU_CAM_DEPTH = R_CAM_IMU.T
t_IMU_CAM_DEPTH = -R_CAM_IMU.T @ t_CAM_IMU

# 3. RGB Optical Frame from IMU Frame (production_extrinsics.yaml active_transforms.T_rgb_airy_imu)
# p_rgb = T_rgb_airy_imu * p_imu
R_RGB_IMU = np.array([
    [ 0.999936, -0.002759, -0.010962],
    [ 0.005675, -0.716106,  0.697968],
    [-0.009776, -0.697985, -0.716045]
], dtype=np.float64)
t_RGB_IMU = np.array([0.063044, -0.056937, -0.030558], dtype=np.float64)

# 4. RGB Optical Frame from Camera Depth Optical Frame (Factory EEPROM T_rgb_depth)
R_RGB_CAM_DEPTH = np.array([
    [ 0.999999, -0.001189,  0.000174],
    [ 0.001189,  0.999999, -0.000658],
    [-0.000174,  0.000658,  0.999999]
], dtype=np.float64)
t_RGB_CAM_DEPTH = np.array([0.0237306, -0.0000932, -0.0004246], dtype=np.float64)

# Camera forward optical axis (+Z in camera frame) expressed in IMU frame
CAM_FWD_AXIS_IMU = R_IMU_CAM_DEPTH @ np.array([0.0, 0.0, 1.0], dtype=np.float64)

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
            if "active_transforms" in _pe_cfg and "T_cam0_airy_imu" in _pe_cfg["active_transforms"]:
                _T_c0_imu = np.array(_pe_cfg["active_transforms"]["T_cam0_airy_imu"]["matrix"], dtype=np.float64)
                R_CAM_IMU = _T_c0_imu[:3, :3]
                t_CAM_IMU = _T_c0_imu[:3, 3]
                R_IMU_CAM_DEPTH = R_CAM_IMU.T
                t_IMU_CAM_DEPTH = -R_CAM_IMU.T @ t_CAM_IMU
                CAM_FWD_AXIS_IMU = R_IMU_CAM_DEPTH @ np.array([0.0, 0.0, 1.0], dtype=np.float64)

            if "active_transforms" in _pe_cfg and "T_rgb_airy_imu" in _pe_cfg["active_transforms"]:
                _T_rgb_imu = np.array(_pe_cfg["active_transforms"]["T_rgb_airy_imu"]["matrix"], dtype=np.float64)
                R_RGB_IMU = _T_rgb_imu[:3, :3]
                t_RGB_IMU = _T_rgb_imu[:3, 3]

            if "factory_transforms" in _pe_cfg and "T_rgb_depth" in _pe_cfg["factory_transforms"]:
                _T_rgb_d = np.array(_pe_cfg["factory_transforms"]["T_rgb_depth"]["matrix"], dtype=np.float64)
                R_RGB_CAM_DEPTH = _T_rgb_d[:3, :3]
                t_RGB_CAM_DEPTH = _T_rgb_d[:3, 3]
    except Exception:
        pass

# Backward compatibility transforms
R_CAM_LIDAR = R_CAM_IMU @ R_IMU_LIDAR
t_CAM_LIDAR = R_CAM_IMU @ t_IMU_LIDAR + t_CAM_IMU
R_LIDAR_CAM = R_CAM_LIDAR.T
t_LIDAR_CAM = -R_CAM_LIDAR.T @ t_CAM_LIDAR
R_RGB_LIDAR = R_RGB_IMU @ R_IMU_LIDAR
t_RGB_LIDAR = R_RGB_IMU @ t_IMU_LIDAR + t_RGB_IMU


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
    safe_z = np.where(zr > 0.15, zr, 1.0)
    xn = pts_rgb[:, 0] / safe_z
    yn = pts_rgb[:, 1] / safe_z
    # Pinhole frustum gate: prevent 6th-order radial polynomial (k3 < 0) from folding
    # wide-angle 360-deg LiDAR points (|xn| > 1.12 or |yn| > 0.72) back into the image plane.
    in_frustum = (zr > 0.15) & (np.abs(xn) <= 1.12) & (np.abs(yn) <= 0.72)
    xn_c = np.where(in_frustum, xn, 0.0)
    yn_c = np.where(in_frustum, yn, 0.0)
    r2 = xn_c * xn_c + yn_c * yn_c
    r4 = r2 * r2
    r6 = r4 * r2
    k1, k2, p1, p2, k3 = d_coeffs[:5]
    radial = 1.0 + k1 * r2 + k2 * r4 + k3 * r6
    xd = xn_c * radial + 2.0 * p1 * xn_c * yn_c + p2 * (r2 + 2.0 * xn_c * xn_c)
    yd = yn_c * radial + p1 * (r2 + 2.0 * yn_c * yn_c) + 2.0 * p2 * xn_c * yn_c
    u = np.where(in_frustum, np.round(fx * xd + cx), -1.0).astype(np.int32)
    v = np.where(in_frustum, np.round(fy * yd + cy), -1.0).astype(np.int32)
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


# ==============================================================================
# ARTIFICIAL VISION — APPLE GLASS (visionOS) WHITE & GOLD AR DESIGN SYSTEM
# Matched to https://prevdc.artificialviz.com/auth
# ==============================================================================
AV_LOGO_CANDIDATES = [
    os.path.join(_GUI_DIR, "assets", "av_logo.png"),
    "/home/scanar/scanarMini/src/scanhub_gui/assets/av_logo.png",
    "/home/scanar/scanarMini/gui/av_logo.png",
]

AV_THEME = {
    "gold_primary": "#BAA164",     # oklch(71.4455% .0838495 87.1018) — Artificial Vision Primary Gold
    "gold_bright": "#E5C983",      # Specular luminous gold highlight for AR glasses legibility
    "gold_muted": "#D8BF7C",       # Warm champagne gold for mono labels
    "white_pure": "#FFFFFF",       # Pure specular white
    "white_warm": "#FCFBF8",       # Warm alabaster white (prevdc foreground token)
    "white_muted": "#D0CCC2",      # Secondary translucent white-ivory text
    "glass_fill": "rgba(255, 255, 255, 0.06)",
    "glass_gold_fill": "rgba(186, 161, 100, 0.14)",
    "glass_border_white": "rgba(255, 255, 255, 0.24)",
    "glass_border_gold": "rgba(186, 161, 100, 0.52)",
    "status_ok": "#22C55E",        # oklch(62% .15 150) — Emerald
    "status_warn": "#F59E0B",      # Amber gold
    "status_err": "#EF4444",       # Coral red
}


def get_av_logo_path() -> str:
    for p in AV_LOGO_CANDIDATES:
        if os.path.isfile(p):
            return p
    return AV_LOGO_CANDIDATES[0]


def load_av_logo_pixmap(size_px: int = 120) -> QPixmap:
    path = get_av_logo_path()
    pix = QPixmap(path)
    if not pix.isNull() and size_px > 0:
        aspect = Qt.AspectRatioMode.KeepAspectRatio if hasattr(Qt, "AspectRatioMode") else Qt.KeepAspectRatio
        smooth = Qt.TransformationMode.SmoothTransformation if hasattr(Qt, "TransformationMode") else Qt.SmoothTransformation
        return pix.scaled(size_px, size_px, aspect, smooth)
    return pix


class ArtificialVisionSplashScreen(QWidget):
    """
    Apple Glass (visionOS) Translucent White & Gold Splash Screen featuring the
    Artificial Vision Monogram Logo and https://prevdc.artificialviz.com/auth brand identity.
    Uses WA_TranslucentBackground + FramelessWindowHint for AR glasses optical visibility.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        flags = (
            (Qt.WindowType.SplashScreen if hasattr(Qt, "WindowType") else Qt.SplashScreen)
            | (Qt.WindowType.FramelessWindowHint if hasattr(Qt, "WindowType") else Qt.FramelessWindowHint)
            | (Qt.WindowType.WindowStaysOnTopHint if hasattr(Qt, "WindowType") else Qt.WindowStaysOnTopHint)
        )
        self.setWindowFlags(flags)
        if hasattr(Qt, "WidgetAttribute"):
            self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        else:
            self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setFixedSize(560, 380)
        self.progress_pct = 0
        self.status_text = "Initializing Artificial Vision Spatial Core..."
        self.logo_pixmap = load_av_logo_pixmap(118)

        # Transparent Apple Glass Close Button in top-right of splash
        self.btn_splash_close = QPushButton("✕", self)
        self.btn_splash_close.setFixedSize(26, 26)
        self.btn_splash_close.setCursor(
            Qt.CursorShape.PointingHandCursor if hasattr(Qt, "CursorShape") else Qt.PointingHandCursor
        )
        self.btn_splash_close.setStyleSheet(
            "QPushButton { background: rgba(255, 255, 255, 0.08); color: #FCFBF8; "
            "border: 1px solid rgba(255, 255, 255, 0.25); border-radius: 13px; font-size: 10px; font-weight: bold; }"
            "QPushButton:hover { background: rgba(186, 161, 100, 0.28); border-color: #BAA164; color: #FFFFFF; }"
        )
        self.btn_splash_close.move(self.width() - 38, 12)
        self.btn_splash_close.clicked.connect(self.close)

        self._center_on_screen()

    def _center_on_screen(self):
        try:
            screen = QApplication.primaryScreen()
            if screen is not None:
                geo = screen.availableGeometry()
                self.move(
                    geo.x() + (geo.width() - self.width()) // 2,
                    geo.y() + (geo.height() - self.height()) // 2,
                )
        except Exception:
            pass

    def set_progress(self, pct: int, status_msg: str = ""):
        self.progress_pct = int(np.clip(pct, 0, 100))
        if status_msg:
            self.status_text = status_msg
        self.update()
        try:
            QApplication.processEvents()
        except Exception:
            pass

    def set_status(self, status_msg: str, pct: int = None):
        target_pct = self.progress_pct if pct is None else pct
        self.set_progress(target_pct, status_msg)

    def mousePressEvent(self, event):
        self.close()

    def paintEvent(self, event):
        painter = QPainter(self)
        try:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            if hasattr(QPainter.RenderHint, "SmoothPixmapTransform"):
                painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)

            w, h = self.width(), self.height()
            cx = w // 2

            # 0. Clear full rect to 100% transparent so corners and glass pass through
            comp_src = (
                QPainter.CompositionMode.CompositionMode_Source
                if hasattr(QPainter, "CompositionMode")
                else QPainter.CompositionMode_Source
            )
            comp_over = (
                QPainter.CompositionMode.CompositionMode_SourceOver
                if hasattr(QPainter, "CompositionMode")
                else QPainter.CompositionMode_SourceOver
            )
            painter.setCompositionMode(comp_src)
            painter.fillRect(0, 0, w, h, QColor(0, 0, 0, 0))
            painter.setCompositionMode(comp_over)

            # 1. Translucent Apple Glass Spatial Card (ultra-low alpha for AR walking visibility)
            glass_rect = QRectF(2.0, 2.0, w - 4.0, h - 4.0)
            glass_grad = QLinearGradient(0, 0, 0, h)
            glass_grad.setColorAt(0.0, QColor(255, 255, 255, 38))
            glass_grad.setColorAt(0.45, QColor(24, 24, 28, 44))
            glass_grad.setColorAt(1.0, QColor(186, 161, 100, 32))
            painter.setPen(QPen(QColor(186, 161, 100, 175), 1.5))
            painter.setBrush(QBrush(glass_grad))
            painter.drawRoundedRect(glass_rect, 20.0, 20.0)

            # Inner Specular White Glass Hairline Rim (Apple visionOS glass refraction edge)
            inner_rect = QRectF(5.0, 5.0, w - 10.0, h - 10.0)
            painter.setPen(QPen(QColor(255, 255, 255, 65), 1.0))
            painter.setBrush(Qt.BrushStyle.NoBrush if hasattr(Qt, "BrushStyle") else Qt.NoBrush)
            painter.drawRoundedRect(inner_rect, 17.0, 17.0)

            # 2. Top Champagne Gold Specular Pill Highlight
            painter.setPen(Qt.PenStyle.NoPen if hasattr(Qt, "PenStyle") else Qt.NoPen)
            painter.setBrush(QColor("#BAA164"))
            painter.drawRoundedRect(QRectF(cx - 52, 2.0, 104.0, 3.0), 1.5, 1.5)

            # 3. Translucent Concentric LiDAR Horizon Rings behind Logo
            logo_cy = 112
            for r_px, alpha_v in [(84, 52), (66, 85)]:
                painter.setPen(QPen(QColor(186, 161, 100, alpha_v), 1.0, Qt.PenStyle.SolidLine))
                painter.setBrush(Qt.BrushStyle.NoBrush if hasattr(Qt, "BrushStyle") else Qt.NoBrush)
                painter.drawEllipse(QPointF(cx, logo_cy), r_px, r_px)

            # Soft radial champagne-gold optical halo
            halo = QRadialGradient(QPointF(cx, logo_cy), 68)
            halo.setColorAt(0.0, QColor(186, 161, 100, 62))
            halo.setColorAt(0.65, QColor(255, 255, 255, 18))
            halo.setColorAt(1.0, QColor(186, 161, 100, 0))
            painter.setPen(Qt.PenStyle.NoPen if hasattr(Qt, "PenStyle") else Qt.NoPen)
            painter.setBrush(QBrush(halo))
            painter.drawEllipse(QPointF(cx, logo_cy), 68, 68)

            # 4. Draw Copied Artificial Vision Gold Monogram Logo
            if not self.logo_pixmap.isNull():
                lw = self.logo_pixmap.width()
                lh = self.logo_pixmap.height()
                painter.drawPixmap(cx - lw // 2, logo_cy - lh // 2, self.logo_pixmap)

            # 5. Brand & Product Typography (Matched to https://prevdc.artificialviz.com/auth)
            align_center = Qt.AlignmentFlag.AlignCenter if hasattr(Qt, "AlignmentFlag") else Qt.AlignCenter

            painter.setPen(QColor("#E5C983"))
            f_brand = QFont("IBM Plex Mono", 8, QFont.Weight.Bold)
            f_brand.setLetterSpacing(
                QFont.SpacingType.AbsoluteSpacing if hasattr(QFont, "SpacingType") else QFont.AbsoluteSpacing,
                3.2,
            )
            painter.setFont(f_brand)
            painter.drawText(QRect(0, 196, w, 18), align_center, "VERIFY  ·  DIRECT  ·  CLARIFY")

            painter.setPen(QColor("#FCFBF8"))
            f_title = QFont("IBM Plex Sans", 24, QFont.Weight.DemiBold)
            painter.setFont(f_title)
            painter.drawText(QRect(0, 216, w, 34), align_center, "scanHUB")

            painter.setPen(QColor(252, 251, 248, 205))
            f_prod = QFont("IBM Plex Sans", 10)
            painter.setFont(f_prod)
            painter.drawText(
                QRect(0, 250, w, 20),
                align_center,
                "an Artificial Vision product",
            )

            # 6. Minimalist Champagne Gold Glass Progress Bar (4px height)
            bar_w = 340
            bar_h = 4
            bar_x = cx - bar_w // 2
            bar_y = 292
            painter.setPen(Qt.PenStyle.NoPen if hasattr(Qt, "PenStyle") else Qt.NoPen)
            painter.setBrush(QColor(255, 255, 255, 45))
            painter.drawRoundedRect(QRectF(bar_x, bar_y, bar_w, bar_h), 2.0, 2.0)
            fill_w = int(bar_w * (self.progress_pct / 100.0))
            if fill_w > 0:
                painter.setBrush(QColor("#BAA164"))
                painter.drawRoundedRect(QRectF(bar_x, bar_y, fill_w, bar_h), 2.0, 2.0)

            # 7. Status Caption
            painter.setPen(QColor("#FCFBF8"))
            painter.setFont(QFont("IBM Plex Sans", 8))
            painter.drawText(QRect(0, 308, w, 18), align_center, self.status_text)
        finally:
            painter.end()


class ClearPassthroughViewport(QFrame):
    """
    Main Left Spatial AR Passthrough Viewport (60%).
    Pure blank black (#000000) passthrough window with no grid, no 3D point cloud,
    and no data distractions.
    In optical see-through AR glasses, pure black emits zero light, turning off
    micro-OLED pixels in this region so the wearer sees completely unobstructed
    physical reality (hallways, stairs, doors) while walking.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self._vp_lock = threading.Lock()
        self.setStyleSheet(
            "background-color: #000000; border: 1px solid rgba(186, 161, 100, 0.42); border-radius: 12px;"
        )

    def update_scene_data(self, data: dict):
        """No-op: Blank passthrough window intentionally does not render distracting clouds."""
        pass

    def update_pose(self, x: float, y: float, yaw: float):
        """No-op: Blank passthrough window does not render spatial overlays."""
        pass

    def paintEvent(self, event):
        painter = QPainter(self)
        try:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            w, h = self.width(), self.height()

            # 1. Blank pure black field (#000000)
            # In optical see-through AR glasses, pure black = 0 emission (100% optical passthrough)
            painter.fillRect(0, 0, w, h, QColor(0, 0, 0))

            # 2. Subtle Champagne Gold border frame matching White & Gold Apple Glass aesthetic
            painter.setPen(QPen(QColor(186, 161, 100, 110), 1.0))
            painter.setBrush(Qt.BrushStyle.NoBrush if hasattr(Qt, "BrushStyle") else Qt.NoBrush)
            painter.drawRoundedRect(QRectF(0.5, 0.5, w - 1.0, h - 1.0), 12.0, 12.0)
        finally:
            painter.end()



class Ros2BridgeWorker(QThread):
    """
    Multi-Threaded ROS 2 Data Ingestion & Live Multi-Modal Processor.
    """
    image_received = pyqtSignal(QImage) # Live Clean RGB (Left) + Isolated Depth (Right)
    pose_received = pyqtSignal(float, float, float) # x, y, yaw
    rtk_status_received = pyqtSignal(int, str)
    sensor_rates_received = pyqtSignal(dict)
    motion_stability_received = pyqtSignal(bool, float)
    livo_hud_received = pyqtSignal(str, float, int, float)
    
    # NavVis Rotate-Before-Scan Ritual Signals
    rotation_progress_received = pyqtSignal(float)
    rotation_completed_received = pyqtSignal()

    # M12.9 Observability & Telemetry Signals
    map_points_updated = pyqtSignal(dict) # Sliced points for right-panel map
    telemetry_updated = pyqtSignal(dict)  # Live coverage %, counters, beam elevation breakdown

    def __init__(self, diag_recorder=None):
        super().__init__()
        self.running = True
        self.bridge = CvBridge() if CvBridge is not None else None
        self.diag_recorder = diag_recorder
        self.lock = threading.RLock()

        # NavVis 2-Stage Ritual Integration State
        self.in_rotation_ritual = False
        self.accumulated_rotation_deg = 0.0
        self.last_imu_gyro_time = None
        self.last_ax = 0.0
        self.last_ay = 0.0
        self.last_az = -9.81

        # Motion & Pose State
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_z = 0.0
        self.current_yaw = 0.0
        self.current_quat = [0.0, 0.0, 0.0, 1.0]
        self.current_lin_speed = 0.0
        self.current_ang_speed = 0.0
        self.using_slam_odom = False
        self.last_slam_time = 0.0
        self.last_lidar_time = 0.0
        self.last_rslidar_time = 0.0
        self.last_camera_time = 0.0
        self.last_depth_time = 0.0
        self.last_imu_msg_time = 0.0
        self.last_slam_msg_time = 0.0
        self.cnt_camera = 0
        self.cnt_lidar = 0
        self.cnt_rslidar = 0
        self.cnt_imu = 0
        self.cnt_slam = 0
        self.cnt_livo_diag = 0
        self.cnt_depth_gated_turn = 0
        self.stable_motion_start = None
        self._last_image_emit = 0.0
        self._last_depth_voxel_time = 0.0
        self._last_map_dispatch_time = 0.0
        self._cached_map_points_data = None
        self.ros_node = None
        self.camera_node = None
        self.slam_node = None
        self._preview_lock = threading.Lock()

        # Rolling timestamp queues for real-time Hz monitoring
        self._ts_rslidar = deque(maxlen=100)
        self._ts_lidar = deque(maxlen=100)
        self._ts_camera = deque(maxlen=150)
        self._ts_depth = deque(maxlen=150)
        self._ts_slam = deque(maxlen=100)

        # Camera Intrinsics
        self.fx_d = 411.833588
        self.fy_d = 411.833588
        self.cx_d = 423.666138
        self.cy_d = 240.990234

        self.fx_rgb = 609.006531
        self.fy_rgb = 609.015625
        self.cx_rgb = 643.868774
        self.cy_rgb = 399.326538
        self.d_rgb = D_RGB_DEFAULT.copy()

        # M12.9 Multi-Factor Gating Parameters (Configurable via GUI)
        self.depth_min_m = 0.30
        self.depth_max_m = 4.00
        self.surface_agreement_m = 0.050 # 50mm tolerance
        self.confidence_threshold = 0.50
        self.max_turn_rate_rad_s = 1.35  # ~77 deg/s angular velocity gate for stereo depth
        self.display_mode = "colorized_fused" # airy_only, stereo_only, fused, colorized_fused, accepted_rejected, observation_diag

        # Session Buffers & Point Stores
        self.is_session_recording = False
        self.session_start_epoch = 0.0
        self.session_duration = 0.0
        self.odom_buffer = deque(maxlen=4000)
        self.trajectory_records = []
        self.airy_sweeps = []
        self.gemini_frames = []
        self.rgb_keyframes = deque(maxlen=600)

        # Co-Temporal Ring Buffers & Non-Blocking Processing Queues
        self.rgb_ring = deque(maxlen=45)            # (sec_aligned, rgb_np) ~2.2s of decoded RGB history
        self._raw_rgb_queue = deque(maxlen=12)      # (sec, arr_time, w, h, step, enc, raw_bytes)
        self._raw_depth_queue = deque(maxlen=12)    # (sec, arr_time, w, h, step, enc, raw_bytes)
        self._latest_raw_depth_tuple = None         # Latest raw depth frame for zero-contention 22 FPS preview
        self._raw_lidar_queue = deque(maxlen=4)     # (sec, num_pts, step, raw_bytes)
        self._last_preview_rgb_stamp = 0.0
        self._last_preview_depth_stamp = 0.0
        self._rgb_seq = 0
        self._depth_seq = 0
        self._last_preview_rgb_seq = -1
        self._last_preview_depth_seq = -1
        self._slam_wall_to_stamp_offset = 0.0
        self._has_slam_clock_offset = False
        self._pred_yaw = 0.0
        self._last_imu_pred_time = 0.0
        self._last_imu_emit_time = 0.0

        # Real-Time Display Map Buffers (World Frame)
        self.live_airy_voxels = {}   # (gx, gy, gz) -> [x, y, z, r, g, b, obs_state]
        self.live_stereo_voxels = {} # (gx, gy, gz) -> [x, y, z, r, g, b, is_accepted]
        self._airy_cache_dirty = True
        self._stereo_cache_dirty = True
        self._airy_cached_arr = np.zeros((0, 7), dtype=np.float32)
        self._stereo_cached_arr = np.zeros((0, 7), dtype=np.float32)
        self.live_elevation_stats = {"floor": 0, "wall": 0, "ceiling": 0, "total": 0}

        # Latest Raw Camera Frames
        self.latest_color_raw = None
        self.latest_depth_raw = None
        self.latest_color_stamp = 0.0
        self.latest_depth_stamp = 0.0

        # Milestone 12: Point Cloud Cleanup (Default OFF during live scan for maximum real-time FPS; active in Post/Preview/Export)
        self.live_cleanup_enabled = False
        self.live_ground_lock_enabled = False
        self.live_cleanup_time_ms = 0.0
        self.m12_params = CleanupParams(
            enable_voxel_downsample=True,
            voxel_leaf_size_m=0.02,
            enable_sor=True,
            sor_mean_k=20,
            sor_std_mul=2.0,
            enable_ror=True,
            ror_radius_m=0.10,
            ror_min_neighbors=4,
            enable_ground_alignment=True,
            ground_dist_threshold_m=0.04,
            ground_max_iterations=400,
            ground_max_tilt_deg=35.0,
            align_ground_to_z_zero=True,
            remove_ground_plane=False,
        )
        self.cleanup_preview_active = False
        self.cleanup_preview_points = None  # (N, 6) float32 [x, y, z, r, g, b]
        self.pre_cleanup_backup_airy = None
        self.pre_cleanup_backup_stereo = None
        self.last_cleanup_stats = ProcessingStats()

        # Collaborative SLAM (C-SLAM) Unlimited Multi-Scanner Engine
        hostname = os.uname().nodename if hasattr(os, "uname") else "ScanHUB"
        self.cslam_engine = CSlamNetworkEngine(
            scanner_name=f"{hostname}-{os.getpid() % 1000:03d}",
            project_name="Default_Project",
            separation_distance_m=15.0,
            min_overlap_ratio=0.18,
        )
        self.cslam_engine.start()

    def set_gating_parameters(self, min_d, max_d, agree_tol, conf_thresh):
        with self.lock:
            self.depth_min_m = float(min_d)
            self.depth_max_m = float(max_d)
            self.surface_agreement_m = float(agree_tol)
            self.confidence_threshold = float(conf_thresh)

    def set_display_mode(self, mode_name):
        with self.lock:
            self.display_mode = mode_name

    def set_live_cleanup_enabled(self, enabled: bool):
        with self.lock:
            self.live_cleanup_enabled = bool(enabled)

    def set_live_ground_lock_enabled(self, enabled: bool):
        with self.lock:
            self.live_ground_lock_enabled = bool(enabled)

    def _get_voxel_arrays_locked(self):
        """Returns cached (N, 7) float32 arrays for airy and stereo voxels without redundant Python loops."""
        if self._airy_cache_dirty:
            if len(self.live_airy_voxels) > 0:
                self._airy_cached_arr = np.asarray(list(self.live_airy_voxels.values()), dtype=np.float32)
            else:
                self._airy_cached_arr = np.zeros((0, 7), dtype=np.float32)
            self._airy_cache_dirty = False
        if self._stereo_cache_dirty:
            if len(self.live_stereo_voxels) > 0:
                self._stereo_cached_arr = np.asarray(list(self.live_stereo_voxels.values()), dtype=np.float32)
            else:
                self._stereo_cached_arr = np.zeros((0, 7), dtype=np.float32)
            self._stereo_cache_dirty = False
        return self._airy_cached_arr, self._stereo_cached_arr

    def ensure_local_cloud_for_interactive_ops(self):
        """
        If no live ROS 2 hardware scan is in memory yet, loads the latest room scan or generates a
        structured 3D room with realistic noise/outliers so M12 Cleanup and C-SLAM can be used immediately.
        """
        with self.lock:
            if len(self.live_airy_voxels) > 200 or len(self.live_stereo_voxels) > 200:
                return
            rng = np.random.default_rng(42)
            # 1. Dense structured floor (64 x 50 = 3200 pts, ~6.2cm spacing -> 8 neighbors within 0.10m ROR radius)
            fx, fy = np.meshgrid(np.linspace(-2.0, 2.0, 64), np.linspace(-1.6, 1.6, 50))
            fx = fx.ravel() + rng.normal(0.0, 0.003, fx.size)
            fy = fy.ravel() + rng.normal(0.0, 0.003, fy.size)
            fz = rng.normal(0.0, 0.004, fx.size) + 0.04 * fx  # Slight 2.3 deg tilt for ground leveling
            floor_pts = np.column_stack([fx, fy, fz])

            # 2. Structured room walls (~6.2cm spacing)
            wx, wz = np.meshgrid(np.linspace(-2.0, 2.0, 64), np.linspace(0.06, 1.5, 24))
            w_north = np.column_stack([wx.ravel(), np.full(wx.size, 1.6) + rng.normal(0, 0.003, wx.size), wz.ravel()])
            w_south = np.column_stack([wx.ravel(), np.full(wx.size, -1.6) + rng.normal(0, 0.003, wx.size), wz.ravel()])

            wy, wz2 = np.meshgrid(np.linspace(-1.6, 1.6, 50), np.linspace(0.06, 1.5, 24))
            w_east = np.column_stack([np.full(wy.size, 2.0) + rng.normal(0, 0.003, wy.size), wy.ravel(), wz2.ravel()])
            w_west = np.column_stack([np.full(wy.size, -2.0) + rng.normal(0, 0.003, wy.size), wy.ravel(), wz2.ravel()])

            # 3. Add 180 isolated statistical/radius outliers
            outliers = rng.uniform(-5.5, 5.5, size=(180, 3))
            all_xyz = np.vstack([floor_pts, w_north, w_south, w_east, w_west, outliers])

            for i in range(len(all_xyz)):
                x, y, z = float(all_xyz[i, 0]), float(all_xyz[i, 1]), float(all_xyz[i, 2])
                gx, gy, gz = int(math.floor(x / 0.02)), int(math.floor(y / 0.02)), int(math.floor(z / 0.02))
                is_floor = abs(z) < 0.20
                r_c = 70 if is_floor else 229
                g_c = 160 if is_floor else 201
                b_c = 210 if is_floor else 131
                self.live_airy_voxels[(gx, gy, gz)] = [x, y, z, r_c, g_c, b_c, 1]
            self._airy_cache_dirty = True

    def extract_raw_3d_points_for_mode(self, mode: str, airy_vals=None, stereo_vals=None) -> np.ndarray:
        """
        Vectorized extraction of (N, 6) float32 array [x, y, z, r, g, b] (RGB 0..255) for the active display mode.
        Clearly distinguishes:
        - `airy_only`: Authoritative Airy 360° LiDAR in Artificial Vision Champagne Gold (229, 201, 131)
        - `stereo_only`: Accepted Orbbec Gemini 336L Stereo Depth in Electric Cyan (34, 211, 238)
        - `fused`: Structural Dual-Sensor Overlay — Airy LiDAR (Gold) + Gemini Stereo (Cyan)
        - `colorized_fused`: True RGB Camera Texture mapped onto 3D geometry (rendered on top of subtle structural slate context)
        """
        if airy_vals is not None and isinstance(airy_vals, np.ndarray):
            a_arr = airy_vals
        elif airy_vals is not None:
            a_arr = np.asarray(airy_vals, dtype=np.float32) if len(airy_vals) > 0 else np.zeros((0, 7), dtype=np.float32)
        else:
            with self.lock:
                a_arr, _ = self._get_voxel_arrays_locked()

        if stereo_vals is not None and isinstance(stereo_vals, np.ndarray):
            s_arr = stereo_vals
        elif stereo_vals is not None:
            s_arr = np.asarray(stereo_vals, dtype=np.float32) if len(stereo_vals) > 0 else np.zeros((0, 7), dtype=np.float32)
        else:
            with self.lock:
                _, s_arr = self._get_voxel_arrays_locked()

        parts = []
        if mode == "airy_only":
            if len(a_arr) > 0:
                out = np.empty((len(a_arr), 6), dtype=np.float32)
                out[:, :3] = a_arr[:, :3]
                out[:, 3] = 229.0
                out[:, 4] = 201.0
                out[:, 5] = 131.0
                parts.append(out)
        elif mode == "stereo_only":
            if len(s_arr) > 0:
                out = np.empty((len(s_arr), 6), dtype=np.float32)
                out[:, :3] = s_arr[:, :3]
                is_acc = s_arr[:, 6] > 0.5
                out[is_acc, 3:6] = [34.0, 211.0, 238.0]   # Accepted in vibrant Electric Cyan
                out[~is_acc, 3:6] = [45.0, 130.0, 150.0]  # Unaccepted/warming in subtle cyan-slate
                parts.append(out)
        elif mode == "fused":
            # Structural Sensor-Layer Comparison: Airy LiDAR = Champagne Gold, Gemini Stereo = Electric Cyan
            if len(a_arr) > 0:
                out_a = np.empty((len(a_arr), 6), dtype=np.float32)
                out_a[:, :3] = a_arr[:, :3]
                out_a[:, 3] = 229.0
                out_a[:, 4] = 201.0
                out_a[:, 5] = 131.0
                parts.append(out_a)
            if len(s_arr) > 0:
                out_s = np.empty((len(s_arr), 6), dtype=np.float32)
                out_s[:, :3] = s_arr[:, :3]
                is_acc = s_arr[:, 6] > 0.5
                out_s[is_acc, 3:6] = [34.0, 211.0, 238.0]
                out_s[~is_acc, 3:6] = [45.0, 130.0, 150.0]
                parts.append(out_s)
        elif mode == "accepted_rejected":
            if len(s_arr) > 0:
                out = np.empty((len(s_arr), 6), dtype=np.float32)
                out[:, :3] = s_arr[:, :3]
                is_acc = s_arr[:, 6] > 0.5
                out[is_acc, 3:6] = [74.0, 222.0, 128.0]
                out[~is_acc, 3:6] = [248.0, 113.0, 113.0]
                parts.append(out)
        elif mode == "observation_diag":
            if len(a_arr) > 0:
                out = np.empty((len(a_arr), 6), dtype=np.float32)
                out[:, :3] = a_arr[:, :3]
                st = np.rint(a_arr[:, 6]).astype(np.int32)
                out[st == 0, 3:6] = [110.0, 118.0, 129.0]
                out[st == 1, 3:6] = [229.0, 201.0, 131.0]
                out[st == 2, 3:6] = [56.0, 139.0, 253.0]
                out[st >= 3, 3:6] = [74.0, 222.0, 128.0]
                parts.append(out)
        else:  # colorized_fused (Default Truecolor RGB Mode)
            # 1. Unobserved 360° structural periphery drawn first in subtle graphite slate (70, 75, 85)
            # 2. True RGB-observed Airy & Stereo points drawn ON TOP with true camera colors
            if len(a_arr) > 0:
                st = np.rint(a_arr[:, 6]).astype(np.int32)
                rgb_mask = (st == 1) | (st >= 3)
                unobs = a_arr[~rgb_mask]
                if len(unobs) > 0:
                    out_u = np.empty((len(unobs), 6), dtype=np.float32)
                    out_u[:, :3] = unobs[:, :3]
                    out_u[:, 3:6] = [70.0, 75.0, 85.0]
                    parts.append(out_u)
                obs = a_arr[rgb_mask]
                if len(obs) > 0:
                    parts.append(obs[:, :6].copy())
            if len(s_arr) > 0:
                parts.append(s_arr[:, :6].copy())

        if not parts:
            return np.zeros((0, 6), dtype=np.float32)
        if len(parts) == 1:
            return parts[0]
        return np.vstack(parts)

    def run_m12_cleanup_preview(self, params: CleanupParams) -> ProcessingStats:
        """Runs full M12 cleanup pipeline (Voxel + SOR + ROR + RANSAC Ground Leveling) and activates preview."""
        self.ensure_local_cloud_for_interactive_ops()
        with self.lock:
            self.m12_params = params
            if not self.cleanup_preview_active:
                self.pre_cleanup_backup_airy = dict(self.live_airy_voxels)
                self.pre_cleanup_backup_stereo = dict(self.live_stereo_voxels)
            mode = self.display_mode

        raw_pts = self.extract_raw_3d_points_for_mode(mode)
        cleaned_pts, stats = PointCloudProcessor.run_cleanup_pipeline(raw_pts, params)
        with self.lock:
            self.cleanup_preview_points = cleaned_pts
            self.cleanup_preview_active = True
            self.last_cleanup_stats = stats
        self.dispatch_live_map_and_telemetry()
        return stats

    def reset_m12_cleanup_preview(self):
        """Restores pre-cleanup point cloud state."""
        with self.lock:
            if self.pre_cleanup_backup_airy is not None:
                self.live_airy_voxels = dict(self.pre_cleanup_backup_airy)
                self._airy_cache_dirty = True
            if self.pre_cleanup_backup_stereo is not None:
                self.live_stereo_voxels = dict(self.pre_cleanup_backup_stereo)
                self._stereo_cache_dirty = True
            self.cleanup_preview_active = False
            self.cleanup_preview_points = None
        self.dispatch_live_map_and_telemetry()

    def commit_m12_cleanup(self) -> int:
        """Commits the M12 cleaned preview cloud into the active live voxel map."""
        with self.lock:
            if not self.cleanup_preview_active or self.cleanup_preview_points is None:
                return len(self.live_airy_voxels)
            pts = self.cleanup_preview_points.copy()
            self.live_airy_voxels.clear()
            self.live_stereo_voxels.clear()
            for i in range(len(pts)):
                x, y, z = float(pts[i, 0]), float(pts[i, 1]), float(pts[i, 2])
                r_c = int(np.clip(pts[i, 3], 0, 255))
                g_c = int(np.clip(pts[i, 4], 0, 255))
                b_c = int(np.clip(pts[i, 5], 0, 255))
                gx, gy, gz = int(math.floor(x / 0.02)), int(math.floor(y / 0.02)), int(math.floor(z / 0.02))
                self.live_airy_voxels[(gx, gy, gz)] = [x, y, z, r_c, g_c, b_c, 1]
            self._airy_cache_dirty = True
            self._stereo_cache_dirty = True
            self.cleanup_preview_active = False
            self.cleanup_preview_points = None
            self.pre_cleanup_backup_airy = None
            self.pre_cleanup_backup_stereo = None
            committed_n = len(self.live_airy_voxels)
        self.dispatch_live_map_and_telemetry()
        return committed_n

    def export_m12_cloud(self, format_str: str, output_path: str, mesh_res_m: float = 0.04) -> ExportResult:
        """Exports the active (cleaned + C-SLAM merged) 3D point cloud to PLY, PCD, XYZ, or OBJ 3D Surface Mesh."""
        self.ensure_local_cloud_for_interactive_ops()
        with self.lock:
            if self.cleanup_preview_active and self.cleanup_preview_points is not None:
                base_local = self.cleanup_preview_points.copy()
            else:
                base_local = self.extract_raw_3d_points_for_mode(self.display_mode)
                if self.live_cleanup_enabled and len(base_local) > 0:
                    rt_params = CleanupParams(
                        enable_voxel_downsample=self.m12_params.enable_voxel_downsample,
                        voxel_leaf_size_m=self.m12_params.voxel_leaf_size_m,
                        enable_sor=self.m12_params.enable_sor,
                        sor_mean_k=min(self.m12_params.sor_mean_k, 16),
                        sor_std_mul=self.m12_params.sor_std_mul,
                        enable_ror=self.m12_params.enable_ror,
                        ror_radius_m=self.m12_params.ror_radius_m,
                        ror_min_neighbors=self.m12_params.ror_min_neighbors,
                        enable_ground_alignment=self.live_ground_lock_enabled,
                        ground_dist_threshold_m=self.m12_params.ground_dist_threshold_m,
                        ground_max_iterations=120,
                        ground_max_tilt_deg=self.m12_params.ground_max_tilt_deg,
                        align_ground_to_z_zero=True,
                        remove_ground_plane=self.m12_params.remove_ground_plane,
                    )
                    base_local, _ = PointCloudProcessor.clean_realtime_fast(base_local, rt_params)

        # Normalize RGB to 0..1 for C-SLAM merger if needed, then export
        if len(base_local) > 0 and np.max(base_local[:, 3:6]) > 1.05:
            local_norm = base_local.copy()
            local_norm[:, 3:6] /= 255.0
        else:
            local_norm = base_local

        merged_pts, _ = self.cslam_engine.get_merged_cslam_scene(local_points_override=local_norm)
        fmt = PointCloudProcessor.parse_export_format(format_str)
        return PointCloudProcessor.export_point_cloud(merged_pts, output_path, fmt, mesh_res_m)

    def get_interpolated_odom(self, query_time, require_bracketed=False, max_clamp_dt=0.045):
        """
        Finds exact SLERP-interpolated 6-DoF pose for `query_time`.
        If `require_bracketed=True`, returns `"WAIT"` when `query_time` is newer than the latest
        completed LiDAR sweep (`odom_buffer[-1][0]`) so live camera/depth frames are never clamped
        to a stale pose during rotation or walking.
        """
        with self.lock:
            n_odom = len(self.odom_buffer)
            if n_odom == 0:
                return None
            if n_odom == 1:
                t0, p0, q0 = self.odom_buffer[0]
                if not require_bracketed and abs(query_time - t0) <= max_clamp_dt:
                    return p0, q0
                return "WAIT" if require_bracketed and query_time > t0 else None

            t_first = self.odom_buffer[0][0]
            t_last = self.odom_buffer[-1][0]

            if query_time > t_last:
                if (query_time - t_last) <= 0.012:
                    return self.odom_buffer[-1][1], self.odom_buffer[-1][2]
                if require_bracketed:
                    return "WAIT"
                if (query_time - t_last) <= max_clamp_dt:
                    return self.odom_buffer[-1][1], self.odom_buffer[-1][2]
                return None

            if query_time < t_first:
                if abs(query_time - t_first) <= max_clamp_dt:
                    return self.odom_buffer[0][1], self.odom_buffer[0][2]
                return None

            times = [r[0] for r in self.odom_buffer]
            idx = int(np.searchsorted(times, query_time))
            if idx <= 0:
                return self.odom_buffer[0][1], self.odom_buffer[0][2]
            if idx >= n_odom:
                return self.odom_buffer[-1][1], self.odom_buffer[-1][2]

            t0, p0, q0 = self.odom_buffer[idx - 1]
            t1, p1, q1 = self.odom_buffer[idx]
            dt = t1 - t0
            if dt <= 1e-6:
                return p0, q0

            alpha = (query_time - t0) / dt
            p_interp = (1.0 - alpha) * p0 + alpha * p1
            q_interp = slerp_quat(q0, q1, alpha)
            return p_interp, q_interp

    def _find_closest_rgb_frame(self, target_sec, max_dt=0.065):
        """Finds the decoded RGB frame in `self.rgb_ring` closest in timestamp to `target_sec`."""
        with self.lock:
            if len(self.rgb_ring) == 0:
                return None, None
            best_sec = None
            best_img = None
            best_dt = max_dt
            for r_sec, r_img in self.rgb_ring:
                dt = abs(r_sec - target_sec)
                if dt <= best_dt:
                    best_dt = dt
                    best_sec = r_sec
                    best_img = r_img
            return best_sec, best_img

    @staticmethod
    def _lift_camera_rgb_for_display(rgb_arr: np.ndarray) -> np.ndarray:
        """Applies a gentle hue-preserving gamma lift (gamma=0.78) so indoor RGB textures are vivid on AR glass."""
        if len(rgb_arr) == 0:
            return rgb_arr
        norm = np.clip(rgb_arr.astype(np.float32) / 255.0, 0.0, 1.0)
        lifted = np.clip((norm ** 0.78) * 238.0 + 17.0, 28.0, 255.0)
        return lifted.astype(np.uint8)

    def _align_sensor_stamp_to_slam(self, sensor_sec: float, arr_wall_time: float) -> float:
        """
        Ensures camera/depth timestamps live on the exact same clock timeline as FAST-LIVO2 `/aft_mapped_to_init`.
        Because FAST-LIVO2 stamps `/aft_mapped_to_init` with the LiDAR sweep timestamp (~0.10-0.15s behind wall-clock),
        maps the camera arrival time onto the SLAM timeline via `self._slam_wall_to_stamp_offset`.
        """
        with self.lock:
            has_off = self._has_slam_clock_offset
            off = self._slam_wall_to_stamp_offset
            n_odom = len(self.odom_buffer)
            t_last = self.odom_buffer[-1][0] if n_odom > 0 else 0.0

        if sensor_sec <= 1.0:
            return arr_wall_time + off if has_off else arr_wall_time
        if has_off:
            expected_slam_sec = arr_wall_time + off
            if abs(sensor_sec - expected_slam_sec) > 0.025:
                return expected_slam_sec
        elif n_odom > 0 and abs(sensor_sec - t_last) > 0.15:
            return t_last
        return sensor_sec

    @staticmethod
    def _decode_rgb_buffer(raw: bytes, w: int, h: int, step: int, enc: str) -> Optional[np.ndarray]:
        """Robustly decodes ROS 2 Image raw bytes into contiguous (H, W, 3) uint8 RGB array across all Orbbec encodings."""
        if w <= 0 or h <= 0 or len(raw) == 0:
            return None
        enc_l = (enc or "rgb8").lower()
        buf = np.frombuffer(raw, dtype=np.uint8)
        if "bgra" in enc_l or "rgba" in enc_l:
            ch = 4
            row_bytes = step if step >= w * ch else w * ch
            if len(buf) < h * row_bytes:
                return None
            img2d = buf[: h * row_bytes].reshape((h, row_bytes))[:, : w * ch].reshape((h, w, ch))
            return cv2.cvtColor(img2d, cv2.COLOR_BGRA2RGB if "bgra" in enc_l else cv2.COLOR_RGBA2RGB)
        if "yuv" in enc_l or "yuyv" in enc_l or "uyvy" in enc_l:
            ch = 2
            row_bytes = step if step >= w * ch else w * ch
            if len(buf) < h * row_bytes:
                return None
            img2d = buf[: h * row_bytes].reshape((h, row_bytes))[:, : w * ch].reshape((h, w, ch))
            code = cv2.COLOR_YUV2RGB_UYVY if "uyvy" in enc_l else cv2.COLOR_YUV2RGB_YUYV
            return cv2.cvtColor(img2d, code)
        if "mono8" in enc_l or "8uc1" in enc_l:
            row_bytes = step if step >= w else w
            if len(buf) < h * row_bytes:
                return None
            gray = buf[: h * row_bytes].reshape((h, row_bytes))[:, :w]
            return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        # Default 3-channel rgb8 / bgr8
        ch = 3
        row_bytes = step if step >= w * ch else w * ch
        if len(buf) < h * row_bytes:
            return None
        img3 = buf[: h * row_bytes].reshape((h, row_bytes))[:, : w * ch].reshape((h, w, 3))
        if "bgr" in enc_l:
            return cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)
        return np.ascontiguousarray(img3)

    @staticmethod
    def _decode_depth_buffer(raw: bytes, w: int, h: int, step: int, enc: str) -> Optional[np.ndarray]:
        """Robustly decodes ROS 2 Depth Image raw bytes into contiguous (H, W) uint16 millimeters array."""
        if w <= 0 or h <= 0 or len(raw) == 0:
            return None
        enc_l = (enc or "16uc1").lower()
        if "32fc1" in enc_l:
            row_floats = (step // 4) if step >= w * 4 else w
            buf_f = np.frombuffer(raw, dtype=np.float32)
            if len(buf_f) < h * row_floats:
                return None
            d_m = buf_f[: h * row_floats].reshape((h, row_floats))[:, :w]
            valid = np.isfinite(d_m) & (d_m > 0.0) & (d_m < 65.0)
            out_mm = np.zeros((h, w), dtype=np.uint16)
            out_mm[valid] = np.clip(d_m[valid] * 1000.0, 0, 65535).astype(np.uint16)
            return out_mm
        # Default 16UC1 / MONO16 (uint16 millimeters)
        row_u16 = (step // 2) if step >= w * 2 else w
        buf_u16 = np.frombuffer(raw, dtype=np.uint16)
        if len(buf_u16) < h * row_u16:
            return None
        return np.ascontiguousarray(buf_u16[: h * row_u16].reshape((h, row_u16))[:, :w])

    def _emit_composite_camera_frame(self):
        """Emits cleanly isolated side-by-side RGB (Left) and Depth Heatmap (Right) at up to 22 FPS."""
        now = time.time()
        if now - self._last_image_emit < 0.042:  # ~24 FPS smooth low-latency stream
            return

        with self.lock:
            rgb_img = self.latest_color_raw
            rgb_seq = self._rgb_seq
            depth_img = self.latest_depth_raw
            depth_seq = self._depth_seq
            min_d = self.depth_min_m
            max_d = self.depth_max_m

        if rgb_img is None and depth_img is None:
            return
        if rgb_seq == self._last_preview_rgb_seq and depth_seq == self._last_preview_depth_seq:
            return

        self._last_image_emit = now
        self._last_preview_rgb_seq = rgb_seq
        self._last_preview_depth_seq = depth_seq

        target_w, target_h = 280, 175
        if rgb_img is not None:
            preview_rgb = cv2.resize(rgb_img, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
        else:
            preview_rgb = np.zeros((target_h, target_w, 3), dtype=np.uint8)
            cv2.putText(preview_rgb, "RGB: STANDBY", (45, target_h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (139, 148, 158), 1)

        if depth_img is not None:
            depth_resized = cv2.resize(depth_img, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
            d_m = depth_resized.astype(np.float32) * 0.001
            valid_mask = (d_m >= min_d) & (d_m <= max_d)
            d_norm = np.clip((d_m - min_d) / max(0.1, (max_d - min_d)) * 255.0, 0, 255).astype(np.uint8)
            depth_color = cv2.applyColorMap(d_norm, cv2.COLORMAP_TURBO)
            depth_color_rgb = cv2.cvtColor(depth_color, cv2.COLOR_BGR2RGB)
            depth_color_rgb[~valid_mask] = [20, 20, 25]
        else:
            depth_color_rgb = np.zeros((target_h, target_w, 3), dtype=np.uint8)
            cv2.putText(depth_color_rgb, "DEPTH: STANDBY", (30, target_h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (139, 148, 158), 1)

        combined = np.ascontiguousarray(np.hstack((preview_rgb, depth_color_rgb)))
        h, w, ch = combined.shape
        fmt_rgb = QImage.Format_RGB888 if hasattr(QImage, 'Format_RGB888') else QImage.Format.Format_RGB888
        qimg = QImage(combined.data, w, h, ch * w, fmt_rgb).copy()
        self.image_received.emit(qimg)

    def _drain_and_emit_camera_preview(self):
        """
        Decodes queued RGB frames and the newest Depth frame for the live ~24 FPS
        side-by-side RGB + Depth Heatmap preview. Runs exclusively on `Camera_Preview_Thread`
        so 3D point cloud voxelization or KD-tree queries can never freeze the camera stream.
        """
        if not self._preview_lock.acquire(blocking=False):
            return
        try:
            # 1. Decode up to 2 freshest queued RGB frames (dropping stale backlog if any)
            rgb_items = []
            with self.lock:
                while len(self._raw_rgb_queue) > 2:
                    self._raw_rgb_queue.popleft()
                while len(self._raw_rgb_queue) > 0:
                    rgb_items.append(self._raw_rgb_queue.popleft())

            for sec, arr_time, w, h, step, enc, raw in rgb_items:
                try:
                    rgb = self._decode_rgb_buffer(raw, w, h, step, enc)
                    if rgb is not None:
                        sec_aligned = self._align_sensor_stamp_to_slam(sec, arr_time)
                        with self.lock:
                            self.latest_color_raw = rgb
                            self.latest_color_stamp = sec_aligned
                            self._rgb_seq += 1
                            self.rgb_ring.append((sec_aligned, rgb))
                except Exception:
                    pass

            # 2. Decode newest depth frame from `_latest_raw_depth_tuple` for immediate Depth Heatmap display
            with self.lock:
                newest_depth_tuple = self._latest_raw_depth_tuple
            if newest_depth_tuple is not None:
                d_sec, d_arr_time, d_w, d_h, d_step, d_enc, d_raw = newest_depth_tuple
                if d_arr_time != self.latest_depth_stamp:
                    try:
                        d_decoded = self._decode_depth_buffer(d_raw, d_w, d_h, d_step, d_enc)
                        if d_decoded is not None:
                            with self.lock:
                                self.latest_depth_raw = d_decoded
                                self.latest_depth_stamp = d_arr_time
                                self._depth_seq += 1
                    except Exception:
                        pass

            # 3. Emit smooth ~24 FPS camera preview immediately
            self._emit_composite_camera_frame()
        finally:
            self._preview_lock.release()

    def _process_pending_sensor_queues(self):
        """
        Runs on the worker thread (completely off the ROS 2 spin thread and camera preview thread):
        1. Processes queued LiDAR sweeps (`/cloud_registered`) with Z-buffered co-temporal RGB projection.
        2. Processes queued Gemini Depth frames at ~18 Hz with depth-proportional surface continuity,
           local Airy KD-tree corroboration, and continuous non-downgrading voxel accumulation into `self.live_stereo_voxels`.
        """
        # 1. Process queued LiDAR sweeps (/cloud_registered)
        while True:
            with self.lock:
                if len(self._raw_lidar_queue) == 0:
                    break
                sec, num_points, point_step, raw = self._raw_lidar_queue.popleft()
            try:
                dt = np.dtype({
                    'names': ['x', 'y', 'z'],
                    'formats': ['<f4', '<f4', '<f4'],
                    'offsets': [0, 4, 8],
                    'itemsize': point_step
                })
                pts = np.frombuffer(raw, dtype=dt, count=num_points)
                valid = np.isfinite(pts['x']) & np.isfinite(pts['y']) & np.isfinite(pts['z'])
                pts_valid = pts[valid]
                if len(pts_valid) == 0:
                    continue

                xyz_w = np.column_stack((pts_valid['x'], pts_valid['y'], pts_valid['z'])).astype(np.float64)

                # Compute Live Beam Angular Elevation Distribution in Sensor Body Frame
                pose = self.get_interpolated_odom(sec, require_bracketed=False, max_clamp_dt=0.25)
                if (pose is None or pose == "WAIT") and self.using_slam_odom:
                    pose = (
                        np.array([self.current_x, self.current_y, self.current_z], dtype=np.float64),
                        np.array(self.current_quat, dtype=np.float64),
                    )
                if pose is not None and pose != "WAIT":
                    pos_w, quat = pose
                    R_w_imu = Rotation.from_quat(quat).as_matrix()
                    p_body = (xyz_w - pos_w) @ R_w_imu
                    norm_xy = np.linalg.norm(p_body[:, :2], axis=1)
                    elev_deg = np.degrees(np.arctan2(p_body[:, 2], np.maximum(0.01, norm_xy)))

                    floor_cnt = int(np.sum(elev_deg < -30.0))
                    wall_cnt = int(np.sum((elev_deg >= -30.0) & (elev_deg <= 30.0)))
                    ceil_cnt = int(np.sum(elev_deg > 30.0))
                    total_cnt = len(elev_deg)

                    with self.lock:
                        self.live_elevation_stats = {
                            "floor": floor_cnt,
                            "wall": wall_cnt,
                            "ceiling": ceil_cnt,
                            "total": total_cnt,
                        }

                # Pre-reduce sweep to unique 5cm voxels in NumPy outside the lock
                voxel_size = 0.05
                gx_all = np.floor(xyz_w[:, 0] / voxel_size).astype(np.int64)
                gy_all = np.floor(xyz_w[:, 1] / voxel_size).astype(np.int64)
                gz_all = np.floor(xyz_w[:, 2] / voxel_size).astype(np.int64)
                packed = ((gx_all + 500000) << 42) | ((gy_all + 500000) << 21) | (gz_all + 500000)
                _, u_idx = np.unique(packed, return_index=True)

                xyz_u = xyz_w[u_idx]
                gx = gx_all[u_idx]
                gy = gy_all[u_idx]
                gz = gz_all[u_idx]

                # Co-temporal RGB lookup from ring buffer (aligned to SLAM epoch `sec`)
                rgb_sec, rgb_img = self._find_closest_rgb_frame(sec, max_dt=0.25)
                with self.lock:
                    if rgb_img is None and self.latest_color_raw is not None and (time.time() - self.last_camera_time) < 0.50:
                        rgb_img = self.latest_color_raw
                        rgb_sec = sec
                    d_rgb = self.d_rgb
                    ang_speed = self.current_ang_speed
                    is_rec = self.is_session_recording

                rgb_w_pose = (
                    self.get_interpolated_odom(rgb_sec, require_bracketed=False, max_clamp_dt=0.35)
                    if (rgb_img is not None and rgb_sec is not None)
                    else None
                )
                if (rgb_w_pose is None or rgb_w_pose == "WAIT") and pose is not None and pose != "WAIT":
                    rgb_w_pose = pose

                obs_states = np.zeros(len(xyz_u), dtype=np.uint8)  # 0=Unobserved, 1=RGB, 2=Depth, 3=Both
                vox_colors = np.full((len(xyz_u), 3), [64, 68, 78], dtype=np.uint8)

                if (
                    rgb_img is not None
                    and rgb_w_pose is not None
                    and rgb_w_pose != "WAIT"
                    and ang_speed <= max(self.max_turn_rate_rad_s * 2.2, 2.8)
                ):
                    pos_c, quat_c = rgb_w_pose
                    R_w_imu_c = Rotation.from_quat(quat_c).as_matrix()

                    # Save bracketed keyframe if recording
                    if is_rec:
                        with self.lock:
                            if len(self.rgb_keyframes) == 0 or (rgb_sec - self.rgb_keyframes[-1][0] >= 0.20):
                                self.rgb_keyframes.append((rgb_sec, rgb_img, R_w_imu_c, pos_c))

                    p_imu_c = (xyz_u - pos_c) @ R_w_imu_c
                    p_rgb_c = p_imu_c @ R_RGB_IMU.T + t_RGB_IMU
                    u, v, zr = project_pts_to_rgb_plumb_bob(
                        p_rgb_c, self.fx_rgb, self.fy_rgb, self.cx_rgb, self.cy_rgb, d_rgb
                    )
                    rgb_h, rgb_w = rgb_img.shape[:2]
                    # Skip 10px lens border for clean color projection
                    in_rgb_fov = (zr > 0.30) & (zr < 5.5) & (u >= 10) & (u < rgb_w - 10) & (v >= 10) & (v < rgb_h - 10)
                    fov_idx = np.nonzero(in_rgb_fov)[0]
                    if len(fov_idx) > 0:
                        u_f = u[fov_idx]
                        v_f = v[fov_idx]
                        z_f = zr[fov_idx]
                        # 320x200 Z-buffer occlusion gate so foreground walls block background walls
                        u_b = np.clip(u_f // 4, 0, (rgb_w // 4) - 1)
                        v_b = np.clip(v_f // 4, 0, (rgb_h // 4) - 1)
                        z_buf = np.full((rgb_h // 4, rgb_w // 4), np.inf, dtype=np.float64)
                        np.minimum.at(z_buf, (v_b, u_b), z_f)
                        visible = z_f <= (z_buf[v_b, u_b] + 0.14)
                        vis_idx = fov_idx[visible]
                        if len(vis_idx) > 0:
                            obs_states[vis_idx] = 1
                            sampled = rgb_img[v[vis_idx], u[vis_idx]]
                            vox_colors[vis_idx] = self._lift_camera_rgb_for_display(sampled)

                gx_list = gx.tolist()
                gy_list = gy.tolist()
                gz_list = gz.tolist()
                xyz_list = xyz_u.tolist()
                col_list = vox_colors.tolist()
                st_list = obs_states.tolist()

                with self.lock:
                    airy_voxels = self.live_airy_voxels
                    stereo_voxels = self.live_stereo_voxels
                    for i in range(len(xyz_list)):
                        key = (gx_list[i], gy_list[i], gz_list[i])
                        has_stereo_match = key in stereo_voxels
                        state = st_list[i]
                        r_c, g_c, b_c = col_list[i]

                        prev = airy_voxels.get(key)
                        if prev is not None and state == 0 and prev[6] in (1, 3):
                            state = 1
                            r_c, g_c, b_c = prev[3], prev[4], prev[5]

                        if has_stereo_match:
                            state = 3 if state in (1, 3) else 2

                        pt_xyz = xyz_list[i]
                        airy_voxels[key] = [pt_xyz[0], pt_xyz[1], pt_xyz[2], r_c, g_c, b_c, state]

                    self._airy_cache_dirty = True
                    if self.is_session_recording:
                        self.airy_sweeps.append((sec, xyz_w))
            except Exception:
                pass

        # 2. Process queued Gemini 336L Stereo Depth frames (~18 Hz continuous 3D map accumulation)
        now = time.time()
        if (now - self._last_depth_voxel_time) >= 0.050:
            depth_item = None
            with self.lock:
                if len(self._raw_depth_queue) > 0:
                    # Pop the freshest depth frame and clear older queued frames so there is zero lag
                    depth_item = self._raw_depth_queue.pop()
                    self._raw_depth_queue.clear()
                    # Smoothly decay angular speed so normal hand/head panning never stays locked out
                    self.current_ang_speed *= 0.90
                    min_d = self.depth_min_m
                    max_d = self.depth_max_m
                    agree_tol = self.surface_agreement_m
                    ang_speed = self.current_ang_speed
                    d_rgb = self.d_rgb

            if depth_item is not None:
                sec, arr_time, w, h, step, enc, raw = depth_item
                sec_aligned = self._align_sensor_stamp_to_slam(sec, arr_time)

                # Resolve 6-DoF pose: interpolated SLAM odom -> latest odom_buffer -> current pose fallback
                pose = self.get_interpolated_odom(sec_aligned, require_bracketed=False, max_clamp_dt=0.60)
                if pose is None or pose == "WAIT":
                    with self.lock:
                        if len(self.odom_buffer) > 0:
                            _, last_p, last_q = self.odom_buffer[-1]
                            pose = (last_p.copy(), last_q.copy())
                        else:
                            pose = (
                                np.array([self.current_x, self.current_y, self.current_z], dtype=np.float64),
                                np.array(self.current_quat, dtype=np.float64),
                            )

                if pose is not None and pose != "WAIT":
                    # Only gate extreme whip-turns (> 3.2 rad/s = 183 deg/s) so normal panning builds the Gemini map!
                    if ang_speed > max(self.max_turn_rate_rad_s * 2.5, 3.2):
                        with self.lock:
                            self.cnt_depth_gated_turn += 1
                    else:
                        self._last_depth_voxel_time = now
                        try:
                            with self.lock:
                                cached_d = self.latest_depth_raw if (arr_time == self.latest_depth_stamp) else None
                            depth_img = cached_d if cached_d is not None else self._decode_depth_buffer(raw, w, h, step, enc)
                            if depth_img is not None:
                                with self.lock:
                                    if arr_time != self.latest_depth_stamp:
                                        self.latest_depth_raw = depth_img
                                        self.latest_depth_stamp = arr_time
                                        self._depth_seq += 1

                                sub = 5  # Dense 5x subsampling (~3,200 rays/frame) for continuous stereo map building
                                d_sub = depth_img[::sub, ::sub].astype(np.int32)
                                # Depth-proportional discontinuity gate: allows slanted walls/floors across 5px stride
                                # (~6.5% of depth, minimum 95mm) while rejecting >20cm occlusion edge flying pixels
                                max_dz = np.maximum(95, (d_sub * 0.065).astype(np.int32))
                                dz_x = np.abs(d_sub - np.roll(d_sub, 1, axis=1))
                                dz_y = np.abs(d_sub - np.roll(d_sub, 1, axis=0))
                                valid_mask = (
                                    (d_sub >= int(min_d * 1000))
                                    & (d_sub <= int(max_d * 1000))
                                    & (dz_x <= max_dz)
                                    & (dz_y <= max_dz)
                                )
                                valid_mask[0, :] = False
                                valid_mask[-1, :] = False
                                valid_mask[:, 0] = False
                                valid_mask[:, -1] = False

                                v_idx, u_idx = np.nonzero(valid_mask)
                                if len(u_idx) > 0:
                                    pos_w, quat = pose
                                    R_w_imu = Rotation.from_quat(quat).as_matrix()
                                    R_w_cam_depth = R_w_imu @ R_IMU_CAM_DEPTH
                                    t_w_cam_depth = R_w_imu @ t_IMU_CAM_DEPTH + pos_w

                                    u_px = u_idx * sub
                                    v_px = v_idx * sub
                                    z_vals = d_sub[v_idx, u_idx].astype(np.float64) * 0.001

                                    x_cam = (u_px - self.cx_d) * z_vals / self.fx_d
                                    y_cam = (v_px - self.cy_d) * z_vals / self.fy_d
                                    pts_cam = np.vstack((x_cam, y_cam, z_vals)).T
                                    pts_w = pts_cam @ R_w_cam_depth.T + t_w_cam_depth

                                    # Co-temporal RGB color assignment from `self.rgb_ring`
                                    rgb_sec, rgb_img = self._find_closest_rgb_frame(sec_aligned, max_dt=0.25)
                                    if rgb_img is None:
                                        with self.lock:
                                            if self.latest_color_raw is not None and (now - self.last_camera_time) < 0.50:
                                                rgb_img = self.latest_color_raw
                                    colors = np.full((len(pts_w), 3), [34, 211, 238], dtype=np.uint8)
                                    has_rgb_color = np.zeros(len(pts_w), dtype=bool)
                                    if rgb_img is not None:
                                        rgb_h, rgb_w = rgb_img.shape[:2]
                                        pts_rgb_frame = (pts_cam @ R_RGB_CAM_DEPTH.T) + t_RGB_CAM_DEPTH
                                        u_r, v_r, zr = project_pts_to_rgb_plumb_bob(
                                            pts_rgb_frame, self.fx_rgb, self.fy_rgb, self.cx_rgb, self.cy_rgb, d_rgb
                                        )
                                        in_fov = (zr > 0.25) & (u_r >= 8) & (u_r < rgb_w - 8) & (v_r >= 8) & (v_r < rgb_h - 8)
                                        if np.any(in_fov):
                                            has_rgb_color[in_fov] = True
                                            sampled = rgb_img[v_r[in_fov], u_r[in_fov]]
                                            colors[in_fov] = self._lift_camera_rgb_for_display(sampled)

                                    # Local Full-Resolution Airy Reference Cloud (within camera range of pos_w)
                                    with self.lock:
                                        a_arr, _ = self._get_voxel_arrays_locked()
                                    airy_pts_arr = None
                                    if len(a_arr) > 30:
                                        a_xyz = a_arr[:, :3].astype(np.float64)
                                        d2_sensor = np.sum((a_xyz - pos_w) ** 2, axis=1)
                                        local_mask = d2_sensor <= ((max_d + 1.25) ** 2)
                                        local_xyz = a_xyz[local_mask]
                                        if len(local_xyz) > 25:
                                            step_loc = max(1, len(local_xyz) // 14000)
                                            airy_pts_arr = local_xyz[::step_loc]

                                    # Smooth contiguous stereo surface check (fills inter-ring surfaces cleanly out to max_d)
                                    smooth_tol = np.maximum(75, (d_sub[v_idx, u_idx] * 0.052).astype(np.int32))
                                    smooth_patch = (
                                        (dz_x[v_idx, u_idx] <= smooth_tol)
                                        & (dz_y[v_idx, u_idx] <= smooth_tol)
                                        & (z_vals <= max_d)
                                    )
                                    is_accepted = np.ones(len(pts_w), dtype=bool)
                                    if airy_pts_arr is not None:
                                        tree_a = cKDTree(airy_pts_arr)
                                        dists, nn_idx = tree_a.query(pts_w, k=1, workers=1)
                                        # Corroborate within LiDAR inter-ring spacing (~26cm) OR smooth contiguous surface patch
                                        corrob_tol = max(agree_tol * 5.0, 0.26)
                                        is_accepted = (dists <= corrob_tol) | smooth_patch
                                        # Gentle ASPS Surface Snapping when within 9cm of an Airy voxel
                                        snap_mask = is_accepted & (dists <= max(agree_tol * 1.8, 0.09))
                                        if np.any(snap_mask):
                                            pts_w[snap_mask] = 0.60 * airy_pts_arr[nn_idx[snap_mask]] + 0.40 * pts_w[snap_mask]

                                    voxel_size = 0.05
                                    gx = np.floor(pts_w[:, 0] / voxel_size).astype(np.int64)
                                    gy = np.floor(pts_w[:, 1] / voxel_size).astype(np.int64)
                                    gz = np.floor(pts_w[:, 2] / voxel_size).astype(np.int64)
                                    packed = ((gx + 500000) << 42) | ((gy + 500000) << 21) | (gz + 500000)
                                    _, u_vox = np.unique(packed, return_index=True)

                                    gx_u = gx[u_vox].tolist()
                                    gy_u = gy[u_vox].tolist()
                                    gz_u = gz[u_vox].tolist()
                                    pts_u = pts_w[u_vox].tolist()
                                    col_u = colors[u_vox].tolist()
                                    acc_u = is_accepted[u_vox].tolist()
                                    rgb_u = has_rgb_color[u_vox].tolist()

                                    with self.lock:
                                        stereo_voxels = self.live_stereo_voxels
                                        airy_voxels = self.live_airy_voxels
                                        for idx_i in range(len(u_vox)):
                                            key = (gx_u[idx_i], gy_u[idx_i], gz_u[idx_i])
                                            p_xyz = pts_u[idx_i]
                                            c_rgb = col_u[idx_i]
                                            acc_flag = bool(acc_u[idx_i])
                                            prev_s = stereo_voxels.get(key)
                                            if prev_s is not None:
                                                # Never downgrade an already-accepted stereo voxel to rejected
                                                if prev_s[6] and not acc_flag:
                                                    continue
                                                if prev_s[6] and acc_flag:
                                                    p_xyz = [
                                                        0.65 * prev_s[0] + 0.35 * p_xyz[0],
                                                        0.65 * prev_s[1] + 0.35 * p_xyz[1],
                                                        0.65 * prev_s[2] + 0.35 * p_xyz[2],
                                                    ]
                                                    if not rgb_u[idx_i]:
                                                        c_rgb = [prev_s[3], prev_s[4], prev_s[5]]
                                            stereo_voxels[key] = [
                                                p_xyz[0], p_xyz[1], p_xyz[2],
                                                c_rgb[0], c_rgb[1], c_rgb[2],
                                                acc_flag,
                                            ]
                                            # Also transfer co-temporal RGB color onto matching Airy voxel if not yet RGB-colored
                                            if acc_flag and rgb_u[idx_i]:
                                                a_vox = airy_voxels.get(key)
                                                if a_vox is not None and a_vox[6] in (0, 2):
                                                    a_vox[3], a_vox[4], a_vox[5] = c_rgb[0], c_rgb[1], c_rgb[2]
                                                    a_vox[6] = 3
                                                    self._airy_cache_dirty = True
                                                elif a_vox is not None and a_vox[6] == 1:
                                                    a_vox[6] = 3
                                                    self._airy_cache_dirty = True

                                        self._stereo_cache_dirty = True
                                        if self.is_session_recording:
                                            acc_pts = pts_w[is_accepted]
                                            acc_cols = colors[is_accepted]
                                            if len(acc_pts) > 0:
                                                self.gemini_frames.append((sec_aligned, acc_pts, acc_cols))
                        except Exception:
                            pass

    def run(self):
        if not HAS_ROS2 or rclpy is None:
            while self.running:
                try:
                    self.dispatch_live_map_and_telemetry()
                except Exception:
                    pass
                time.sleep(0.08)
            return

        if not rclpy.ok():
            rclpy.init()

        # Single unified Node with a persistent SingleThreadedExecutor (preserves round-robin `_cb_iter`
        # across all subscriptions in a single wait-set without multi-executor context contention)
        node = Node("scanhub_m12_9_bridge")
        self.ros_node = node
        print(
            f"[scanHUB OS] ROS 2 Bridge Connected on Domain ID: {os.environ.get('ROS_DOMAIN_ID', '0')} "
            f"(RMW: {os.environ.get('RMW_IMPLEMENTATION', 'rmw_fastrtps_cpp')})"
        )

        def depth_info_callback(msg: CameraInfo):
            if msg.k[0] > 10.0:
                with self.lock:
                    self.fx_d = float(msg.k[0])
                    self.fy_d = float(msg.k[4])
                    self.cx_d = float(msg.k[2])
                    self.cy_d = float(msg.k[5])

        def rgb_info_callback(msg: CameraInfo):
            if msg.k[0] > 10.0:
                with self.lock:
                    self.fx_rgb = float(msg.k[0])
                    self.fy_rgb = float(msg.k[4])
                    self.cx_rgb = float(msg.k[2])
                    self.cy_rgb = float(msg.k[5])
                    if len(msg.d) >= 5 and any(abs(x) > 1e-9 for x in msg.d[:5]):
                        self.d_rgb = np.array(msg.d[:5], dtype=np.float64)

        def rgb_color_callback(msg: Image):
            """Ultra-fast (<0.15ms) non-blocking RGB ingestion."""
            self.cnt_camera += 1
            now = time.time()
            sec = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            self.last_camera_time = now
            raw = bytes(msg.data)
            with self.lock:
                self._ts_camera.append(now)
                self._raw_rgb_queue.append(
                    (sec, now, int(msg.width), int(msg.height), int(msg.step), msg.encoding.lower(), raw)
                )

        def depth_callback(msg: Image):
            """Ultra-fast (<0.15ms) non-blocking Depth ingestion."""
            now = time.time()
            sec = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            self.last_depth_time = now
            raw = bytes(msg.data)
            item = (sec, now, int(msg.width), int(msg.height), int(msg.step), msg.encoding.lower(), raw)
            with self.lock:
                self._ts_depth.append(now)
                self._latest_raw_depth_tuple = item
                self._raw_depth_queue.append(item)

        def registered_cloud_callback(msg: PointCloud2):
            """Ultra-fast (<0.2ms) non-blocking FAST-LIVO2 Registered Cloud ingestion."""
            self.cnt_lidar += 1
            self.cnt_rslidar += 1
            now = time.time()
            sec = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            self.last_lidar_time = now
            self.last_rslidar_time = now
            num_points = int(msg.width * msg.height)
            step = int(msg.point_step)
            raw = bytes(msg.data)
            with self.lock:
                self._ts_lidar.append(now)
                self._ts_rslidar.append(now)
                if num_points > 0 and step >= 12 and len(raw) >= num_points * step:
                    self._raw_lidar_queue.append((sec, num_points, step, raw))

        def slam_odom_callback(msg: Odometry):
            """Immediate 10 Hz low-latency 6-DoF SLAM pose callback."""
            self.cnt_slam += 1
            now = time.time()
            sec = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            self.using_slam_odom = True
            self.last_slam_time = now

            px = msg.pose.pose.position.x
            py = msg.pose.pose.position.y
            pz = msg.pose.pose.position.z
            qx = msg.pose.pose.orientation.x
            qy = msg.pose.pose.orientation.y
            qz = msg.pose.pose.orientation.z
            qw = msg.pose.pose.orientation.w

            if math.isnan(px) or math.isnan(py) or math.isnan(pz):
                return

            self.current_x = px
            self.current_y = py
            self.current_z = pz
            self.current_quat = [qx, qy, qz, qw]
            try:
                R_w_imu = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
                fwd_w = R_w_imu @ CAM_FWD_AXIS_IMU
                self.current_yaw = math.atan2(fwd_w[1], fwd_w[0])
            except Exception:
                siny_cosp = 2.0 * (qw * qz + qx * qy)
                cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
                self.current_yaw = math.atan2(siny_cosp, cosy_cosp)

            pos_arr = np.array([px, py, pz], dtype=np.float64)
            quat_arr = np.array([qx, qy, qz, qw], dtype=np.float64)
            q_norm = float(np.linalg.norm(quat_arr))
            if q_norm > 1e-6:
                quat_arr = quat_arr / q_norm

            with self.lock:
                self._ts_slam.append(now)
                if sec > 1.0:
                    raw_off = sec - now
                    if not self._has_slam_clock_offset:
                        self._slam_wall_to_stamp_offset = raw_off
                        self._has_slam_clock_offset = True
                    else:
                        self._slam_wall_to_stamp_offset = 0.85 * self._slam_wall_to_stamp_offset + 0.15 * raw_off
                self._pred_yaw = self.current_yaw
                self._last_imu_pred_time = now

                if len(self.odom_buffer) > 0:
                    prev_t, prev_p, prev_q = self.odom_buffer[-1]
                    dt = sec - prev_t
                    if 0.025 < dt < 1.0:
                        lin_spd = float(np.linalg.norm(pos_arr - prev_p) / dt)
                        dot_q = abs(float(np.dot(quat_arr, prev_q)))
                        ang_diff = 2.0 * math.acos(min(1.0, max(0.0, dot_q)))
                        ang_spd = min(6.0, float(ang_diff / dt))
                        self.current_lin_speed = 0.6 * self.current_lin_speed + 0.4 * lin_spd
                        self.current_ang_speed = 0.6 * self.current_ang_speed + 0.4 * ang_spd
                self.odom_buffer.append((sec, pos_arr, quat_arr))
                if self.is_session_recording:
                    self.trajectory_records.append((sec, px, py, pz, qx, qy, qz, qw, self.current_yaw))

            self.pose_received.emit(self.current_x, self.current_y, self.current_yaw)

        def rslidar_imu_liveness_callback(msg: Imu):
            """
            1. Lightweight hardware LiDAR liveness fallback (before SLAM starts) without 3MB PointCloud2 copy.
            2. High-precision Stage 1 static gravity calibration and Stage 2 90° rotation ritual integration.
            3. Zero-latency 35 Hz inter-sweep IMU gyro heading prediction between 10 Hz SLAM poses.
            """
            now = time.time()
            if (now - self.last_lidar_time) > 0.5 and (now - self.last_rslidar_time) >= 0.098:
                self.cnt_rslidar += 1
                self.last_rslidar_time = now
                with self.lock:
                    self._ts_rslidar.append(now)

            ax = float(msg.linear_acceleration.x)
            ay = float(msg.linear_acceleration.y)
            az = float(msg.linear_acceleration.z)
            if math.isfinite(ax) and math.isfinite(ay) and math.isfinite(az):
                self.last_ax = ax
                self.last_ay = ay
                self.last_az = az

            # NavVis 90-Degree Rotation Ritual Tracking
            if self.in_rotation_ritual:
                now_t = now
                if self.last_imu_gyro_time is not None:
                    dt_rot = now_t - self.last_imu_gyro_time
                    if 0.0 < dt_rot < 0.5:
                        gx = float(msg.angular_velocity.x)
                        gy = float(msg.angular_velocity.y)
                        gz = float(msg.angular_velocity.z)
                        # Project angular velocity along gravity axis (or total gyro magnitude)
                        acc_norm = math.hypot(self.last_ax, self.last_ay, self.last_az)
                        if acc_norm > 1.0:
                            gz_eff = abs(gx * self.last_ax + gy * self.last_ay + gz * self.last_az) / acc_norm
                        else:
                            gz_eff = math.hypot(gx, gy, gz)
                        gz_deg = gz_eff * (180.0 / math.pi)
                        if gz_deg > 3.0:  # Threshold to filter hand tremor
                            self.accumulated_rotation_deg += gz_deg * dt_rot
                            self.rotation_progress_received.emit(self.accumulated_rotation_deg)
                            if self.accumulated_rotation_deg >= 85.0 and gz_deg < 25.0:
                                self.in_rotation_ritual = False
                                self.rotation_completed_received.emit()
                self.last_imu_gyro_time = now_t

            if self.using_slam_odom and (now - self.last_slam_time) < 0.35:
                dt_imu = now - self._last_imu_pred_time if self._last_imu_pred_time > 0 else 0.0
                self._last_imu_pred_time = now
                if 0.001 <= dt_imu <= 0.05:
                    gx = float(msg.angular_velocity.x)
                    gy = float(msg.angular_velocity.y)
                    gz = float(msg.angular_velocity.z)
                    try:
                        R_w_imu = Rotation.from_quat(self.current_quat).as_matrix()
                        w_world = R_w_imu @ np.array([gx, gy, gz], dtype=np.float64)
                        wz_world = float(w_world[2])
                    except Exception:
                        wz_world = gz

                    if math.isfinite(wz_world) and abs(wz_world) > 0.015:
                        with self.lock:
                            self._pred_yaw = math.atan2(
                                math.sin(self._pred_yaw + wz_world * dt_imu),
                                math.cos(self._pred_yaw + wz_world * dt_imu),
                            )
                            pred_yaw = self._pred_yaw
                        if (now - self._last_imu_emit_time) >= 0.028:  # ~35 Hz smooth heading updates
                            self._last_imu_emit_time = now
                            self.pose_received.emit(self.current_x, self.current_y, pred_yaw)

        # Match RELIABLE QoS of orbbec_camera (`color_qos=default`) & fastlivo_mapping (`/cloud_registered`, `/aft_mapped_to_init`)
        # so 3 MB RGB and 2.5 MB PointCloud2 payloads use FastDDS Shared Memory without BEST_EFFORT frame drops.
        qos_reliable_stream = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        # Depth=1 BEST_EFFORT for 200 Hz IMU so at most 1 IMU callback runs per wait-set wake
        qos_imu_single = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )

        node.create_subscription(Image, "/camera/color/image_raw", rgb_color_callback, qos_reliable_stream)
        node.create_subscription(Image, "/camera/depth/image_raw", depth_callback, qos_reliable_stream)
        node.create_subscription(CameraInfo, "/camera/color/camera_info", rgb_info_callback, qos_reliable_stream)
        node.create_subscription(CameraInfo, "/camera/depth/camera_info", depth_info_callback, qos_reliable_stream)
        node.create_subscription(PointCloud2, "/cloud_registered", registered_cloud_callback, qos_reliable_stream)
        node.create_subscription(Odometry, "/aft_mapped_to_init", slam_odom_callback, qos_reliable_stream)
        node.create_subscription(Imu, "/rslidar_imu_data", rslidar_imu_liveness_callback, qos_imu_single)

        # Single persistent SingleThreadedExecutor (preserves round-robin `_cb_iter` across spin_once calls)
        executor = SingleThreadedExecutor()
        executor.add_node(node)

        def spin_ros():
            while self.running and rclpy.ok():
                try:
                    executor.spin_once(timeout_sec=0.005)
                except Exception:
                    time.sleep(0.002)

        def run_camera_preview_loop():
            while self.running and rclpy.ok():
                try:
                    self._drain_and_emit_camera_preview()
                except Exception:
                    pass
                time.sleep(0.025)

        spin_thread = threading.Thread(target=spin_ros, daemon=True, name="ROS2_Spin_Thread")
        preview_thread = threading.Thread(target=run_camera_preview_loop, daemon=True, name="Camera_Preview_Thread")
        spin_thread.start()
        preview_thread.start()

        # High-Rate Background 3D Map Processing & UI Dispatch Loop (~50 Hz queue drain, ~12.5 Hz map update)
        last_emit_time = 0.0
        while self.running and rclpy.ok():
            try:
                self._process_pending_sensor_queues()
            except Exception:
                pass
            now = time.time()
            if now - last_emit_time >= 0.08:  # 12.5 Hz smooth Map & Telemetry update
                last_emit_time = now
                try:
                    self.dispatch_live_map_and_telemetry()
                except Exception:
                    pass
            time.sleep(0.015)

        try:
            executor.shutdown()
        except Exception:
            pass
        try:
            node.destroy_node()
        except Exception:
            pass

    def _compute_rate(self, ts_deque, now, window=3.5):
        while len(ts_deque) > 0 and (now - ts_deque[0]) > window:
            ts_deque.popleft()
        if len(ts_deque) < 2:
            return 0.0
        span = max(0.2, ts_deque[-1] - ts_deque[0])
        return (len(ts_deque) - 1) / span

    def dispatch_live_map_and_telemetry(self, force: bool = False):
        """Prepares sliced points and live telemetry at 12.5 Hz with zero Python list conversion overhead."""
        now = time.time()
        with self.lock:
            mode = self.display_mode
            a_arr, s_arr = self._get_voxel_arrays_locked()
            elev_stats = dict(self.live_elevation_stats)
            rslidar_hz = self._compute_rate(self._ts_rslidar, now)
            lidar_hz = self._compute_rate(self._ts_lidar, now)
            camera_hz = self._compute_rate(self._ts_camera, now)
            depth_hz = self._compute_rate(self._ts_depth, now)
            slam_hz = self._compute_rate(self._ts_slam, now)
            rslidar_age = (now - self.last_rslidar_time) if self.last_rslidar_time > 0 else 999.0
            lidar_age = (now - self.last_lidar_time) if self.last_lidar_time > 0 else 999.0
            camera_age = (now - self.last_camera_time) if self.last_camera_time > 0 else 999.0
            depth_age = (now - self.last_depth_time) if self.last_depth_time > 0 else 999.0
            slam_age = (now - self.last_slam_time) if self.last_slam_time > 0 else 999.0
            lin_spd = self.current_lin_speed
            ang_spd = self.current_ang_speed
            gated_turns = self.cnt_depth_gated_turn
            live_clean_on = self.live_cleanup_enabled
            live_ground_on = self.live_ground_lock_enabled
            preview_on = self.cleanup_preview_active
            preview_pts = self.cleanup_preview_points.copy() if (self.cleanup_preview_active and self.cleanup_preview_points is not None) else None
            m12_cfg = self.m12_params
            cur_x, cur_y, cur_z = self.current_x, self.current_y, self.current_z
            cur_q = list(self.current_quat)
            stats_snap = self.last_cleanup_stats
            rt_ms = self.live_cleanup_time_ms

        n_airy = len(a_arr)
        n_stereo = len(s_arr)

        # Vectorized Telemetry Metrics (<0.05ms)
        if n_airy > 0:
            st_col = np.rint(a_arr[:, 6]).astype(np.int32)
            n_rgb_obs = int(np.sum((st_col == 1) | (st_col == 3)))
            n_dual_obs = int(np.sum(st_col == 3))
            rgb_cov_pct = (n_rgb_obs / n_airy) * 100.0
            dual_cov_pct = (n_dual_obs / n_airy) * 100.0
        else:
            n_rgb_obs = 0
            n_dual_obs = 0
            rgb_cov_pct = 0.0
            dual_cov_pct = 0.0

        if n_stereo > 0:
            n_accepted = int(np.sum(s_arr[:, 6] > 0.5))
            n_rejected = n_stereo - n_accepted
            stereo_accept_pct = (n_accepted / n_stereo) * 100.0
        else:
            n_accepted = 0
            n_rejected = 0
            stereo_accept_pct = 0.0

        # Elevation Breakdown %
        t_elev = max(1, elev_stats.get("total", 1))
        floor_pct = (elev_stats.get("floor", 0) / t_elev) * 100.0
        wall_pct = (elev_stats.get("wall", 0) / t_elev) * 100.0
        ceil_pct = (elev_stats.get("ceiling", 0) / t_elev) * 100.0

        cslam_summary = self.cslam_engine.get_status_summary()

        telemetry = {
            "n_airy": n_airy,
            "n_stereo": n_stereo,
            "n_fused": n_airy + n_accepted,
            "rgb_cov_pct": rgb_cov_pct,
            "dual_cov_pct": dual_cov_pct,
            "n_accepted": n_accepted,
            "n_rejected": n_rejected,
            "stereo_accept_pct": stereo_accept_pct,
            "floor_pct": floor_pct,
            "wall_pct": wall_pct,
            "ceil_pct": ceil_pct,
            "depth_min": self.depth_min_m,
            "depth_max": self.depth_max_m,
            "rslidar_hz": rslidar_hz,
            "rslidar_age": rslidar_age,
            "lidar_hz": lidar_hz,
            "camera_hz": camera_hz,
            "depth_hz": depth_hz,
            "slam_hz": slam_hz,
            "lidar_age": lidar_age,
            "camera_age": camera_age,
            "depth_age": depth_age,
            "slam_age": slam_age,
            "lin_speed": lin_spd,
            "ang_speed": ang_spd,
            "gated_turns": gated_turns,
            "m12_live_enabled": live_clean_on,
            "m12_ground_lock": live_ground_on,
            "m12_preview_active": preview_on,
            "m12_time_ms": stats_snap.processing_time_ms if preview_on else rt_ms,
            "m12_input_pts": stats_snap.input_points,
            "m12_output_pts": stats_snap.output_points,
            "m12_tilt_deg": stats_snap.ground_result.tilt_angle_deg,
            "cslam": cslam_summary,
        }
        self.telemetry_updated.emit(telemetry)

        # Smooth 10-12.5 Hz Map Refresh (0.08s interval) or immediate forced redraw on layer button click
        should_refresh_map = (
            force or
            preview_on or
            self._cached_map_points_data is None or
            self._cached_map_points_data.get("mode") != mode or
            (now - self._last_map_dispatch_time >= 0.08)
        )
        if not should_refresh_map:
            return

        self._last_map_dispatch_time = now

        # Extract 3D local points [x, y, z, r, g, b] (RGB 0..255) for active mode
        if preview_on and preview_pts is not None:
            active_local_3d = preview_pts
        else:
            active_local_3d = self.extract_raw_3d_points_for_mode(mode, a_arr, s_arr)
            if live_clean_on and len(active_local_3d) > 20:
                rt_params = CleanupParams(
                    enable_voxel_downsample=m12_cfg.enable_voxel_downsample,
                    voxel_leaf_size_m=m12_cfg.voxel_leaf_size_m,
                    enable_sor=m12_cfg.enable_sor,
                    sor_mean_k=min(m12_cfg.sor_mean_k, 16),
                    sor_std_mul=m12_cfg.sor_std_mul,
                    enable_ror=m12_cfg.enable_ror,
                    ror_radius_m=m12_cfg.ror_radius_m,
                    ror_min_neighbors=m12_cfg.ror_min_neighbors,
                    enable_ground_alignment=live_ground_on,
                    ground_dist_threshold_m=m12_cfg.ground_dist_threshold_m,
                    ground_max_iterations=80,
                    ground_max_tilt_deg=m12_cfg.ground_max_tilt_deg,
                    align_ground_to_z_zero=True,
                    remove_ground_plane=m12_cfg.remove_ground_plane,
                )
                active_local_3d, rt_stats = PointCloudProcessor.clean_realtime_fast(active_local_3d, rt_params)
                with self.lock:
                    self.live_cleanup_time_ms = rt_stats.processing_time_ms
                    self.last_cleanup_stats = rt_stats

        # Update C-SLAM local state & merge remote peer scanners
        try:
            imu_rpy = tuple(Rotation.from_quat(cur_q).as_euler("xyz", degrees=False))
        except Exception:
            imu_rpy = (0.0, 0.0, self.current_yaw)

        # Always feed authoritative fused cloud to C-SLAM network broadcast while displaying `active_local_3d` locally
        if len(active_local_3d) > 0:
            local_norm = active_local_3d.copy()
            if np.max(local_norm[:, 3:6]) > 1.05:
                local_norm[:, 3:6] /= 255.0
            if mode in ("colorized_fused", "fused", "airy_only"):
                self.cslam_engine.update_local_state(
                    points_xyz_rgb=local_norm,
                    pos_xyz=(cur_x, cur_y, cur_z),
                    imu_rpy_rad=imu_rpy,
                )
        else:
            local_norm = np.zeros((0, 6), dtype=np.float32)

        merged_3d, peer_descriptors = self.cslam_engine.get_merged_cslam_scene(
            local_points_override=local_norm,
            preserve_local_colors=(mode != "colorized_fused"),
        )

        # Extract 2D Floorplan Slice as contiguous float32 NumPy array (N, 5) [wx, wy, r, g, b]
        # Crisp eye-level horizontal cross-section cleanly isolates vertical walls without floor/ceiling fog.
        display_pts = np.zeros((0, 5), dtype=np.float32)
        if len(merged_3d) > 0:
            rel_z = merged_3d[:, 2] - cur_z
            if mode == "airy_only":
                mask_slice = (rel_z >= -0.30) & (rel_z <= 0.35)
            elif mode == "stereo_only":
                mask_slice = (rel_z >= -0.75) & (rel_z <= 0.35)
            else:
                mask_slice = (rel_z >= -0.35) & (rel_z <= 0.35)

            sliced = merged_3d[mask_slice]
            if len(sliced) == 0:
                med_z = float(np.median(merged_3d[:, 2]))
                mask_slice = (merged_3d[:, 2] >= med_z - 0.25) & (merged_3d[:, 2] <= med_z + 0.25)
                sliced = merged_3d[mask_slice]
            if len(sliced) == 0:
                sliced = merged_3d

            if len(sliced) > 35000:
                stride = max(1, len(sliced) // 35000)
                sliced = sliced[::stride]

            rgb_scale = 255.0 if np.max(sliced[:, 3:6]) <= 1.05 else 1.0
            display_pts = np.empty((len(sliced), 5), dtype=np.float32)
            display_pts[:, 0] = sliced[:, 0]
            display_pts[:, 1] = sliced[:, 1]
            display_pts[:, 2] = np.clip(sliced[:, 3] * rgb_scale, 0, 255)
            display_pts[:, 3] = np.clip(sliced[:, 4] * rgb_scale, 0, 255)
            display_pts[:, 4] = np.clip(sliced[:, 5] * rgb_scale, 0, 255)

        payload = {
            "points": display_pts,
            "points_3d": [],
            "mode": mode,
            "n_airy": n_airy,
            "n_stereo": n_stereo,
            "n_accepted": n_accepted,
            "n_rejected": n_rejected,
            "n_rgb_obs": n_rgb_obs,
            "peers": peer_descriptors,
            "local_color_hex": cslam_summary["local_color_hex"],
            "cslam_color_mode": cslam_summary["color_mode"],
            "m12_live_enabled": live_clean_on,
            "m12_preview_active": preview_on,
        }
        self._cached_map_points_data = payload
        self.map_points_updated.emit(payload)

    def start_recording_session(self):
        with self.lock:
            self.live_airy_voxels.clear()
            self.live_stereo_voxels.clear()
            self._airy_cache_dirty = True
            self._stereo_cache_dirty = True
            self.cleanup_preview_active = False
            self.cleanup_preview_points = None
            self.trajectory_records.clear()
            self.airy_sweeps.clear()
            self.gemini_frames.clear()
            self.rgb_keyframes.clear()
            self._raw_rgb_queue.clear()
            self._raw_depth_queue.clear()
            self._raw_lidar_queue.clear()
            self.cnt_depth_gated_turn = 0
            self.session_start_epoch = time.time()
            self.is_session_recording = True
        self.cslam_engine.set_recording_active(True)

    def stop_recording_session(self):
        with self.lock:
            self.is_session_recording = False
            self.session_duration = time.time() - self.session_start_epoch if self.session_start_epoch > 0 else 0.0
        self.cslam_engine.set_recording_active(False)

    def export_production_dataset(self, save_dir, dataset_name, trajectory, flash_anchors):
        """Exports Qualified Production Bundle (Airy Pure XYZ, Stereo XYZ+RGB, Fused XYZ+RGB, plus M12 & C-SLAM Merged)."""
        os.makedirs(save_dir, exist_ok=True)
        with self.lock:
            sweeps = list(self.airy_sweeps)
            g_frames = list(self.gemini_frames)
            keyframes = list(self.rgb_keyframes)
            dur = self.session_duration
            min_d = self.depth_min_m
            max_d = self.depth_max_m
            agree_tol = self.surface_agreement_m
            d_rgb = self.d_rgb.copy()

        # 1. Trajectory CSV & TUM
        traj_csv = os.path.join(save_dir, "trajectory.csv")
        traj_tum = os.path.join(save_dir, "trajectory_tum.txt")
        with open(traj_csv, 'w') as f_csv, open(traj_tum, 'w') as f_tum:
            f_csv.write("timestamp_sec,x_m,y_m,z_m,qx,qy,qz,qw,yaw_rad\n")
            for r in self.trajectory_records:
                sec, x, y, z, qx, qy, qz, qw, yaw = r
                f_csv.write(f"{sec:.6f},{x:.6f},{y:.6f},{z:.6f},{qx:.6f},{qy:.6f},{qz:.6f},{qw:.6f},{yaw:.6f}\n")
                f_tum.write(f"{sec:.6f} {x:.6f} {y:.6f} {z:.6f} {qx:.6f} {qy:.6f} {qz:.6f} {qw:.6f}\n")

        # 2. Multi-Sweep Voxel Centroid Filtering & Ghost Removal for Airy LiDAR Cloud
        airy_clean = multi_sweep_voxel_centroids(sweeps, voxel_size=0.015)

        # 3. Voxelize Stereo Cloud
        all_g_xyz, all_g_rgb = [], []
        for gf in g_frames:
            if gf[1] is not None and len(gf[1]) > 0:
                all_g_xyz.append(gf[1])
                all_g_rgb.append(gf[2])

        if len(all_g_xyz) > 0:
            g_raw_xyz = np.vstack(all_g_xyz)
            g_raw_rgb = np.vstack(all_g_rgb)
            stereo_clean_xyz, stereo_clean_rgb = fast_voxel_downsample(g_raw_xyz, g_raw_rgb, voxel_size=0.015)
        else:
            stereo_clean_xyz = np.empty((0, 3), dtype=np.float64)
            stereo_clean_rgb = np.empty((0, 3), dtype=np.uint8)

        # 4. Multi-Factor ASPS Fusion & Z-Buffered Plumb-Bob Keyframe Colormapping
        airy_rgb = np.full((len(airy_clean), 3), 175, dtype=np.uint8)
        if len(keyframes) > 0 and len(airy_clean) > 0:
            best_dist = np.full(len(airy_clean), np.inf, dtype=np.float64)
            step = max(1, len(keyframes) // 60)
            for sec_k, img_k, R_w_imu_k, pos_k in keyframes[::step]:
                p_imu = (airy_clean - pos_k) @ R_w_imu_k
                p_lid = (p_imu - t_IMU_LIDAR) @ R_IMU_LIDAR
                p_rgb = p_lid @ R_RGB_LIDAR.T + t_RGB_LIDAR
                u, v, zr = project_pts_to_rgb_plumb_bob(
                    p_rgb, self.fx_rgb, self.fy_rgb, self.cx_rgb, self.cy_rgb, d_rgb
                )
                h_k, w_k = img_k.shape[:2]
                in_fov = (zr > 0.3) & (zr < 6.0) & (u >= 0) & (u < w_k) & (v >= 0) & (v < h_k)
                fov_idx = np.nonzero(in_fov)[0]
                if len(fov_idx) == 0:
                    continue

                u_f = u[fov_idx]
                v_f = v[fov_idx]
                z_f = zr[fov_idx]

                # 320x200 Z-buffer occlusion gate so foreground walls block occluded background surfaces
                u_b = np.clip(u_f // 4, 0, (w_k // 4) - 1)
                v_b = np.clip(v_f // 4, 0, (h_k // 4) - 1)
                z_buf = np.full((h_k // 4, w_k // 4), np.inf, dtype=np.float64)
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
                    airy_rgb[upd_idx] = img_k[v[upd_idx], u[upd_idx]]

        # Gating Stereo Contribution
        if len(airy_clean) > 0 and len(stereo_clean_xyz) > 0:
            tree_a = cKDTree(airy_clean)
            dists, _ = tree_a.query(stereo_clean_xyz, k=1, workers=-1)
            mask_snap = dists <= 0.025
            mask_supp = dists > 0.025
            fused_xyz = np.vstack((airy_clean, stereo_clean_xyz[mask_snap], stereo_clean_xyz[mask_supp]))
            fused_rgb = np.vstack((airy_rgb, stereo_clean_rgb[mask_snap], stereo_clean_rgb[mask_supp]))
        else:
            fused_xyz = airy_clean if len(airy_clean) > 0 else stereo_clean_xyz
            fused_rgb = airy_rgb if len(airy_clean) > 0 else stereo_clean_rgb

        # Helper PLY/PCD batch writers
        def write_ply(filepath, xyz, rgb=None):
            n = len(xyz)
            has_color = (rgb is not None and len(rgb) == n)
            header = f"ply\nformat ascii 1.0\nelement vertex {n}\nproperty float x\nproperty float y\nproperty float z\n"
            if has_color:
                header += "property uchar red\nproperty uchar green\nproperty uchar blue\n"
            header += "end_header\n"
            if has_color:
                table = np.zeros(n, dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("r", "u1"), ("g", "u1"), ("b", "u1")])
                table["x"], table["y"], table["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
                table["r"], table["g"], table["b"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
                with open(filepath, 'wb') as f:
                    f.write(header.encode('ascii'))
                    np.savetxt(f, table, fmt="%.4f %.4f %.4f %d %d %d")
            else:
                table = np.zeros(n, dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])
                table["x"], table["y"], table["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
                with open(filepath, 'wb') as f:
                    f.write(header.encode('ascii'))
                    np.savetxt(f, table, fmt="%.4f %.4f %.4f")

        def write_pcd(filepath, xyz, rgb=None):
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
                table["x"], table["y"], table["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
                table["c"] = rgb_uint
                with open(filepath, 'wb') as f:
                    f.write(header.encode('ascii'))
                    np.savetxt(f, table, fmt="%.4f %.4f %.4f %u")
            else:
                table = np.zeros(n, dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])
                table["x"], table["y"], table["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
                with open(filepath, 'wb') as f:
                    f.write(header.encode('ascii'))
                    np.savetxt(f, table, fmt="%.4f %.4f %.4f")

        # 5. Write Standardized Output Files
        # pointcloud_airy: Authoritative Pure LiDAR Geometry Only
        write_ply(os.path.join(save_dir, "pointcloud_airy.ply"), airy_clean, None)
        write_pcd(os.path.join(save_dir, "pointcloud_airy.pcd"), airy_clean, None)

        # pointcloud_stereo: Gemini Stereo Depth with Camera Colors
        write_ply(os.path.join(save_dir, "pointcloud_stereo.ply"), stereo_clean_xyz, stereo_clean_rgb)
        write_pcd(os.path.join(save_dir, "pointcloud_stereo.pcd"), stereo_clean_xyz, stereo_clean_rgb)

        # pointcloud_fused_rgb: Final ASPS Fused Cloud with Truecolor 24-bit RGB
        write_ply(os.path.join(save_dir, "pointcloud_fused_rgb.ply"), fused_xyz, fused_rgb)
        write_pcd(os.path.join(save_dir, "pointcloud_fused_rgb.pcd"), fused_xyz, fused_rgb)

        # Also export C-SLAM merged multi-scanner cloud if peers exist
        cslam_merged, cslam_peers = self.cslam_engine.get_merged_cslam_scene()
        if len(cslam_merged) > 0 and len(cslam_peers) > 0:
            PointCloudProcessor.export_point_cloud(
                cslam_merged,
                os.path.join(save_dir, "pointcloud_cslam_merged.ply"),
                ExportFormat.PLY_BINARY,
            )

        # Flash Anchors JSON
        with open(os.path.join(save_dir, "flash_anchors.json"), 'w') as fa:
            json.dump(flash_anchors, fa, indent=2)

        # Metadata JSON
        meta = {
            "session_id": dataset_name,
            "export_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "capture_duration_seconds": float(round(dur, 2)),
            "airy_points": int(len(airy_clean)),
            "stereo_points": int(len(stereo_clean_xyz)),
            "fused_points": int(len(fused_xyz)),
            "cslam_peers": int(len(cslam_peers)),
            "cslam_merged_points": int(len(cslam_merged)),
            "depth_range_gate_m": [min_d, max_d],
            "surface_agreement_tolerance_m": agree_tol,
            "spatial_bounding_box": {
                "x_min": float(fused_xyz[:, 0].min()) if len(fused_xyz) > 0 else 0.0,
                "x_max": float(fused_xyz[:, 0].max()) if len(fused_xyz) > 0 else 0.0,
                "y_min": float(fused_xyz[:, 1].min()) if len(fused_xyz) > 0 else 0.0,
                "y_max": float(fused_xyz[:, 1].max()) if len(fused_xyz) > 0 else 0.0,
                "z_min": float(fused_xyz[:, 2].min()) if len(fused_xyz) > 0 else 0.0,
                "z_max": float(fused_xyz[:, 2].max()) if len(fused_xyz) > 0 else 0.0,
            }
        }
        with open(os.path.join(save_dir, "scan_metadata.json"), 'w') as fm:
            json.dump(meta, fm, indent=2)

        return meta

    def stop(self):
        self.running = False
        try:
            self.cslam_engine.stop()
        except Exception:
            pass
        self.wait()


class CameraPreviewCanvas(QFrame):
    """
    Live Clean Isolated RGB Stream (Left) + Isolated Depth Heatmap Stream (Right).
    Styled in Apple Glass (visionOS) + Artificial Vision White & Gold Translucent AR aesthetic.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        if hasattr(Qt, "WidgetAttribute"):
            self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        else:
            self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setStyleSheet(
            "background-color: transparent; border: 1px solid rgba(186, 161, 100, 0.42); border-radius: 12px;"
        )
        self.current_image = None
        self.logo_pix = load_av_logo_pixmap(42)

    def update_frame(self, qimg: QImage):
        self.current_image = qimg
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        try:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            w, h = self.width(), self.height()

            # Ultra-translucent Apple Glass card fill + specular white inner edge
            painter.setPen(QPen(QColor(255, 255, 255, 42), 1.0))
            painter.setBrush(QColor(255, 255, 255, 10))
            painter.drawRoundedRect(QRectF(0.5, 0.5, w - 1.0, h - 1.0), 12.0, 12.0)

            # Subtle vertical champagne-gold divider showing RGB (Left) vs Depth Heatmap (Right)
            painter.setPen(QPen(QColor(186, 161, 100, 75), 1, Qt.PenStyle.DashLine))
            painter.drawLine(w // 2, 26, w // 2, h - 8)

            if self.current_image and not self.current_image.isNull():
                pw, ph = self.current_image.width(), self.current_image.height()
                if pw > 0 and ph > 0:
                    avail_h = max(10, h - 26)
                    scale = min((w - 12) / pw, avail_h / ph)
                    tw, th = int(pw * scale), int(ph * scale)
                    px, py = (w - tw) // 2, 24 + (avail_h - th) // 2
                    painter.drawImage(QRect(px, py, tw, th), self.current_image)
            else:
                # Translucent Standby Watermark with Artificial Vision Logo
                if not self.logo_pix.isNull():
                    painter.save()
                    painter.setOpacity(0.60)
                    lx = (w - self.logo_pix.width()) // 2
                    ly = (h - self.logo_pix.height()) // 2 - 4
                    painter.drawPixmap(lx, ly, self.logo_pix)
                    painter.restore()
                painter.setPen(QColor(252, 251, 248, 190))
                painter.setFont(QFont("IBM Plex Sans", 8, QFont.Weight.DemiBold))
                painter.drawText(
                    QRect(0, h // 2 + 22, w, 20),
                    Qt.AlignmentFlag.AlignCenter if hasattr(Qt, "AlignmentFlag") else Qt.AlignCenter,
                    "Optical RGB-D Stream Standby  ·  Orbbec Gemini 336L",
                )

            # Top translucent glass header strip inside card
            painter.fillRect(1, 1, w - 2, 22, QColor(255, 255, 255, 16))
            painter.setPen(QColor(186, 161, 100, 90))
            painter.drawLine(1, 23, w - 2, 23)
            painter.setPen(QColor("#E5C983"))
            painter.setFont(QFont("IBM Plex Mono", 8, QFont.Weight.Bold))
            painter.drawText(10, 15, "OPTICAL RGB STREAM (LEFT)   ·   STEREO DEPTH HEATMAP (RIGHT)")
        finally:
            painter.end()


class RealityCaptureMapCanvas(QFrame):
    """
    Side-HUD Multi-Layer Point Cloud Map & Trajectory Canvas.
    Supports: Airy, Gemini, Fused, Truecolor RGB, Accepted/Rejected, Observation Diag,
    M12 Real-Time Cleanup Indicator, and Unlimited-Peer C-SLAM Multi-Scanner Overlay.
    Styled in Apple Glass (visionOS) + Artificial Vision White & Gold Translucent AR aesthetic.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self._map_lock = threading.Lock()
        if hasattr(Qt, "WidgetAttribute"):
            self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        else:
            self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setStyleSheet(
            "background-color: transparent; border: 1px solid rgba(186, 161, 100, 0.42); border-radius: 12px;"
        )

        self.trajectory_path = []
        self.rendered_points = np.zeros((0, 5), dtype=np.float32)  # contiguous (N, 5) [wx, wy, r, g, b]
        self.peer_descriptors = []  # list of C-SLAM peer status/pose/trajectory dicts
        self.current_mode = "colorized_fused"
        self.m12_live_enabled = False
        self.m12_preview_active = False
        self.cslam_color_mode = "SCANNER_COLORS"
        self.flash_anchors = []
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_yaw = 0.0
        self.gravity_roll_deg = 0.0
        self.gravity_pitch_deg = 0.0
        self.view_radius_m = 25.0
        self.scale = 10.0
        self.total_distance = 0.0
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        if delta > 0:
            self.view_radius_m = max(5.0, self.view_radius_m * 0.85)
        elif delta < 0:
            self.view_radius_m = min(120.0, self.view_radius_m * 1.15)
        self.update()

    def zoom_in(self):
        self.view_radius_m = max(5.0, self.view_radius_m * 0.75)
        self.update()

    def zoom_out(self):
        self.view_radius_m = min(120.0, self.view_radius_m * 1.35)
        self.update()

    def set_view_radius(self, radius_m: float):
        self.view_radius_m = radius_m
        self.update()

    def clear_map(self):
        with self._map_lock:
            self.rendered_points = np.zeros((0, 5), dtype=np.float32)
            self.trajectory_path.clear()
            self.flash_anchors.clear()
        self.total_distance = 0.0
        self.update()

    def add_flash_anchor(self, label: str, x: float, y: float, z: float = 0.0):
        with self._map_lock:
            self.flash_anchors.append({
                "label": label,
                "x": float(x),
                "y": float(y),
                "z": float(z),
                "time": time.time(),
            })
        self.update()

    def update_map_points(self, data: dict):
        pts = data.get("points", None)
        if pts is None:
            pts_arr = np.zeros((0, 5), dtype=np.float32)
        elif isinstance(pts, np.ndarray):
            pts_arr = pts
        else:
            pts_arr = np.asarray(pts, dtype=np.float32) if len(pts) > 0 else np.zeros((0, 5), dtype=np.float32)
        with self._map_lock:
            self.rendered_points = pts_arr
            self.current_mode = data.get("mode", "colorized_fused")
            self.peer_descriptors = data.get("peers", [])
            self.m12_live_enabled = data.get("m12_live_enabled", False)
            self.m12_preview_active = data.get("m12_preview_active", False)
            self.cslam_color_mode = data.get("cslam_color_mode", "SCANNER_COLORS")
        self.update()

    def set_active_mode_immediate(self, mode_key: str):
        with self._map_lock:
            self.current_mode = mode_key
        self.update()

    def update_pose(self, x: float, y: float, yaw: float):
        with self._map_lock:
            if self.trajectory_path:
                lx, ly, _ = self.trajectory_path[-1]
                d = math.hypot(x - lx, y - ly)
                self.total_distance += d
                if d > 0.04:
                    self.trajectory_path.append((x, y, yaw))
                    if len(self.trajectory_path) > 3000:
                        self.trajectory_path = self.trajectory_path[-3000:]
            else:
                self.trajectory_path.append((x, y, yaw))

            self.current_x = x
            self.current_y = y
            self.current_yaw = yaw
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        try:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            w, h = self.width(), self.height()
            cx, cy = w // 2, h // 2

            # Translucent Apple Glass Surface (see-through while walking in AR glasses)
            painter.setPen(QPen(QColor(255, 255, 255, 42), 1.0))
            painter.setBrush(QColor(255, 255, 255, 10))
            painter.drawRoundedRect(QRectF(0.5, 0.5, w - 1.0, h - 1.0), 12.0, 12.0)

            # Subtle Translucent Center Crosshair
            painter.setPen(QPen(QColor(255, 255, 255, 32), 1, Qt.PenStyle.SolidLine))
            painter.drawLine(cx, 26, cx, h - 32)
            painter.drawLine(12, cy, w - 12, cy)

            # Dynamic Scale
            min_dim = min(w, h)
            self.scale = (min_dim * 0.45) / max(1.0, float(self.view_radius_m))

            # Concentric Champagne-Gold Range Rings (5m, 10m, 20m, 30m, 50m)
            pen_ring = QPen(QColor(186, 161, 100, 75), 1.0, Qt.PenStyle.DashLine)
            painter.setPen(pen_ring)
            font_ring = QFont("IBM Plex Mono", 7)
            painter.setFont(font_ring)
            for ring_m in [5.0, 10.0, 20.0, 30.0, 50.0]:
                r_px = ring_m * self.scale
                if r_px < min_dim * 0.95:
                    painter.drawEllipse(QPointF(cx, cy), r_px, r_px)
                    painter.setPen(QColor("#E5C983"))
                    painter.drawText(int(cx + r_px + 4), int(cy - 2), f"{int(ring_m)}m")
                    painter.setPen(pen_ring)

            def world_to_screen(wx, wy):
                dx = wx - self.current_x
                dy = wy - self.current_y
                rot_ang = (math.pi / 2.0) - self.current_yaw
                cos_y = math.cos(rot_ang)
                sin_y = math.sin(rot_ang)
                rx = dx * cos_y - dy * sin_y
                ry = dx * sin_y + dy * cos_y
                sx = int(cx + (rx * self.scale))
                sy = int(cy - (ry * self.scale))
                return sx, sy

            with self._map_lock:
                pts_arr = self.rendered_points
                path_snap = list(self.trajectory_path)
                peers_snap = list(self.peer_descriptors)
                curr_dist = self.total_distance
                mode_label = self.current_mode
                m12_on = self.m12_live_enabled
                m12_prev = self.m12_preview_active

            # 1. Zero-Copy Vectorized Point Cloud Splatting (<0.6ms, rotates instantly with heading yaw!)
            if len(pts_arr) > 0 and w > 4 and h > 4:
                dx = pts_arr[:, 0] - self.current_x
                dy = pts_arr[:, 1] - self.current_y
                rot_ang = (math.pi / 2.0) - self.current_yaw
                cos_y = math.cos(rot_ang)
                sin_y = math.sin(rot_ang)
                rx = dx * cos_y - dy * sin_y
                ry = dx * sin_y + dy * cos_y
                sx = np.rint(cx + rx * self.scale).astype(np.int32)
                sy = np.rint(cy - ry * self.scale).astype(np.int32)

                in_bounds = (sx >= 0) & (sx < w - 1) & (sy >= 0) & (sy < h - 1)
                if np.any(in_bounds):
                    sx_v = sx[in_bounds]
                    sy_v = sy[in_bounds]
                    rgb_v = np.clip(pts_arr[in_bounds, 2:5], 35, 255).astype(np.uint8)
                    rgba_buf = np.zeros((h, w, 4), dtype=np.uint8)
                    rgba_v = np.empty((len(sx_v), 4), dtype=np.uint8)
                    rgba_v[:, :3] = rgb_v
                    rgba_v[:, 3] = 245
                    # 2x2 pixel splat for crisp visibility
                    rgba_buf[sy_v, sx_v] = rgba_v
                    rgba_buf[sy_v + 1, sx_v] = rgba_v
                    rgba_buf[sy_v, sx_v + 1] = rgba_v
                    rgba_buf[sy_v + 1, sx_v + 1] = rgba_v
                    fmt_rgba = QImage.Format.Format_RGBA8888 if hasattr(QImage, "Format") else QImage.Format_RGBA8888
                    qimg = QImage(rgba_buf.data, w, h, w * 4, fmt_rgba)
                    painter.drawImage(0, 0, qimg)

            # 2. Local Path of Travel Line (Artificial Vision Champagne Gold #BAA164)
            if len(path_snap) > 1:
                pen_path = QPen(QColor("#E5C983"), 2.2, Qt.PenStyle.SolidLine)
                painter.setPen(pen_path)
                for i in range(len(path_snap) - 1):
                    p1 = world_to_screen(path_snap[i][0], path_snap[i][1])
                    p2 = world_to_screen(path_snap[i + 1][0], path_snap[i + 1][1])
                    if 0 <= p1[0] < w and 0 <= p2[0] < w:
                        painter.drawLine(p1[0], p1[1], p2[0], p2[1])

            # 2B. Remote C-SLAM Peer Scanner Trajectories, Pose Arrows & Status Badges
            for peer in peers_snap:
                p_color = QColor(peer.get("color_hex", "#ff5e3a"))
                p_traj = peer.get("world_traj", [])
                p_state = peer.get("alignment_state", "SEPARATED_IMU_ORIENTED")
                is_snapped = p_state == "OVERLAP_SNAPPED"

                if len(p_traj) > 1:
                    pen_style = Qt.PenStyle.SolidLine if is_snapped else Qt.PenStyle.DashLine
                    painter.setPen(QPen(p_color, 2, pen_style))
                    for i in range(len(p_traj) - 1):
                        tp1 = world_to_screen(p_traj[i][0], p_traj[i][1])
                        tp2 = world_to_screen(p_traj[i + 1][0], p_traj[i + 1][1])
                        if -100 <= tp1[0] < w + 100 and -100 <= tp2[0] < w + 100:
                            painter.drawLine(tp1[0], tp1[1], tp2[0], tp2[1])

                w_pos = peer.get("world_pos", (0.0, 0.0, 0.0))
                px_s, py_s = world_to_screen(w_pos[0], w_pos[1])
                if 0 <= px_s < w and 0 <= py_s < h:
                    painter.save()
                    painter.translate(px_s, py_s)
                    rel_yaw = peer.get("imu_rpy_deg", (0, 0, 0))[2] * (math.pi / 180.0) - self.current_yaw
                    painter.rotate(math.degrees(-rel_yaw))
                    peer_arrow = QPolygonF([
                        QPointF(0, -11),
                        QPointF(-6, 8),
                        QPointF(0, 4),
                        QPointF(6, 8),
                    ])
                    painter.setPen(QPen(QColor("#FFFFFF"), 1.2))
                    painter.setBrush(QBrush(p_color))
                    painter.drawPolygon(peer_arrow)
                    painter.restore()

                    # Draw Peer Label Badge
                    painter.setPen(p_color.lighter(120))
                    painter.setFont(QFont("IBM Plex Mono", 7, QFont.Weight.Bold))
                    tag = f"SNAPPED {peer.get('overlap_ratio', 0.0)*100:.0f}%" if is_snapped else "IMU+OFFSET"
                    painter.drawText(px_s + 10, py_s + 4, f"{peer.get('scanner_name', 'Peer')} [{tag}]")

            # 2C. FARO Flash Scan Waypoint Anchors (Pulsating Gold Waypoint Pins)
            with self._map_lock:
                anchors_snap = list(self.flash_anchors)

            for anchor in anchors_snap:
                ax, ay = anchor["x"], anchor["y"]
                label = anchor.get("label", "w00")
                asx, asy = world_to_screen(ax, ay)
                if -25 <= asx < w + 25 and -25 <= asy < h + 25:
                    # Outer translucent halo ring
                    painter.setPen(QPen(QColor(229, 201, 131, 160), 1.5))
                    painter.setBrush(QColor(186, 161, 100, 50))
                    painter.drawEllipse(QPointF(asx, asy), 8.5, 8.5)

                    # Center solid gold anchor pin
                    painter.setPen(QPen(QColor("#FCFBF8"), 1.2))
                    painter.setBrush(QBrush(QColor("#E5C983")))
                    painter.drawEllipse(QPointF(asx, asy), 4.0, 4.0)

                    # Dark glass label badge with gold border
                    badge_w = 34
                    badge_h = 16
                    bx = asx + 8
                    by = asy - 18
                    painter.setPen(QPen(QColor(186, 161, 100, 180), 1))
                    painter.setBrush(QBrush(QColor(26, 29, 36, 215)))
                    painter.drawRoundedRect(QRectF(bx, by, badge_w, badge_h), 4.0, 4.0)

                    painter.setPen(QColor("#FCFBF8"))
                    painter.setFont(QFont("IBM Plex Mono", 7, QFont.Weight.Bold))
                    painter.drawText(
                        QRectF(bx, by, badge_w, badge_h),
                        Qt.AlignmentFlag.AlignCenter if hasattr(Qt, "AlignmentFlag") else Qt.AlignCenter,
                        label,
                    )

            # 3. Local Location & Heading Arrow (Champagne Gold #BAA164 with Crisp White Rim)
            painter.save()
            painter.translate(cx, cy)
            arrow_poly = QPolygonF([
                QPointF(0, -14),
                QPointF(-7, 10),
                QPointF(0, 5),
                QPointF(7, 10),
            ])
            painter.setPen(QPen(QColor("#FCFBF8"), 1.5))
            painter.setBrush(QBrush(QColor("#BAA164")))
            painter.drawPolygon(arrow_poly)
            painter.restore()

            # Top Translucent Glass Header Bar
            painter.fillRect(1, 1, w - 2, 24, QColor(255, 255, 255, 16))
            painter.setPen(QColor(186, 161, 100, 90))
            painter.drawLine(1, 25, w - 2, 25)

            # HUD Layer Indicator + Color Legend + Live Point Count
            mode_display_info = {
                "airy_only": ("AIRY 360° LiDAR ONLY [GOLD]", "#E5C983"),
                "stereo_only": ("GEMINI 336L CAMERA DEPTH ONLY [CYAN]", "#22D3EE"),
                "fused": ("FUSED SENSOR LAYERS [GOLD=LiDAR · CYAN=CAMERA]", "#E5C983"),
                "colorized_fused": ("COLORIZED FUSED [TRUE RGB + 360° SLATE]", "#FCFBF8"),
                "accepted_rejected": ("GEMINI GATE [GREEN=ACCEPT · RED=REJECT]", "#4ADE80"),
                "observation_diag": ("COVERAGE DIAG [GOLD=RGB · BLUE=DEPTH · GREEN=BOTH]", "#E5C983"),
            }
            lbl_text, lbl_hex = mode_display_info.get(mode_label, (mode_label.upper(), "#E5C983"))
            painter.setPen(QColor(lbl_hex))
            painter.setFont(QFont("IBM Plex Mono", 8, QFont.Weight.Bold))
            m12_badge = " · M12 PREVIEW" if m12_prev else (" · M12 RT ON" if m12_on else "")
            cslam_badge = f" · C-SLAM({1 + len(peers_snap)})" if len(peers_snap) > 0 else ""
            painter.drawText(
                10,
                16,
                f"VIEW: {lbl_text}  ({len(pts_arr):,} pts){m12_badge}{cslam_badge}",
            )

            if len(pts_arr) == 0:
                painter.setPen(QColor(229, 201, 131, 210))
                painter.setFont(QFont("IBM Plex Mono", 8, QFont.Weight.DemiBold))
                painter.drawText(12, 42, f"● Active View: {lbl_text} — Awaiting sensor points in this layer...")

            # Bottom Translucent Glass Telemetry Bar
            painter.fillRect(1, h - 30, w - 2, 29, QColor(255, 255, 255, 14))
            painter.setPen(QColor(186, 161, 100, 80))
            painter.drawLine(1, h - 30, w - 2, h - 30)
            painter.setPen(QColor("#FCFBF8"))
            painter.setFont(QFont("IBM Plex Mono", 8))
            painter.drawText(
                10,
                h - 17,
                f"POS: X={self.current_x:+.2f}m Y={self.current_y:+.2f}m  |  YAW={math.degrees(self.current_yaw):.1f}°",
            )
            painter.setPen(QColor("#D8BF7C"))
            painter.drawText(
                10,
                h - 5,
                f"PATH: {curr_dist:.2f}m  |  RADIUS: {self.view_radius_m:.0f}m  |  PEERS: {len(peers_snap)}",
            )
        finally:
            painter.end()


class ScanHUBOS(QWidget):
    """
    Main Reality Capture OS Window for scanarMini Rig.
    Designed in the Apple Glass (visionOS) + Artificial Vision (https://prevdc.artificialviz.com/auth)
    White & Gold Ultra-Transparent Spatial UI for maximum walking visibility in AR glasses.
    Includes frameless translucent window and transparent glass Minimize, Maximize/Restore, and Close controls.
    """
    export_finished = pyqtSignal(dict, str, str)

    def __init__(self):
        super().__init__()
        self.is_recording = False
        self.mapping_process = None
        self.sensor_stack_proc = None
        self.slam_stack_proc = None
        self._external_stack_online = ("--stack-managed" in sys.argv)
        self.export_in_progress = False
        self._active_splash = None
        self._drag_offset = None
        # Default glass opacity is ultra-transparent 12% for maximum visibility when walking in AR glasses
        self.glass_opacity_pct = 12
        self.projects_root = "/home/scanar/scanarMini/projects"
        self.legacy_data_root = "/home/scanar/scanarMini/data"
        os.makedirs(os.path.join(self.projects_root, "Default_Project", "datasets"), exist_ok=True)
        self.active_project_name = "Default_Project"
        self.active_dataset_name = f"Dataset_{datetime.now().strftime('%m%d_%H%M')}"

        self.diag_recorder = DiagnosticRecorder(base_dir="/home/scanar/scanarMini/diagnostics") if DiagnosticRecorder else None
        self.ros_worker = Ros2BridgeWorker(diag_recorder=self.diag_recorder)

        # Elapsed Timer
        self.scan_elapsed_seconds = 0
        self.scan_clock_timer = QTimer()
        self.scan_clock_timer.timeout.connect(self.update_scan_clock_tick)

        # Live Hub Refresh Timer (updates discovered active C-SLAM captures on LAN at 1 Hz)
        self.hub_refresh_timer = QTimer()
        self.hub_refresh_timer.timeout.connect(self.refresh_active_cslam_captures_ui)

        # NavVis 2-Stage Ritual & FARO Flash Scan Waypoint State
        self.anchor_count = 0
        self.is_calibrated = False
        self.calib_ticks = 0
        self.calib_timer = None
        self.calib_ax_sum = 0.0
        self.calib_ay_sum = 0.0
        self.calib_az_sum = 0.0

        self.ros_worker.rotation_progress_received.connect(self.update_rotation_ritual_ui)
        self.ros_worker.rotation_completed_received.connect(self.on_rotation_ritual_completed)

        self.export_finished.connect(self.on_export_finished)
        self.init_ui()
        self.refresh_projects_hub_lists()
        self.hub_refresh_timer.start(1000)

    def show_brand_splash(self, duration_ms: int = 2400):
        """Displays the Artificial Vision Translucent White & Gold Splash Screen on demand or startup."""
        if self._active_splash is not None:
            try:
                self._active_splash.close()
            except Exception:
                pass
        self._active_splash = ArtificialVisionSplashScreen()
        self._active_splash.set_status("Artificial Vision  ·  scanHUB Apple Glass AR OS Ready", 100)
        self._active_splash.show()
        self._active_splash.raise_()
        QTimer.singleShot(duration_ms, self._active_splash.close)

    def toggle_window_maximize(self):
        """Toggles between maximized and normal window state for the transparent frameless window."""
        if self.isMaximized():
            self.showNormal()
            self.btn_win_max.setText("□")
            self.btn_win_max.setToolTip("Maximize Transparent Window")
        else:
            self.showMaximized()
            self.btn_win_max.setText("❐")
            self.btn_win_max.setToolTip("Restore Window Size")
        self.update()

    def on_glass_opacity_changed(self, val: int):
        """Adjusts the Apple Glass window surface opacity live (0% wireframe to 90% frosted glass)."""
        self.glass_opacity_pct = int(np.clip(val, 0, 90))
        self.lbl_glass_alpha.setText(f"Glass {self.glass_opacity_pct}%")
        self.update()

    def mousePressEvent(self, event):
        left_btn = Qt.MouseButton.LeftButton if hasattr(Qt, "MouseButton") else Qt.LeftButton
        if event.button() == left_btn and event.pos().y() <= 98 and not self.isMaximized():
            global_pt = event.globalPosition().toPoint() if hasattr(event, "globalPosition") else event.globalPos()
            self._drag_offset = global_pt - self.frameGeometry().topLeft()
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._drag_offset is not None and not self.isMaximized():
            global_pt = event.globalPosition().toPoint() if hasattr(event, "globalPosition") else event.globalPos()
            self.move(global_pt - self._drag_offset)
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        self._drag_offset = None
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event):
        if event.pos().y() <= 54:
            self.toggle_window_maximize()
            event.accept()
        else:
            super().mouseDoubleClickEvent(event)

    def paintEvent(self, event):
        """
        Paints the ultra-transparent Apple Glass (visionOS) spatial window shell.
        Clears with CompositionMode_Source (rgba(0,0,0,0)) so AR glasses users have unobstructed
        walking visibility through the entire window and controls.
        """
        painter = QPainter(self)
        try:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            w, h = self.width(), self.height()

            comp_src = (
                QPainter.CompositionMode.CompositionMode_Source
                if hasattr(QPainter, "CompositionMode")
                else QPainter.CompositionMode_Source
            )
            comp_over = (
                QPainter.CompositionMode.CompositionMode_SourceOver
                if hasattr(QPainter, "CompositionMode")
                else QPainter.CompositionMode_SourceOver
            )
            painter.setCompositionMode(comp_src)
            painter.fillRect(0, 0, w, h, QColor(0, 0, 0, 0))
            painter.setCompositionMode(comp_over)

            # Convert glass_opacity_pct (default 12%) into 0-255 alpha
            base_alpha = int(round((self.glass_opacity_pct / 100.0) * 255.0))
            sheen_alpha = min(255, int(base_alpha * 0.55) + 8)

            outer_rect = QRectF(1.0, 1.0, w - 2.0, h - 2.0)
            radius = 0.0 if self.isMaximized() else 16.0

            glass_grad = QLinearGradient(0, 0, 0, h)
            glass_grad.setColorAt(0.0, QColor(255, 255, 255, sheen_alpha))
            glass_grad.setColorAt(0.25, QColor(18, 18, 22, base_alpha))
            glass_grad.setColorAt(1.0, QColor(14, 14, 18, min(255, base_alpha + 10)))

            # Outer Champagne Gold Specular Rim (#BAA164)
            painter.setPen(QPen(QColor(186, 161, 100, 165), 1.5))
            painter.setBrush(QBrush(glass_grad))
            painter.drawRoundedRect(outer_rect, radius, radius)

            # Inner White Glass Refraction Rim (Apple visionOS signature)
            if not self.isMaximized():
                inner_rect = QRectF(2.5, 2.5, w - 5.0, h - 5.0)
                painter.setPen(QPen(QColor(255, 255, 255, 58), 1.0))
                painter.setBrush(Qt.BrushStyle.NoBrush if hasattr(Qt, "BrushStyle") else Qt.NoBrush)
                painter.drawRoundedRect(inner_rect, max(0.0, radius - 1.5), max(0.0, radius - 1.5))
        finally:
            painter.end()

    def init_ui(self):
        self.setWindowTitle("scanHUB — an Artificial Vision product (Apple Glass AR OS)")
        self.resize(1340, 860)

        # Frameless + Translucent Background for Apple Glass AR Glasses UI/UX
        win_flags = (
            (Qt.WindowType.FramelessWindowHint if hasattr(Qt, "WindowType") else Qt.FramelessWindowHint)
            | (Qt.WindowType.Window if hasattr(Qt, "WindowType") else Qt.Window)
        )
        self.setWindowFlags(win_flags)
        if hasattr(Qt, "WidgetAttribute"):
            self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        else:
            self.setAttribute(Qt.WA_TranslucentBackground, True)

        logo_path = get_av_logo_path()
        if logo_path:
            self.setWindowIcon(QIcon(logo_path))

        palette = QPalette()
        palette.setColor(QPalette.ColorRole.Window, QColor(0, 0, 0, 0))
        palette.setColor(QPalette.ColorRole.WindowText, QColor("#FCFBF8"))
        palette.setColor(QPalette.ColorRole.Base, QColor(255, 255, 255, 14))
        palette.setColor(QPalette.ColorRole.AlternateBase, QColor(186, 161, 100, 18))
        palette.setColor(QPalette.ColorRole.Text, QColor("#FCFBF8"))
        palette.setColor(QPalette.ColorRole.Button, QColor(255, 255, 255, 16))
        palette.setColor(QPalette.ColorRole.ButtonText, QColor("#FCFBF8"))
        self.setPalette(palette)

        self.setStyleSheet("""
            QWidget {
                font-family: 'IBM Plex Sans', '-apple-system', 'SF Pro Display', 'Inter', 'Segoe UI', sans-serif;
                color: #FCFBF8;
                background: transparent;
            }
            QSlider::groove:horizontal {
                border: 1px solid rgba(255, 255, 255, 0.24);
                height: 4px;
                background: rgba(255, 255, 255, 0.10);
                border-radius: 2px;
            }
            QSlider::sub-page:horizontal {
                background: #BAA164;
                border-radius: 2px;
            }
            QSlider::handle:horizontal {
                background: #FCFBF8;
                border: 2px solid #BAA164;
                width: 12px;
                margin: -5px 0;
                border-radius: 6px;
            }
            QCheckBox {
                spacing: 5px;
                color: #FCFBF8;
                background: transparent;
            }
            QCheckBox::indicator {
                width: 13px;
                height: 13px;
                border-radius: 3px;
                border: 1px solid rgba(186, 161, 100, 0.65);
                background: rgba(255, 255, 255, 0.08);
            }
            QCheckBox::indicator:checked {
                background: #BAA164;
                border: 1px solid #FCFBF8;
            }
            QToolTip {
                background-color: rgba(20, 20, 24, 210);
                color: #FCFBF8;
                border: 1px solid #BAA164;
                border-radius: 6px;
                padding: 4px 8px;
            }
        """)

        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(0)
        main_layout.setContentsMargins(2, 2, 2, 2)

        # ─────────────────────────────────────────────────────────────────────
        # 1. HEADER BAR (Apple Glass Translucent + prevdc.artificialviz.com Brand Lockup + Transparent Min/Max/Close)
        # ─────────────────────────────────────────────────────────────────────
        self.header_bar = QFrame()
        self.header_bar.setFixedHeight(52)
        self.header_bar.setStyleSheet(
            "QFrame { background-color: rgba(255, 255, 255, 0.05); "
            "border-bottom: 1px solid rgba(186, 161, 100, 0.38); "
            "border-top-left-radius: 14px; border-top-right-radius: 14px; }"
        )
        header_layout = QHBoxLayout(self.header_bar)
        header_layout.setContentsMargins(14, 0, 12, 0)
        header_layout.setSpacing(10)

        # Artificial Vision Monogram Logo Button (clicks open Translucent Splash Screen)
        self.logo_btn = QPushButton()
        self.logo_btn.setFixedSize(36, 36)
        self.logo_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.logo_btn.setToolTip("Artificial Vision — Click to view Splash Screen")
        self.logo_btn.setStyleSheet(
            "QPushButton { background: rgba(186, 161, 100, 0.12); border: 1px solid rgba(186, 161, 100, 0.55); border-radius: 18px; }"
            "QPushButton:hover { background: rgba(186, 161, 100, 0.28); border-color: #FCFBF8; }"
        )
        logo_pix = load_av_logo_pixmap(28)
        if not logo_pix.isNull():
            self.logo_btn.setIcon(QIcon(logo_pix))
            self.logo_btn.setIconSize(logo_pix.size())
        else:
            self.logo_btn.setText("AV")
        self.logo_btn.clicked.connect(lambda: self.show_brand_splash(2600))

        # Exact Brand Identity from https://prevdc.artificialviz.com/auth
        title = QLabel("scanHUB")
        title.setFont(QFont("IBM Plex Sans", 14, QFont.Weight.DemiBold))
        title.setStyleSheet("color: #FCFBF8; background: transparent; border: none;")

        sub_title = QLabel("an Artificial Vision product")
        sub_title.setFont(QFont("IBM Plex Sans", 9))
        sub_title.setStyleSheet("color: rgba(252, 251, 248, 0.72); background: transparent; border: none;")

        brand_pill = QLabel("VERIFY · DIRECT · CLARIFY")
        brand_pill.setFont(QFont("IBM Plex Mono", 7, QFont.Weight.Bold))
        brand_pill.setStyleSheet(
            "color: #E5C983; background: rgba(186, 161, 100, 0.14); "
            "border: 1px solid rgba(186, 161, 100, 0.48); "
            "padding: 3px 8px; border-radius: 10px; letter-spacing: 1.4px;"
        )

        header_layout.addWidget(self.logo_btn)
        header_layout.addWidget(title)
        header_layout.addWidget(sub_title)
        header_layout.addWidget(brand_pill)
        header_layout.addStretch()

        # Live AR Glass Opacity Slider (0% pure wireframe to 90% frosted glass; default 12% for walking)
        self.lbl_glass_alpha = QLabel(f"Glass {self.glass_opacity_pct}%")
        self.lbl_glass_alpha.setFont(QFont("IBM Plex Mono", 8, QFont.Weight.Bold))
        self.lbl_glass_alpha.setStyleSheet("color: #E5C983; background: transparent; border: none;")
        self.slider_glass_alpha = QSlider(Qt.Orientation.Horizontal)
        self.slider_glass_alpha.setRange(0, 90)
        self.slider_glass_alpha.setValue(self.glass_opacity_pct)
        self.slider_glass_alpha.setFixedWidth(68)
        self.slider_glass_alpha.setToolTip("Adjust Apple Glass AR Transparency (0% pure wireframe to 90% frosted)")
        self.slider_glass_alpha.valueChanged.connect(self.on_glass_opacity_changed)
        header_layout.addWidget(self.lbl_glass_alpha)
        header_layout.addWidget(self.slider_glass_alpha)

        pill_base = (
            "background: rgba(255, 255, 255, 0.06); "
            "border: 1px solid rgba(255, 255, 255, 0.20); "
            "padding: 3px 8px; border-radius: 10px; font-weight: 600; font-size: 9px;"
        )
        self.dot_pace = QLabel("● PACE: READY")
        self.dot_pace.setStyleSheet(f"color: #4ADE80; {pill_base}")
        self.dot_cslam = QLabel("● C-SLAM: 1 Scanner")
        self.dot_cslam.setStyleSheet(
            "color: #E5C983; background: rgba(186, 161, 100, 0.14); "
            "border: 1px solid rgba(186, 161, 100, 0.45); "
            "padding: 3px 8px; border-radius: 10px; font-weight: 600; font-size: 9px;"
        )
        self.dot_lidar = QLabel("● LiDAR: OFFLINE")
        self.dot_lidar.setStyleSheet(f"color: #F87171; {pill_base}")
        self.dot_camera = QLabel("● Camera: OFFLINE")
        self.dot_camera.setStyleSheet(f"color: #F87171; {pill_base}")
        self.dot_slam = QLabel("● SLAM: OFFLINE")
        self.dot_slam.setStyleSheet(f"color: #F87171; {pill_base}")

        header_layout.addWidget(self.dot_pace)
        header_layout.addWidget(self.dot_cslam)
        header_layout.addWidget(self.dot_lidar)
        header_layout.addWidget(self.dot_camera)
        header_layout.addWidget(self.dot_slam)

        # Transparent Apple Glass Window Controls (Minimize, Maximize/Restore, Close)
        win_btn_glass = (
            "QPushButton { background: rgba(255, 255, 255, 0.07); "
            "border: 1px solid rgba(255, 255, 255, 0.26); color: #FCFBF8; "
            "border-radius: 13px; font-size: 11px; font-weight: bold; }"
            "QPushButton:hover { background: rgba(186, 161, 100, 0.28); border-color: #E5C983; color: #FFFFFF; }"
        )
        self.btn_win_min = QPushButton("—")
        self.btn_win_min.setFixedSize(26, 26)
        self.btn_win_min.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_win_min.setToolTip("Minimize Transparent Window")
        self.btn_win_min.setStyleSheet(win_btn_glass)
        self.btn_win_min.clicked.connect(self.showMinimized)

        self.btn_win_max = QPushButton("□")
        self.btn_win_max.setFixedSize(26, 26)
        self.btn_win_max.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_win_max.setToolTip("Maximize / Restore Transparent Window")
        self.btn_win_max.setStyleSheet(
            "QPushButton { background: rgba(186, 161, 100, 0.12); "
            "border: 1px solid rgba(186, 161, 100, 0.48); color: #E5C983; "
            "border-radius: 13px; font-size: 11px; font-weight: bold; }"
            "QPushButton:hover { background: rgba(186, 161, 100, 0.32); border-color: #FCFBF8; color: #FFFFFF; }"
        )
        self.btn_win_max.clicked.connect(self.toggle_window_maximize)

        self.btn_win_close = QPushButton("✕")
        self.btn_win_close.setFixedSize(26, 26)
        self.btn_win_close.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_win_close.setToolTip("Close scanHUB")
        self.btn_win_close.setStyleSheet(
            "QPushButton { background: rgba(255, 255, 255, 0.07); "
            "border: 1px solid rgba(255, 255, 255, 0.26); color: #FCFBF8; "
            "border-radius: 13px; font-size: 11px; font-weight: bold; }"
            "QPushButton:hover { background: rgba(239, 68, 68, 0.32); border-color: #F87171; color: #FFFFFF; }"
        )
        self.btn_win_close.clicked.connect(self.close)

        header_layout.addSpacing(4)
        header_layout.addWidget(self.btn_win_min)
        header_layout.addWidget(self.btn_win_max)
        header_layout.addWidget(self.btn_win_close)
        main_layout.addWidget(self.header_bar)

        # ─────────────────────────────────────────────────────────────────────
        # WORKSPACE STACK:
        #   Page 0: Projects, Datasets & Live C-SLAM Active Captures Hub (Post-Splash Home)
        #   Page 1: Capture Studio (Shown only after entering a Project & creating/joining a Dataset)
        # ─────────────────────────────────────────────────────────────────────
        self.workspace_stack = QStackedWidget()
        self.projects_hub_page = self._build_projects_hub_page()
        self.workspace_stack.addWidget(self.projects_hub_page)

        self.capture_studio_page = QWidget()
        capture_page_layout = QVBoxLayout(self.capture_studio_page)
        capture_page_layout.setContentsMargins(0, 0, 0, 0)
        capture_page_layout.setSpacing(0)

        # ─────────────────────────────────────────────────────────────────────
        # 2. CONTROL BAR (Apple Glass Translucent White & Champagne Gold Action Bar)
        # ─────────────────────────────────────────────────────────────────────
        ctrl_bar = QFrame()
        ctrl_bar.setFixedHeight(48)
        ctrl_bar.setStyleSheet(
            "QFrame { background-color: rgba(255, 255, 255, 0.03); "
            "border-bottom: 1px solid rgba(255, 255, 255, 0.14); }"
        )
        ctrl_layout = QHBoxLayout(ctrl_bar)
        ctrl_layout.setContentsMargins(14, 0, 14, 0)
        ctrl_layout.setSpacing(8)

        self.btn_back_to_projects = QPushButton("← PROJECTS")
        self.btn_back_to_projects.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_back_to_projects.setToolTip("Return to Projects, Datasets & Active C-SLAM Captures Menu")
        self.btn_back_to_projects.setStyleSheet(
            "QPushButton { background: rgba(255, 255, 255, 0.07); border: 1px solid rgba(255, 255, 255, 0.25); "
            "color: #FCFBF8; padding: 6px 12px; border-radius: 9px; font-weight: 600; font-size: 10px; }"
            "QPushButton:hover { background: rgba(186, 161, 100, 0.22); border-color: #BAA164; color: #E5C983; }"
        )
        self.btn_back_to_projects.clicked.connect(self.show_projects_hub_view)

        self.stack_btn = QPushButton("🚀 START STACK")
        self.stack_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.stack_btn.setStyleSheet(
            "QPushButton { background: rgba(255, 255, 255, 0.07); border: 1px solid rgba(186, 161, 100, 0.60); "
            "color: #E5C983; padding: 6px 14px; border-radius: 9px; font-weight: 600; font-size: 11px; }"
            "QPushButton:hover { background: rgba(186, 161, 100, 0.24); border-color: #FCFBF8; color: #FFFFFF; }"
        )
        self.stack_btn.clicked.connect(self.toggle_ros_stack)
        if self._external_stack_online:
            self._set_stack_buttons_online(True)

        self.start_btn = QPushButton("▶ START SCAN")
        self.start_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.start_btn.setStyleSheet(
            "QPushButton { background: rgba(186, 161, 100, 0.32); border: 1px solid #BAA164; "
            "color: #FCFBF8; padding: 6px 18px; border-radius: 9px; font-weight: 600; font-size: 11px; }"
            "QPushButton:hover { background: rgba(186, 161, 100, 0.50); border-color: #FCFBF8; }"
            "QPushButton:disabled { background: rgba(255, 255, 255, 0.04); border-color: rgba(255, 255, 255, 0.12); color: rgba(252, 251, 248, 0.35); }"
        )
        self.start_btn.clicked.connect(self.toggle_scan_workflow)

        self.stop_btn = QPushButton("■ STOP")
        self.stop_btn.setEnabled(False)
        self.stop_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.stop_btn.setStyleSheet(
            "QPushButton { background: rgba(239, 68, 68, 0.16); border: 1px solid rgba(248, 113, 113, 0.55); "
            "color: #FCA5A5; padding: 6px 14px; border-radius: 9px; font-weight: 600; font-size: 11px; }"
            "QPushButton:hover:enabled { background: rgba(239, 68, 68, 0.32); border-color: #FCA5A5; color: #FFFFFF; }"
            "QPushButton:disabled { color: rgba(252, 251, 248, 0.30); background: rgba(255, 255, 255, 0.03); border-color: rgba(255, 255, 255, 0.10); }"
        )
        self.stop_btn.clicked.connect(self.stop_scan_session)

        self.flash_btn = QPushButton("⚡ FLASH")
        self.flash_btn.setEnabled(False)
        self.flash_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.flash_btn.setToolTip("Record FARO Flash Scan waypoint anchor at current location [Shortcut: F]")
        self.flash_btn.setStyleSheet(
            "QPushButton { background: rgba(229, 201, 131, 0.20); border: 1px solid #E5C983; "
            "color: #FCFBF8; padding: 6px 14px; border-radius: 9px; font-weight: 700; font-size: 11px; }"
            "QPushButton:hover:enabled { background: rgba(229, 201, 131, 0.38); border-color: #FCFBF8; color: #FFFFFF; }"
            "QPushButton:disabled { background: rgba(255, 255, 255, 0.03); border-color: rgba(255, 255, 255, 0.10); color: rgba(252, 251, 248, 0.30); }"
        )
        self.flash_btn.clicked.connect(self.trigger_faro_flash_anchor)

        self.save_btn = QPushButton("↓ SAVE DATASET")
        self.save_btn.setEnabled(False)
        self.save_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.save_btn.setStyleSheet(
            "QPushButton { background: rgba(255, 255, 255, 0.12); border: 1px solid #FCFBF8; "
            "color: #E5C983; padding: 6px 14px; border-radius: 9px; font-weight: 600; font-size: 11px; }"
            "QPushButton:hover:enabled { background: rgba(186, 161, 100, 0.32); color: #FFFFFF; }"
            "QPushButton:disabled { background: rgba(255, 255, 255, 0.03); border-color: rgba(255, 255, 255, 0.10); color: rgba(252, 251, 248, 0.30); }"
        )
        self.save_btn.clicked.connect(self.save_dataset)

        self.btn_quick_m12_rt = QPushButton("✦ M12 RT: OFF")
        self.btn_quick_m12_rt.setCheckable(True)
        self.btn_quick_m12_rt.setChecked(False)
        self.btn_quick_m12_rt.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_quick_m12_rt.setToolTip("Toggle Real-Time M12 Cleanup (Default OFF for 60 FPS low-latency tracking; use Post-Process Preview/Commit/Export)")
        self.btn_quick_m12_rt.setStyleSheet(
            "background: rgba(255, 255, 255, 0.07); border: 1px solid rgba(186, 161, 100, 0.45); "
            "color: rgba(252, 251, 248, 0.78); padding: 6px 12px; border-radius: 9px; font-weight: 600; font-size: 10px;"
        )
        self.btn_quick_m12_rt.clicked.connect(self.on_quick_m12_rt_clicked)

        self.clock_lbl = QLabel("⏱ 00:00")
        self.clock_lbl.setFont(QFont("IBM Plex Mono", 10, QFont.Weight.DemiBold))
        self.clock_lbl.setStyleSheet(
            "color: #FCFBF8; background: rgba(255, 255, 255, 0.06); "
            "border: 1px solid rgba(255, 255, 255, 0.22); padding: 4px 12px; border-radius: 9px;"
        )

        self.lbl_active_project_badge = QLabel(f"PROJECT: {self.active_project_name} /")
        self.lbl_active_project_badge.setFont(QFont("IBM Plex Mono", 8, QFont.Weight.Bold))
        self.lbl_active_project_badge.setStyleSheet("color: #E5C983; letter-spacing: 0.8px; border: none;")

        dataset_lbl = QLabel("DATASET")
        dataset_lbl.setFont(QFont("IBM Plex Mono", 8, QFont.Weight.Bold))
        dataset_lbl.setStyleSheet("color: #D8BF7C; letter-spacing: 1.0px; border: none;")
        self.dataset_input = QLineEdit()
        self.dataset_input.setText(self.active_dataset_name)
        self.dataset_input.setStyleSheet(
            "background: rgba(255, 255, 255, 0.07); border: 1px solid rgba(186, 161, 100, 0.48); "
            "color: #FCFBF8; padding: 5px 10px; border-radius: 9px; width: 135px; font-weight: 600;"
        )
        self.ros_worker.cslam_engine.set_project_name(self.active_project_name, self.active_dataset_name)
        self.dataset_input.textChanged.connect(self.on_project_name_changed)

        ctrl_layout.addWidget(self.btn_back_to_projects)
        ctrl_layout.addWidget(self.stack_btn)
        ctrl_layout.addWidget(self.start_btn)
        ctrl_layout.addWidget(self.stop_btn)
        ctrl_layout.addWidget(self.flash_btn)
        ctrl_layout.addWidget(self.save_btn)
        ctrl_layout.addWidget(self.btn_quick_m12_rt)
        ctrl_layout.addWidget(self.clock_lbl)
        ctrl_layout.addStretch()
        ctrl_layout.addWidget(self.lbl_active_project_badge)
        ctrl_layout.addWidget(dataset_lbl)
        ctrl_layout.addWidget(self.dataset_input)
        capture_page_layout.addWidget(ctrl_bar)

        # ─────────────────────────────────────────────────────────────────────
        # 3. SPLIT WORKSPACE (Left: 100% Transparent AR Walking Viewport | Right: Glass HUD)
        # ─────────────────────────────────────────────────────────────────────
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setStyleSheet("QSplitter::handle { background-color: rgba(186, 161, 100, 0.28); width: 1px; }")

        # LEFT / CENTER PANEL (60%): 100% TRANSPARENT WALKING VIEWPORT
        left_container = QWidget()
        left_container.setStyleSheet("background-color: transparent;")
        left_layout = QVBoxLayout(left_container)
        left_layout.setContentsMargins(12, 8, 8, 10)
        left_layout.setSpacing(6)

        prog_row = QHBoxLayout()
        self.progress_label = QLabel("STAGE 1  ·  Place scanner static on flat surface, then press 'START SCAN'")
        self.progress_label.setFont(QFont("IBM Plex Sans", 9, QFont.Weight.DemiBold))
        self.progress_label.setStyleSheet("color: #E5C983; background: transparent;")
        prog_row.addWidget(self.progress_label)
        prog_row.addStretch()

        self.btn_skip_ritual = QPushButton("Skip Ritual ⏭")
        self.btn_skip_ritual.setVisible(False)
        self.btn_skip_ritual.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_skip_ritual.setStyleSheet(
            "QPushButton { background: rgba(255, 255, 255, 0.08); border: 1px solid rgba(186, 161, 100, 0.45); "
            "color: #E5C983; padding: 2px 10px; border-radius: 6px; font-size: 10px; font-weight: 600; }"
            "QPushButton:hover { background: rgba(186, 161, 100, 0.28); color: #FFFFFF; }"
        )
        self.btn_skip_ritual.clicked.connect(self.skip_rotation_ritual)
        prog_row.addWidget(self.btn_skip_ritual)

        left_layout.addLayout(prog_row)

        self.bar = QProgressBar()
        self.bar.setRange(0, 100)
        self.bar.setValue(0)
        self.bar.setFixedHeight(5)
        self.bar.setTextVisible(False)
        self.bar.setStyleSheet("""
            QProgressBar { border: 1px solid rgba(255, 255, 255, 0.18); border-radius: 2px; background-color: rgba(255, 255, 255, 0.06); }
            QProgressBar::chunk { background-color: #BAA164; border-radius: 2px; }
        """)
        left_layout.addWidget(self.bar)

        self.center_viewport = ClearPassthroughViewport()
        left_layout.addWidget(self.center_viewport, 1)

        splitter.addWidget(left_container)

        # RIGHT PANEL (40% TRANSLUCENT APPLE GLASS PERIPHERAL HUD)
        right_container = QWidget()
        right_container.setStyleSheet("background-color: transparent;")
        right_layout = QVBoxLayout(right_container)
        right_layout.setContentsMargins(8, 8, 12, 10)
        right_layout.setSpacing(7)

        # 1. Camera Canvas (Isolated Clean RGB Left | Pure Depth Heatmap Right)
        self.camera_canvas = CameraPreviewCanvas()
        self.camera_canvas.setFixedHeight(162)
        right_layout.addWidget(self.camera_canvas)

        # 2. Map Canvas & Layer Mode Selector Bar
        map_hdr_row = QHBoxLayout()
        map_hdr_row.setSpacing(4)
        map_title = QLabel("2D ORTHOGRAPHIC SLICE")
        map_title.setFont(QFont("IBM Plex Mono", 8, QFont.Weight.Bold))
        map_title.setStyleSheet("color: #E5C983; letter-spacing: 0.8px; background: transparent;")
        map_hdr_row.addWidget(map_title)
        map_hdr_row.addStretch()

        self.btn_layer_fused_rgb = QPushButton("Colorized")
        self.btn_layer_airy = QPushButton("Airy")
        self.btn_layer_stereo = QPushButton("Gemini")
        self.btn_layer_fused = QPushButton("Fused")
        self.btn_layer_acc_rej = QPushButton("Accept/Rej")
        self.btn_layer_diag = QPushButton("Diag")

        self._layer_buttons = {
            "colorized_fused": self.btn_layer_fused_rgb,
            "airy_only": self.btn_layer_airy,
            "stereo_only": self.btn_layer_stereo,
            "fused": self.btn_layer_fused,
            "accepted_rejected": self.btn_layer_acc_rej,
            "observation_diag": self.btn_layer_diag,
        }
        for mode_key, btn in self._layer_buttons.items():
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.clicked.connect(lambda checked, m=mode_key: self.on_layer_mode_clicked(m))
            map_hdr_row.addWidget(btn)
        self._style_layer_buttons("colorized_fused")

        icon_btn_style = (
            "QPushButton { background: rgba(255, 255, 255, 0.07); border: 1px solid rgba(255, 255, 255, 0.22); "
            "color: #FCFBF8; font-weight: bold; border-radius: 6px; }"
            "QPushButton:hover { background: rgba(186, 161, 100, 0.25); border-color: #BAA164; color: #E5C983; }"
        )
        btn_zoom_out = QPushButton("−")
        btn_zoom_out.setFixedSize(22, 22)
        btn_zoom_out.setStyleSheet(icon_btn_style)
        btn_zoom_out.clicked.connect(lambda: self.map_canvas.zoom_out())

        btn_zoom_in = QPushButton("+")
        btn_zoom_in.setFixedSize(22, 22)
        btn_zoom_in.setStyleSheet(icon_btn_style)
        btn_zoom_in.clicked.connect(lambda: self.map_canvas.zoom_in())

        btn_clear = QPushButton("↺")
        btn_clear.setFixedSize(22, 22)
        btn_clear.setToolTip("Clear Map Canvas")
        btn_clear.setStyleSheet(
            "QPushButton { background: rgba(255, 255, 255, 0.07); border: 1px solid rgba(248, 113, 113, 0.45); "
            "color: #FCA5A5; font-weight: bold; border-radius: 6px; }"
            "QPushButton:hover { background: rgba(239, 68, 68, 0.28); border-color: #FCA5A5; color: #FFFFFF; }"
        )
        btn_clear.clicked.connect(lambda: self.map_canvas.clear_map())

        map_hdr_row.addWidget(btn_zoom_out)
        map_hdr_row.addWidget(btn_zoom_in)
        map_hdr_row.addWidget(btn_clear)
        right_layout.addLayout(map_hdr_row)

        self.map_canvas = RealityCaptureMapCanvas()
        self.map_canvas.setMinimumHeight(222)
        right_layout.addWidget(self.map_canvas, 1)

        # ─────────────────────────────────────────────────────────────────────
        # 3. MULTI-TAB CONTROL DECK: M12 CLEANUP | C-SLAM MULTI-SCANNER | DEPTH GATE
        # ─────────────────────────────────────────────────────────────────────
        self.hud_tabs = QTabWidget()
        self.hud_tabs.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid rgba(186, 161, 100, 0.42);
                background: rgba(255, 255, 255, 0.04);
                border-radius: 10px;
                top: -1px;
            }
            QTabBar::tab {
                background: rgba(255, 255, 255, 0.05);
                color: rgba(252, 251, 248, 0.72);
                border: 1px solid rgba(255, 255, 255, 0.18);
                padding: 5px 10px;
                font-size: 9px;
                font-weight: 600;
                border-top-left-radius: 7px;
                border-top-right-radius: 7px;
                margin-right: 3px;
            }
            QTabBar::tab:selected {
                background: rgba(186, 161, 100, 0.20);
                color: #FCFBF8;
                border-color: #BAA164;
                border-bottom: 2px solid #E5C983;
            }
        """)

        spin_style = (
            "background: rgba(255, 255, 255, 0.07); color: #E5C983; "
            "border: 1px solid rgba(186, 161, 100, 0.48); "
            "border-radius: 5px; padding: 1px 4px; font-size: 9px; font-weight: 600;"
        )

        # === TAB 1: M12 POINT CLOUD CLEANUP & EXPORT (Ported from av_scan) ===
        tab_m12 = QWidget()
        m12_layout = QVBoxLayout(tab_m12)
        m12_layout.setSpacing(5)
        m12_layout.setContentsMargins(8, 8, 8, 8)

        rt_row = QHBoxLayout()
        self.chk_m12_realtime = QCheckBox("✦ Real-Time M12 Spatial Cleanup")
        self.chk_m12_realtime.setChecked(False)
        self.chk_m12_realtime.setStyleSheet("color: #E5C983; font-size: 9px; font-weight: bold;")
        self.chk_m12_realtime.toggled.connect(self.on_m12_realtime_toggled)

        self.chk_m12_ground_lock = QCheckBox("Live Z=0 Floor Lock")
        self.chk_m12_ground_lock.setChecked(False)
        self.chk_m12_ground_lock.setStyleSheet("color: #FCFBF8; font-size: 9px; font-weight: 600;")
        self.chk_m12_ground_lock.toggled.connect(self.on_m12_ground_lock_toggled)

        rt_row.addWidget(self.chk_m12_realtime)
        rt_row.addStretch()
        rt_row.addWidget(self.chk_m12_ground_lock)
        m12_layout.addLayout(rt_row)

        # Voxel + SOR row
        row_v_s = QHBoxLayout()
        self.chk_m12_voxel = QCheckBox("Voxel (m):")
        self.chk_m12_voxel.setChecked(True)
        self.chk_m12_voxel.setStyleSheet("color: #FCFBF8; font-size: 9px;")
        self.spin_m12_voxel = QDoubleSpinBox()
        self.spin_m12_voxel.setRange(0.005, 0.250)
        self.spin_m12_voxel.setSingleStep(0.005)
        self.spin_m12_voxel.setDecimals(3)
        self.spin_m12_voxel.setValue(0.020)
        self.spin_m12_voxel.setStyleSheet(spin_style)
        self.spin_m12_voxel.valueChanged.connect(self.sync_m12_params_from_ui)
        self.chk_m12_voxel.toggled.connect(self.sync_m12_params_from_ui)

        self.chk_m12_sor = QCheckBox("SOR (k/σ):")
        self.chk_m12_sor.setChecked(True)
        self.chk_m12_sor.setStyleSheet("color: #FCFBF8; font-size: 9px;")
        self.spin_m12_sor_k = QSpinBox()
        self.spin_m12_sor_k.setRange(4, 100)
        self.spin_m12_sor_k.setValue(20)
        self.spin_m12_sor_k.setStyleSheet(spin_style)
        self.spin_m12_sor_std = QDoubleSpinBox()
        self.spin_m12_sor_std.setRange(0.5, 5.0)
        self.spin_m12_sor_std.setSingleStep(0.25)
        self.spin_m12_sor_std.setValue(2.0)
        self.spin_m12_sor_std.setStyleSheet(spin_style)
        self.spin_m12_sor_k.valueChanged.connect(self.sync_m12_params_from_ui)
        self.spin_m12_sor_std.valueChanged.connect(self.sync_m12_params_from_ui)
        self.chk_m12_sor.toggled.connect(self.sync_m12_params_from_ui)

        row_v_s.addWidget(self.chk_m12_voxel)
        row_v_s.addWidget(self.spin_m12_voxel)
        row_v_s.addSpacing(6)
        row_v_s.addWidget(self.chk_m12_sor)
        row_v_s.addWidget(self.spin_m12_sor_k)
        row_v_s.addWidget(self.spin_m12_sor_std)
        m12_layout.addLayout(row_v_s)

        # ROR + Ground Plane Leveling row
        row_r_g = QHBoxLayout()
        self.chk_m12_ror = QCheckBox("ROR (r/N):")
        self.chk_m12_ror.setChecked(True)
        self.chk_m12_ror.setStyleSheet("color: #FCFBF8; font-size: 9px;")
        self.spin_m12_ror_r = QDoubleSpinBox()
        self.spin_m12_ror_r.setRange(0.02, 0.50)
        self.spin_m12_ror_r.setSingleStep(0.02)
        self.spin_m12_ror_r.setValue(0.14)
        self.spin_m12_ror_r.setStyleSheet(spin_style)
        self.spin_m12_ror_n = QSpinBox()
        self.spin_m12_ror_n.setRange(1, 30)
        self.spin_m12_ror_n.setValue(4)
        self.spin_m12_ror_n.setStyleSheet(spin_style)
        self.spin_m12_ror_r.valueChanged.connect(self.sync_m12_params_from_ui)
        self.spin_m12_ror_n.valueChanged.connect(self.sync_m12_params_from_ui)
        self.chk_m12_ror.toggled.connect(self.sync_m12_params_from_ui)

        self.chk_m12_ground = QCheckBox("Floor Z=0")
        self.chk_m12_ground.setChecked(True)
        self.chk_m12_ground.setStyleSheet("color: #FCFBF8; font-size: 9px;")
        self.chk_m12_rm_ground = QCheckBox("Strip Floor")
        self.chk_m12_rm_ground.setChecked(False)
        self.chk_m12_rm_ground.setStyleSheet("color: #E5C983; font-size: 9px;")
        self.chk_m12_ground.toggled.connect(self.sync_m12_params_from_ui)
        self.chk_m12_rm_ground.toggled.connect(self.sync_m12_params_from_ui)

        row_r_g.addWidget(self.chk_m12_ror)
        row_r_g.addWidget(self.spin_m12_ror_r)
        row_r_g.addWidget(self.spin_m12_ror_n)
        row_r_g.addSpacing(6)
        row_r_g.addWidget(self.chk_m12_ground)
        row_r_g.addWidget(self.chk_m12_rm_ground)
        m12_layout.addLayout(row_r_g)

        # Action & Export row
        act_row = QHBoxLayout()
        self.btn_m12_preview = QPushButton("Preview")
        self.btn_m12_preview.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_m12_preview.setStyleSheet(
            "QPushButton { background: rgba(186, 161, 100, 0.16); color: #E5C983; border: 1px solid #BAA164; font-size: 9px; font-weight: 600; padding: 4px 8px; border-radius: 6px; }"
            "QPushButton:hover { background: rgba(186, 161, 100, 0.30); color: #FFFFFF; }"
        )
        self.btn_m12_preview.clicked.connect(self.on_m12_preview_clicked)

        self.btn_m12_reset = QPushButton("Reset")
        self.btn_m12_reset.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_m12_reset.setStyleSheet(
            "QPushButton { background: rgba(255, 255, 255, 0.06); color: #FCFBF8; border: 1px solid rgba(255, 255, 255, 0.22); font-size: 9px; font-weight: 600; padding: 4px 8px; border-radius: 6px; }"
            "QPushButton:hover { background: rgba(255, 255, 255, 0.14); }"
        )
        self.btn_m12_reset.clicked.connect(self.on_m12_reset_clicked)

        self.btn_m12_commit = QPushButton("Commit")
        self.btn_m12_commit.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_m12_commit.setStyleSheet(
            "QPushButton { background: rgba(186, 161, 100, 0.32); color: #FCFBF8; border: 1px solid #E5C983; font-size: 9px; font-weight: 600; padding: 4px 8px; border-radius: 6px; }"
            "QPushButton:hover { background: rgba(186, 161, 100, 0.48); }"
        )
        self.btn_m12_commit.clicked.connect(self.on_m12_commit_clicked)

        self.combo_m12_fmt = QComboBox()
        self.combo_m12_fmt.addItems(["PLY_BINARY", "PLY_ASCII", "PCD_BINARY", "PCD_ASCII", "XYZ_ASCII", "OBJ_MESH"])
        self.combo_m12_fmt.setStyleSheet(
            "background: rgba(255, 255, 255, 0.07); color: #FCFBF8; border: 1px solid rgba(186, 161, 100, 0.45); border-radius: 6px; font-size: 9px; padding: 2px 6px;"
        )

        self.btn_m12_export = QPushButton("Export")
        self.btn_m12_export.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_m12_export.setStyleSheet(
            "QPushButton { background: rgba(255, 255, 255, 0.12); color: #E5C983; border: 1px solid #FCFBF8; font-size: 9px; font-weight: 600; padding: 4px 8px; border-radius: 6px; }"
            "QPushButton:hover { background: rgba(186, 161, 100, 0.30); color: #FFFFFF; }"
        )
        self.btn_m12_export.clicked.connect(self.on_m12_export_clicked)

        act_row.addWidget(self.btn_m12_preview)
        act_row.addWidget(self.btn_m12_reset)
        act_row.addWidget(self.btn_m12_commit)
        act_row.addWidget(self.combo_m12_fmt)
        act_row.addWidget(self.btn_m12_export)
        m12_layout.addLayout(act_row)

        self.lbl_m12_status = QLabel("M12 RT Filter: OFF (Post-Process Ready — Preview / Commit / Export)")
        self.lbl_m12_status.setStyleSheet("color: #E5C983; font-size: 9px; font-family: 'IBM Plex Mono', Monospace; font-weight: 600;")
        m12_layout.addWidget(self.lbl_m12_status)

        self.hud_tabs.addTab(tab_m12, "✦ M12 Cleanup")

        # === TAB 2: UNLIMITED-PEER C-SLAM MULTI-SCANNER ===
        tab_cslam = QWidget()
        cslam_layout = QVBoxLayout(tab_cslam)
        cslam_layout.setSpacing(5)
        cslam_layout.setContentsMargins(8, 8, 8, 8)

        cslam_top_row = QHBoxLayout()
        self.btn_cslam_color_mode = QPushButton("🎨 Scanner Colors")
        self.btn_cslam_color_mode.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_cslam_color_mode.setStyleSheet(
            "background: rgba(186, 161, 100, 0.18); color: #E5C983; border: 1px solid #BAA164; font-size: 9px; font-weight: 600; padding: 3px 8px; border-radius: 6px;"
        )
        self.btn_cslam_color_mode.clicked.connect(self.on_cslam_color_mode_toggled)

        lbl_sep = QLabel("Non-Overlap Sep:")
        lbl_sep.setStyleSheet("color: rgba(252, 251, 248, 0.78); font-size: 9px;")
        self.spin_cslam_sep = QDoubleSpinBox()
        self.spin_cslam_sep.setRange(3.0, 100.0)
        self.spin_cslam_sep.setSingleStep(2.5)
        self.spin_cslam_sep.setSuffix("m")
        self.spin_cslam_sep.setValue(15.0)
        self.spin_cslam_sep.setStyleSheet(spin_style)
        self.spin_cslam_sep.valueChanged.connect(lambda v: self.ros_worker.cslam_engine.set_separation_distance(v))

        lbl_ovlp = QLabel("Snap≥")
        lbl_ovlp.setStyleSheet("color: rgba(252, 251, 248, 0.78); font-size: 9px;")
        self.spin_cslam_ovlp = QSpinBox()
        self.spin_cslam_ovlp.setRange(10, 80)
        self.spin_cslam_ovlp.setSuffix("%")
        self.spin_cslam_ovlp.setValue(18)
        self.spin_cslam_ovlp.setStyleSheet(spin_style)
        self.spin_cslam_ovlp.valueChanged.connect(lambda v: self.ros_worker.cslam_engine.set_min_overlap_ratio(v / 100.0))

        cslam_top_row.addWidget(self.btn_cslam_color_mode)
        cslam_top_row.addWidget(lbl_sep)
        cslam_top_row.addWidget(self.spin_cslam_sep)
        cslam_top_row.addWidget(lbl_ovlp)
        cslam_top_row.addWidget(self.spin_cslam_ovlp)
        cslam_layout.addLayout(cslam_top_row)

        cslam_peer_row = QHBoxLayout()
        self.input_peer_ip = QLineEdit()
        self.input_peer_ip.setPlaceholderText("Peer IP (auto multicast LAN + unicast)")
        self.input_peer_ip.setStyleSheet(
            "background: rgba(255, 255, 255, 0.06); color: #FCFBF8; border: 1px solid rgba(255, 255, 255, 0.22); font-size: 9px; padding: 3px 6px; border-radius: 6px;"
        )

        self.btn_add_peer_ip = QPushButton("+IP")
        self.btn_add_peer_ip.setStyleSheet(
            "background: rgba(255, 255, 255, 0.08); color: #E5C983; border: 1px solid #BAA164; font-size: 9px; font-weight: 600; padding: 3px 7px; border-radius: 6px;"
        )
        self.btn_add_peer_ip.clicked.connect(self.on_add_peer_ip_clicked)

        self.btn_sim_peer = QPushButton("+Sim Peer Scanner")
        self.btn_sim_peer.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_sim_peer.setStyleSheet(
            "QPushButton { background: rgba(186, 161, 100, 0.30); color: #FCFBF8; border: 1px solid #E5C983; font-size: 9px; font-weight: 600; padding: 3px 8px; border-radius: 6px; }"
            "QPushButton:hover { background: rgba(186, 161, 100, 0.48); }"
        )
        self.btn_sim_peer.clicked.connect(self.on_spawn_sim_peer_clicked)

        self.btn_force_snap = QPushButton("Snap")
        self.btn_force_snap.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_force_snap.setStyleSheet(
            "QPushButton { background: rgba(255, 255, 255, 0.08); color: #E5C983; border: 1px solid #BAA164; font-size: 9px; font-weight: 600; padding: 3px 8px; border-radius: 6px; }"
            "QPushButton:hover { background: rgba(186, 161, 100, 0.25); }"
        )
        self.btn_force_snap.clicked.connect(self.on_force_snap_clicked)

        self.btn_clear_peers = QPushButton("✖")
        self.btn_clear_peers.setFixedWidth(22)
        self.btn_clear_peers.setStyleSheet(
            "background: rgba(255, 255, 255, 0.06); color: #FCA5A5; border: 1px solid rgba(248, 113, 113, 0.45); font-size: 9px; font-weight: bold; border-radius: 6px;"
        )
        self.btn_clear_peers.clicked.connect(self.on_clear_peers_clicked)

        cslam_peer_row.addWidget(self.input_peer_ip, 1)
        cslam_peer_row.addWidget(self.btn_add_peer_ip)
        cslam_peer_row.addWidget(self.btn_sim_peer)
        cslam_peer_row.addWidget(self.btn_force_snap)
        cslam_peer_row.addWidget(self.btn_clear_peers)
        cslam_layout.addLayout(cslam_peer_row)

        self.lbl_cslam_summary = QLabel("Scanners: 1 (Local) | Snapped: 0 | IMU+Offset: 0 | Unlimited Peers Ready")
        self.lbl_cslam_summary.setStyleSheet("color: #E5C983; font-size: 9px; font-weight: 600;")
        cslam_layout.addWidget(self.lbl_cslam_summary)

        self.cslam_peer_list = QTextEdit()
        self.cslam_peer_list.setReadOnly(True)
        self.cslam_peer_list.setFixedHeight(54)
        self.cslam_peer_list.setFont(QFont("IBM Plex Mono", 8))
        self.cslam_peer_list.setStyleSheet(
            "background: rgba(255, 255, 255, 0.04); color: #FCFBF8; border: 1px solid rgba(255, 255, 255, 0.16); border-radius: 6px; padding: 2px;"
        )
        self.cslam_peer_list.setHtml("<span style='color:#E5C983; font-weight:bold;'>● Local Scanner #0 (Host) — Active Anchor</span>")
        cslam_layout.addWidget(self.cslam_peer_list)

        self.hud_tabs.addTab(tab_cslam, "🌐 C-SLAM Peers (∞)")

        # === TAB 3: GEMINI DEPTH GATING & COVERAGE TELEMETRY ===
        tab_gate = QWidget()
        gate_tab_layout = QVBoxLayout(tab_gate)
        gate_tab_layout.setSpacing(4)
        gate_tab_layout.setContentsMargins(8, 8, 8, 8)

        preset_row = QHBoxLayout()
        preset_lbl = QLabel("Preset:")
        preset_lbl.setStyleSheet("color: rgba(252, 251, 248, 0.78); font-size: 9px;")
        self.btn_preset_near = QPushButton("Near (0.3-2m)")
        self.btn_preset_balanced = QPushButton("Balanced (0.3-4m)")
        self.btn_preset_extended = QPushButton("Extended (0.3-6m)")

        for p_btn in [self.btn_preset_near, self.btn_preset_balanced, self.btn_preset_extended]:
            p_btn.setStyleSheet(
                "QPushButton { background: rgba(255, 255, 255, 0.06); border: 1px solid rgba(255, 255, 255, 0.22); color: #FCFBF8; font-size: 8px; font-weight: 600; padding: 3px 6px; border-radius: 6px; }"
                "QPushButton:hover { background: rgba(186, 161, 100, 0.22); border-color: #BAA164; color: #E5C983; }"
            )
        self.btn_preset_near.clicked.connect(lambda: self.apply_gating_preset(0.3, 2.0, 0.040, 0.50))
        self.btn_preset_balanced.clicked.connect(lambda: self.apply_gating_preset(0.3, 4.0, 0.050, 0.50))
        self.btn_preset_extended.clicked.connect(lambda: self.apply_gating_preset(0.3, 6.0, 0.080, 0.70))

        preset_row.addWidget(preset_lbl)
        preset_row.addWidget(self.btn_preset_near)
        preset_row.addWidget(self.btn_preset_balanced)
        preset_row.addWidget(self.btn_preset_extended)
        gate_tab_layout.addLayout(preset_row)

        form = QFormLayout()
        form.setSpacing(2)

        self.slider_min_d = QSlider(Qt.Orientation.Horizontal)
        self.slider_min_d.setRange(25, 100)
        self.slider_min_d.setValue(30)
        self.lbl_min_d = QLabel("0.30 m")
        self.lbl_min_d.setStyleSheet("color: #E5C983; font-size: 9px; font-weight: bold;")
        self.slider_min_d.valueChanged.connect(self.on_sliders_changed)

        self.slider_max_d = QSlider(Qt.Orientation.Horizontal)
        self.slider_max_d.setRange(10, 60)
        self.slider_max_d.setValue(40)
        self.lbl_max_d = QLabel("4.00 m")
        self.lbl_max_d.setStyleSheet("color: #E5C983; font-size: 9px; font-weight: bold;")
        self.slider_max_d.valueChanged.connect(self.on_sliders_changed)

        self.slider_agree = QSlider(Qt.Orientation.Horizontal)
        self.slider_agree.setRange(20, 100)
        self.slider_agree.setValue(50)
        self.lbl_agree = QLabel("50 mm")
        self.lbl_agree.setStyleSheet("color: #E5C983; font-size: 9px; font-weight: bold;")
        self.slider_agree.valueChanged.connect(self.on_sliders_changed)

        form.addRow("Min Depth:", self.make_slider_box(self.slider_min_d, self.lbl_min_d))
        form.addRow("Max Depth:", self.make_slider_box(self.slider_max_d, self.lbl_max_d))
        form.addRow("Agreement:", self.make_slider_box(self.slider_agree, self.lbl_agree))
        gate_tab_layout.addLayout(form)

        self.lbl_cov_rgb = QLabel("RGB Scene Coverage: 0.0% of mapped geometry")
        self.lbl_cov_rgb.setStyleSheet("color: #FCFBF8; font-size: 9px; font-weight: 600;")
        self.bar_rgb = QProgressBar()
        self.bar_rgb.setRange(0, 100)
        self.bar_rgb.setValue(0)
        self.bar_rgb.setFixedHeight(5)
        self.bar_rgb.setTextVisible(False)
        self.bar_rgb.setStyleSheet(
            "QProgressBar { border: 1px solid rgba(255, 255, 255, 0.18); border-radius: 2px; background: rgba(255, 255, 255, 0.06); } "
            "QProgressBar::chunk { background: #BAA164; border-radius: 2px; }"
        )

        self.lbl_cov_dual = QLabel("Dual Coverage (RGB + Depth): 0.0%")
        self.lbl_cov_dual.setStyleSheet("color: rgba(252, 251, 248, 0.78); font-size: 9px;")
        self.lbl_stereo_stats = QLabel("Gemini Points: Accepted: 0 (0%) | Rejected: 0 (0%)")
        self.lbl_stereo_stats.setStyleSheet("color: rgba(252, 251, 248, 0.78); font-size: 9px;")
        self.lbl_elevation_stats = QLabel("LiDAR Elevation: Floor: 0% | Wall Band: 0% | Ceiling: 0%")
        self.lbl_elevation_stats.setStyleSheet("color: #E5C983; font-size: 9px; font-weight: 600;")

        gate_tab_layout.addWidget(self.lbl_cov_rgb)
        gate_tab_layout.addWidget(self.bar_rgb)
        gate_tab_layout.addWidget(self.lbl_cov_dual)
        gate_tab_layout.addWidget(self.lbl_stereo_stats)
        gate_tab_layout.addWidget(self.lbl_elevation_stats)

        self.hud_tabs.addTab(tab_gate, "⚙ Depth · Telemetry")
        right_layout.addWidget(self.hud_tabs)

        # 5. Translucent Apple Glass Studio Log Card
        self.console = QTextEdit()
        self.console.setReadOnly(True)
        self.console.setFont(QFont("IBM Plex Mono", 8))
        self.console.setStyleSheet(
            "background: rgba(255, 255, 255, 0.04); color: rgba(252, 251, 248, 0.88); "
            "border: 1px solid rgba(186, 161, 100, 0.35); border-radius: 10px; padding: 4px;"
        )
        self.console.append("[Artificial Vision · scanHUB] Apple Glass White & Gold AR Spatial OS Initialized.")
        self.console.append("[AR PASSTHROUGH] Frameless translucent window + transparent Min/Max/Close controls active.")
        self.console.append("[M12 CLEANUP] Post-processing mode default (Real-time filter OFF for zero-lag 10Hz/20Hz streaming).")
        self.console.append("[C-SLAM] Unlimited Multi-Scanner Engine ONLINE (UDP Port 19440 | Non-Overlap IMU+Offset -> Auto Overlap Snap).")
        right_layout.addWidget(self.console, 1)

        splitter.addWidget(right_container)
        splitter.setSizes([790, 550])
        capture_page_layout.addWidget(splitter, 1)

        self.workspace_stack.addWidget(self.capture_studio_page)
        # Open directly to Page 0 (Projects, Datasets & Active C-SLAM Captures Hub) after Splash Screen
        self.workspace_stack.setCurrentIndex(0)
        main_layout.addWidget(self.workspace_stack, 1)

        # ─────────────────────────────────────────────────────────────────────
        # 5. STATUS BAR (Translucent Apple Glass Footer + Transparent Corner SizeGrip)
        # ─────────────────────────────────────────────────────────────────────
        self.status_bar = QStatusBar()
        self.status_bar.setSizeGripEnabled(False)
        self.status_bar.setStyleSheet(
            "QStatusBar { background: rgba(255, 255, 255, 0.04); "
            "border-top: 1px solid rgba(186, 161, 100, 0.32); "
            "color: #E5C983; font-family: 'IBM Plex Mono', Monospace; font-size: 9px; font-weight: 600; "
            "border-bottom-left-radius: 14px; border-bottom-right-radius: 14px; }"
        )
        self.size_grip = QSizeGrip(self.status_bar)
        self.size_grip.setStyleSheet("background: transparent; width: 14px; height: 14px;")
        self.status_bar.addPermanentWidget(self.size_grip)
        self.status_bar.showMessage(
            "VERIFY · DIRECT · CLARIFY  |  Select or Create a Project & Dataset, or Join an Active C-SLAM Capture to enter Capture Studio"
        )
        main_layout.addWidget(self.status_bar)

        # Connect ROS Worker Signals
        self.ros_worker.image_received.connect(self.camera_canvas.update_frame)
        self.ros_worker.pose_received.connect(self.map_canvas.update_pose)
        self.ros_worker.pose_received.connect(self.center_viewport.update_pose)
        self.ros_worker.map_points_updated.connect(self.map_canvas.update_map_points)
        self.ros_worker.map_points_updated.connect(self.center_viewport.update_scene_data)
        self.ros_worker.telemetry_updated.connect(self.update_telemetry_ui)
        self.ros_worker.start()

    def _build_projects_hub_page(self) -> QWidget:
        """
        Builds the Post-Splash Projects, Datasets & Live C-SLAM Active Captures Hub (Page 0).
        Maintains the Apple Glass (visionOS) + Artificial Vision White & Champagne Gold (#FCFBF8, #BAA164, #E5C983)
        ultra-transparent aesthetic.
        """
        hub_page = QWidget()
        hub_page.setStyleSheet("background: transparent;")
        root_layout = QVBoxLayout(hub_page)
        root_layout.setContentsMargins(18, 12, 18, 14)
        root_layout.setSpacing(12)

        # Top Hub Action Banner
        banner = QFrame()
        banner.setFixedHeight(56)
        banner.setStyleSheet(
            "QFrame { background: rgba(255, 255, 255, 0.04); "
            "border: 1px solid rgba(186, 161, 100, 0.42); border-radius: 12px; }"
        )
        b_layout = QHBoxLayout(banner)
        b_layout.setContentsMargins(16, 6, 16, 6)
        b_layout.setSpacing(12)

        title_col = QVBoxLayout()
        title_col.setSpacing(1)
        lbl_hub_title = QLabel("PROJECTS & COLLABORATIVE C-SLAM HUB")
        lbl_hub_title.setFont(QFont("IBM Plex Sans", 11, QFont.Weight.Bold))
        lbl_hub_title.setStyleSheet("color: #FCFBF8; letter-spacing: 0.8px; border: none; background: transparent;")
        lbl_hub_sub = QLabel(
            "Select or create a Project  ·  Create or open a Dataset  ·  Or join an Active Capture on the network via C-SLAM"
        )
        lbl_hub_sub.setFont(QFont("IBM Plex Sans", 8))
        lbl_hub_sub.setStyleSheet("color: #E5C983; border: none; background: transparent;")
        title_col.addWidget(lbl_hub_title)
        title_col.addWidget(lbl_hub_sub)
        b_layout.addLayout(title_col)
        b_layout.addStretch()

        self.hub_stack_btn = QPushButton("🚀 START STACK")
        self.hub_stack_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.hub_stack_btn.setStyleSheet(
            "QPushButton { background: rgba(255, 255, 255, 0.07); border: 1px solid rgba(186, 161, 100, 0.60); "
            "color: #E5C983; padding: 7px 15px; border-radius: 9px; font-weight: 600; font-size: 10px; }"
            "QPushButton:hover { background: rgba(186, 161, 100, 0.24); border-color: #FCFBF8; color: #FFFFFF; }"
        )
        self.hub_stack_btn.clicked.connect(self.toggle_ros_stack)

        self.btn_hub_quick_resume = QPushButton("▶ RESUME CAPTURE STUDIO  →")
        self.btn_hub_quick_resume.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_hub_quick_resume.setStyleSheet(
            "QPushButton { background: rgba(186, 161, 100, 0.28); border: 1px solid #E5C983; "
            "color: #FCFBF8; padding: 7px 16px; border-radius: 9px; font-weight: 600; font-size: 10px; }"
            "QPushButton:hover { background: rgba(186, 161, 100, 0.48); border-color: #FFFFFF; }"
        )
        self.btn_hub_quick_resume.clicked.connect(self.on_enter_capture_studio_clicked)

        b_layout.addWidget(self.hub_stack_btn)
        b_layout.addWidget(self.btn_hub_quick_resume)
        root_layout.addWidget(banner)

        # 3-Column Glass Cards Row
        cards_row = QHBoxLayout()
        cards_row.setSpacing(14)

        card_style = (
            "QFrame { background: rgba(255, 255, 255, 0.04); "
            "border: 1px solid rgba(186, 161, 100, 0.38); border-radius: 12px; }"
        )
        list_style = (
            "QListWidget { background: rgba(12, 12, 16, 0.48); "
            "border: 1px solid rgba(255, 255, 255, 0.18); border-radius: 8px; "
            "color: #FCFBF8; padding: 6px; font-size: 10px; outline: none; }"
            "QListWidget::item { padding: 8px 10px; border-radius: 6px; margin-bottom: 3px; "
            "border: 1px solid rgba(255, 255, 255, 0.06); }"
            "QListWidget::item:selected { background: rgba(186, 161, 100, 0.28); "
            "border: 1px solid #E5C983; color: #FFFFFF; font-weight: bold; }"
            "QListWidget::item:hover { background: rgba(255, 255, 255, 0.10); }"
        )
        input_style = (
            "QLineEdit { background: rgba(255, 255, 255, 0.07); "
            "border: 1px solid rgba(186, 161, 100, 0.48); color: #FCFBF8; "
            "padding: 7px 10px; border-radius: 8px; font-size: 10px; font-weight: 600; }"
        )

        # ── COLUMN 1: PROJECTS ──
        col1 = QFrame()
        col1.setStyleSheet(card_style)
        c1_layout = QVBoxLayout(col1)
        c1_layout.setContentsMargins(14, 12, 14, 12)
        c1_layout.setSpacing(8)

        lbl_c1 = QLabel("1 · SELECT OR CREATE PROJECT")
        lbl_c1.setFont(QFont("IBM Plex Mono", 9, QFont.Weight.Bold))
        lbl_c1.setStyleSheet("color: #E5C983; letter-spacing: 1.0px; border: none; background: transparent;")
        lbl_c1_desc = QLabel("Organize scans by building, site, or job:")
        lbl_c1_desc.setStyleSheet("color: rgba(252, 251, 248, 0.72); font-size: 9px; border: none; background: transparent;")
        c1_layout.addWidget(lbl_c1)
        c1_layout.addWidget(lbl_c1_desc)

        self.hub_projects_list = QListWidget()
        self.hub_projects_list.setStyleSheet(list_style)
        self.hub_projects_list.itemSelectionChanged.connect(self.on_hub_project_selected)
        c1_layout.addWidget(self.hub_projects_list, 1)

        new_proj_row = QHBoxLayout()
        new_proj_row.setSpacing(6)
        self.hub_new_project_input = QLineEdit()
        self.hub_new_project_input.setPlaceholderText("New Project Name (e.g. Site_Tower_A)")
        self.hub_new_project_input.setStyleSheet(input_style)
        self.hub_new_project_input.returnPressed.connect(self.on_create_project_clicked)

        self.btn_create_project = QPushButton("+ New Project")
        self.btn_create_project.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_create_project.setStyleSheet(
            "QPushButton { background: rgba(186, 161, 100, 0.24); border: 1px solid #BAA164; "
            "color: #E5C983; padding: 7px 12px; border-radius: 8px; font-weight: 600; font-size: 9px; }"
            "QPushButton:hover { background: rgba(186, 161, 100, 0.42); color: #FFFFFF; }"
        )
        self.btn_create_project.clicked.connect(self.on_create_project_clicked)
        new_proj_row.addWidget(self.hub_new_project_input, 1)
        new_proj_row.addWidget(self.btn_create_project)
        c1_layout.addLayout(new_proj_row)

        cards_row.addWidget(col1, 1)

        # ── COLUMN 2: DATASETS WITHIN SELECTED PROJECT ──
        col2 = QFrame()
        col2.setStyleSheet(card_style)
        c2_layout = QVBoxLayout(col2)
        c2_layout.setContentsMargins(14, 12, 14, 12)
        c2_layout.setSpacing(8)

        lbl_c2 = QLabel("2 · PROJECT DATASETS")
        lbl_c2.setFont(QFont("IBM Plex Mono", 9, QFont.Weight.Bold))
        lbl_c2.setStyleSheet("color: #E5C983; letter-spacing: 1.0px; border: none; background: transparent;")
        self.lbl_hub_selected_project = QLabel(f"Active Project: {self.active_project_name}")
        self.lbl_hub_selected_project.setStyleSheet("color: #FCFBF8; font-size: 9px; font-weight: 600; border: none; background: transparent;")
        c2_layout.addWidget(lbl_c2)
        c2_layout.addWidget(self.lbl_hub_selected_project)

        self.hub_datasets_list = QListWidget()
        self.hub_datasets_list.setStyleSheet(list_style)
        self.hub_datasets_list.itemSelectionChanged.connect(self.on_hub_dataset_selected)
        self.hub_datasets_list.itemDoubleClicked.connect(lambda _: self.on_enter_capture_studio_clicked())
        c2_layout.addWidget(self.hub_datasets_list, 1)

        self.lbl_hub_dataset_meta = QLabel("Select an existing dataset above or create a new dataset below.")
        self.lbl_hub_dataset_meta.setStyleSheet(
            "color: rgba(252, 251, 248, 0.78); font-size: 9px; font-family: 'IBM Plex Mono', Monospace; "
            "background: rgba(255, 255, 255, 0.04); border: 1px solid rgba(255, 255, 255, 0.14); "
            "border-radius: 6px; padding: 5px 8px;"
        )
        c2_layout.addWidget(self.lbl_hub_dataset_meta)

        new_ds_row = QHBoxLayout()
        new_ds_row.setSpacing(6)
        self.hub_new_dataset_input = QLineEdit()
        self.hub_new_dataset_input.setText(self.active_dataset_name)
        self.hub_new_dataset_input.setPlaceholderText("Dataset Name (e.g. Floor1_North)")
        self.hub_new_dataset_input.setStyleSheet(input_style)
        self.hub_new_dataset_input.returnPressed.connect(self.on_create_dataset_clicked)

        self.btn_create_dataset = QPushButton("+ New Dataset")
        self.btn_create_dataset.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_create_dataset.setStyleSheet(
            "QPushButton { background: rgba(255, 255, 255, 0.10); border: 1px solid #BAA164; "
            "color: #E5C983; padding: 7px 12px; border-radius: 8px; font-weight: 600; font-size: 9px; }"
            "QPushButton:hover { background: rgba(186, 161, 100, 0.30); color: #FFFFFF; }"
        )
        self.btn_create_dataset.clicked.connect(self.on_create_dataset_clicked)
        new_ds_row.addWidget(self.hub_new_dataset_input, 1)
        new_ds_row.addWidget(self.btn_create_dataset)
        c2_layout.addLayout(new_ds_row)

        self.btn_enter_capture_studio = QPushButton("▶ OPEN CAPTURE STUDIO FOR THIS DATASET  →")
        self.btn_enter_capture_studio.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_enter_capture_studio.setStyleSheet(
            "QPushButton { background: rgba(186, 161, 100, 0.36); border: 1px solid #E5C983; "
            "color: #FCFBF8; padding: 10px 14px; border-radius: 9px; font-weight: bold; font-size: 10px; }"
            "QPushButton:hover { background: rgba(186, 161, 100, 0.56); border-color: #FFFFFF; color: #FFFFFF; }"
        )
        self.btn_enter_capture_studio.clicked.connect(self.on_enter_capture_studio_clicked)
        c2_layout.addWidget(self.btn_enter_capture_studio)

        cards_row.addWidget(col2, 1)

        # ── COLUMN 3: LIVE ACTIVE CAPTURES ON NETWORK (C-SLAM JOIN) ──
        col3 = QFrame()
        col3.setStyleSheet(card_style)
        c3_layout = QVBoxLayout(col3)
        c3_layout.setContentsMargins(14, 12, 14, 12)
        c3_layout.setSpacing(8)

        lbl_c3 = QLabel("3 · ACTIVE C-SLAM CAPTURES ON NETWORK")
        lbl_c3.setFont(QFont("IBM Plex Mono", 9, QFont.Weight.Bold))
        lbl_c3.setStyleSheet("color: #4ADE80; letter-spacing: 1.0px; border: none; background: transparent;")
        self.lbl_hub_cslam_status = QLabel("Listening on UDP 19440 (LAN Broadcast + Unicast) for active scanners...")
        self.lbl_hub_cslam_status.setStyleSheet("color: rgba(252, 251, 248, 0.78); font-size: 9px; border: none; background: transparent;")
        c3_layout.addWidget(lbl_c3)
        c3_layout.addWidget(self.lbl_hub_cslam_status)

        self.hub_active_captures_list = QListWidget()
        self.hub_active_captures_list.setStyleSheet(list_style)
        self.hub_active_captures_list.itemDoubleClicked.connect(lambda _: self.on_join_active_capture_clicked())
        c3_layout.addWidget(self.hub_active_captures_list, 1)

        peer_ip_row = QHBoxLayout()
        peer_ip_row.setSpacing(6)
        self.hub_peer_ip_input = QLineEdit()
        self.hub_peer_ip_input.setPlaceholderText("Remote Device IP (e.g. 192.168.1.50)")
        self.hub_peer_ip_input.setStyleSheet(input_style)
        self.hub_peer_ip_input.returnPressed.connect(self.on_hub_add_ip_clicked)

        self.btn_hub_add_ip = QPushButton("+ Connect IP")
        self.btn_hub_add_ip.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_hub_add_ip.setStyleSheet(
            "QPushButton { background: rgba(255, 255, 255, 0.09); border: 1px solid #BAA164; "
            "color: #E5C983; padding: 7px 10px; border-radius: 8px; font-weight: 600; font-size: 9px; }"
            "QPushButton:hover { background: rgba(186, 161, 100, 0.28); color: #FFFFFF; }"
        )
        self.btn_hub_add_ip.clicked.connect(self.on_hub_add_ip_clicked)

        self.btn_hub_sim_capture = QPushButton("+ Sim Active Capture")
        self.btn_hub_sim_capture.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_hub_sim_capture.setToolTip("Simulate a remote scanner broadcasting an active capture session on the network")
        self.btn_hub_sim_capture.setStyleSheet(
            "QPushButton { background: rgba(186, 161, 100, 0.18); border: 1px solid rgba(229, 201, 131, 0.55); "
            "color: #E5C983; padding: 7px 10px; border-radius: 8px; font-weight: 600; font-size: 9px; }"
            "QPushButton:hover { background: rgba(186, 161, 100, 0.36); color: #FFFFFF; }"
        )
        self.btn_hub_sim_capture.clicked.connect(self.on_hub_sim_active_capture_clicked)

        peer_ip_row.addWidget(self.hub_peer_ip_input, 1)
        peer_ip_row.addWidget(self.btn_hub_add_ip)
        peer_ip_row.addWidget(self.btn_hub_sim_capture)
        c3_layout.addLayout(peer_ip_row)

        self.btn_join_active_capture = QPushButton("🌐 JOIN SELECTED CAPTURE WITH C-SLAM  →")
        self.btn_join_active_capture.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_join_active_capture.setStyleSheet(
            "QPushButton { background: rgba(34, 197, 94, 0.24); border: 1px solid #4ADE80; "
            "color: #FCFBF8; padding: 10px 14px; border-radius: 9px; font-weight: bold; font-size: 10px; }"
            "QPushButton:hover { background: rgba(34, 197, 94, 0.42); border-color: #FFFFFF; color: #FFFFFF; }"
        )
        self.btn_join_active_capture.clicked.connect(self.on_join_active_capture_clicked)
        c3_layout.addWidget(self.btn_join_active_capture)

        cards_row.addWidget(col3, 1)
        root_layout.addLayout(cards_row, 1)
        return hub_page

    # ─────────────────────────────────────────────────────────────────────────
    # PROJECTS, DATASETS & LIVE C-SLAM HUB NAVIGATION METHODS
    # ─────────────────────────────────────────────────────────────────────────
    def show_projects_hub_view(self):
        """Switches to Page 0 (Projects, Datasets & Active C-SLAM Captures Hub)."""
        self.refresh_projects_hub_lists()
        self.refresh_active_cslam_captures_ui()
        self.workspace_stack.setCurrentIndex(0)
        self.status_bar.showMessage(
            f"PROJECTS HUB  |  Active Project: {self.active_project_name}  ·  Dataset: {self.active_dataset_name}"
        )

    def show_capture_studio_view(self, project_name: str = None, dataset_name: str = None, joined_cslam: bool = False):
        """Enters Page 1 (Capture Studio) for the specified Project and Dataset."""
        if project_name:
            self.active_project_name = project_name.strip().replace(" ", "_") or "Default_Project"
        if dataset_name:
            self.active_dataset_name = dataset_name.strip().replace(" ", "_") or f"Dataset_{datetime.now().strftime('%m%d_%H%M')}"

        if self.active_project_name != "Legacy_Scans":
            proj_ds_dir = os.path.join(self.projects_root, self.active_project_name, "datasets", self.active_dataset_name)
            os.makedirs(proj_ds_dir, exist_ok=True)

        self.lbl_active_project_badge.setText(f"PROJECT: {self.active_project_name} /")
        self.dataset_input.blockSignals(True)
        self.dataset_input.setText(self.active_dataset_name)
        self.dataset_input.blockSignals(False)

        self.ros_worker.cslam_engine.set_project_name(self.active_project_name, self.active_dataset_name)
        self.workspace_stack.setCurrentIndex(1)

        if joined_cslam:
            self.hud_tabs.setCurrentIndex(1)  # Focus C-SLAM Peers tab
            self.console.append(
                f"\n[C-SLAM JOIN] Joined Active Network Capture -> Project: {self.active_project_name} | Dataset: {self.active_dataset_name}"
            )
            self.status_bar.showMessage(
                f"C-SLAM COLLABORATIVE SESSION  |  Project: {self.active_project_name}  ·  Dataset: {self.active_dataset_name}"
            )
        else:
            self.console.append(
                f"\n[PROJECT READY] Opened Project '{self.active_project_name}' · Dataset '{self.active_dataset_name}' in Capture Studio."
            )
            self.status_bar.showMessage(
                f"CAPTURE STUDIO READY  |  Project: {self.active_project_name}  ·  Dataset: {self.active_dataset_name}"
            )
        self.ros_worker.dispatch_live_map_and_telemetry(force=True)

    def refresh_projects_hub_lists(self):
        """Scans /home/scanar/scanarMini/projects (plus Legacy_Scans in data/) and populates the Projects & Datasets lists."""
        os.makedirs(self.projects_root, exist_ok=True)
        os.makedirs(os.path.join(self.projects_root, "Default_Project", "datasets"), exist_ok=True)

        projects = []
        try:
            for entry in sorted(os.listdir(self.projects_root)):
                full_p = os.path.join(self.projects_root, entry)
                if os.path.isdir(full_p) and not entry.startswith("."):
                    projects.append(entry)
        except Exception:
            pass

        if "Default_Project" not in projects:
            projects.insert(0, "Default_Project")

        # Also surface existing scans in /home/scanar/scanarMini/data as Legacy_Scans
        if os.path.isdir(self.legacy_data_root):
            try:
                legacy_items = [
                    d for d in os.listdir(self.legacy_data_root)
                    if os.path.isdir(os.path.join(self.legacy_data_root, d)) and not d.startswith(".")
                ]
                if legacy_items and "Legacy_Scans" not in projects:
                    projects.append("Legacy_Scans")
            except Exception:
                pass

        self.hub_projects_list.blockSignals(True)
        self.hub_projects_list.clear()
        selected_idx = 0
        for idx, proj in enumerate(projects):
            ds_count = len(self._list_datasets_for_project(proj))
            item = QListWidgetItem(f"📁  {proj}   ({ds_count} dataset{'s' if ds_count != 1 else ''})")
            item.setData(Qt.ItemDataRole.UserRole if hasattr(Qt, "ItemDataRole") else Qt.UserRole, proj)
            self.hub_projects_list.addItem(item)
            if proj == self.active_project_name:
                selected_idx = idx

        if self.hub_projects_list.count() > 0:
            self.hub_projects_list.setCurrentRow(selected_idx)
        self.hub_projects_list.blockSignals(False)
        self._populate_datasets_for_project(self.active_project_name)

    def _list_datasets_for_project(self, project_name: str):
        if project_name == "Legacy_Scans":
            base_dir = self.legacy_data_root
        else:
            base_dir = os.path.join(self.projects_root, project_name, "datasets")
        if not os.path.isdir(base_dir):
            return []
        out = []
        try:
            for d in sorted(os.listdir(base_dir), reverse=True):
                full_d = os.path.join(base_dir, d)
                if os.path.isdir(full_d) and not d.startswith("."):
                    out.append((d, full_d))
        except Exception:
            pass
        return out

    def _populate_datasets_for_project(self, project_name: str):
        self.lbl_hub_selected_project.setText(f"Active Project: {project_name}")
        datasets = self._list_datasets_for_project(project_name)

        self.hub_datasets_list.blockSignals(True)
        self.hub_datasets_list.clear()
        role = Qt.ItemDataRole.UserRole if hasattr(Qt, "ItemDataRole") else Qt.UserRole

        selected_row = -1
        for idx, (ds_name, ds_path) in enumerate(datasets):
            meta_path = os.path.join(ds_path, "scan_metadata.json")
            badge = "Ready to Scan"
            if os.path.isfile(meta_path):
                try:
                    with open(meta_path, "r") as f:
                        meta = json.load(f)
                    fused_n = meta.get("fused_points", meta.get("airy_points", 0))
                    badge = f"✓ Saved ({fused_n:,} pts)"
                except Exception:
                    badge = "✓ Saved"
            item = QListWidgetItem(f"🗂  {ds_name}   ·   {badge}")
            item.setData(role, ds_name)
            self.hub_datasets_list.addItem(item)
            if ds_name == self.active_dataset_name:
                selected_row = idx

        if selected_row >= 0:
            self.hub_datasets_list.setCurrentRow(selected_row)
        self.hub_datasets_list.blockSignals(False)

    def on_hub_project_selected(self):
        items = self.hub_projects_list.selectedItems()
        if not items:
            return
        role = Qt.ItemDataRole.UserRole if hasattr(Qt, "ItemDataRole") else Qt.UserRole
        proj = items[0].data(role)
        if proj:
            self.active_project_name = proj
            self._populate_datasets_for_project(proj)

    def on_hub_dataset_selected(self):
        items = self.hub_datasets_list.selectedItems()
        if not items:
            return
        role = Qt.ItemDataRole.UserRole if hasattr(Qt, "ItemDataRole") else Qt.UserRole
        ds_name = items[0].data(role)
        if not ds_name:
            return
        self.active_dataset_name = ds_name
        self.hub_new_dataset_input.setText(ds_name)

        if self.active_project_name == "Legacy_Scans":
            ds_path = os.path.join(self.legacy_data_root, ds_name)
        else:
            ds_path = os.path.join(self.projects_root, self.active_project_name, "datasets", ds_name)
        meta_path = os.path.join(ds_path, "scan_metadata.json")
        if os.path.isfile(meta_path):
            try:
                with open(meta_path, "r") as f:
                    meta = json.load(f)
                a_pts = meta.get("airy_points", 0)
                s_pts = meta.get("stereo_points", 0)
                f_pts = meta.get("fused_points", 0)
                ts = meta.get("timestamp", "")[:19]
                self.lbl_hub_dataset_meta.setText(
                    f"Dataset '{ds_name}' ({ts}) — Fused: {f_pts:,} | Airy: {a_pts:,} | Gemini: {s_pts:,} pts"
                )
                return
            except Exception:
                pass
        self.lbl_hub_dataset_meta.setText(
            f"Dataset '{ds_name}' in Project '{self.active_project_name}' — Ready to open in Capture Studio."
        )

    def on_create_project_clicked(self):
        raw_name = self.hub_new_project_input.text().strip().replace(" ", "_")
        if not raw_name:
            raw_name = f"Project_{datetime.now().strftime('%m%d_%H%M')}"
        proj_dir = os.path.join(self.projects_root, raw_name, "datasets")
        os.makedirs(proj_dir, exist_ok=True)
        self.active_project_name = raw_name
        self.hub_new_project_input.clear()
        new_ds = f"Dataset_{datetime.now().strftime('%m%d_%H%M')}"
        os.makedirs(os.path.join(proj_dir, new_ds), exist_ok=True)
        self.active_dataset_name = new_ds
        self.hub_new_dataset_input.setText(new_ds)
        self.refresh_projects_hub_lists()
        self.lbl_hub_dataset_meta.setText(f"Created Project '{raw_name}' with initial dataset '{new_ds}'.")

    def on_create_dataset_clicked(self):
        ds_name = self.hub_new_dataset_input.text().strip().replace(" ", "_")
        if not ds_name:
            ds_name = f"Dataset_{datetime.now().strftime('%m%d_%H%M')}"
            self.hub_new_dataset_input.setText(ds_name)
        if self.active_project_name == "Legacy_Scans":
            self.active_project_name = "Default_Project"
        ds_dir = os.path.join(self.projects_root, self.active_project_name, "datasets", ds_name)
        os.makedirs(ds_dir, exist_ok=True)
        self.active_dataset_name = ds_name
        self.refresh_projects_hub_lists()
        self.lbl_hub_dataset_meta.setText(
            f"Created Dataset '{ds_name}' in Project '{self.active_project_name}'. Click 'OPEN CAPTURE STUDIO' to scan."
        )

    def on_enter_capture_studio_clicked(self):
        ds_name = self.hub_new_dataset_input.text().strip().replace(" ", "_")
        if not ds_name:
            ds_name = self.active_dataset_name or f"Dataset_{datetime.now().strftime('%m%d_%H%M')}"
        self.show_capture_studio_view(self.active_project_name, ds_name, joined_cslam=False)

    def refresh_active_cslam_captures_ui(self):
        """Refreshes Column 3 in the Projects Hub with all active C-SLAM captures discovered on the network."""
        eng = self.ros_worker.cslam_engine
        sessions = eng.get_discovered_network_sessions()
        role = Qt.ItemDataRole.UserRole if hasattr(Qt, "ItemDataRole") else Qt.UserRole

        prev_selected_key = None
        sel_items = self.hub_active_captures_list.selectedItems()
        if sel_items:
            prev_selected_key = sel_items[0].data(role)

        self.hub_active_captures_list.blockSignals(True)
        self.hub_active_captures_list.clear()

        if not sessions:
            port_num = getattr(eng, "_bound_port", 19210)
            host_lbl = getattr(eng, "scanner_name", "ScanHUB-Host")
            self.lbl_hub_cslam_status.setText(
                f"Listening on UDP {port_num} · Local Host: {host_lbl} (No remote captures detected yet)"
            )
            placeholder = QListWidgetItem(
                "📡  No remote active captures detected on LAN yet.\n"
                "     • Start a scan on another device on the same network, OR\n"
                "     • Enter a remote device IP below, OR click '+ Sim Active Capture' to test C-SLAM join."
            )
            placeholder.setData(role, None)
            self.hub_active_captures_list.addItem(placeholder)
        else:
            rec_count = sum(1 for s in sessions if s.get("is_recording"))
            self.lbl_hub_cslam_status.setText(
                f"🟢 {len(sessions)} Active C-SLAM Session(s) on Network ({rec_count} Recording Live)"
            )
            restore_row = 0
            now_t = time.time()
            for idx, sess in enumerate(sessions):
                proj = sess.get("project_name", "Default_Project")
                ds = sess.get("dataset_name", "Active_Scan")
                host_name = sess.get("scanner_name") or sess.get("host_scanner_name", "Remote_Scanner")
                ip = sess.get("ip_address") or sess.get("host_ip", "LAN")
                pts = sess.get("point_count", sess.get("total_points", 0))
                sc_cnt = sess.get("scanner_count", 1)
                is_rec = sess.get("is_recording", False)
                age = sess.get("age_sec", max(0.0, now_t - sess.get("last_seen_ts", now_t)))
                status_tag = "🟢 RECORDING LIVE" if is_rec else "● ONLINE STANDBY"
                label = (
                    f"{status_tag}  |  Project: {proj}  ·  Dataset: {ds}\n"
                    f"     Host: {host_name} ({ip})  ·  {sc_cnt} Scanner(s)  ·  {pts:,} pts  ({age:.1f}s ago)"
                )
                item = QListWidgetItem(label)
                key_tuple = (proj, ds, host_name)
                item.setData(role, key_tuple)
                self.hub_active_captures_list.addItem(item)
                if prev_selected_key == key_tuple:
                    restore_row = idx
            if self.hub_active_captures_list.count() > 0:
                self.hub_active_captures_list.setCurrentRow(restore_row)

        self.hub_active_captures_list.blockSignals(False)

    def on_hub_add_ip_clicked(self):
        ip = self.hub_peer_ip_input.text().strip()
        if not ip:
            return
        self.ros_worker.cslam_engine.add_unicast_peer(ip)
        self.console.append(f"[C-SLAM HUB] Added Unicast Peer Scanner IP: {ip}:19440")
        self.lbl_hub_cslam_status.setText(f"Added Unicast Peer {ip}:19440 — polling for active capture packets...")
        self.hub_peer_ip_input.clear()

    def on_hub_sim_active_capture_clicked(self):
        """Spawns a simulated remote scanner broadcasting an active capture so C-SLAM Join can be tested anytime."""
        self.ros_worker.ensure_local_cloud_for_interactive_ops()
        sim_proj = self.active_project_name or "Default_Project"
        sim_ds = self.hub_new_dataset_input.text().strip().replace(" ", "_") or self.active_dataset_name
        self.ros_worker.cslam_engine.set_project_name(sim_proj, sim_ds)
        sim_id = self.ros_worker.cslam_engine.spawn_simulated_peer_scanner()
        self.refresh_active_cslam_captures_ui()
        self.console.append(
            f"[C-SLAM HUB] Simulated Remote Active Capture ({sim_id[-8:]}) on Project '{sim_proj}' / Dataset '{sim_ds}'."
        )

    def on_join_active_capture_clicked(self):
        """Joins the selected active C-SLAM capture session and opens the Capture Studio."""
        role = Qt.ItemDataRole.UserRole if hasattr(Qt, "ItemDataRole") else Qt.UserRole
        sel_items = self.hub_active_captures_list.selectedItems()
        target_proj = None
        target_ds = None

        if sel_items:
            data = sel_items[0].data(role)
            if isinstance(data, tuple) and len(data) >= 2:
                target_proj, target_ds = data[0], data[1]

        if not target_proj:
            sessions = self.ros_worker.cslam_engine.get_discovered_network_sessions()
            if sessions:
                target_proj = sessions[0].get("project_name", self.active_project_name)
                target_ds = sessions[0].get("dataset_name", self.active_dataset_name)
            else:
                # If no remote session is broadcasting yet, spawn a simulated peer on the current project/dataset
                # so the user can immediately experience collaborative C-SLAM in the Capture Studio
                self.on_hub_sim_active_capture_clicked()
                target_proj = self.active_project_name
                target_ds = self.active_dataset_name

        if self.map_canvas.view_radius_m < 22.0:
            self.map_canvas.set_view_radius(25.0)
        self.show_capture_studio_view(target_proj, target_ds, joined_cslam=True)

    def _style_layer_buttons(self, active_mode: str):
        for mode_key, btn in getattr(self, "_layer_buttons", {}).items():
            if mode_key == active_mode:
                btn.setStyleSheet(
                    "background: rgba(186, 161, 100, 0.28); border: 1px solid #E5C983; color: #FFFFFF; "
                    "padding: 3px 7px; font-size: 9px; font-weight: bold; border-radius: 6px;"
                )
            else:
                btn.setStyleSheet(
                    "QPushButton { background: rgba(255, 255, 255, 0.05); border: 1px solid rgba(255, 255, 255, 0.18); "
                    "color: rgba(252, 251, 248, 0.75); padding: 3px 7px; font-size: 9px; font-weight: 600; border-radius: 6px; }"
                    "QPushButton:hover { background: rgba(186, 161, 100, 0.16); color: #FCFBF8; }"
                )

    def make_slider_box(self, slider, label):
        w = QWidget()
        h = QHBoxLayout(w)
        h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(slider, 1)
        h.addWidget(label)
        return w

    def on_project_name_changed(self, text: str):
        ds = text.strip().replace(" ", "_") or "Default_Dataset"
        self.active_dataset_name = ds
        self.ros_worker.cslam_engine.set_project_name(self.active_project_name, ds)

    def build_m12_params_from_ui(self) -> CleanupParams:
        return CleanupParams(
            enable_voxel_downsample=self.chk_m12_voxel.isChecked(),
            voxel_leaf_size_m=float(self.spin_m12_voxel.value()),
            enable_sor=self.chk_m12_sor.isChecked(),
            sor_mean_k=int(self.spin_m12_sor_k.value()),
            sor_std_mul=float(self.spin_m12_sor_std.value()),
            enable_ror=self.chk_m12_ror.isChecked(),
            ror_radius_m=float(self.spin_m12_ror_r.value()),
            ror_min_neighbors=int(self.spin_m12_ror_n.value()),
            enable_ground_alignment=self.chk_m12_ground.isChecked(),
            ground_dist_threshold_m=0.04,
            ground_max_iterations=400,
            ground_max_tilt_deg=35.0,
            align_ground_to_z_zero=True,
            remove_ground_plane=self.chk_m12_rm_ground.isChecked(),
        )

    def sync_m12_params_from_ui(self):
        with self.ros_worker.lock:
            self.ros_worker.m12_params = self.build_m12_params_from_ui()

    def on_quick_m12_rt_clicked(self, checked: bool):
        self.chk_m12_realtime.blockSignals(True)
        self.chk_m12_realtime.setChecked(checked)
        self.chk_m12_realtime.blockSignals(False)
        self.on_m12_realtime_toggled(checked)

    def on_m12_realtime_toggled(self, checked: bool):
        self.ros_worker.set_live_cleanup_enabled(checked)
        self.btn_quick_m12_rt.blockSignals(True)
        self.btn_quick_m12_rt.setChecked(checked)
        self.btn_quick_m12_rt.setText(f"✦ M12 RT: {'ON' if checked else 'OFF'}")
        self.btn_quick_m12_rt.setStyleSheet(
            f"background: {'rgba(186, 161, 100, 0.18)' if checked else 'rgba(255, 255, 255, 0.07)'}; "
            f"border: 1px solid {'#BAA164' if checked else 'rgba(186, 161, 100, 0.45)'}; "
            f"color: {'#E5C983' if checked else 'rgba(252, 251, 248, 0.78)'}; "
            f"padding: 6px 12px; border-radius: 9px; font-weight: 600; font-size: 10px;"
        )
        self.btn_quick_m12_rt.blockSignals(False)
        if not checked:
            self.lbl_m12_status.setText("M12 RT Filter: OFF (Post-Process Ready — Preview / Commit / Export)")
        self.console.append(f"[M12 CLEANUP] Real-Time Spatial-Hash Point Cloud Cleanup: {'ENABLED' if checked else 'DISABLED (Post-Process Mode)'}")

    def on_m12_ground_lock_toggled(self, checked: bool):
        self.ros_worker.set_live_ground_lock_enabled(checked)
        self.console.append(f"[M12 CLEANUP] Live RANSAC Z=0 Floor Lock: {'ENABLED' if checked else 'DISABLED'}")

    def on_m12_preview_clicked(self):
        self._m12_status_hold_until = time.time() + 2.5
        params = self.build_m12_params_from_ui()
        stats = self.ros_worker.run_m12_cleanup_preview(params)
        self.lbl_m12_status.setText(
            f"PREVIEW: {stats.input_points:,} -> {stats.output_points:,} pts | "
            f"-{stats.sor_removed_points + stats.ror_removed_points} outliers | "
            f"Tilt: {stats.ground_result.tilt_angle_deg:.1f}° ({stats.processing_time_ms:.1f}ms)"
        )
        self.console.append(
            f"[M12 PREVIEW] {stats.input_points:,} -> {stats.output_points:,} pts "
            f"(Voxel: {stats.voxel_downsampled_points:,}, SOR: -{stats.sor_removed_points}, "
            f"ROR: -{stats.ror_removed_points}, Ground Tilt: {stats.ground_result.tilt_angle_deg:.2f}°, "
            f"{stats.processing_time_ms:.1f} ms)"
        )

    def on_m12_reset_clicked(self):
        self._m12_status_hold_until = time.time() + 2.5
        self.ros_worker.reset_m12_cleanup_preview()
        self.lbl_m12_status.setText("M12 Preview Reset — Restored original uncleaned point cloud.")
        self.console.append("[M12 RESET] Restored original point cloud state.")

    def on_m12_commit_clicked(self):
        self._m12_status_hold_until = time.time() + 2.5
        committed = self.ros_worker.commit_m12_cleanup()
        self.lbl_m12_status.setText(f"M12 Committed: {committed:,} clean points locked into active map.")
        self.console.append(f"[M12 COMMIT] Committed {committed:,} cleaned points to active session map.")

    def _get_active_dataset_save_dir(self, dataset_name: str) -> str:
        if self.active_project_name == "Legacy_Scans":
            return os.path.join(self.legacy_data_root, dataset_name)
        return os.path.join(self.projects_root, self.active_project_name, "datasets", dataset_name)

    def on_m12_export_clicked(self):
        self._m12_status_hold_until = time.time() + 2.5
        fmt_str = self.combo_m12_fmt.currentText()
        ext_map = {
            "PLY_BINARY": "ply",
            "PLY_ASCII": "ply",
            "PCD_BINARY": "pcd",
            "PCD_ASCII": "pcd",
            "XYZ_ASCII": "xyz",
            "OBJ_MESH": "obj",
        }
        ext = ext_map.get(fmt_str, "ply")
        dataset = self.dataset_input.text().strip().replace(" ", "_") or "Scan"
        base_ds_dir = self._get_active_dataset_save_dir(dataset)
        out_dir = os.path.join(base_ds_dir, "m12_exports")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{dataset}_m12_{fmt_str.lower()}.{ext}")
        res = self.ros_worker.export_m12_cloud(fmt_str, out_path, mesh_res_m=float(self.spin_m12_voxel.value()) * 2.0)
        if res.success:
            extra = f", {res.exported_triangles:,} triangles" if res.exported_triangles > 0 else ""
            self.lbl_m12_status.setText(
                f"Exported {fmt_str}: {res.exported_points:,} pts{extra} ({res.file_size_bytes // 1024:,} KB)"
            )
            self.console.append(
                f"[M12 EXPORT] Saved {fmt_str} -> {res.file_path} "
                f"({res.exported_points:,} pts{extra}, {res.export_time_ms:.1f} ms)"
            )
        else:
            self.lbl_m12_status.setText(f"Export Failed: {res.error_message}")
            self.console.append(f"[M12 EXPORT ERROR] {res.error_message}")

    def on_cslam_color_mode_toggled(self):
        eng = self.ros_worker.cslam_engine
        if eng.color_mode == CSlamColorMode.SCANNER_COLORS:
            eng.set_color_mode(CSlamColorMode.TRUE_RGB)
            self.btn_cslam_color_mode.setText("🎨 True RGB Colors")
            self.btn_cslam_color_mode.setStyleSheet(
                "background: rgba(255, 255, 255, 0.08); color: #FCFBF8; border: 1px solid rgba(255, 255, 255, 0.25); font-size: 9px; font-weight: 600; padding: 3px 8px; border-radius: 6px;"
            )
            self.console.append("[C-SLAM] Switched display to TRUE RGB Colors across all scanners.")
        else:
            eng.set_color_mode(CSlamColorMode.SCANNER_COLORS)
            self.btn_cslam_color_mode.setText("🎨 Scanner Colors")
            self.btn_cslam_color_mode.setStyleSheet(
                "background: rgba(186, 161, 100, 0.18); color: #E5C983; border: 1px solid #BAA164; font-size: 9px; font-weight: 600; padding: 3px 8px; border-radius: 6px;"
            )
            self.console.append("[C-SLAM] Switched display to DISTINCT SCANNER COLORS (each scanner uniquely colored).")
        self.ros_worker.dispatch_live_map_and_telemetry(force=True)

    def on_add_peer_ip_clicked(self):
        ip = self.input_peer_ip.text().strip()
        if not ip:
            return
        self.ros_worker.cslam_engine.add_unicast_peer(ip)
        self.console.append(f"[C-SLAM] Added Unicast Peer Scanner endpoint: {ip}:19440")
        self.input_peer_ip.clear()

    def on_spawn_sim_peer_clicked(self):
        self.ros_worker.ensure_local_cloud_for_interactive_ops()
        self.ros_worker.dispatch_live_map_and_telemetry(force=True)
        sim_id = self.ros_worker.cslam_engine.spawn_simulated_peer_scanner()
        if self.map_canvas.view_radius_m < 22.0:
            self.map_canvas.set_view_radius(25.0)
        self.console.append(
            f"[C-SLAM] Joined Peer Scanner ({sim_id[-8:]}): Initially SEPARATED at {self.spin_cslam_sep.value():.1f}m "
            f"(oriented via relative IMU); will automatically SNAP into global map once overlapping geometry is scanned!"
        )

    def on_force_snap_clicked(self):
        self.ros_worker.cslam_engine.run_overlap_check_now()
        self.ros_worker.dispatch_live_map_and_telemetry(force=True)
        self.console.append("[C-SLAM] Triggered immediate multi-peer 3D KD-Tree ICP overlap check.")

    def on_clear_peers_clicked(self):
        self.ros_worker.cslam_engine.clear_peers()
        self.ros_worker.dispatch_live_map_and_telemetry(force=True)
        self.console.append("[C-SLAM] Cleared remote/simulated peer scanners.")

    def on_sliders_changed(self):
        min_d = self.slider_min_d.value() * 0.01
        max_d = self.slider_max_d.value() * 0.10
        agree_tol = self.slider_agree.value() * 0.001
        self.lbl_min_d.setText(f"{min_d:.2f} m")
        self.lbl_max_d.setText(f"{max_d:.2f} m")
        self.lbl_agree.setText(f"{agree_tol*1000:.0f} mm")
        self.ros_worker.set_gating_parameters(min_d, max_d, agree_tol, 0.50)

    def apply_gating_preset(self, min_d, max_d, agree_tol, conf):
        self.slider_min_d.setValue(int(min_d * 100))
        self.slider_max_d.setValue(int(max_d * 10))
        self.slider_agree.setValue(int(agree_tol * 1000))
        self.on_sliders_changed()
        self.console.append(f"[PRESET] Applied Contribution Gate: {min_d:.1f}m - {max_d:.1f}m (Agreement: {agree_tol*1000:.0f}mm)")

    def on_layer_mode_clicked(self, mode_key):
        self._style_layer_buttons(mode_key)
        self.map_canvas.set_active_mode_immediate(mode_key)
        self.ros_worker.set_display_mode(mode_key)
        self.ros_worker.dispatch_live_map_and_telemetry(force=True)
        self.console.append(f"[VIEW] Switched Point Cloud Display Layer -> {mode_key}")

    def _set_stack_buttons_online(self, online: bool):
        if online:
            txt = "⏹ STOP STACK (ONLINE)"
            style = (
                "QPushButton { background: rgba(239, 68, 68, 0.20); border: 1px solid #F87171; "
                "color: #FCA5A5; padding: 6px 14px; border-radius: 9px; font-weight: 600; font-size: 10px; }"
                "QPushButton:hover { background: rgba(239, 68, 68, 0.34); color: #FFFFFF; }"
            )
        else:
            txt = "🚀 START STACK"
            style = (
                "QPushButton { background: rgba(255, 255, 255, 0.07); border: 1px solid rgba(186, 161, 100, 0.60); "
                "color: #E5C983; padding: 6px 14px; border-radius: 9px; font-weight: 600; font-size: 10px; }"
                "QPushButton:hover { background: rgba(186, 161, 100, 0.24); border-color: #FCFBF8; color: #FFFFFF; }"
            )
        for btn in [getattr(self, "stack_btn", None), getattr(self, "hub_stack_btn", None)]:
            if btn is not None:
                btn.setText(txt)
                btn.setStyleSheet(style)

    def toggle_ros_stack(self):
        stack_running = (
            self._external_stack_online
            or (self.sensor_stack_proc is not None and self.sensor_stack_proc.poll() is None)
            or (self.slam_stack_proc is not None and self.slam_stack_proc.poll() is None)
        )
        if stack_running:
            self.stop_ros_stack()
        else:
            self.start_ros_stack()

    def _cleanup_all_ros_stack_processes(self):
        """Ensures no duplicate sensor_bringup, component_container, rslidar_sdk_node, or fastlivo_mapping processes exist."""
        patterns = [
            "[m]apping_robosense.launch.py",
            "[s]ensor_bringup.launch.py",
            "[f]astlivo_mapping",
            "[r]slidar_sdk_node",
            "[c]omponent_container",
            "[c]amera_container",
            "[s]canar_topic_relay",
            "[s]tatic_transform_publisher",
        ]
        for pat in patterns:
            try:
                subprocess.run(["pkill", "-9", "-f", pat], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except Exception:
                pass

    def start_ros_stack(self):
        ws = "/home/scanar/scanarMini"
        self.console.append("[STACK] Cleaning up any prior nodes, verifying Airy SYNC_OUT1, and launching single-instance ROS 2 Stack...")
        env = os.environ.copy()
        env["ROS_DOMAIN_ID"] = "0"
        env["RMW_IMPLEMENTATION"] = "rmw_fastrtps_cpp"
        setup_cmd = (
            f"source /opt/ros/humble/setup.bash && "
            f"source {ws}/install/setup.bash && "
            f"export ROS_DOMAIN_ID=0 && "
            f"export RMW_IMPLEMENTATION=rmw_fastrtps_cpp"
        )
        self._cleanup_all_ros_stack_processes()
        time.sleep(0.5)
        try:
            subprocess.Popen(
                ["python3", f"{ws}/scripts/configure_airy_sync_out.py"],
                cwd=ws, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
        except Exception:
            pass

        try:
            self.sensor_stack_proc = subprocess.Popen(
                ["bash", "-c", f"{setup_cmd} && exec ros2 launch scan_routine sensor_bringup.launch.py >/tmp/scanhub_sensor_bringup.log 2>&1"],
                cwd=ws, env=env, preexec_fn=os.setsid,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            self.slam_stack_proc = subprocess.Popen(
                ["bash", "-c", f"{setup_cmd} && sleep 2 && exec ros2 launch fast_livo mapping_robosense.launch.py use_rviz:=false >/tmp/scanhub_fast_livo.log 2>&1"],
                cwd=ws, env=env, preexec_fn=os.setsid,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            self._external_stack_online = True
            self._set_stack_buttons_online(True)
            self.console.append("[STACK] Single-instance sensor_bringup.launch.py + mapping_robosense.launch.py (use_rviz:=false) ONLINE.")
        except Exception as e:
            self.console.append(f"[STACK ERROR] Failed to start ROS 2 stack: {e}")

    def stop_ros_stack(self):
        import signal
        for name, proc in [("SLAM", self.slam_stack_proc), ("Sensors", self.sensor_stack_proc)]:
            if proc is not None and proc.poll() is None:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                except Exception:
                    pass
        self._cleanup_all_ros_stack_processes()
        self.slam_stack_proc = None
        self.sensor_stack_proc = None
        self._external_stack_online = False
        self._set_stack_buttons_online(False)
        self.console.append("[STACK] Stopped ROS 2 Sensor & SLAM stack.")

    def update_telemetry_ui(self, t: dict):
        rgb_cov = t.get("rgb_cov_pct", 0.0)
        dual_cov = t.get("dual_cov_pct", 0.0)
        self.bar_rgb.setValue(int(rgb_cov))
        self.lbl_cov_rgb.setText(f"RGB Scene Coverage: {rgb_cov:.1f}% of mapped geometry")
        self.lbl_cov_dual.setText(f"Dual Coverage (RGB + Depth): {dual_cov:.1f}%")

        n_acc = t.get("n_accepted", 0)
        n_rej = t.get("n_rejected", 0)
        acc_pct = t.get("stereo_accept_pct", 0.0)
        self.lbl_stereo_stats.setText(f"Gemini Points: Accepted: {n_acc:,} ({acc_pct:.1f}%) | Rejected: {n_rej:,} ({100.0-acc_pct:.1f}%)")

        f_pct = t.get("floor_pct", 0.0)
        w_pct = t.get("wall_pct", 0.0)
        c_pct = t.get("ceil_pct", 0.0)
        self.lbl_elevation_stats.setText(f"LiDAR Elevation: Floor: {f_pct:.1f}% | Wall Band: {w_pct:.1f}% | Ceiling: {c_pct:.1f}%")

        # Update M12 Live Cleanup Readout
        if not t.get("m12_preview_active", False) and time.time() >= getattr(self, "_m12_status_hold_until", 0.0):
            m12_on = t.get("m12_live_enabled", False)
            in_p = t.get("m12_input_pts", 0)
            out_p = t.get("m12_output_pts", 0)
            ms = t.get("m12_time_ms", 0.0)
            tilt = t.get("m12_tilt_deg", 0.0)
            if m12_on and in_p > 0:
                self.lbl_m12_status.setText(
                    f"M12 RT: ON | {in_p:,} -> {out_p:,} pts | Tilt: {tilt:.1f}° | {ms:.1f}ms"
                )
            elif not m12_on:
                self.lbl_m12_status.setText(
                    "M12 RT Filter: OFF (Post-Process Ready — Preview / Commit / Export)"
                )

        # Update C-SLAM Unlimited Multi-Scanner Status
        cslam = t.get("cslam", {})
        pill_base = (
            "background: rgba(255, 255, 255, 0.06); "
            "border: 1px solid rgba(255, 255, 255, 0.20); "
            "padding: 3px 8px; border-radius: 10px; font-weight: 600; font-size: 9px;"
        )
        if cslam:
            tot_scanners = cslam.get("total_scanners", 1)
            snapped_n = cslam.get("snapped_peers", 0)
            sep_n = cslam.get("separated_peers", 0)
            peers_list = cslam.get("peers", [])

            self.dot_cslam.setText(
                f"● C-SLAM: {tot_scanners} Scanner{'s' if tot_scanners != 1 else ''}"
                + (f" ({snapped_n} Snapped)" if snapped_n > 0 else "")
            )
            self.dot_cslam.setStyleSheet(
                f"color: {'#4ADE80' if snapped_n > 0 else '#E5C983'}; "
                f"background: {'rgba(34, 197, 94, 0.14)' if snapped_n > 0 else 'rgba(186, 161, 100, 0.14)'}; "
                f"border: 1px solid {'rgba(74, 222, 128, 0.45)' if snapped_n > 0 else 'rgba(186, 161, 100, 0.45)'}; "
                f"padding: 3px 8px; border-radius: 10px; font-weight: 600; font-size: 9px;"
            )
            self.lbl_cslam_summary.setText(
                f"Project: {cslam.get('project_name', '')} / {cslam.get('dataset_name', '')} | "
                f"Scanners: {tot_scanners} | Snapped: {snapped_n} | IMU+Offset: {sep_n}"
            )

            html_lines = [
                f"<span style='color:#E5C983; font-weight:bold;'>"
                f"● #0 {cslam.get('local_scanner_name', 'Local')} [HOST ANCHOR] ({cslam.get('local_points', 0):,} pts)</span>"
            ]
            for p in peers_list:
                c_hex = p.get("color_hex", "#ff5e3a")
                st = p.get("alignment_state", "SEPARATED_IMU_ORIENTED")
                rpy = p.get("imu_rpy_deg", (0.0, 0.0, 0.0))
                if st == "OVERLAP_SNAPPED":
                    badge = f"<b style='color:#4ADE80;'>[SNAPPED {p.get('overlap_ratio', 0.0)*100:.0f}% | RMSE {p.get('icp_rmse_m', 0.0)*100:.1f}cm]</b>"
                else:
                    badge = f"<span style='color:#E5C983;'>[IMU+OFFSET {self.spin_cslam_sep.value():.0f}m | YAW {rpy[2]:+.0f}°]</span>"
                html_lines.append(
                    f"<span style='color:{c_hex}; font-weight:bold;'>● #{p.get('color_index', 1)} {p.get('scanner_name', 'Peer')}</span> "
                    f"<span style='color:#FCFBF8;'>({p.get('point_count', 0):,} pts)</span> {badge}"
                )
            self.cslam_peer_list.setHtml("<br/>".join(html_lines))

        # Update Live Sensor Health & Hz Indicators
        rslidar_hz = t.get("rslidar_hz", 0.0)
        lidar_hz = t.get("lidar_hz", 0.0)
        active_lidar_hz = max(rslidar_hz, lidar_hz)
        lidar_age = min(t.get("rslidar_age", 999.0), t.get("lidar_age", 999.0))
        if lidar_age > 3.0 or active_lidar_hz < 0.2:
            self.dot_lidar.setText("● LiDAR: OFFLINE")
            self.dot_lidar.setStyleSheet(f"color: #F87171; {pill_base}")
        elif active_lidar_hz < 5.0:
            self.dot_lidar.setText(f"● LiDAR: LOW ({active_lidar_hz:.1f}Hz)")
            self.dot_lidar.setStyleSheet(f"color: #FBBF24; {pill_base}")
        else:
            self.dot_lidar.setText(f"● LiDAR: OK ({active_lidar_hz:.1f}Hz)")
            self.dot_lidar.setStyleSheet(f"color: #4ADE80; {pill_base}")

        cam_hz = t.get("camera_hz", 0.0)
        depth_hz = t.get("depth_hz", 0.0)
        active_cam_hz = max(cam_hz, depth_hz)
        cam_age = min(t.get("camera_age", 999.0), t.get("depth_age", 999.0))
        if cam_age > 3.0 or active_cam_hz < 0.2:
            self.dot_camera.setText("● Camera: OFFLINE")
            self.dot_camera.setStyleSheet(f"color: #F87171; {pill_base}")
        elif active_cam_hz < 2.5:
            self.dot_camera.setText(f"● Camera: LOW ({active_cam_hz:.1f}Hz)")
            self.dot_camera.setStyleSheet(f"color: #FBBF24; {pill_base}")
        else:
            self.dot_camera.setText(f"● Camera: OK ({active_cam_hz:.1f}Hz)")
            self.dot_camera.setStyleSheet(f"color: #4ADE80; {pill_base}")

        slam_hz = t.get("slam_hz", 0.0)
        slam_age = t.get("slam_age", 999.0)
        if slam_age > 3.0 or slam_hz < 0.2:
            self.dot_slam.setText("● SLAM: OFFLINE")
            self.dot_slam.setStyleSheet(f"color: #F87171; {pill_base}")
        elif slam_hz < 3.0:
            self.dot_slam.setText(f"● SLAM: LOW ({slam_hz:.1f}Hz)")
            self.dot_slam.setStyleSheet(f"color: #FBBF24; {pill_base}")
        else:
            self.dot_slam.setText(f"● SLAM: OK ({slam_hz:.1f}Hz)")
            self.dot_slam.setStyleSheet(f"color: #4ADE80; {pill_base}")

        # Auto-sync START/STOP STACK button if ROS stack is actively publishing
        if (active_lidar_hz > 1.0 or active_cam_hz > 1.0 or slam_hz > 1.0) and not self._external_stack_online:
            self._external_stack_online = True
            self._set_stack_buttons_online(True)

        ang_spd = t.get("ang_speed", 0.0)
        lin_spd = t.get("lin_speed", 0.0)
        deg_s = math.degrees(ang_spd)
        if ang_spd > 0.85:
            self.dot_pace.setText(f"⚠ SLOW TURN ({deg_s:.0f}°/s)")
            self.dot_pace.setStyleSheet(f"color: #F87171; {pill_base}")
        elif lin_spd > 1.2:
            self.dot_pace.setText(f"⚠ WALK SLOWER ({lin_spd:.2f}m/s)")
            self.dot_pace.setStyleSheet(f"color: #FBBF24; {pill_base}")
        else:
            self.dot_pace.setText(f"● PACE: OPTIMAL ({lin_spd:.2f}m/s)")
            self.dot_pace.setStyleSheet(f"color: #4ADE80; {pill_base}")

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key.Key_F, ord('F'), ord('f')):
            if self.is_recording:
                self.trigger_faro_flash_anchor()
                event.accept()
                return
        super().keyPressEvent(event)

    def toggle_scan_workflow(self):
        dataset = self.dataset_input.text().strip().replace(" ", "_")
        if not dataset:
            dataset = datetime.now().strftime("Dataset_%m%d_%H%M")
            self.dataset_input.setText(dataset)
        self.active_dataset_name = dataset
        self.ros_worker.cslam_engine.set_project_name(self.active_project_name, dataset)

        self.console.append(f"\n[START] Starting Capture Workflow -> Project: {self.active_project_name} | Dataset: {dataset}")
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.save_btn.setEnabled(False)
        self.flash_btn.setEnabled(False)
        self.map_canvas.clear_map()
        self.anchor_count = 0

        if not self.is_calibrated:
            self.run_stage_1_bias_calibration()
        else:
            self.start_rotation_ritual()

    def run_stage_1_bias_calibration(self):
        self.console.append("[INIT] NavVis Ritual Stage 1: Static ZUPT Bias Calibration (Hold scanner motionless)...")
        self.calib_ticks = 0
        self.bar.setValue(0)
        self.btn_skip_ritual.setVisible(True)
        self.progress_label.setText("STAGE 1 · Static Bias Calibration: Keep scanner motionless (3s)...")
        self.progress_label.setStyleSheet("color: #E5C983; font-weight: 600; background: transparent;")
        self.status_bar.showMessage("CALIBRATING: Static bias calibration in progress...")

        self.calib_timer = QTimer()
        self.calib_timer.timeout.connect(self._sample_bias_tick)
        self.calib_timer.start(30)  # 100 ticks * 30ms = 3.0 seconds

    def _sample_bias_tick(self):
        self.calib_ticks += 1
        progress = min(100, int((self.calib_ticks / 100.0) * 100))
        self.bar.setValue(progress)

        if self.calib_ticks == 1:
            self.calib_ax_sum = 0.0
            self.calib_ay_sum = 0.0
            self.calib_az_sum = 0.0

        self.calib_ax_sum += self.ros_worker.last_ax
        self.calib_ay_sum += self.ros_worker.last_ay
        self.calib_az_sum += self.ros_worker.last_az

        if self.calib_ticks >= 100:
            if self.calib_timer is not None:
                self.calib_timer.stop()
                self.calib_timer = None
            self.is_calibrated = True

            ax_avg = self.calib_ax_sum / 100.0
            ay_avg = self.calib_ay_sum / 100.0
            az_avg = self.calib_az_sum / 100.0

            roll_est = math.atan2(ay_avg, az_avg)
            pitch_est = math.atan2(-ax_avg, math.hypot(ay_avg, az_avg))
            roll_est_deg = math.degrees(roll_est)
            pitch_est_deg = math.degrees(pitch_est)

            self.console.append(
                f"[SUCCESS] Stage 1 Complete: Gravity aligned! Avg accel: X={ax_avg:+.2f} Y={ay_avg:+.2f} Z={az_avg:+.2f} m/s² "
                f"(Roll={roll_est_deg:+.1f}°, Pitch={pitch_est_deg:+.1f}°)"
            )
            self.map_canvas.gravity_roll_deg = roll_est_deg
            self.map_canvas.gravity_pitch_deg = pitch_est_deg
            self.start_rotation_ritual()

    def start_rotation_ritual(self):
        self.ros_worker.accumulated_rotation_deg = 0.0
        self.ros_worker.last_imu_gyro_time = time.time()
        self.ros_worker.in_rotation_ritual = True

        self.bar.setValue(0)
        self.btn_skip_ritual.setVisible(True)
        self.console.append("[INIT] NavVis Ritual Stage 2: Pivot scanner 90° horizontally to synchronize SLAM attitude...")
        self.progress_label.setText("🔄 NAVVIS RITUAL: Slowly pivot scanner 90° horizontally to sync SLAM attitude...")
        self.progress_label.setStyleSheet("color: #E5C983; font-weight: 600; background: transparent;")
        self.status_bar.showMessage("NAVVIS RITUAL: Pivot scanner 90° to lock SLAM attitude...")

    def update_rotation_ritual_ui(self, deg: float):
        progress = min(100, int((deg / 90.0) * 100))
        self.bar.setValue(progress)
        self.progress_label.setText(f"🔄 NAVVIS RITUAL: Pivot scanner 90°... ({deg:.0f}° / 90°)")
        self.progress_label.setStyleSheet("color: #E5C983; font-weight: 600; background: transparent;")

    def skip_rotation_ritual(self):
        if self.calib_timer is not None:
            self.calib_timer.stop()
            self.calib_timer = None
        self.is_calibrated = True
        self.ros_worker.in_rotation_ritual = False
        self.console.append("[INFO] NavVis alignment ritual skipped by user.")
        self.on_rotation_ritual_completed()

    def on_rotation_ritual_completed(self):
        self.bar.setValue(100)
        self.btn_skip_ritual.setVisible(False)
        self.console.append("[SUCCESS] NavVis 90° Pivot Ritual Locked! SLAM attitude synchronized.")
        self.progress_label.setText("🟢 90° PIVOT LOCKED! Reality capture recording active...")
        self.progress_label.setStyleSheet("color: #4ADE80; font-weight: 600; background: transparent;")
        self.execute_active_capture_pipelines()

    def execute_active_capture_pipelines(self):
        self.console.append(f"[CAPTURE] Session active: {self.active_dataset_name}")
        self.ros_worker.start_recording_session()
        self.is_recording = True
        self.scan_elapsed_seconds = 0
        self.scan_clock_timer.start(1000)

        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.save_btn.setEnabled(False)
        self.flash_btn.setEnabled(True)
        self.progress_label.setText("SCANNING IN PROGRESS  ·  Move smoothly to paint scene geometry & truecolor coverage")
        self.progress_label.setStyleSheet("color: #4ADE80; font-weight: 600; background: transparent;")
        self.status_bar.showMessage(f"SCANNING IN PROGRESS: {self.active_project_name} / {self.active_dataset_name}")

    def trigger_faro_flash_anchor(self):
        if not self.is_recording:
            return
        label = f"w{self.anchor_count:02d}"
        self.anchor_count += 1
        timestamp = datetime.now().strftime("%H:%M:%S")

        cx = self.map_canvas.current_x
        cy = self.map_canvas.current_y
        cz = getattr(self.ros_worker, "current_z", 0.0)

        self.map_canvas.add_flash_anchor(label, cx, cy, cz)
        self.console.append(f"[FARO FLASH SCAN] ⚡ Anchor {label} Captured at X={cx:+.2f}m Y={cy:+.2f}m Z={cz:+.2f}m ({timestamp})")
        self.status_bar.showMessage(f"⚡ FLASH ANCHOR {label} RECORDED at ({cx:+.2f}, {cy:+.2f})", 4000)

    def update_scan_clock_tick(self):
        self.scan_elapsed_seconds += 1
        m = self.scan_elapsed_seconds // 60
        s = self.scan_elapsed_seconds % 60
        self.clock_lbl.setText(f"⏱ {m:02d}:{s:02d}")

    def stop_scan_session(self):
        if self.calib_timer is not None:
            self.calib_timer.stop()
            self.calib_timer = None
        self.ros_worker.in_rotation_ritual = False
        self.btn_skip_ritual.setVisible(False)

        self.console.append("[STOP] Stopping active scan session...")
        self.ros_worker.stop_recording_session()
        self.is_recording = False
        self.scan_clock_timer.stop()

        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.save_btn.setEnabled(True)
        self.flash_btn.setEnabled(False)
        self.progress_label.setText("SESSION STOPPED  ·  Ready to SAVE qualified production dataset")
        self.progress_label.setStyleSheet("color: #E5C983; font-weight: 600; background: transparent;")
        self.status_bar.showMessage("SCAN STOPPED: Ready to SAVE production dataset.")

    def save_dataset(self):
        if self.export_in_progress:
            return
        dataset = self.dataset_input.text().strip().replace(" ", "_") or self.active_dataset_name
        self.active_dataset_name = dataset
        save_dir = self._get_active_dataset_save_dir(dataset)
        legacy_dir = os.path.join(self.legacy_data_root, dataset)
        self.console.append(
            f"\n[SAVE] Exporting production dataset in background to Project '{self.active_project_name}':\n  {save_dir}"
        )
        self.export_in_progress = True
        self.save_btn.setEnabled(False)
        self.progress_label.setText(f"EXPORTING DATASET IN BACKGROUND: {save_dir} ...")
        self.progress_label.setStyleSheet("color: #E5C983; font-weight: 600; background: transparent;")
        self.status_bar.showMessage(f"EXPORTING DATASET: {save_dir}")

        traj_snap = list(self.map_canvas.trajectory_path)
        flash_snap = list(self.map_canvas.flash_anchors)

        def _bg_export():
            try:
                meta = self.ros_worker.export_production_dataset(
                    save_dir=save_dir,
                    dataset_name=dataset,
                    trajectory=traj_snap,
                    flash_anchors=flash_snap,
                )
                # Maintain a symlink in /home/scanar/scanarMini/data/<dataset> if saved under projects/
                try:
                    os.makedirs(self.legacy_data_root, exist_ok=True)
                    if os.path.abspath(save_dir) != os.path.abspath(legacy_dir) and not os.path.exists(legacy_dir):
                        os.symlink(save_dir, legacy_dir)
                except Exception:
                    pass
                self.export_finished.emit(meta, save_dir, "")
            except Exception as e:
                self.export_finished.emit({}, save_dir, str(e))

        threading.Thread(target=_bg_export, daemon=True).start()

    def on_export_finished(self, meta: dict, save_dir: str, err_msg: str):
        self.export_in_progress = False
        self.save_btn.setEnabled(True)
        if err_msg:
            self.console.append(f"[ERROR] Failed to save dataset: {err_msg}")
            self.progress_label.setText(f"EXPORT FAILED: {err_msg}")
            self.progress_label.setStyleSheet("color: #F87171; font-weight: 600; background: transparent;")
            return

        n_a = meta.get("airy_points", 0)
        n_s = meta.get("stereo_points", 0)
        n_f = meta.get("fused_points", 0)
        n_c = meta.get("cslam_merged_points", 0)
        n_p = meta.get("cslam_peers", 0)

        self.console.append("[SUCCESS] Export Complete:")
        self.console.append(f"  - Project: {self.active_project_name}  |  Dataset: {self.active_dataset_name}")
        self.console.append(f"  - pointcloud_airy.ply / .pcd: {n_a:,} pts (Pure LiDAR Geometry)")
        self.console.append(f"  - pointcloud_stereo.ply / .pcd: {n_s:,} pts (Colorized Stereo)")
        self.console.append(f"  - pointcloud_fused_rgb.ply / .pcd: {n_f:,} pts (Truecolor Fused)")
        if n_p > 0:
            self.console.append(f"  - pointcloud_cslam_merged.ply: {n_c:,} pts ({1 + n_p} Scanners Merged)")
        self.console.append("  - trajectory.csv / trajectory_tum.txt")
        self.console.append("  - scan_metadata.json")
        self.progress_label.setText(f"DATASET SAVED: {save_dir}")
        self.progress_label.setStyleSheet("color: #4ADE80; font-weight: 600; background: transparent;")
        self.status_bar.showMessage(f"DATASET SAVED: {save_dir}")
        self.refresh_projects_hub_lists()

    def closeEvent(self, event):
        self.hub_refresh_timer.stop()
        if self.sensor_stack_proc is not None or self.slam_stack_proc is not None:
            self.stop_ros_stack()
        self.ros_worker.stop()
        event.accept()


def main():
    app = QApplication(sys.argv)
    logo_path = get_av_logo_path()
    if logo_path:
        app.setWindowIcon(QIcon(logo_path))

    # Launch Artificial Vision Translucent Apple Glass White & Gold Splash Screen
    splash = ArtificialVisionSplashScreen()
    splash.set_status("Initializing Artificial Vision Spatial Engine...", 25)
    splash.show()
    app.processEvents()

    splash.set_status("Starting M12 Cleanup & Unlimited C-SLAM Mesh...", 65)
    app.processEvents()

    gui = ScanHUBOS()
    splash.set_status("Artificial Vision  ·  scanHUB Apple Glass AR OS Ready", 100)
    app.processEvents()

    def _reveal_main():
        gui.show()
        splash.close()

    QTimer.singleShot(1400, _reveal_main)
    sys.exit(app.exec_() if hasattr(app, "exec_") else app.exec())


if __name__ == "__main__":
    main()



