#!/usr/bin/env python3
"""
Milestone 12: Point Cloud Cleanup, Post-Processing, Real-Time Spatial-Hash Filter & Multi-Format Export.
Ported from av_scan (core/processing/point_cloud_processor.hpp & .cpp) and vectorized for ScanHUB OS.
"""

import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Tuple, Optional, List
import numpy as np
from scipy.spatial import cKDTree


class ExportFormat(Enum):
    PLY_ASCII = "PLY_ASCII"
    PLY_BINARY = "PLY_BINARY"
    PCD_ASCII = "PCD_ASCII"
    PCD_BINARY = "PCD_BINARY"
    XYZ_ASCII = "XYZ_ASCII"
    OBJ_MESH = "OBJ_MESH"
    E57_STRUCTURED = "E57_STRUCTURED"


@dataclass
class CleanupParams:
    # Voxel Grid Downsampling
    enable_voxel_downsample: bool = True
    voxel_leaf_size_m: float = 0.02

    # Statistical Outlier Removal (SOR)
    enable_sor: bool = True
    sor_mean_k: int = 20
    sor_std_mul: float = 2.0

    # Radius Outlier Removal (ROR)
    enable_ror: bool = True
    ror_radius_m: float = 0.10
    ror_min_neighbors: int = 4

    # Ground Plane Detection & Z=0 Alignment
    enable_ground_alignment: bool = True
    ground_dist_threshold_m: float = 0.04
    ground_max_iterations: int = 400
    ground_max_tilt_deg: float = 35.0
    align_ground_to_z_zero: bool = True
    remove_ground_plane: bool = False


@dataclass
class GroundPlaneResult:
    found: bool = False
    plane_coeffs: List[float] = field(default_factory=lambda: [0.0, 0.0, 1.0, 0.0])
    inlier_count: int = 0
    inlier_ratio: float = 0.0
    tilt_deg: float = 0.0
    floor_height_m: float = 0.0
    alignment_transform: np.ndarray = field(default_factory=lambda: np.eye(4, dtype=np.float64))

    @property
    def detected(self) -> bool:
        return self.found

    @property
    def tilt_angle_deg(self) -> float:
        return self.tilt_deg


@dataclass
class ProcessingStats:
    input_point_count: int = 0
    output_point_count: int = 0
    voxel_removed_count: int = 0
    sor_removed_count: int = 0
    ror_removed_count: int = 0
    ground_removed_count: int = 0
    computation_time_ms: float = 0.0
    bbox_min: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    bbox_max: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    ground_plane: GroundPlaneResult = field(default_factory=GroundPlaneResult)

    @property
    def total_removed_count(self) -> int:
        return max(0, self.input_point_count - self.output_point_count)

    @property
    def input_points(self) -> int:
        return self.input_point_count

    @property
    def output_points(self) -> int:
        return self.output_point_count

    @property
    def voxel_downsampled_points(self) -> int:
        return max(0, self.input_point_count - self.voxel_removed_count)

    @property
    def sor_removed_points(self) -> int:
        return self.sor_removed_count

    @property
    def ror_removed_points(self) -> int:
        return self.ror_removed_count

    @property
    def processing_time_ms(self) -> float:
        return self.computation_time_ms

    @property
    def ground_result(self) -> GroundPlaneResult:
        return self.ground_plane


@dataclass
class ExportResult:
    success: bool = False
    file_path: str = ""
    format_name: str = ""
    exported_points: int = 0
    exported_triangles: int = 0
    file_size_bytes: int = 0
    export_time_ms: float = 0.0
    error_message: str = ""


class PointCloudProcessor:
    """
    Milestone 12 Point Cloud Cleanup, Real-Time Fast Filter, and Multi-Format Exporter.
    Accepts NumPy arrays of shape (N, 3), (N, 6) [x,y,z,r,g,b], or (N, 7) [x,y,z,r,g,b,intensity].
    """

    @staticmethod
    def _filter_finite(points: np.ndarray) -> np.ndarray:
        if points is None or len(points) == 0:
            return np.zeros((0, 6), dtype=np.float32)
        pts = np.asarray(points, dtype=np.float32)
        if pts.ndim != 2 or pts.shape[1] < 3:
            return np.zeros((0, 6), dtype=np.float32)
        mask = np.all(np.isfinite(pts[:, :3]), axis=1)
        return pts[mask]

    @staticmethod
    def compute_bounding_box(points: np.ndarray) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        pts = PointCloudProcessor._filter_finite(points)
        if len(pts) == 0:
            return (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)
        bmin = np.min(pts[:, :3], axis=0)
        bmax = np.max(pts[:, :3], axis=0)
        return (float(bmin[0]), float(bmin[1]), float(bmin[2])), (float(bmax[0]), float(bmax[1]), float(bmax[2]))

    @staticmethod
    def voxel_downsample(points: np.ndarray, leaf_size_m: float = 0.02) -> np.ndarray:
        """
        3D Voxel Grid Downsampling preserving true RGB and intensity centroid averages.
        Matches av_scan PointCloudProcessor::voxelDownsample.
        """
        pts = PointCloudProcessor._filter_finite(points)
        if len(pts) == 0 or leaf_size_m <= 1e-4:
            return pts.copy()

        inv_leaf = 1.0 / float(leaf_size_m)
        coords = np.floor(pts[:, :3].astype(np.float64) * inv_leaf).astype(np.int64)

        # Shift to non-negative relative coordinates for fast 1D/lexicographical hashing
        mins = coords.min(axis=0)
        rel = coords - mins
        maxs = rel.max(axis=0) + 1

        # Check if 64-bit integer packing fits without overflow
        span_prod = float(maxs[0]) * float(maxs[1]) * float(maxs[2])
        if span_prod < 9e17:
            keys = rel[:, 0] * (maxs[1] * maxs[2]) + rel[:, 1] * maxs[2] + rel[:, 2]
            _, inv, counts = np.unique(keys, return_inverse=True, return_counts=True)
        else:
            _, inv, counts = np.unique(coords, axis=0, return_inverse=True, return_counts=True)

        num_voxels = len(counts)
        cols = pts.shape[1]
        out = np.zeros((num_voxels, cols), dtype=np.float32)
        counts_f = counts.astype(np.float64)

        for c in range(cols):
            summed = np.bincount(inv, weights=pts[:, c].astype(np.float64), minlength=num_voxels)
            out[:, c] = (summed / counts_f).astype(np.float32)

        return out

    @staticmethod
    def remove_statistical_outliers(
        points: np.ndarray,
        mean_k: int = 20,
        std_dev_mul_thresh: float = 2.0,
    ) -> np.ndarray:
        """
        Statistical Outlier Removal (SOR) via 3D KD-Tree k-NN distance distribution.
        Matches av_scan PointCloudProcessor::removeStatisticalOutliers.
        """
        pts = PointCloudProcessor._filter_finite(points)
        n_pts = len(pts)
        if n_pts <= 4 or mean_k < 2:
            return pts.copy()

        k_query = min(int(mean_k) + 1, n_pts)
        if k_query <= 1:
            return pts.copy()

        tree = cKDTree(pts[:, :3])
        n_workers = -1 if n_pts >= 25000 else 1
        dists, _ = tree.query(pts[:, :3], k=k_query, workers=n_workers)
        # Exclude self-distance at index 0
        if dists.ndim == 1:
            mean_dists = dists
        else:
            mean_dists = np.mean(dists[:, 1:], axis=1)

        global_mean = float(np.mean(mean_dists))
        global_std = float(np.std(mean_dists, ddof=1)) if n_pts > 1 else 0.0
        max_allowed = global_mean + float(std_dev_mul_thresh) * global_std

        keep_mask = mean_dists <= max_allowed
        return pts[keep_mask]

    @staticmethod
    def remove_radius_outliers(
        points: np.ndarray,
        radius_m: float = 0.10,
        min_neighbors: int = 4,
    ) -> np.ndarray:
        """
        Radius Outlier Removal (ROR) via 3D KD-Tree fixed-radius neighbor counting.
        Matches av_scan PointCloudProcessor::removeRadiusOutliers.
        """
        pts = PointCloudProcessor._filter_finite(points)
        n_pts = len(pts)
        if n_pts == 0 or radius_m <= 1e-4 or min_neighbors < 1:
            return pts.copy()

        tree = cKDTree(pts[:, :3])
        n_workers = -1 if n_pts >= 25000 else 1
        # return_length=True counts all points within radius_m including self (+1)
        counts = tree.query_ball_point(pts[:, :3], r=float(radius_m), return_length=True, workers=n_workers)
        neighbor_counts = np.asarray(counts, dtype=np.int32) - 1
        keep_mask = neighbor_counts >= int(min_neighbors)
        return pts[keep_mask]

    @staticmethod
    def _rotation_matrix_from_two_vectors(v_from: np.ndarray, v_to: np.ndarray) -> np.ndarray:
        """
        Computes 3x3 rotation matrix aligning unit vector v_from to unit vector v_to.
        Matches Eigen::Quaterniond::FromTwoVectors(best_normal, z_axis).
        """
        u = np.asarray(v_from, dtype=np.float64)
        v = np.asarray(v_to, dtype=np.float64)
        u_norm = np.linalg.norm(u)
        v_norm = np.linalg.norm(v)
        if u_norm < 1e-9 or v_norm < 1e-9:
            return np.eye(3, dtype=np.float64)
        u = u / u_norm
        v = v / v_norm

        dot = float(np.clip(np.dot(u, v), -1.0, 1.0))
        if dot > 1.0 - 1e-9:
            return np.eye(3, dtype=np.float64)
        if dot < -1.0 + 1e-9:
            # 180 deg rotation around an orthogonal axis
            ortho = np.array([1.0, 0.0, 0.0]) if abs(u[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
            axis = np.cross(u, ortho)
            axis /= np.linalg.norm(axis)
            return -np.eye(3, dtype=np.float64) + 2.0 * np.outer(axis, axis)

        cross = np.cross(u, v)
        vx = np.array([
            [0.0, -cross[2], cross[1]],
            [cross[2], 0.0, -cross[0]],
            [-cross[1], cross[0], 0.0],
        ], dtype=np.float64)
        rot = np.eye(3, dtype=np.float64) + vx + (vx @ vx) * (1.0 / (1.0 + dot))
        return rot

    @staticmethod
    def detect_and_align_ground_plane(
        points: np.ndarray,
        distance_threshold_m: float = 0.04,
        max_iterations: int = 400,
        max_tilt_deg: float = 35.0,
        align_to_z_zero: bool = True,
        remove_ground_inliers: bool = False,
    ) -> Tuple[np.ndarray, GroundPlaneResult]:
        """
        RANSAC Ground Plane Detection & Automatic Z=0 Floor Leveling.
        Matches av_scan PointCloudProcessor::detectAndAlignGroundPlane.
        """
        res = GroundPlaneResult()
        pts = PointCloudProcessor._filter_finite(points)
        n_pts = len(pts)
        if n_pts < 30:
            return pts.copy(), res

        xyz = pts[:, :3].astype(np.float64)
        z_vals = xyz[:, 2]
        # Focus candidate sampling on lower 60% of Z range to lock onto floor rather than ceiling
        z_cutoff = float(np.percentile(z_vals, 60.0))
        candidate_indices = np.flatnonzero(z_vals <= z_cutoff)
        if len(candidate_indices) < 10:
            candidate_indices = np.arange(n_pts)

        # Subsample evaluation cloud to <= 15,000 points for sub-5ms vectorized RANSAC
        max_eval = 15000
        step = max(1, len(candidate_indices) // max_eval)
        eval_indices = candidate_indices[::step]
        eval_xyz = xyz[eval_indices]

        min_cos_tilt = float(np.cos(np.radians(max_tilt_deg)))
        rng = np.random.default_rng(42)

        num_iters = max(10, int(max_iterations))
        idx_triplets = rng.integers(0, len(candidate_indices), size=(num_iters, 3))
        p1 = xyz[candidate_indices[idx_triplets[:, 0]]]
        p2 = xyz[candidate_indices[idx_triplets[:, 1]]]
        p3 = xyz[candidate_indices[idx_triplets[:, 2]]]

        normals = np.cross(p2 - p1, p3 - p1)
        norms = np.linalg.norm(normals, axis=1)
        valid = norms > 1e-6
        if not np.any(valid):
            return pts.copy(), res

        normals = normals[valid] / norms[valid, None]
        p1 = p1[valid]

        # Canonical +Z up
        flip = normals[:, 2] < 0.0
        normals[flip] = -normals[flip]

        # Tilt constraint relative to +Z up
        tilt_ok = normals[:, 2] >= min_cos_tilt
        if not np.any(tilt_ok):
            return pts.copy(), res

        normals = normals[tilt_ok]
        p1 = p1[tilt_ok]
        d_vals = -np.sum(normals * p1, axis=1)  # (M,)

        # Vectorized inlier evaluation: (M, K)
        dists_eval = np.abs(normals @ eval_xyz.T + d_vals[:, None])
        inlier_counts = np.sum(dists_eval <= float(distance_threshold_m), axis=1)
        best_idx = int(np.argmax(inlier_counts))
        best_inliers = int(inlier_counts[best_idx])

        if best_inliers < 10:
            return pts.copy(), res

        best_normal = normals[best_idx].copy()
        best_d = float(d_vals[best_idx])

        # Refine plane normal & offset using PCA / SVD on all inliers in candidate_indices
        cand_xyz = xyz[candidate_indices]
        cand_dists = np.abs(cand_xyz @ best_normal + best_d)
        refined_inliers = cand_xyz[cand_dists <= float(distance_threshold_m)]
        if len(refined_inliers) >= 10:
            centroid = np.mean(refined_inliers, axis=0)
            centered = refined_inliers - centroid
            cov = (centered.T @ centered) / float(len(refined_inliers))
            eigvals, eigvecs = np.linalg.eigh(cov)
            pca_n = eigvecs[:, 0]  # smallest eigenvalue
            if pca_n[2] < 0.0:
                pca_n = -pca_n
            if pca_n[2] >= min_cos_tilt:
                best_normal = pca_n / np.linalg.norm(pca_n)
                best_d = -float(np.dot(best_normal, centroid))

        # Full-cloud inlier evaluation
        full_dists = np.abs(xyz @ best_normal + best_d)
        is_inlier = full_dists <= float(distance_threshold_m)
        total_inliers = int(np.sum(is_inlier))

        res.found = True
        res.plane_coeffs = [float(best_normal[0]), float(best_normal[1]), float(best_normal[2]), float(best_d)]
        res.inlier_count = total_inliers
        res.inlier_ratio = float(total_inliers) / float(n_pts)
        res.tilt_deg = float(np.degrees(np.arccos(np.clip(best_normal[2], -1.0, 1.0))))
        res.floor_height_m = float(-best_d / max(1e-6, best_normal[2]))

        R_align = PointCloudProcessor._rotation_matrix_from_two_vectors(best_normal, np.array([0.0, 0.0, 1.0]))
        T_align = np.eye(4, dtype=np.float64)
        T_align[:3, :3] = R_align
        T_align[2, 3] = best_d
        res.alignment_transform = T_align

        out_pts = pts.copy()
        if remove_ground_inliers:
            out_pts = out_pts[~is_inlier]

        if align_to_z_zero and len(out_pts) > 0:
            aligned_xyz = (out_pts[:, :3].astype(np.float64) @ R_align.T)
            aligned_xyz[:, 2] += best_d
            out_pts[:, :3] = aligned_xyz.astype(np.float32)

        return out_pts, res

    @staticmethod
    def run_cleanup_pipeline(
        points: np.ndarray,
        params: Optional[CleanupParams] = None,
    ) -> Tuple[np.ndarray, ProcessingStats]:
        """
        Executes the full Milestone 12 cleanup pipeline:
        Voxel Downsample -> SOR -> ROR -> RANSAC Ground Plane Alignment.
        Matches av_scan PointCloudProcessor::runCleanupPipeline.
        """
        if params is None:
            params = CleanupParams()

        t0 = time.perf_counter()
        stats = ProcessingStats()
        working = PointCloudProcessor._filter_finite(points)
        stats.input_point_count = len(working)
        prev_count = len(working)

        if params.enable_voxel_downsample and params.voxel_leaf_size_m > 0.001:
            working = PointCloudProcessor.voxel_downsample(working, params.voxel_leaf_size_m)
            stats.voxel_removed_count = max(0, prev_count - len(working))
            prev_count = len(working)

        if params.enable_sor and params.sor_mean_k >= 2:
            working = PointCloudProcessor.remove_statistical_outliers(working, params.sor_mean_k, params.sor_std_mul)
            stats.sor_removed_count = max(0, prev_count - len(working))
            prev_count = len(working)

        if params.enable_ror and params.ror_radius_m > 0.001 and params.ror_min_neighbors >= 1:
            working = PointCloudProcessor.remove_radius_outliers(working, params.ror_radius_m, params.ror_min_neighbors)
            stats.ror_removed_count = max(0, prev_count - len(working))
            prev_count = len(working)

        if params.enable_ground_alignment and len(working) >= 30:
            working, gp_res = PointCloudProcessor.detect_and_align_ground_plane(
                working,
                distance_threshold_m=params.ground_dist_threshold_m,
                max_iterations=params.ground_max_iterations,
                max_tilt_deg=params.ground_max_tilt_deg,
                align_to_z_zero=params.align_ground_to_z_zero,
                remove_ground_inliers=params.remove_ground_plane,
            )
            stats.ground_plane = gp_res
            stats.ground_removed_count = max(0, prev_count - len(working))

        stats.output_point_count = len(working)
        stats.bbox_min, stats.bbox_max = PointCloudProcessor.compute_bounding_box(working)
        stats.computation_time_ms = (time.perf_counter() - t0) * 1000.0
        return working, stats

    @staticmethod
    def clean_realtime_fast(
        points: np.ndarray,
        params: Optional[CleanupParams] = None,
    ) -> Tuple[np.ndarray, ProcessingStats]:
        """
        Real-time fast O(N) spatial-hash M12 cleanup pipeline designed for live 10-20 Hz map updates.
        Performs voxel deduplication, single-pass bounded ROR + local SOR filtering, and optional
        fast ground-plane Z=0 lock without stalling live capture.
        Matches av_scan PointCloudProcessor::cleanRealtimeFast.
        """
        if params is None:
            params = CleanupParams(
                enable_voxel_downsample=True,
                voxel_leaf_size_m=0.015,
                enable_sor=True,
                sor_mean_k=12,
                sor_std_mul=2.0,
                enable_ror=True,
                ror_radius_m=0.12,
                ror_min_neighbors=3,
                enable_ground_alignment=False,
            )

        t0 = time.perf_counter()
        stats = ProcessingStats()
        working = PointCloudProcessor._filter_finite(points)
        stats.input_point_count = len(working)

        if len(working) == 0:
            return working, stats

        if params.enable_voxel_downsample and params.voxel_leaf_size_m > 0.001:
            working = PointCloudProcessor.voxel_downsample(working, params.voxel_leaf_size_m)
            stats.voxel_removed_count = max(0, stats.input_point_count - len(working))

        n_pts = len(working)
        if n_pts > 8 and (params.enable_ror or params.enable_sor):
            ror_r = float(params.ror_radius_m) if (params.enable_ror and params.ror_radius_m > 0.01) else 0.12
            cell_size = max(0.05, ror_r)
            sor_r = cell_size * 1.6
            target_k = max(4, min(int(params.sor_mean_k), 16))
            min_nb = max(1, int(params.ror_min_neighbors))
            isolated_penalty_dist = float(cell_size * 3.0)

            tree = cKDTree(working[:, :3])
            n_workers = -1 if n_pts >= 25000 else 1
            # Single bounded k-NN query with distance_upper_bound = sor_r evaluates both ROR and SOR simultaneously!
            dists, _ = tree.query(
                working[:, :3],
                k=max(target_k + 1, min_nb + 2),
                distance_upper_bound=sor_r,
                workers=n_workers,
            )
            neighbor_dists = dists[:, 1:]  # exclude self

            if params.enable_ror:
                ror_counts = np.sum(neighbor_dists <= ror_r, axis=1)
                ror_keep = ror_counts >= min_nb
            else:
                ror_keep = np.ones(n_pts, dtype=bool)

            if params.enable_sor:
                sor_neighbors = neighbor_dists[:, :target_k]
                valid_mask = np.isfinite(sor_neighbors)
                valid_counts = np.sum(valid_mask, axis=1)
                safe_dists = np.where(valid_mask, sor_neighbors, 0.0)
                sor_mean_dist = np.where(
                    valid_counts > 0,
                    np.sum(safe_dists, axis=1) / np.maximum(1, valid_counts),
                    isolated_penalty_dist,
                )
                valid_sor = sor_mean_dist[ror_keep]
                if len(valid_sor) > 4:
                    mean_d = float(np.mean(valid_sor))
                    std_d = float(np.std(valid_sor))
                    sor_thresh = mean_d + float(params.sor_std_mul) * std_d
                    sor_keep = sor_mean_dist <= sor_thresh
                else:
                    sor_keep = np.ones(n_pts, dtype=bool)
            else:
                sor_keep = np.ones(n_pts, dtype=bool)

            stats.ror_removed_count = int(np.sum(~ror_keep))
            stats.sor_removed_count = int(np.sum(ror_keep & (~sor_keep)))
            working = working[ror_keep & sor_keep]

        if params.enable_ground_alignment and len(working) >= 30:
            prev_count = len(working)
            working, gp_res = PointCloudProcessor.detect_and_align_ground_plane(
                working,
                distance_threshold_m=params.ground_dist_threshold_m,
                max_iterations=min(int(params.ground_max_iterations), 120),
                max_tilt_deg=params.ground_max_tilt_deg,
                align_to_z_zero=params.align_ground_to_z_zero,
                remove_ground_inliers=params.remove_ground_plane,
            )
            stats.ground_plane = gp_res
            stats.ground_removed_count = max(0, prev_count - len(working))

        stats.output_point_count = len(working)
        stats.bbox_min, stats.bbox_max = PointCloudProcessor.compute_bounding_box(working)
        stats.computation_time_ms = (time.perf_counter() - t0) * 1000.0
        return working, stats

    @staticmethod
    def parse_export_format(format_str: str) -> ExportFormat:
        s = (format_str or "").upper()
        if "E57" in s:
            return ExportFormat.E57_STRUCTURED
        if "PLY" in s and "BIN" in s:
            return ExportFormat.PLY_BINARY
        if "PLY" in s:
            return ExportFormat.PLY_ASCII
        if "PCD" in s and "BIN" in s:
            return ExportFormat.PCD_BINARY
        if "PCD" in s:
            return ExportFormat.PCD_ASCII
        if "XYZ" in s or "PTS" in s:
            return ExportFormat.XYZ_ASCII
        if "OBJ" in s or "MESH" in s:
            return ExportFormat.OBJ_MESH
        return ExportFormat.PLY_ASCII

    @staticmethod
    def export_format_extension(fmt: ExportFormat) -> str:
        if fmt == ExportFormat.E57_STRUCTURED:
            return ".e57"
        if fmt in (ExportFormat.PLY_ASCII, ExportFormat.PLY_BINARY):
            return ".ply"
        if fmt in (ExportFormat.PCD_ASCII, ExportFormat.PCD_BINARY):
            return ".pcd"
        if fmt == ExportFormat.XYZ_ASCII:
            return ".xyz"
        if fmt == ExportFormat.OBJ_MESH:
            return ".obj"
        return ".ply"

    @staticmethod
    def export_format_name(fmt: ExportFormat) -> str:
        names = {
            ExportFormat.PLY_ASCII: "PLY (ASCII)",
            ExportFormat.PLY_BINARY: "PLY (Binary Little-Endian)",
            ExportFormat.PCD_ASCII: "PCD (ASCII)",
            ExportFormat.PCD_BINARY: "PCD (Binary)",
            ExportFormat.XYZ_ASCII: "XYZRGBI (ASCII)",
            ExportFormat.OBJ_MESH: "Wavefront OBJ (3D Surface Mesh)",
            ExportFormat.E57_STRUCTURED: "ASTM E57 (Structured Station + 360° Panorama)",
        }
        return names.get(fmt, "PLY (ASCII)")

    @staticmethod
    def _extract_xyz_rgb_intensity(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        pts = PointCloudProcessor._filter_finite(points)
        n_pts = len(pts)
        xyz = pts[:, :3].astype(np.float32)
        if pts.shape[1] >= 6:
            rgb_raw = pts[:, 3:6].astype(np.float32)
            if np.max(rgb_raw) <= 1.05:
                rgb_u8 = np.clip(rgb_raw * 255.0, 0, 255).astype(np.uint8)
            else:
                rgb_u8 = np.clip(rgb_raw, 0, 255).astype(np.uint8)
        else:
            rgb_u8 = np.full((n_pts, 3), 200, dtype=np.uint8)

        if pts.shape[1] >= 7:
            intensity = pts[:, 6].astype(np.float32)
        else:
            intensity = np.mean(rgb_u8.astype(np.float32), axis=1)
        return xyz, rgb_u8, intensity

    @staticmethod
    def export_point_cloud(
        points: np.ndarray,
        output_path: str,
        fmt: ExportFormat = ExportFormat.PLY_ASCII,
        mesh_grid_resolution_m: float = 0.04,
    ) -> ExportResult:
        """
        Exports point cloud to PLY (ASCII/Binary), PCD (ASCII/Binary), XYZ, or reconstructed Wavefront OBJ 3D Surface Mesh.
        Matches av_scan PointCloudProcessor::exportPointCloud.
        """
        t0 = time.perf_counter()
        res = ExportResult(file_path=output_path, format_name=PointCloudProcessor.export_format_name(fmt))

        pts = PointCloudProcessor._filter_finite(points)
        if len(pts) == 0:
            res.error_message = "Cannot export empty point cloud."
            return res

        parent_dir = os.path.dirname(os.path.abspath(output_path))
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)

        xyz, rgb_u8, intensity = PointCloudProcessor._extract_xyz_rgb_intensity(pts)
        n_pts = len(xyz)

        try:
            if fmt == ExportFormat.PLY_ASCII:
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(
                        "ply\n"
                        "format ascii 1.0\n"
                        "comment Generated by ScanHUB Milestone 12 PointCloudProcessor\n"
                        f"element vertex {n_pts}\n"
                        "property float x\n"
                        "property float y\n"
                        "property float z\n"
                        "property uchar red\n"
                        "property uchar green\n"
                        "property uchar blue\n"
                        "property float intensity\n"
                        "end_header\n"
                    )
                    lines = [
                        f"{xyz[i,0]:.5f} {xyz[i,1]:.5f} {xyz[i,2]:.5f} {rgb_u8[i,0]} {rgb_u8[i,1]} {rgb_u8[i,2]} {intensity[i]:.2f}\n"
                        for i in range(n_pts)
                    ]
                    f.writelines(lines)
                res.success = True
                res.exported_points = n_pts

            elif fmt == ExportFormat.PLY_BINARY:
                header = (
                    "ply\n"
                    "format binary_little_endian 1.0\n"
                    "comment Generated by ScanHUB Milestone 12 PointCloudProcessor\n"
                    f"element vertex {n_pts}\n"
                    "property float x\n"
                    "property float y\n"
                    "property float z\n"
                    "property uchar red\n"
                    "property uchar green\n"
                    "property uchar blue\n"
                    "property float intensity\n"
                    "end_header\n"
                ).encode("ascii")
                vertex_dtype = np.dtype([
                    ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                    ("r", "u1"), ("g", "u1"), ("b", "u1"),
                    ("intensity", "<f4"),
                ])
                buf = np.empty(n_pts, dtype=vertex_dtype)
                buf["x"] = xyz[:, 0]
                buf["y"] = xyz[:, 1]
                buf["z"] = xyz[:, 2]
                buf["r"] = rgb_u8[:, 0]
                buf["g"] = rgb_u8[:, 1]
                buf["b"] = rgb_u8[:, 2]
                buf["intensity"] = intensity
                with open(output_path, "wb") as f:
                    f.write(header)
                    f.write(buf.tobytes())
                res.success = True
                res.exported_points = n_pts

            elif fmt in (ExportFormat.PCD_ASCII, ExportFormat.PCD_BINARY):
                rgb_packed_u32 = (
                    (rgb_u8[:, 0].astype(np.uint32) << 16)
                    | (rgb_u8[:, 1].astype(np.uint32) << 8)
                    | rgb_u8[:, 2].astype(np.uint32)
                )
                rgb_float = rgb_packed_u32.view(np.float32)
                header_str = (
                    "# .PCD v0.7 - Point Cloud Data file format\n"
                    "VERSION 0.7\n"
                    "FIELDS x y z intensity rgb\n"
                    "SIZE 4 4 4 4 4\n"
                    "TYPE F F F F F\n"
                    "COUNT 1 1 1 1 1\n"
                    f"WIDTH {n_pts}\n"
                    "HEIGHT 1\n"
                    "VIEWPOINT 0 0 0 1 0 0 0\n"
                    f"POINTS {n_pts}\n"
                    f"DATA {'binary' if fmt == ExportFormat.PCD_BINARY else 'ascii'}\n"
                )
                if fmt == ExportFormat.PCD_ASCII:
                    with open(output_path, "w", encoding="utf-8") as f:
                        f.write(header_str)
                        lines = [
                            f"{xyz[i,0]:.5f} {xyz[i,1]:.5f} {xyz[i,2]:.5f} {intensity[i]:.2f} {rgb_float[i]:.8e}\n"
                            for i in range(n_pts)
                        ]
                        f.writelines(lines)
                else:
                    pcd_dtype = np.dtype([
                        ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                        ("intensity", "<f4"), ("rgb", "<f4"),
                    ])
                    buf = np.empty(n_pts, dtype=pcd_dtype)
                    buf["x"] = xyz[:, 0]
                    buf["y"] = xyz[:, 1]
                    buf["z"] = xyz[:, 2]
                    buf["intensity"] = intensity
                    buf["rgb"] = rgb_float
                    with open(output_path, "wb") as f:
                        f.write(header_str.encode("ascii"))
                        f.write(buf.tobytes())
                res.success = True
                res.exported_points = n_pts

            elif fmt == ExportFormat.XYZ_ASCII:
                with open(output_path, "w", encoding="utf-8") as f:
                    lines = [
                        f"{xyz[i,0]:.5f} {xyz[i,1]:.5f} {xyz[i,2]:.5f} {rgb_u8[i,0]} {rgb_u8[i,1]} {rgb_u8[i,2]} {intensity[i]:.2f}\n"
                        for i in range(n_pts)
                    ]
                    f.writelines(lines)
                res.success = True
                res.exported_points = n_pts

            elif fmt == ExportFormat.OBJ_MESH:
                cell = max(0.01, float(mesh_grid_resolution_m))
                mesh_pts = PointCloudProcessor.voxel_downsample(pts, cell)
                if len(mesh_pts) == 0:
                    mesh_pts = pts
                m_xyz, m_rgb_u8, _ = PointCloudProcessor._extract_xyz_rgb_intensity(mesh_pts)
                m_n = len(m_xyz)

                inv_cell = 1.0 / cell
                vox_coords = np.floor(m_xyz.astype(np.float64) * inv_cell).astype(np.int64)
                vox_to_idx = {}
                for i in range(m_n):
                    vox_to_idx[(int(vox_coords[i, 0]), int(vox_coords[i, 1]), int(vox_coords[i, 2]))] = i

                def find_surface_neighbor(kx: int, ky: int, kz: int, axis: int) -> int:
                    for delta in (0, 1):
                        for sign in (-1, 1):
                            d = delta * sign
                            q = (
                                kx + (d if axis == 0 else 0),
                                ky + (d if axis == 1 else 0),
                                kz + (d if axis == 2 else 0),
                            )
                            idx = vox_to_idx.get(q, -1)
                            if idx != -1:
                                return idx
                            if delta == 0:
                                break
                    return -1

                max_edge_sq = (cell * 2.75) ** 2

                def valid_tri(i0: int, i1: int, i2: int) -> bool:
                    if i0 < 0 or i1 < 0 or i2 < 0 or i0 == i1 or i1 == i2 or i0 == i2:
                        return False
                    p0, p1, p2 = m_xyz[i0], m_xyz[i1], m_xyz[i2]
                    if np.sum((p1 - p0) ** 2) > max_edge_sq:
                        return False
                    if np.sum((p2 - p1) ** 2) > max_edge_sq:
                        return False
                    if np.sum((p0 - p2) ** 2) > max_edge_sq:
                        return False
                    cross = np.cross(p1 - p0, p2 - p0)
                    return float(np.dot(cross, cross)) > 1e-10

                triangles = []
                for (kx, ky, kz), v00 in vox_to_idx.items():
                    # XY plane quads (floors/ceilings)
                    v10 = find_surface_neighbor(kx + 1, ky, kz, 2)
                    v01 = find_surface_neighbor(kx, ky + 1, kz, 2)
                    if v10 != -1 and v01 != -1:
                        if valid_tri(v00, v10, v01):
                            triangles.append((v00, v10, v01))
                        v11 = find_surface_neighbor(kx + 1, ky + 1, kz, 2)
                        if v11 != -1 and valid_tri(v10, v11, v01):
                            triangles.append((v10, v11, v01))

                    # XZ plane quads (front/back walls)
                    v10 = find_surface_neighbor(kx + 1, ky, kz, 1)
                    v01 = find_surface_neighbor(kx, ky, kz + 1, 1)
                    if v10 != -1 and v01 != -1:
                        if valid_tri(v00, v10, v01):
                            triangles.append((v00, v10, v01))
                        v11 = find_surface_neighbor(kx + 1, ky, kz + 1, 1)
                        if v11 != -1 and valid_tri(v10, v11, v01):
                            triangles.append((v10, v11, v01))

                    # YZ plane quads (side walls)
                    v10 = find_surface_neighbor(kx, ky + 1, kz, 0)
                    v01 = find_surface_neighbor(kx, ky, kz + 1, 0)
                    if v10 != -1 and v01 != -1:
                        if valid_tri(v00, v10, v01):
                            triangles.append((v00, v10, v01))
                        v11 = find_surface_neighbor(kx, ky + 1, kz + 1, 0)
                        if v11 != -1 and valid_tri(v10, v11, v01):
                            triangles.append((v10, v11, v01))

                normals = np.zeros((m_n, 3), dtype=np.float32)
                if triangles:
                    tris_arr = np.asarray(triangles, dtype=np.int32)
                    p0 = m_xyz[tris_arr[:, 0]]
                    p1 = m_xyz[tris_arr[:, 1]]
                    p2 = m_xyz[tris_arr[:, 2]]
                    fn = np.cross(p1 - p0, p2 - p0)
                    np.add.at(normals, tris_arr[:, 0], fn)
                    np.add.at(normals, tris_arr[:, 1], fn)
                    np.add.at(normals, tris_arr[:, 2], fn)

                norms_len = np.linalg.norm(normals, axis=1)
                valid_n = norms_len > 1e-6
                normals[valid_n] /= norms_len[valid_n, None]
                normals[~valid_n] = np.array([0.0, 0.0, 1.0], dtype=np.float32)

                m_rgb_f = m_rgb_u8.astype(np.float32) / 255.0
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(
                        "# Wavefront OBJ generated by ScanHUB Milestone 12 PointCloudProcessor\n"
                        f"# Vertices: {m_n} | Triangles: {len(triangles)}\n"
                        "o ScanHUB_Surface_Mesh\n"
                    )
                    for i in range(m_n):
                        f.write(
                            f"v {m_xyz[i,0]:.5f} {m_xyz[i,1]:.5f} {m_xyz[i,2]:.5f} "
                            f"{m_rgb_f[i,0]:.4f} {m_rgb_f[i,1]:.4f} {m_rgb_f[i,2]:.4f}\n"
                        )
                    for i in range(m_n):
                        f.write(f"vn {normals[i,0]:.5f} {normals[i,1]:.5f} {normals[i,2]:.5f}\n")
                    for a, b, c in triangles:
                        i1, i2, i3 = a + 1, b + 1, c + 1
                        f.write(f"f {i1}//{i1} {i2}//{i2} {i3}//{i3}\n")

                res.success = True
                res.exported_points = m_n
                res.exported_triangles = len(triangles)

            elif fmt == ExportFormat.E57_STRUCTURED:
                import cv2 as _cv2
                try:
                    from .av_structure_engine import (
                        StationGenerationConfig,
                        VirtualStationGenerator,
                        VirtualStationSelector,
                    )
                    from .e57_structured_writer import E57StationPayload, write_structured_e57
                except ImportError:
                    from av_structure_engine import (
                        StationGenerationConfig,
                        VirtualStationGenerator,
                        VirtualStationSelector,
                    )
                    from e57_structured_writer import E57StationPayload, write_structured_e57

                st_cfg = StationGenerationConfig(scan_rows=256, scan_cols=1024, pano_width=2048, pano_height=1024)
                stations = VirtualStationSelector.select_stations(st_cfg, points_xyz=xyz)
                gen = VirtualStationGenerator(st_cfg)
                payloads = []
                total_val = 0
                for st in stations:
                    s_res = gen.synthesize_station(st, xyz, rgb_u8)
                    pano_bgr = _cv2.cvtColor(s_res.panorama_rgb, _cv2.COLOR_RGB2BGR)
                    ok_j, jpg_b = _cv2.imencode(".jpg", pano_bgr, [int(_cv2.IMWRITE_JPEG_QUALITY), 93])
                    payloads.append(
                        E57StationPayload(
                            station_id=st.station_id,
                            station_name=st.station_id,
                            description=f"AV Structure Virtual Station ({st.source_mode})",
                            translation_xyz=(float(st.position_xyz[0]), float(st.position_xyz[1]), float(st.position_xyz[2])),
                            rotation_wxyz=st.rotation_wxyz,
                            xyz_local=s_res.scan_xyz_local,
                            invalid_state=s_res.scan_invalid_state,
                            rgb=s_res.scan_rgb,
                            panorama_jpeg_bytes=jpg_b.tobytes() if ok_j else None,
                            panorama_width=st_cfg.pano_width,
                            panorama_height=st_cfg.pano_height,
                        )
                    )
                    total_val += s_res.valid_scan_points

                write_structured_e57(output_path, payloads)
                res.success = True
                res.exported_points = total_val

            if os.path.exists(output_path):
                res.file_size_bytes = os.path.getsize(output_path)

        except Exception as exc:
            res.success = False
            res.error_message = str(exc)

        res.export_time_ms = (time.perf_counter() - t0) * 1000.0
        return res
