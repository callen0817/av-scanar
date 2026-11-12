# scanarMini Engineering Roadmap (M0 – M30)

## Executive Summary & Architectural Separation of Concerns

The development of **scanarMini** is structured across two major phases:
1. **M0 – M15: MVP Qualification Phase** (Isolate, Bring-Up, Physically Validate Individual Sensors, LIO, Visual Tracking, Full LIVO, and Acceptance Testing).
2. **M16 – M30: Commercial & Multi-Rig Production Phase** (Filtering, Loop Closure, Multi-Device Synchronization, Multi-Rig Fusion, Production Exports, and Hardening).

### Three-Tier Coordinate Hierarchy
To prevent spatial distortion and preserve geometric integrity, the architecture strictly decouples:

```
┌────────────────────────────────────────────────────────┐
│ 1. CALIBRATED SENSOR TF (Immutable Physical Geometry)  │
│    - Fixed mechanical CAD & factory Kalibr matrices    │
│    - Never modified by local tilt or runtime alignment │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│ 2. SLAM / WORLD FRAME (Gravity-Aligned Coordinates)   │
│    - Established at startup via static gravity vector  │
│    - Origin: gravity_aligned_world -> scanar_base_link │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│ 3. GUI VIEWPORT (Operator Screen Manipulation)         │
│    - Canvas zoom, pan, Q/E rotation, W/A/S/D offsets   │
│    - Purely cosmetic projection for visualization      │
└────────────────────────────────────────────────────────┘
```

---

## Phase 1: MVP Qualification Roadmap (M0 – M15)

| Milestone | Scope | Key Deliverables & Pass Criteria | Status |
| :--- | :--- | :--- | :--- |
| **M0** | Workspace & Provenance | Clean isolation of `/home/scanar/scanarMini/`; read-only reference to `/home/scanR/`. | ✅ Complete |
| **M1** | Production Drivers | ROS 2 Humble drivers for RoboSense Airy (`rslidar_sdk`) and Orbbec Gemini 336L. | ✅ Complete |
| **M2** | Airy LiDAR Driver | 10 Hz `/rslidar_points` packet reception, UDP binding, PointCloud2 translation. | ✅ Complete |
| **M3** | Airy IMU Parser | Compiled with `-DENABLE_IMU_DATA_PARSE=ON`, UDP port 6688, `/rslidar_imu_data` active. | ✅ Complete |
| **M4** | Gemini 336L Feeds | RGB, Depth, Left/Right IR streams live via dedicated USB 3.2 bus. | ✅ Complete |
| **M5** | Factory Calibration | Kalibr camera matrices, factory DIFOP IMU-LiDAR transform, stereo baseline. | ✅ Complete |
| **M6** | Production TF Tree | Mathematically inverted static TF tree ($T_{\text{rslidar}\leftarrow\text{cam0\_optical}}$). | ✅ Complete |
| **M7** | ScanHUB GUI | Integrated OLED dark mode dashboard, split live feeds, real-time map canvas. | ✅ Complete |
| **M8** | Physical Sensor Qualification | Sub-gate qualification on USB 3.2 and dedicated PCIe Ethernet (100% throughput). | ✅ Complete |
| **M9** | Timing & Sync Qualification | Hardware strobe signal measurement, Airy Pin 2 -> Gemini Pin 6 VSYNC trigger lock. | ✅ Complete |
| **M10** | Pure Airy LIO SLAM Suite | Pure Airy LiDAR + Airy IMU FAST-LIVO2 state estimation (frontend drift < 2.0%). | ✅ Complete |
| **M11** | GUI ↔ SLAM Physical Workflow | Real-time map canvas, 1:1 motion vector lock, Flash Scans & dataset persistence. | ✅ Complete |
| **M12** | **Point Cloud RGB Colorization, Cleanup & Unlimited C-SLAM** | **Projection validation ($T_{\text{RGB}\leftarrow\text{LiDAR}}$), M12 Voxel/SOR/ROR/RANSAC Ground Leveling, Unlimited-Peer UDP C-SLAM.** | ✅ Complete |
| **M13** | Independent Visual Tracking | Stereo IR + IMU visual odometry verification under aggressive LiDAR-degraded motion. | 🔴 Scheduled |
| **M14** | Multi-Session Loop Closure & Export | Automated session management, backend loop closure, LAS/E57/PLY production export. | 🔴 Scheduled |
| **M15** | **Virtual Stations, Structured Scans & Panoramas (`AV Structure`)** | **Shared-center virtual scan stations, structured spherical 3D scans, multi-view RGB keyframe 360° panoramas, ASTM E2807-11 Structured `.e57` export.** | ✅ Complete |
| **M15B** | Commercial MVP Acceptance | Repeatable physical testing: static drift, 5m scale, 90° rotation, square loop closure. | 🔴 **CURRENT** |

---

## Milestone M8: Detailed Sub-Gate Test Plan

### M8.0 — Production Interface Qualification [PASSED]
* **Gemini 336L USB 3.2 Verification**:
  * Bus Topology: Direct connection on SuperSpeed root hub (Bus 02, Dev 10).
  * Negotiated Speed: **5000M (USB 3.2 Gen 1)** (not traversing USB 2.0 480M hub).
  * Device Serial: `CPC6463000XW` verified from hardware descriptors.
  * Supported Modes: Enumerate resolutions up to $1280 \times 800$ @ 30–60 FPS across RGB, Depth, and Left/Right IR.
* **RoboSense Airy Ethernet Verification**:
  * Interface: Dedicated PCIe interface `enP8p1s0` (no USB-Ethernet dongle in path).
  * Link Speed: 100Mb/s Full Duplex (standard Airy 100BASE-TX).
  * MTU: 1500 bytes.
  * Network Error Statistics: 0 errors, 0 dropped, 0 missed packets across $> 117\text{M}$ packets.

### M8.1 — Airy LiDAR Raw Geometry (Standalone)
* Bypass all manual GUI view adjustments (force Roll=0, Pitch=0, Yaw=0, XYZ=0).
* Verify physical axis correspondence using reference targets:
  * Distinct object $+1\text{ m}$ forward $\rightarrow$ appears at $\approx +X$ in LiDAR frame.
  * Distinct object $+1\text{ m}$ left $\rightarrow$ appears at $\approx +Y$ in LiDAR frame.
  * Distinct object $+1\text{ m}$ above sensor $\rightarrow$ appears at $\approx +Z$ in LiDAR frame.
* Physical rotation test: Rotate scanner $\pm 90^\circ$ and verify point cloud rotates consistently in sensor frame.

### M8.2 — Airy Native IMU Qualification (Standalone)
* Record $\ge 60\text{ s}$ of `/rslidar_imu_data` on dedicated Ethernet.
* Verify rate $\approx 200\text{ Hz}$ ($\Delta t \approx 5\text{ ms}$), zero non-monotonic timestamps, zero duplicates.
* **Empirical Gravity Vector**:
  * Normal base-down level pose: Expected dominant linear acceleration along **$-Z \approx -9.81\text{ m/s}^2$** ($\|a\| \approx 9.81\text{ m/s}^2$).
  * Inverted upside-down pose: Dominant linear acceleration along **$+Z \approx +9.81\text{ m/s}^2$**.
  * Left-side: $\approx +X$; Right-side: $\approx -X$; Camera-down: $\approx +Y$; Rear-down: $\approx -Y$.
* **Gyroscope Axis Dominance**:
  * Roll motion $\rightarrow$ Gyroscope $X$ dominant.
  * Pitch motion $\rightarrow$ Gyroscope $Y$ dominant.
  * Yaw motion $\rightarrow$ Gyroscope $Z$ dominant.

### M8.3 — Airy LiDAR + IMU Rigid-Frame Consistency
* Apply the factory DIFOP transformation matrix: $T_{\text{airy\_lidar}\leftarrow\text{airy\_imu}}$.
* Confirm transformed IMU gravity aligns with physical vertical in LiDAR coordinate frame.
* Verify transformed angular rates match physical LiDAR rotation direction during yaw, pitch, and roll motions.

### M8.4A — Gemini Capability Sweep (Projector OFF)
* Interrogate supported resolutions and frame rates on dedicated USB 3.2 bus.
* Test sustainable multi-stream combinations without frame drops:
  * RGB: $640 \times 400$ vs $1280 \times 720 / 1280 \times 800$ @ 30 FPS.
  * Depth: $640 \times 400$ vs $1280 \times 800$ @ 30 FPS.
  * Stereo IR: $640 \times 400$ @ 30–60 FPS.
* Query mode-dependent factory `camera_info` and EEPROM intrinsic parameters.

### M8.4B — Gemini Production Characterization (Projector ON)
* Enable active infrared speckle projector (`enable_laser: true`).
* Verify active stereo depth point density and absence of motion blur under adaptive exposure ($\le 12\text{ ms}$).

### M8.5 — Combined Sensor Throughput Benchmark
* Run all streams concurrently for 2–5 minutes:
  * Airy LiDAR (`/rslidar_points` @ 10 Hz)
  * Airy IMU (`/rslidar_imu_data` @ 200 Hz)
  * RGB (`/camera/color/image_raw` @ 30 Hz)
  * Depth (`/camera/depth/image_raw` @ 30 Hz)
  * Left IR (`/cam0/image_raw` @ 30 Hz)
  * Right IR (`/cam1/image_raw` @ 30 Hz)
* Benchmark CPU load, RAM usage, zero-drop packet transport, and timestamp continuity.

---

## Milestone M15: Virtual Stations, Structured Scans & Panoramas (`AV Structure`) [COMPLETE]

Transforms continuous mobile LiDAR + RGB reality capture (and third-party point clouds) into navigable terrestrial-grade **Virtual Scan Stations** for `AV Scan`, `AV Walkthrough`, and downstream BIM/CAD tools (Autodesk ReCap, CloudCompare, Revit, Faro Scene) without requiring a dedicated 360° camera.

### M15.1 — Intelligent Multi-Trigger RGB Keyframe Capture
* Continuous full-resolution RGB keyframe selection (`keyframes/kf_XXXX.jpg` + `keyframes_manifest.json`) driven by:
  * **Translation Trigger**: Every $0.5\text{–}1.0\text{ m}$ of trajectory motion (default $0.65\text{ m}$).
  * **Rotation Trigger**: Every $15\text{–}20^\circ$ of 3D camera rotation (default $17.5^\circ$; $12^\circ$ near anchored stations for smooth $360^\circ$ station spins).
  * **Directional Coverage Trigger**: Captures previously unobserved world azimuth $\times$ elevation sectors.
  * **Quality Gate**: Laplacian variance sharpness check + luma exposure check to reject motion-blurred or washed-out frames.
  * **Post-SLAM Pose Refinement**: Re-interpolates camera poses (`SLERP` + linear) against final SLAM trajectory prior to synthesis.

### M15.2 — Virtual Scan Station Placement (3 Modes)
* **Manual / Flash Anchors (`MANUAL_ONLY`)**: Triggered live via `[F] FARO FLASH`, `⚡ Drop Station`, or right-clicking any world coordinate on the 2D Map Canvas (with live $360^\circ$ RGB azimuth coverage ring).
* **Automatic Trajectory Stations (`AUTO_ONLY`)**: Placed along trajectory at configurable spacing (default $3.5\text{ m}$), locally scored for RGB keyframe coverage.
* **Hybrid Mode (`HYBRID`, Default)**: Anchors all manual/Flash stations first, then fills uncovered trajectory spans automatically while maintaining clearance.

### M15.3 — Shared-Center Dual Output Synthesis & Non-Fabrication Rule
* **Structured Spherical 3D Scan** (`rows × cols`, e.g. $256 \times 1024$ or $512 \times 2048$): Gravity-leveled station frame with foreground Z-buffer occlusion culling and explicit `cartesianInvalidState` (`0` = valid measurement, `2` = unobserved/occluded).
* **$360^\circ$ Equirectangular Panorama** ($2048 \times 1024$ / $4096 \times 2048$):
  * **Primary Path**: Geometry-assisted multi-view reprojection of calibrated Plumb-Bob (`Orbbec Gemini 336L`) or spherical (`Ouster Rev8`) keyframes onto visible 3D surfaces with per-keyframe Z-buffer parallax occlusion testing and vignette/baseline/sharpness feathering.
  * **Fallback Path**: Range-adaptive surfel splatting directly from colorized point clouds (`pointcloud_fused_rgb.ply` or third-party `.ply`/`.xyz`).
* **Strict Reality-Capture Integrity**: Unobserved or missing regions are **never** AI-inpainted or fabricated—preserved explicitly as `PanoramaProvenance.UNOBSERVED` (`0`) and `cartesianInvalidState = 2`.

### M15.4 — ASTM E2807-11 Structured `.e57` & Station Bundle Export
* Zero-dependency binary ASTM E2807-11 writer and Castagnoli CRC-32C page validator (`src/scanhub_gui/e57_structured_writer.py`).
* Pairs each station's `data3D` structured spherical scan with its aligned `images2D` `sphericalRepresentation` JPEG panorama via `associatedData3DGuid`.
* Exports standalone `stations/<station_id>/` bundle (`panorama_rgb.jpg`, `depth_map_mm.png`, `coverage_mask.png`, `structured_scan.ply`, `station_metadata.json`, `stations_manifest.json`) and reusable CLI `scripts/generate_virtual_stations.py`.

---

## Phase 2: Commercial & Multi-Rig Production Roadmap (M16 – M30)

### M16: Point-Cloud Conditioning
* Implement adaptive voxel grid filtering (2 cm – 5 cm selectable resolution).
* Statistical Outlier Removal (SOR) and Radius Outlier Removal (ROR).
* Invalid/noise return rejection, intensity/confidence threshold gating.
* Quantitative benchmarking of filter latency on Jetson Orin GPU/CPU.

### M17: Ground & Gravity Refinement
* Gravity-aligned spatial world frame initialization (`gravity_aligned_world`).
* Real-time floor-plane segmentation and optional planar floor constraint.
* Long-term level horizon consistency enforcement without mutating rigid sensor extrinsics.

### M18: LiDAR Deskew & Temporal Optimization
* Exact per-point timestamp interpolation and trajectory deskewing.
* High-rate IMU pre-integration and cubic spline interpolation for high-acceleration motion.
* Quantitative evaluation of motion distortion at angular velocities $> 60^\circ/\text{s}$.

### M19: Architectural Geometry Quality Stack
* Dynamic object (pedestrian, moving vehicle) rejection filter.
* Sharp edge preservation along doorframes, corners, and window mullions.
* Planar surface smoothing and overlap cleanup across multi-pass scans.

### M20: Pose-Graph & Loop-Closure Optimization
* Keyframe selection and submap management.
* Scan-context / LiDAR descriptor loop closure detection.
* GTSAM / Ceres pose-graph optimization backend for global closure residual minimization.

### M21: Architectural & Orthogonality Refinement
* Dominant wall normal detection and optional Manhattan-World structural constraints.
* Automatic floor/ceiling parallelism extraction.
* Operator-selectable orthogonality preservation mode for CAD/BIM floor plan generation.

### M22: High-Fidelity RGB Colorization Refinement
* Occlusion culling using ray-tracing / z-buffer projection.
* Optimal camera viewpoint angle weighting and auto-exposure color normalization.
* Seamless multi-view color blending and sharp color boundary preservation.

### M23: Metrology & Accuracy Characterization
* Calibrated target distance tests at 1 m, 5 m, and 10 m baselines.
* Wall-to-wall dimension accuracy, floor flatness, and vertical wall plumb error verification.
* CloudCompare point-to-point comparison against terrestrial reference LiDAR scans (Faro/Leica).

### M24: Multi-Device Hardware Synchronization
* Shared timing architecture across multiple scanarMini rigs.
* Defined Master/Slave trigger topology (PPS, hardware VSYNC strobe, or IEEE 1588 PTP).
* Cross-device timestamp verification, hardware device ID tagging, and synchronized multi-bag recording.

### M25: Multi-Device Mapping & Fusion
* Real-time and post-processed multi-rig spatial co-registration.
* Shared visual/LiDAR control points or pre-calibrated rig-to-rig transforms.
* Multi-agent trajectory fusion, submap merging, and overlap conflict resolution.

### M26: Multi-Session / Revisit Scanning
* Relocalization within pre-existing point cloud maps.
* Append-mode scanning for multi-day building surveys.
* Coordinate system persistence and immutable project origin management.

### M27: Production Export Pipeline
* Export formats: Clean structured/unstructured E57, LAS/LAZ, XYZRGB, PLY, and PCD.
* Complete metadata bundling: sensor serial numbers, calibration hashes, timestamp epoch, and trajectory.
* Automated export validation and CRC checksum generation.

### M28: Production System Hardening
* Cold-boot automation: zero manual ROS commands required from power-on.
* Hardware watchdog, automatic driver reconnection on USB/Ethernet drop, and disk space protection.
* Calibration version check and wrong-device lockout gate.

### M29: Embedded Performance Optimization
* CUDA/TensorRT acceleration for image processing and point cloud projection.
* Zero-copy ROS 2 intra-process communication tuning.
* Thermal, CPU, GPU, and RAM profiling under continuous $> 60$-minute scans on Jetson Orin.

### M30: Commercial QA & Release Gating
* Automated regression suite executing golden dataset replays.
* Field acceptance test protocols for operator handoff.
* Complete documentation, user manuals, and production release gating.
