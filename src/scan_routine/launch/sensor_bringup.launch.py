import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, ExecuteProcess
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

def generate_launch_description():
    # --- File Paths ---
    # Hardware Configs
    rslidar_config = '/home/scanar/scanarMini/src/rslidar_sdk/config/drone_config.yaml'
    
    # Orbbec Gemini 336L official launch file
    orbbec_launch_file = os.path.join(
        get_package_share_directory('orbbec_camera'),
        'launch',
        'gemini_330_series.launch.py'
    )

    # --- Node Definitions ---

    # 1. RoboSense Airy Lidar & Internal IMU Driver Node
    rslidar_node = Node(
        package='rslidar_sdk',
        executable='rslidar_sdk_node',
        name='rslidar_sdk_node',
        namespace='rslidar_sdk',
        output='screen',
        parameters=[{'config_path': rslidar_config}]
    )

    # 2. Orbbec Gemini 336L Stereo IR & RGB Camera Driver
    camera_node = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(orbbec_launch_file),
        launch_arguments={
            'camera_name': 'camera',
            'enable_left_ir': 'false',
            'enable_right_ir': 'false',
            'enable_color': 'true',
            'enable_depth': 'true',
            'enable_point_cloud': 'false',
            'enable_colored_point_cloud': 'false',
            'enable_laser': 'true', # Production: Laser ON by default for active stereo depth
            'enable_ir_auto_exposure': 'true', # Production: Auto-exposure enabled for robust tracking
            'ir_exposure': '12000', # Production: Max exposure capped at 12ms (12000 us)
            'sync_mode': 'SECONDARY', # Hardware Secondary (Strict Triggered Slave)
            'time_domain': 'system', # Hardware arrival timestamps matched to Linux system clock epoch
            'color_width': '1280',
            'color_height': '800',
            'color_fps': '30',
            'depth_width': '848',
            'depth_height': '480',
            'depth_fps': '30',
            'left_ir_width': '848',
            'left_ir_height': '480',
            'left_ir_fps': '30',
            'right_ir_width': '848',
            'right_ir_height': '480',
            'right_ir_fps': '30',
        }.items()
    )

    # 2.5. Verify Airy Hardware SYNC_OUT1 pulse step (120 deg) & pulse width (1 ms)
    verify_sync_process = ExecuteProcess(
        cmd=['python3', '/home/scanar/scanarMini/src/scan_routine/scripts/verify_airy_sync.py'],
        output='screen'
    )

    # 3. Topic Remapping Process using our custom pure-python relay script (eliminates dependency on topic_tools)
    cam_remapper_process = ExecuteProcess(
        cmd=['python3', '/home/scanar/scanarMini/src/scan_routine/scripts/topic_relay.py'],
        output='screen'
    )

    # 4. Static Transform Publishers (Fully compliant with Bag D and factory CAD specifications)
    
    # scanar_base_link -> airy_lidar (Identical origin definition)
    tf_base_to_lidar = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='tf_base_to_lidar',
        arguments=['0', '0', '0', '0', '0', '0', '1', 'scanar_base_link', 'rslidar']
    )

    # airy_lidar -> airy_imu (Factory DIFOP Calibration)
    tf_lidar_to_imu = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='tf_lidar_to_imu',
        arguments=['0.004250', '0.004180', '-0.004460', 
                   '0.71145376', '-0.70271848', '0.0018789', '0.00409338', 
                   'rslidar', 'airy_imu']
    )

    # T_cam0_lidar translation: [+0.043444, -0.063060, -0.029978] m (scanarV4 45-deg)
    # Inverted T_lidar_cam0 for ROS static TF:
    # Translation: [+0.065332, +0.044062, -0.023515] m
    # Quaternion [qx, qy, qz, qw]: [-0.275342, 0.259035, -0.651888, 0.657366]
    tf_lidar_to_cam0 = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='tf_lidar_to_cam0',
        arguments=['0.065332', '0.044062', '-0.023515', 
                   '-0.275342', '0.259035', '-0.651888', '0.657366', 
                   'rslidar', 'camera_left_ir_optical_frame']
    )

    # camera_left_ir_optical_frame -> camera_right_ir_optical_frame (Cam1 Stereo Baseline 95.2793 mm)
    tf_cam0_to_cam1 = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='tf_cam0_to_cam1',
        arguments=['-0.0952793', '0.0', '0.0', 
                   '0.0', '0.0', '0.0', '1.0', 
                   'camera_left_ir_optical_frame', 'camera_right_ir_optical_frame']
    )

    # camera_left_ir_optical_frame -> camera_color_optical_frame (RGB Optical CAD Offset)
    tf_cam0_to_color = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='tf_cam0_to_color',
        arguments=['0.0237306', '-0.0000932', '-0.0004246', 
                   '-0.000658', '-0.000174', '-0.001189', '0.999999', 
                   'camera_left_ir_optical_frame', 'camera_color_optical_frame']
    )

    return LaunchDescription([
        verify_sync_process,
        rslidar_node,
        camera_node,
        cam_remapper_process,
        tf_base_to_lidar,
        tf_lidar_to_imu,
        tf_lidar_to_cam0,
        tf_cam0_to_cam1,
        tf_cam0_to_color
    ])
