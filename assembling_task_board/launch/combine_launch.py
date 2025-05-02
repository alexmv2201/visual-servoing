from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution


def generate_launch_description():
    # Launch the RealSense Intel node
    realsense_launch_file = PathJoinSubstitution([
        FindPackageShare('realsense2_camera'), 'launch', 'rs_launch.py'
    ])
    realsense_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(realsense_launch_file)
    )
    
    # Launch YOLOBot
    yolo_node = Node(
        package="yolov11",
        executable="YOLO_node.py",
        output="log",
    )
    
    # Lightning Node
    lightning_node = Node(
        package="assembling_task_board",
        executable="lightGlue_node",
        output="screen",
    )
    
    # Lightning Node
    twister_node = Node(
        package="assembling_task_board",
        executable="twistPublisher_node",
        output="screen",
    )

    # Control Switcher Node
    control_switcher_node = Node(
        package="assembling_task_board",
        executable="controlSwitcher_node",
        output="screen",
    )
    
       
    # Gripper Node
    gripper_node = Node(
        package="assembling_task_board",
        executable="gripper_node",
        output="screen",
    )
    
    return LaunchDescription([
        realsense_launch,
        yolo_node,
        lightning_node,
        control_switcher_node,
        twister_node,
        gripper_node
    ])
