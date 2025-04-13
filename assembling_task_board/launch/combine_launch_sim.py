from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution


def generate_launch_description():
    # Launch YOLOBot
    yolo_sim_node = Node(
        package="yolov11",
        executable="YOLO_sim_node.py",
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
        executable="controlSwitcher_sim_node",
        output="screen",
    )
    
    return LaunchDescription([
        yolo_sim_node,
        lightning_node,
        control_switcher_node,
        twister_node
    ])
