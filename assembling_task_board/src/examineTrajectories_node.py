import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose
from sensor_msgs.msg import JointState
import yaml
import os

import ament_index_python.packages

package_name = 'assembling_task_board'  # Replace with your package name
package_path = ament_index_python.packages.get_package_share_directory(package_name)


class Watching_DriveToPositionsNode(Node):
    def __init__(self):
        super().__init__("drive_to_positions")

        self.yaml_file = os.path.join(package_path, 'Task Board information', 'IK solutions', 'IK solutions.yaml')

        # List of objects to look for
        self.current_object = ["Ket_Square_12", "Ket_Square_12_Female","Ket_Square_8", "Ket_Square_8_Female","USB_Male", "USB_Female","Gear_Large", "Gear_Shaft", "Gear_Medium","Gear_Shaft", "Gear_Small","Gear_Shaft","Waterproof_Male", "Waterproof_Female", "M16_Nut", "M16_Screw"]

        # Publisher to publish joint positions
        self.target_joints_publisher = self.create_publisher(JointState, 'target_joints', 10)

        # Subscription to wait for target pose reached
        self.target_pose_reached_subscriber = self.create_subscription(
            Pose, 'target_pose_reached', self.target_pose_reached_callback, 10
        )

        # Load YAML file
        self.joint_positions = self.load_yaml_file()

        # Initialize index for tracking the current object
        self.current_index = 0

        # Start publishing the first joint position
        self.process_next_object()

    def load_yaml_file(self):
        """Load the YAML file and return its contents."""
        try:
            with open(self.yaml_file, 'r') as file:
                data = yaml.safe_load(file)
                self.get_logger().info("Successfully loaded YAML file.")
                return data
        except FileNotFoundError:
            self.get_logger().error(f"YAML file not found: {self.yaml_file}")
        except yaml.YAMLError as e:
            self.get_logger().error(f"Error parsing YAML file: {e}")
        return {}

    def process_next_object(self):
        """Process the next object in the list by publishing its joint positions."""
        if self.current_index < len(self.current_object):
            obj = self.current_object[self.current_index]
            if obj in self.joint_positions and 'joint_positions' in self.joint_positions[obj]:
                joint_position = self.joint_positions[obj]['joint_positions']
                
                # Ensure the joint positions are numeric
                try:
                    joint_position = [float(pos) for pos in joint_position]
                except ValueError as e:
                    self.get_logger().error(f"Invalid joint position values for {obj}: {joint_position}. Error: {e}")
                    self.current_index += 1
                    self.process_next_object()  # Skip to the next object
                    return

                msg = JointState()
                msg.header.stamp = self.get_clock().now().to_msg()
                msg.name = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]  # Adjust joint names
                msg.position = joint_position
                self.target_joints_publisher.publish(msg)
                self.get_logger().info(f"Published joint position for {obj}: {joint_position}")
            else:
                self.get_logger().warning(f"Joint position for {obj} not found or malformed in YAML file.")
                self.current_index += 1  # Skip to the next object
                self.process_next_object()  # Immediately process the next object if this one is invalid
        else:
            self.get_logger().info("Finished publishing all joint positions.")

    def target_pose_reached_callback(self, msg):
        """Callback when target pose is reached, to move to the next object."""
        self.get_logger().info("Received target pose reached message.")
        self.current_index += 1
        self.process_next_object()


def main(args=None):
    rclpy.init(args=args)
    node = Watching_DriveToPositionsNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
