import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool
import yaml
import os
from ament_index_python.packages import get_package_share_directory


class DriveToPositionsNode(Node):
    def __init__(self):
        super().__init__("drive_to_positions")
        package_share = get_package_share_directory('assembling_task_board')

        self.input_yaml_file = os.path.join(package_share, 'Task Board Information', 'Object Information', 'Object Information.yaml')
        self.output_yaml_file = os.path.join(package_share, 'Task Board Information', 'IK solutions', 'IK solutions.yaml')

        #Load named target positions from the input YAML file
        self.target_positions = self.load_named_target_positions()

        if not self.target_positions:
            self.get_logger().error("No target positions specified in the input YAML. Exiting.")
            rclpy.shutdown()
            return

        self.target_names = list(self.target_positions.keys())
        self.current_position_index = 0
        self.ik_solutions = {}  # Store received IK solutions
        self.Object_loader = False
        self.Task_board = False
        self.driving_back = False

        # Publisher for target poses
        self.target_pose_publisher = self.create_publisher(Pose, "/joint_space", 10)
        self.target_joints_publisher = self.create_publisher(JointState, 'target_joints', 10)

        # Subscriber for user input
        self.user_input_subscriber = self.create_subscription(
            Bool, "/user_input", self.user_input_callback, 10
        )

        # Subscriber for IK solutions
        self.ik_solution_subscriber = self.create_subscription(
            JointState, "/ik_solutions", self.ik_solution_callback, 10
        )
        # Start moving to the first position
        self.publish_target_pose()

    def load_named_target_positions(self):
        try:
            with open(self.input_yaml_file, "r") as file:
                config = yaml.safe_load(file)  # Load the YAML file

                # Define priority keys
                start_key = "start_position"
                object_loader_key = "Object_loader"
                task_board_key = "Task_board"

                # Extract objects based on superior_pos dependencies
                def get_objects_with_superior(superior):
                    return [
                        key for key, value in config.items()
                        if "superior_pos" in value and superior in value["superior_pos"]
                    ]

                # Sort keys based on the hierarchy
                sorted_keys = [key for key in [start_key, object_loader_key, task_board_key] if key in config]
                sorted_keys += get_objects_with_superior(object_loader_key)  # Objects that depend on Object_loader
                sorted_keys += get_objects_with_superior(task_board_key)  # Objects that depend on Task_board

                # Add remaining objects that were not included yet
                remaining_objects = sorted(
                    [key for key in config.keys() if key not in sorted_keys]
                )
                sorted_keys += remaining_objects

                # Extract only the "pos" values for each named position
                parsed_config = {
                    key: {
                        "pos": config[key].get("Approximated_Position", []),
                        "superior_pos": config[key].get("superior_pos", [])
                    }
                    for key in sorted_keys
                }

                return parsed_config  # Return dictionary with sorted positions
        except Exception as e:
            self.get_logger().error(f"Failed to load input YAML file: {e}")
            return {}

    def publish_target_pose(self):
        if self.current_position_index >= len(self.target_names):
            self.get_logger().info("All positions have been published. Exiting.")
            rclpy.shutdown()
            return

        current_name = self.target_names[self.current_position_index]

        current_sup =  self.target_positions[current_name]["superior_pos"]
        if current_sup:  # Check if the list is not empty
            current_sup = current_sup[0]
        else:
            current_sup = None  # Or handle it differently if needed
        self.get_logger().info(f"current sup: {current_sup}")
        if current_sup == "Object_loader":
            self.Object_loader = True
            self.get_logger().info(f"Current target '{current_name}' requires a custom pose. Reading from YAML file.")
            

        if current_sup == "Task_board":
            self.Object_loader = False
            self.Task_board = True
            self.get_logger().info(f"Current target '{current_name}' requires a custom pose. Reading from YAML file.")


        self.get_logger().info(f"current sup target pose: {self.Object_loader}")

        position = self.target_positions[current_name]["pos"]



        self.get_logger().info(f"Publishing target pose '{current_name}' at index {self.current_position_index}.")

        # Create the Pose message
        pose = Pose()
        pose.position.x = position[0]
        pose.position.y = position[1]
        pose.position.z = position[2]

        # Set the orientation (fixed orientation)
        from tf_transformations import quaternion_from_euler, quaternion_multiply
        straight_down_quat = quaternion_from_euler(0, 3.141592, 0)  # 180 degrees pitch
        rotation_45_z_quat = quaternion_from_euler(0, 0, 0)  # 45 degrees yaw
        final_quat = quaternion_multiply(rotation_45_z_quat, straight_down_quat)

        pose.orientation.x = final_quat[0]
        pose.orientation.y = final_quat[1]
        pose.orientation.z = final_quat[2]
        pose.orientation.w = final_quat[3]

        # Publish the pose
        self.target_pose_publisher.publish(pose)
        self.get_logger().info(f"Waiting for user input to continue for pose '{current_name}'.")

    def read_and_drive_to_pose(self):
        # Path to your YAML file


        try:
            with open(self.output_yaml_file, "r") as file:
                data = yaml.safe_load(file)

            # Find the pose for the name "Object_loader:Ket_Square_16"
            self.get_logger().info(f"current sup read drive: {self.Object_loader}")
            if self.Object_loader == True:
                key = "Object_loader"
            elif self.Task_board == True:
                key = "Task_board"
            if key in data:
                joint_positions = data[key]["joint_positions"]
                self.driving_back = True
                self.drive_to_ik(joint_positions)

        except Exception as e:
            self.get_logger().error(f"Failed to read YAML file: {e}")

    def drive_to_ik(self, joint_position):
        self.get_logger().info(f"Driving to approx Position")
        joints = joint_position
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]  # Adjust the joint names accordingly
        msg.position = joints  # Use the list of joint positions directly
        self.target_joints_publisher.publish(msg)
    

    def user_input_callback(self, msg):
        
        current_name = self.target_names[self.current_position_index]
        if msg.data == True:
            self.get_logger().info(f"User confirmed pose '{current_name}'.")

            # Check if an IK solution has been received for the current position
            if current_name in self.ik_solutions:
                self.save_to_yaml()
            else:
                self.get_logger().warning(
                    f"No IK solution received for pose '{current_name}'. Skipping save."
                )
        else:
            self.get_logger().info(f"User skipped pose '{current_name}'.")

        # Move to the next position
        
        if current_name not in ("Object_loader", "start_position", "Task_board"):
            self.get_logger().info(f"Driving back before'{current_name}'.")
            self.read_and_drive_to_pose()
        else:
            self.current_position_index += 1
            self.publish_target_pose()
            

    def ik_solution_callback(self, msg):
        if self.driving_back == False:
            current_name = self.target_names[self.current_position_index]
            
            joint_positions = list(msg.position)
            self.get_logger().info(f"Received IK solution for pose '{current_name}': {joint_positions}.")
        
            # Save the joint positions to the ik_solutions dictionary
            self.ik_solutions[current_name] = joint_positions  # Save as a list of floats
        else:
            self.driving_back = False
            self.current_position_index += 1
            self.publish_target_pose()


    def save_to_yaml(self):
        try:
            # Load existing data if the output YAML already exists
            if os.path.exists(self.output_yaml_file):
                with open(self.output_yaml_file, "r") as file:
                    output_data = yaml.safe_load(file) or {}
            else:
                output_data = {}

            current_name = self.target_names[self.current_position_index]

            # Ensure the IK solution is a list
            joint_positions = list(self.ik_solutions[current_name])  # Explicit conversion to list

            # Save the current IK solution
            output_data[current_name] = {
                "joint_positions": joint_positions
            }

            # Write back to the output YAML file
            with open(self.output_yaml_file, "w") as file:
                yaml.dump(output_data, file, default_flow_style=False)

            self.get_logger().info(
                f"Saved IK solution for pose '{current_name}' to {self.output_yaml_file}."
            )
        except Exception as e:
            self.get_logger().error(f"Failed to save IK solution to YAML: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = DriveToPositionsNode()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
