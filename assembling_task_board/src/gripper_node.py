import subprocess
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Float32MultiArray


class GripperControlNode(Node):
    def __init__(self):
        super().__init__('gripper_control_node')

        self.create_subscription(Float32MultiArray, '/gripper_client', self.run_gripper_command, 10)
        
        self.get_logger().info("Starting Gripper Control Node...")

        # Path to the gripper_close.py script
        script_path = "/root/catkin_ws/src/weiss_gripper_ieg76/src/gripper_close.py"

        # Run the chmod command to make the script executable
        try:
            subprocess.run(
                ["docker", "exec", "-i", "weiss_gripper", "sudo", "chmod", "+x", "/root/catkin_ws/src/weiss_gripper_ieg76/src/gripper_close.py"],
                check=True
            )
            print(f"Successfully made {script_path} executable.")
        except subprocess.CalledProcessError as e:
            print(f"Failed to make {script_path} executable: {e}")
    
        #self.run_gripper_test()

    def run_gripper_command(self, msg):

        gripper_pos, gripper_force = int(msg.data[0]), int(msg.data[1])

        """Runs the gripper close command."""
        self.get_logger().info("Running gripper close command...")

        command = [
        "docker", "exec", "-i", "weiss_gripper",
        "bash", "-c", f"source /opt/ros/noetic/setup.bash && source /root/catkin_ws/devel/setup.bash && rosrun weiss_gripper_ieg76 gripper_close.py --position {gripper_pos} --force {gripper_force} "
        ]
        
        # Run the gripper close command
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()

        # Capture the output and error
        if stdout:
            self.get_logger().info(f"Output: {stdout.decode()}")  # Print what the script prints
        if stderr:
            self.get_logger().error(f"Error: {stderr.decode()}")  # Print errors if any
        
        self.get_logger().info("I have sent it")

    def run_gripper_test(self):

        gripper_pos, gripper_force = 30, 70

        """Runs the gripper close command."""
        self.get_logger().info("Running gripper close command...")

        command = [
        "docker", "exec", "-i", "weiss_gripper",
        "bash", "-c", f"source /opt/ros/noetic/setup.bash && source /root/catkin_ws/devel/setup.bash && rosrun weiss_gripper_ieg76 gripper_close.py --position {gripper_pos} --force {gripper_force} "
        ]
        
        # Run the gripper close command
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()

        # Capture the output and error
        if stdout:
            self.get_logger().info(f"Output: {stdout.decode()}")  # Print what the script prints
        if stderr:
            self.get_logger().error(f"Error: {stderr.decode()}")  # Print errors if any
        
        self.get_logger().info("I have sent it")




def main(args=None):
    rclpy.init(args=args)
    node = GripperControlNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
