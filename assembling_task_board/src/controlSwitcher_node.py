import rclpy
from rclpy.node import Node
from controller_manager_msgs.srv import SwitchController
from builtin_interfaces.msg import Duration
from std_msgs.msg import String  # Using a simple message type for input
from moveit_msgs.srv import ServoCommandType  # Import the new service
import time

class ControllerSwitcher(Node):
    def __init__(self):
        super().__init__("controller_switcher")

        # Create client for controller manager
        self.client = self.create_client(SwitchController, "/controller_manager/switch_controller")

        # Create client for /servo_node/switch_command_type
        self.servo_client = self.create_client(ServoCommandType, "/servo_node/switch_command_type")
        request = ServoCommandType.Request()
        request.command_type = 1  # Set command type to 1

        self.get_logger().info("Switching servo command type to 1...")

        # Call service asynchronously
        while rclpy.ok():
            future = self.servo_client.call_async(request)
            rclpy.spin_until_future_complete(self, future)

            if future.result() and future.result().success:
                self.get_logger().info("Successfully switched servo command type.")
                break

            self.get_logger().warn("Failed to switch servo command type. Retrying...")
            time.sleep(0.2)  # Avoid spamming


        # Wait until service is available
        while not self.client.wait_for_service(timeout_sec=2.0):
            self.get_logger().warn("Waiting for /controller_manager/switch_controller service...")

        # Create a subscriber for controller switch requests
        self.subscription = self.create_subscription(
            String,
            "/controller_switch_request",
            self.controller_request_callback,
            10
        )
        self.get_logger().info("Controller Switcher is ready and listening on /controller_switch_request")

    def controller_request_callback(self, msg):

        try:
            # Parse the incoming message
            parts = msg.data.split(";")
            activate, deactivate = [], []

            for part in parts:
                if part.startswith("activate:"):
                    activate = part[len("activate:"):].split(",")
                elif part.startswith("deactivate:"):
                    deactivate = part[len("deactivate:"):].split(",")

            # Call the switch_controllers function with parsed controllers
            self.switch_controllers(activate, deactivate)

        except Exception as e:
            self.get_logger().error(f"Failed to parse controller switch request: {str(e)}")

    def switch_controllers(self, activate, deactivate):

        if not self.client.service_is_ready():
            self.get_logger().error("Service is not ready, cannot switch controllers")
            return

        request = SwitchController.Request()
        request.activate_controllers = activate
        request.deactivate_controllers = deactivate
        request.strictness = 2  # STRICT = 2
        request.timeout = Duration(sec=2, nanosec=0)  # Corrected timeout

        self.get_logger().info(f"Requesting switch: activate {activate}, deactivate {deactivate}")

        # Call service asynchronously
        future = self.client.call_async(request)
        future.add_done_callback(self.switch_controllers_callback)

    def switch_controllers_callback(self, future):
        """Callback function for handling the service response."""
        try:
            response = future.result()
            if response and response.ok:
                self.get_logger().info("Successfully switched controllers")
            else:
                self.get_logger().error("Controller switch failed")
        except Exception as e:
            self.get_logger().error(f"Service call failed: {str(e)}")

def main():
    rclpy.init()
    node = ControllerSwitcher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
