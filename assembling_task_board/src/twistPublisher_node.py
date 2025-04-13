import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TwistStamped
from std_msgs.msg import Float32MultiArray, String
from rclpy.time import Time
from moveit_msgs.srv import ServoCommandType
from rclpy.task import Future
import numpy as np
from control_msgs.msg import JointJog  # Correct message type
from std_msgs.msg import Header
import time

class TwistPublisher(Node):
    def __init__(self):
        super().__init__('twist_publisher')
        self.publisher_ = self.create_publisher(TwistStamped, '/servo_node/delta_twist_cmds', 10)
        self.publisher_joint_= self.create_publisher(JointJog, '/servo_node/delta_joint_cmds', 10)
        self.create_subscription(Float32MultiArray, '/twister_', self.error_callback, 10)
        self.control_switcher_publisher = self.create_publisher(String, "/controller_switch_request", 10)

        
        # Timer to publish velocity commands at 20 Hz
        self.timer = self.create_timer(0.05, self.publish_last_twist)
        self.alpha = 0.02  # Smoothing factor (adjust as needed)
        # Store last twist command and timestamp
        self.last_twist = TwistStamped()
        self.last_twist.header.frame_id = 'base_link'
        self.last_twist.twist.linear.x = 0.0
        self.last_twist.twist.linear.y = 0.0
        self.last_twist.twist.linear.z = 0.0

        self.last_twist.twist.angular.x = 0.0
        self.last_twist.twist.angular.y = 0.0
        self.last_twist.twist.angular.z = 0.0
        self.last_msg_time = None  # Initialize last message time


        
        self.demo_twist_publisher()
        self.stopped = False

   

    def demo_twist_publisher(self):
        """Demo function: Publishes TwistStamped at 50 Hz (every 20ms) for 0.1 seconds (5 iterations)."""
        time.sleep(3)
        msg = String()
        msg.data = "activate:forward_position_controller;deactivate:scaled_joint_trajectory_controller"
        self.control_switcher_publisher.publish(msg)

        twist_stamped = TwistStamped()
        twist_stamped.header.frame_id = 'base_link'  # Adjust as needed
        twist_stamped.twist.linear.x = -0.5  # Move forward
        twist_stamped.twist.linear.y = 0.0
        twist_stamped.twist.linear.z = 0.0
        twist_stamped.twist.angular.x = 0.0
        twist_stamped.twist.angular.y = 0.0
        twist_stamped.twist.angular.z = 0.0  # Rotate around Z
        
        self.get_logger().info("Starting demo twist publisher at 50Hz for 0.1s...")

        for _ in range(20):  # Publish for 5 iterations (0.1s)
            now = self.get_clock().now()
            twist_stamped.header.stamp = now.to_msg()
            self.publisher_.publish(twist_stamped)
            self.get_logger().info("Publishing demo TwistStamped message.")
 


    def error_callback(self, msg_err):
        """Process new error data and update the velocity command smoothly."""
        if len(msg_err.data) < 3:
            self.get_logger().warn("Invalid message received on /twister_")
            return

        error_x = msg_err.data[0]
        error_y = msg_err.data[1]  
        error_z = msg_err.data[2]

        # Smooth transition using exponential moving average
        self.last_twist.twist.linear.x = (1 - self.alpha) * self.last_twist.twist.linear.x + self.alpha * error_x
        self.last_twist.twist.linear.y = (1 - self.alpha) * self.last_twist.twist.linear.y + self.alpha * error_y
        self.last_twist.twist.linear.z = (1 - self.alpha) * self.last_twist.twist.linear.z + self.alpha * error_z

        self.last_twist.twist.angular.x = 0.0
        self.last_twist.twist.angular.y = 0.0
        self.last_twist.twist.angular.z = 0.0

        now = self.get_clock().now()
        self.last_msg_time = now  # Store the last message time
        self.last_twist.header.stamp = now.to_msg()

        self.get_logger().info(f'Updated TwistStamped: ({self.last_twist.twist.linear.x:.2f}, {self.last_twist.twist.linear.y:.2f})')

    def publish_last_twist(self):
        """Continuously publish the last known velocity only if the last message was recent."""
        if self.last_msg_time is None:
            return  # No message has been received yet

        now = self.get_clock().now()
        time_since_last_msg = (now - self.last_msg_time).nanoseconds * 1e-9  # Convert to seconds

        if time_since_last_msg <= 0.2:  # Only publish if last message was within 1 second
            self.last_twist.header.stamp = now.to_msg()
            
            self.publisher_.publish(self.last_twist)
            self.get_logger().info(f'Publishing TwistStamped: ({self.last_twist.twist.linear.x}, {self.last_twist.twist.linear.y})')
            self.stopped = False
        else:
            if self.stopped == False:
                self.last_twist.header.frame_id = 'base_link'
                self.last_twist.twist.linear.x = 0.0
                self.last_twist.twist.linear.y = 0.0
                self.last_twist.twist.linear.z = 0.0

                self.last_twist.twist.angular.x = 0.0
                self.last_twist.twist.angular.y = 0.0
                self.last_twist.twist.angular.z = 0.0
                self.get_logger().info("No recent error message received. Stopping publishing.")
            self.stopped = True

def main(args=None):
    rclpy.init(args=args)
    node = TwistPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
