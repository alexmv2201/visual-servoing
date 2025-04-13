import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose, PoseArray
from std_msgs.msg import Float32MultiArray, Float32
import subprocess
import time
import math
import numpy as np

class PoseCreatorNode(Node):
    def __init__(self):
        super().__init__('pose_creator_node')

        # Subscriber to get x, y, z input coordinates (assumed in a Float32MultiArray)
        self.subscription = self.create_subscription(
            Float32MultiArray,
            'input_coordinates',
            self.coordinates_callback,
            10)
        
        # Publisher to publish the Pose
        self.pose_publisher = self.create_publisher(Pose, 'joint_space', 10)
        self.set_speed_scale_pub = self.create_publisher(Float32, '/speed_scale', 10)
        self.target_pose_array_publisher = self.create_publisher(PoseArray, 'task_space', 10)
        self.get_logger().info("Pose Creator Node has been started")

    def coordinates_callback(self, msg):
        speed_scale = Float32(data=0.1)
        self.set_speed_scale_pub.publish(speed_scale)
        # Assuming the input is a Float32MultiArray with [x, y, z]
        if len(msg.data) == 3:
            x = msg.data[0]
            y = msg.data[1]
            z = msg.data[2]

            # Create a Pose message
            pose = Pose()

            # Set position (x, y, z)
            pose.position.x = x
            pose.position.y = y
            pose.position.z = z

            from tf_transformations import quaternion_from_euler, quaternion_multiply
            straight_down_quat = quaternion_from_euler(0, 3.141592,0)  # 180 degrees pitch
            rotation_45_z_quat = quaternion_from_euler(0, 0, 0)  # 45 degrees yaw
            final_quat = quaternion_multiply(rotation_45_z_quat, straight_down_quat)

            pose.orientation.x = final_quat[0]
            pose.orientation.y = final_quat[1]
            pose.orientation.z = final_quat[2]
            pose.orientation.w = final_quat[3]

            # Publish the pose
            self.pose_publisher.publish(pose)

            pose_array = PoseArray()
    
            pose_array.poses.append(pose)
            
            # Set the header for the PoseArray
            pose_array.header.stamp = self.get_clock().now().to_msg()
            pose_array.header.frame_id = "base_link"  # Replace with the appropriate frame_id
            #self.target_pose_array_publisher.publish(pose_array)


            self.get_logger().info(f'Published Pose: x={x}, y={y}, z={z}, orientation={pose.orientation}')
        else:
            self.get_logger().warn("Received incorrect number of coordinates, expected 3")
    
    def euler_to_quaternion(self, roll, pitch, yaw):
        """Convert Euler angles (in degrees) to a quaternion."""
        roll_rad = math.radians(roll)
        pitch_rad = math.radians(pitch)
        yaw_rad = math.radians(yaw)

        qx = math.sin(roll_rad / 2) * math.cos(pitch_rad / 2) * math.cos(yaw_rad / 2) - math.cos(roll_rad / 2) * math.sin(pitch_rad / 2) * math.sin(yaw_rad / 2)
        qy = math.cos(roll_rad / 2) * math.sin(pitch_rad / 2) * math.cos(yaw_rad / 2) + math.sin(roll_rad / 2) * math.cos(pitch_rad / 2) * math.sin(yaw_rad / 2)
        qz = math.cos(roll_rad / 2) * math.cos(pitch_rad / 2) * math.sin(yaw_rad / 2) - math.sin(roll_rad / 2) * math.sin(pitch_rad / 2) * math.cos(yaw_rad / 2)
        qw = math.cos(roll_rad / 2) * math.cos(pitch_rad / 2) * math.cos(yaw_rad / 2) + math.sin(roll_rad / 2) * math.sin(pitch_rad / 2) * math.sin(yaw_rad / 2)

        return [qx, qy, qz, qw]
    
    def driveToBNC(self):
        pose = Pose()

        # Set position (x, y, z)
        pose.position.x = 0.4
        pose.position.y = -0.4
        pose.position.z = 1.3
        q = self.euler_to_quaternion(180.0, 0.0, 0.0)  # 180 degrees around the x-axis
        pose.orientation.x = q[0]
        pose.orientation.y = q[1]
        pose.orientation.z = q[2]
        pose.orientation.w = q[3]
        self.pose_publisher.publish(pose)

def main(args=None):
    rclpy.init(args=args)
    node = PoseCreatorNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Node stopped by user')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
