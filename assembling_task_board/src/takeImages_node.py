import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool
from geometry_msgs.msg import Pose, PoseArray
import math
from time import sleep  # Import sleep function
import tf_transformations as tf

class TakeImages(Node):
    def __init__(self):
        super().__init__('take_images_node')

        # Declare and retrieve the 'manual_start' parameter
        self.declare_parameter('manual_start', True)
        self.manual_start = self.get_parameter('manual_start').get_parameter_value().bool_value

        # Declare and retrieve the 'yolo_object' parameter
        self.declare_parameter('yolo_object', 'Ket_Square_8_')
        self.yolo_object = self.get_parameter('yolo_object').get_parameter_value().string_value

        # Log the retrieved parameter values
        self.get_logger().info(f'manual_start: {self.manual_start}')
        self.get_logger().info(f'yolo_object: {self.yolo_object}')

        # Publishers
        self.target_pose_publisher = self.create_publisher(Pose, 'joint_space', 10)
        self.target_pose_array_publisher = self.create_publisher(PoseArray, 'task_space', 10)
        self.ask_current_pose_publisher = self.create_publisher(Bool, 'ask_current_pose', 10)
        self.trigger_yolo_publisher = self.create_publisher(String, 'GetPositionPublisher', 10)
        self.take_image_publisher = self.create_publisher(String, 'take_image', 10)

        # Subscriptions
        self.current_pose_subscription = self.create_subscription(
            Pose, 'current_pose_publisher', self.current_pose_callback, 10)

        self.target_pose_reached_subscription = self.create_subscription(
            Pose, 'target_pose_reached', self.target_reached_callback, 10)

        # General variables
        self.trigger = Bool()
        self.camera_angle = 0.26  # Radians
        self.x_offset = 0.0
        self.yolo_started = False
        self.take_yolo_images_bool = True
        self.max_amount_images_per_z_value = 4
        self.current_amount_images_per_z_value = 0
        self.increasing_z_value = 0.001
        self.max_z_value = 0.7
        self.increasing_x_value = 0.02
        self.increasing_y_value = 0.02
        self.z_increased = 0
        self.current_pose = Pose()
        self.original_pose = Pose()
        self.asked_current_pose = False

        self.take_yolo_images()

    def take_yolo_images(self):
        self.get_logger().info('Taking Yolo Images function started')
        if not self.yolo_started:
            if not self.manual_start:
                self.get_logger().info('Drive_to_start function started')
                pose = self.create_pose(0.3472, 0.5, 1.3, 3.141, 0.0, 0.0)
                self.target_pose_publisher.publish(pose)
            else:
                self.trigger.data = True
                self.asked_current_pose = True
                self.ask_current_pose_publisher.publish(self.trigger)


    def change_pose_for_yolo(self, current_pose):
        next_pose = Pose()
        pose_array = PoseArray()
        message = String()
        message.data = self.yolo_object
        next_pose = current_pose

        if self.current_amount_images_per_z_value < self.max_amount_images_per_z_value:
            if self.current_amount_images_per_z_value == 0:
                if not self.yolo_started:
                    self.original_pose = current_pose
                    self.yolo_started = True
            elif self.current_amount_images_per_z_value == 1:
                next_pose.position.x += self.increasing_x_value
                next_pose.position.y += self.increasing_y_value
                sleep(1)  # Wait for 1 second before proceeding
                self.take_image_publisher.publish(message)
            elif self.current_amount_images_per_z_value == 2:
                next_pose.position.x -= 2 * self.increasing_x_value
                next_pose.position.y -= 2 * self.increasing_y_value
                sleep(1)  # Wait for 1 second before proceeding
                self.take_image_publisher.publish(message)
                

            self.current_amount_images_per_z_value += 1
        else:
            self.z_increased += 1
            yaw = 0.3 * self.z_increased
            next_pose = self.create_pose(
                current_pose.position.x, 
                current_pose.position.y, 
                current_pose.position.z, 
                roll=3.141, 
                pitch=0,
                yaw=0.5 * self.z_increased
            )
            next_pose.position.z = self.original_pose.position.z + self.increasing_z_value
            next_pose.position.x = self.original_pose.position.x
            next_pose.position.y = self.original_pose.position.y
            self.increasing_x_value += (0.001 * self.z_increased) / 2
            self.increasing_y_value += (0.001 * self.z_increased) / 2
            self.increasing_y_value = -self.increasing_y_value
            self.increasing_z_value += (self.z_increased * 0.005)
            sleep(1)  # Wait for 1 second before proceeding
            self.take_image_publisher.publish(message)
            self.current_amount_images_per_z_value = 0

        # Populate PoseArray with the current and next poses
        pose_array.poses.append(current_pose)
        pose_array.poses.append(next_pose)
        pose_array.header.stamp = self.get_clock().now().to_msg()
        pose_array.header.frame_id = 'world'
        #self.target_pose_publisher.publish(next_pose)
        self.target_pose_array_publisher.publish(pose_array)

        self.get_logger().info('Published PoseArray with current and next Pose for Yolo')

    def drive_to_start(self):
        self.get_logger().info('Drive_to_start function started')
        self.x_offset = 0.0573 - math.sin(self.camera_angle) * 0.1
        pose = self.create_pose(0.3472, -0.4, 1.15, 3.141, 0.0, 0.0)
        self.target_pose_publisher.publish(pose)

    def current_pose_callback(self, msg):
        if self.asked_current_pose:
            self.get_logger().info('current_pose_function called')
            self.current_pose = msg

            self.get_logger().info(f'Current Pose: [position: ({msg.position.x:.2f}, {msg.position.y:.2f}, {msg.position.z:.2f}), '
                                   f'orientation: ({msg.orientation.x:.2f}, {msg.orientation.y:.2f}, {msg.orientation.z:.2f}, {msg.orientation.w:.2f})]')

            self.asked_current_pose = False
            self.change_pose_for_yolo(self.current_pose)

    def target_reached_callback(self, msg):
        self.current_pose = msg
        self.get_logger().info(f'Target reached. Current Pose: [position: ({msg.position.x:.2f}, {msg.position.y:.2f}, {msg.position.z:.2f}), '
                               f'orientation: ({msg.orientation.x:.2f}, {msg.orientation.y:.2f}, {msg.orientation.z:.2f}, {msg.orientation.w:.2f})]')

        self.change_pose_for_yolo(self.current_pose)

    def create_pose(self, x, y, z, roll, pitch, yaw):
        
        pose = Pose()
        pose.position.x = x
        pose.position.y = y
        pose.position.z = z

        quaternion = tf.quaternion_from_euler(roll, pitch, yaw)
        pose.orientation.x = quaternion[0]
        pose.orientation.y = quaternion[1]
        pose.orientation.z = quaternion[2]
        pose.orientation.w = quaternion[3]
        return pose


def main(args=None):
    rclpy.init(args=args)
    node = TakeImages()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
