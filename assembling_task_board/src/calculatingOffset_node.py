from geometry_msgs.msg import Pose, PoseArray
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import os
from yolov8_msgs.msg import InferenceResult
from yolov8_msgs.msg import Yolov8Inference
from std_msgs.msg import Float32MultiArray, String, Bool
import numpy as np
import cv2
from PIL import Image as PILImage
import yaml
import math
from functions.calculate_camera_pose_sim import estimate_camera_pose

bridge = CvBridge()



class Offset(Node):

    def __init__(self):     
        super().__init__('offset')
        
        self.create_subscription(Float32MultiArray, '/Pose_inference_result', self.yolo_callback, 10)
        self.aks_current_pose_pub_ = self.create_publisher(Bool, "/ask_current_pose", 1)
        self.trigger_yolo_publisher = self.create_publisher(String, 'GetPositionPublisher', 10)

        self.create_subscription(Pose, '/current_pose_publisher', self.currentPose_callback, 10)
        self.create_subscription(
            Bool, "/user_input", self.user_input_callback, 10
        )
        self.image_subscription = self.create_subscription(
            Image,
            '/camera/camera/color/image_raw',
            self.camera_callback,
            10)
        
        # --- Change Object and Path here ---
        self.current_object = "Ket_Square_8_Female"

        self.folder_path = f"/home/alex/workspaces/final/src/assembling_task_board/Task Board information/Zielposen/{self.current_object}"
        self.yaml_path = "/home/alex/workspaces/final/src/assembling_task_board/Task Board information/Object Information/Object Information.yaml"
        self.current_height = None
        self.current_pose = None
        self.current_yaw = None
        self.specific_offset_x = 0
        self.specific_offset_y = 0
        self.get_yolo_data = False
        self.superior_object = None
        self.Task_board_height = 1.035
        self.Object_loader_height = 1.015
        self.effective_height = None
        self.take_image = False


    def quaternion_to_euler(self, x, y, z, w):
        yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        return yaw

    def currentPose_callback(self, msg):
        self.current_pose = msg
        self.current_height = msg.position.z
        self.current_yaw = self.quaternion_to_euler(msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w)
        if self.superior_object == "Object_loader":
            self.effective_height = self.current_pose.position.z - self.Object_loader_height
        elif self.superior_object == "Task_board":
            self.effective_height = self.current_pose.position.z - self.Task_board_height
        self.get_yolo_data = True
        msg = String()
        msg.data = self.current_object
        self.trigger_yolo_publisher.publish(msg)
        self.get_logger().info(f"setting get_yolo_data to True")

    def yolo_callback(self, msg):
        if self.get_yolo_data == True:
            self.get_yolo_data = False
            print(f"yolo data: {msg.data}")
            if len(msg.data) >= 4:
                self.get_logger().info(f"yolo data: {msg.data}")
                self.last_bounding_box = msg.data[:4]  # Store the bounding bo
                self.calculate_offset()

    def camera_callback(self, data):
        if self.take_image == True:
            self.take_image = False
            # Convert ROS image message to OpenCV format
            bridge = CvBridge()
            img = bridge.imgmsg_to_cv2(data, "bgr8")

            
            os.makedirs(self.folder_path, exist_ok=True)  # Ensure the folder exists

            image_name = f"{self.current_object}.png"
            image_path = os.path.join(self.folder_path, image_name)

            # Save the image
            cv2.imwrite(image_path, img)
            self.get_logger().info(f"Image saved at: {image_path}")

    def user_input_callback(self, msg):
        msg = Bool()
        msg.data = True
        self.aks_current_pose_pub_.publish(msg)
        self.get_logger().info(f"Asking current pose")
        # Load the YAML file
        with open(self.yaml_path, "r") as file:
            data = yaml.safe_load(file)
        # Access the position data
        self.superior_object = data[self.current_object].get('superior_pos', [])[0]
        self.take_image = True
        
    def calculate_offset(self):

        self.get_logger().info(f"Height: {self.current_pose.position.z}, effective height: {self.effective_height}")

        # Load the YAML file
        with open(self.yaml_path, "r") as file:
            data = yaml.safe_load(file)

        # Ensure the object exists in the YAML data
        if self.current_object not in data:
            data[self.current_object] = {}

        # Update translation_data
        data[self.current_object]["Ground_Truth_BBox"] = [self.last_bounding_box[0], self.last_bounding_box[1], self.last_bounding_box[2], self.last_bounding_box[3]]

        # Update specific_height, ensuring the key exists
        if "specific_height" not in data[self.current_object]:
            data[self.current_object]["specific_height"] = [float(self.effective_height), 0.05]
        else:
            data[self.current_object]["specific_height"][0] = float(self.effective_height)
        
        image_name = f"{self.current_object}.png"
        image_path = os.path.join(self.folder_path, image_name)

        # Save the image path to YAML
        data[self.current_object]["Zielpose"] = image_path

        # Save the modified YAML file
        with open(self.yaml_path, "w") as file:
            yaml.safe_dump(data, file, default_flow_style=False)

        self.get_logger().info(f"Updated YAML: specific_offset={data[self.current_object]['specific_offset']}, specific_height={data[self.current_object]['specific_height']}")
        

def main(args=None):
    rclpy.init(args=None)
    offset = Offset()
    rclpy.spin(offset)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
