#!/usr/bin/env python3

from ultralytics import YOLO
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import os
from yolov11_msgs.msg import InferenceResult
from yolov11_msgs.msg import Yolov11Inference
from std_msgs.msg import Float32MultiArray, String
import cv2
import ament_index_python.packages

bridge = CvBridge()
package_name = 'yolov11'  # Replace with your package name
package_path = ament_index_python.packages.get_package_share_directory(package_name)
class CameraSubscriber(Node):

    def __init__(self):
        super().__init__('camera_subscriber')
        yolo_path = os.path.join(package_path, 'scripts','active_Model', 'active.pt')
        self.model = YOLO(yolo_path)

        self.process_images = False
        self.positions = []
        self.max_attempts = 5
        self.counter = 0 
        self.j = 0
        self.i = 0
        self.take_image = False
        self.object = None
        self.take_pose_image = False
        self.Assembly_called = False

        self.get_position_subscription = self.create_subscription(
            String,  
            '/GetPositionPublisher',
            self.publish_position_callback,
            10)
        
        self.get_position_subscription = self.create_subscription(
            String,  
            '/GetPositionPublisher_Assembly',
            self.publish_position_Assembly_callback,
            10)

        self.image_subscription = self.create_subscription(
            Image,
            '/camera/camera/color/image_raw',
            self.camera_callback,
            10)
        
        self.take_image = self.create_subscription(
            String,
            'take_image',
            self.take_image_callback,
            10)
        
        self.take_pose_image = self.create_subscription(
            String,
            'take_pose_image',
            self.take_pose_image_callback,
            10)

        self.img_pub = self.create_publisher(Image, "/inference_result", 1)
        self.results_publisher = self.create_publisher(Float32MultiArray, "/Pose_inference_result", 1)
        self.results_publisher_Assembly = self.create_publisher(Float32MultiArray, "/Pose_inference_result_Assembly", 1)
        self.results_publisher_CNN = self.create_publisher(Float32MultiArray, "/For_CNN_inference_result", 1)

    

    def publish_position_callback(self, msg):
        self.get_logger().info("YOLO has been called")
        self.object = msg.data
        self.process_images = True
        self.counter = 0
        self.positions.clear()
    
    def publish_position_Assembly_callback(self, msg):
        self.get_logger().info("YOLO has been called by Assembly")
        self.object = msg.data
        self.process_images = True
        self.counter = 0
        self.Assembly_called = True
        self.positions.clear()

    def take_image_callback(self, msg):
        # Convert ROS Image message to OpenCV image
        self.get_logger().info(f'Take Image called')
        self.take_image = True
        self.object = msg.data
        print(self.object)

    def take_pose_image_callback(self, msg):
        # Convert ROS Image message to OpenCV image (assuming msg.data is the string)
        self.take_pose_image = True
        self.massage = msg.data  # Assuming msg.data is the string you are passing
        # Split the string using ';' as the delimiter
        split_message = self.massage.split(";")
        # Extract pose and object from the split result
        self.pose = split_message[0]
        self.object = split_message[1]
       
    def take_images(self, img):
        # Define folder name based on the object
        if self.object is None or self.object.strip() == "":
            self.get_logger().error("test Object name is None or empty. Cannot save image.")
            return
            
        folder_name = self.object
        folder_path = os.path.join('/home/alex/Images_for_yolo/Automated_Gazebo_Images/YOLO/Original_Gazebo', folder_name)
        print(folder_path)
        
        # Check if the folder exists, if not, create it
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # Count how many files are already in the folder
        self.i = len([name for name in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, name))])
        # Generate a file name using the current count

        
        file_name = f'{self.object}_{self.i}.png'
        file_path = os.path.join(folder_path, file_name)

        # Save the image
        cv2.imwrite(file_path, img)
        self.get_logger().info(f'Image saved at {file_path}')

        # Increment file count and reset the flag
        self.i += 1
        self.take_image = False

    def take_pose_images(self, img):
        if self.object is None or self.object.strip() == "":
            self.get_logger().error("dd Object name is None or empty. Cannot save image.")
            return

        folder_name = self.object
        folder_path = os.path.join('/home/alex/Images_for_yolo/Automated_Gazebo_Images/Ignition/Pose_CNN', folder_name)
        print(folder_path)

        # Check if the folder exists, if not, create it
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # Generate a file name using the current pose
        images_folder_name = os.path.join(folder_path, 'images')
        if not os.path.exists(images_folder_name):
            os.makedirs(images_folder_name)
        
        current_pose = self.pose.split(",")
        current_rot = current_pose[2]
        current_hight = current_pose[3]
        
        base_file_name = f'{self.object}_{current_rot}_{current_hight}.png'
        file_name = base_file_name
        file_path = os.path.join(images_folder_name, file_name)
        
        # Check if the file already exists and append '_i' if necessary
        counter = 1
        while os.path.exists(file_path):
            file_name = f'{self.object}_{current_rot}_{current_hight}_{counter}.png'
            file_path = os.path.join(images_folder_name, file_name)
            counter += 1

        # Save the image
        cv2.imwrite(file_path, img)
        self.get_logger().info(f'Image saved at {file_path}')

        # Save the pose in the 'YOLO_Pose' folder
        pose_folder_name = os.path.join(folder_path, 'poses')
        if not os.path.exists(pose_folder_name):
            os.makedirs(pose_folder_name)
        
        base_pose_file_name = f'{self.object}_{current_rot}_{current_hight}.txt'
        pose_file_name = base_pose_file_name
        pose_file_path = os.path.join(pose_folder_name, pose_file_name)

        # Check if the pose file already exists and append '_i' if necessary
        counter = 1
        while os.path.exists(pose_file_path):
            pose_file_name = f'{self.object}_{current_rot}_{current_hight}_{counter}.txt'
            pose_file_path = os.path.join(pose_folder_name, pose_file_name)
            counter += 1

        # Save the pose
        with open(pose_file_path, 'w') as file:
            file.write(self.pose)

        self.get_logger().info(f'Pose saved at {pose_file_path}')

        # Increment file count and reset the flag
        self.i += 1
        self.take_pose_image = False

    def camera_callback(self, data):
        self.j += 1
        img = bridge.imgmsg_to_cv2(data, "bgr8")

        if self.take_image == True:
            self.get_logger().info(f'Calling take image')
            self.take_images(img)

        if self.take_pose_image == True:
            self.get_logger().info(f'Calling take pose image')
            self.take_pose_image = False
            self.take_pose_images(img)
            
        # YOLOv8 inference
        results = self.model(img)
        best_detection = None
        

        all_detections = []  # List to store all detection data

        for r in results:
            boxes = r.boxes

            for box in boxes:
                b = box.xyxy[0].to('cpu').detach().numpy().copy()  # Get box coordinates
                c = box.cls
                detected_class_name = self.model.names[int(c)]
                confidence = box.conf.item()  # Get confidence score

                # Store all detections matching self.object
                if detected_class_name == self.object:
                    center_x = (b[0] + b[2]) / 2.0
                    center_y = (b[1] + b[3]) / 2.0
                    length = b[2] - b[0]
                    height = b[3] - b[1]

                    # Store detection data for publishing
                    all_detections.extend([center_x, center_y, length, height, confidence])

                    # Accumulate position information for averaging
                    self.counter += 1
                    self.positions.append((center_x, center_y, length, height))


        annotated_frame = results[0].plot()
        img_msg = bridge.cv2_to_imgmsg(annotated_frame)
        self.img_pub.publish(img_msg)


        # Return if no images need processing
        if not self.process_images:
            return
        
        if all_detections:
            result_array = Float32MultiArray(data=all_detections)
            self.results_publisher_Assembly.publish(result_array)
            self.results_publisher.publish(result_array)
      
        
if __name__ == '__main__':
    rclpy.init(args=None)
    print("starting")
    camera_subscriber = CameraSubscriber()
    rclpy.spin(camera_subscriber)
    rclpy.shutdown()
