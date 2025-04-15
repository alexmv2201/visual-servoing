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
import torch
from lightglue import LightGlue, SuperPoint
from lightglue.utils import load_image, rbd
import yaml
import ament_index_python.packages

bridge = CvBridge()

package_name = 'assembling_task_board'  # Replace with your package name
package_path = ament_index_python.packages.get_package_share_directory(package_name)

class Lightning(Node):

    def __init__(self):     
        super().__init__('lightning')

        self.img_pub = self.create_publisher(Image, "/matched_image", 1)
        self.results_publisher = self.create_publisher(Float32MultiArray, "/Lightning_rot_est", 1)

        self.create_subscription(Float32MultiArray, '/Pose_inference_result', self.yolo_callback, 10)

        self.create_subscription(Bool, '/calulateMatchedRotation', self.match_images, 10)

        self.selfPublisher = self.create_publisher(Bool, '/calulateMatchedRotation', 10)

        self.trigger_yolo_publisher = self.create_publisher(String, 'GetPositionPublisher_Assembly', 10)

        self.current_object_subscription = self.create_subscription(
            String,  
            '/current_object',
            self.current_object_callback,
            10)
        
        self.image_subscription = self.create_subscription(
            Image,
            '/camera/camera/color/image_raw',
            self.camera_callback,
            10)
    
        self.bool_match_images = False
        self.yolo_result = None
        self.current_object = None
        self.ground_Truth_data = None
        self.get_Camera_image = False
        self.image_path = None
        self.valid_Bbox_data = None
        self.bbox_data = None
        self.last_bbox_data = []
        self.center_x = None
        self.center_y = None
        self.x_min, self.y_min, self.x_max, self.y_max = None, None, None, None
        self.x_min_2, self.y_min_2, self.x_max_2, self.y_max_2 = None, None, None, None

    def match_images(self, msg):
        if self.valid_Bbox_data == True:
            self.get_logger().info(f"Match images")
            self.bool_match_images = True
            msg = String()
            msg.data = self.current_object
            self.trigger_yolo_publisher.publish(msg)
        else:
            self.get_logger().error(f"No Bbox data")
    
    def current_object_callback(self, msg):
        try: 
            self.last_bbox_data = []
            self.current_object = msg.data
            yaml_file_path = os.path.join(package_path, 'Task Board information', 'Object Information', 'Object Information.yaml')
            with open(yaml_file_path, "r") as file:
                data = yaml.safe_load(file)

            self.bbox_data = data[self.current_object].get('Ground_Truth_BBox', [0])
            self.center_x, self.center_y, width, height = self.bbox_data
            self.x_min = self.center_x - (width / 2)
            self.x_max = self.center_x + (width / 2)
            self.y_min = self.center_y - (height / 2)
            self.y_max = self.center_y + (height / 2)
            image_path = data[self.current_object].get('Zielpose', [0])
            relative_index = image_path.find("Task Board information")
            relative_path = image_path[relative_index:]
            new_path = os.path.join(package_path, relative_path)
            new_path = os.path.join(package_path, relative_path.lstrip("/"))
            self.image_path = new_path

            self.get_logger().info(f"Current Object: {msg.data} got this bbox data: {self.bbox_data}")
            self.valid_Bbox_data = True
        except:
            self.get_logger().error(f"Error in calling_lightning: No Bbox data")
            self.valid_Bbox_data = False

    def extract_yolo_data(self, msg):
        # Parse YOLO data
        data = msg.data
        num_objects = len(data) // 5  # Each object has (x, y, w, h, prob)

        objects = [(data[i * 5], data[i * 5 + 1], data[i * 5 + 2], data[i * 5 + 3], data[i * 5 + 4])
                for i in range(num_objects)]

        # Step 1: Get the ground truth bounding box center
        if self.last_bbox_data == []:
            gt_center_x = self.center_x
            gt_center_y = self.center_y


        # Step 2: Find the object closest to the ground truth bounding box
        best_object = None
        if self.current_object == "Gear_Shaft":
            best_distance = 20  # Define maximal distance to ground truth bbox center
        elif self.current_object == "Waterproof_Female":
            best_distance = 120 # Define maximal distance to ground truth bbox center
        else:
            best_distance = 45 # Define maximal distance to ground truth bbox center

        for obj in objects:
            bbox_center_x, bbox_center_y, _, _, prob = obj

            # Compute Euclidean distance between YOLO bbox center and the ground truth bbox center
            dist = np.sqrt((bbox_center_x - gt_center_x) ** 2 + (bbox_center_y - gt_center_y) ** 2)
            self.get_logger().info(f"dist: {dist} GT center x: {gt_center_x} bbox x: {bbox_center_x} GT center y: {gt_center_y} bbox x: {bbox_center_y}")
            # If this object is closer to the ground truth bbox center, update the best object
            if dist < best_distance:
                self.get_logger().info(f"New best dist {dist}")
                best_distance = dist
                best_object = obj

        return best_object  # Return the (x, y, w, h, prob) of the selected object

    def yolo_callback(self, msg):
        if self.bool_match_images:
            self.get_logger().info(f"Yolo data: {msg}")
            self.bool_match_images = False
            self.yolo_result = self.extract_yolo_data(msg)  # Pass msg to extract data
            if self.yolo_result is None:
                self.get_logger().info(f"No good yolo match")
                msg = String()
                msg.data = self.current_object
                self.trigger_yolo_publisher.publish(msg)
                self.bool_match_images = True
                return
            x, y, width, height = self.yolo_result[:4]
            self.x_min_2 = x - width // 2
            self.x_max_2 = x + width // 2
            self.y_min_2 = y - height // 2
            self.y_max_2 = y + height // 2
            self.get_logger().info(f"Current Object: {msg.data} got this x_min_2: {self.x_min_2}")
            self.get_Camera_image = True
    
    def camera_callback(self, Image):
        if self.get_Camera_image == True:
            self.get_Camera_image = False
            img = bridge.imgmsg_to_cv2(Image, "bgr8")
            self.calling_lightning(img)


    def calling_lightning(self, image1):
        try:
            self.get_logger().info(f"Will try to match. Yolo result:")
            crop_width = 380
            crop_height = 478
            safety = 5
            # Center of the crop (320, 300)
            center_x, center_y = 320, 240
            x_offset = center_x - (crop_width / 2)
            y_offset = center_y - (crop_height / 2)

            # Define bounding boxes
            x_min = self.x_min - x_offset - safety
            x_max = self.x_max - x_offset + safety
            y_min = self.y_min - y_offset - safety
            y_max = self.y_max - y_offset + safety
            x_min_2 = self.x_min_2 - x_offset - safety
            x_max_2 = self.x_max_2 - x_offset + safety
            y_min_2 = self.y_min_2 - y_offset - safety
            y_max_2 = self.y_max_2 - y_offset + safety

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # Read the image
            image0 = cv2.imread(self.image_path)

            def apply_bounding_box(image, x_min, x_max, y_min, y_max):
                """
                Apply a bounding box mask to the image, making everything outside the box white.
                """
                # Ensure coordinates are within bounds and cast them to integers
                x_min, x_max = int(max(0, x_min)), int(min(image.shape[1], x_max))
                y_min, y_max = int(max(0, y_min)), int(min(image.shape[0], y_max))

                # Create a mask where the region inside the bounding box is True (1) and outside is False (0)
                mask = np.zeros(image.shape[:2], dtype=np.uint8)  # Only need 2D mask (height x width)

                # Apply the bounding box to the mask
                mask[y_min:y_max, x_min:x_max] = 1  # Set the area inside the bounding box to 1, outside to 0

                # Set the pixels outside the bounding box to white (255 for RGB)
                # Use broadcasting to apply the mask to all color channels
                image[mask == 0] = [255, 255, 255]  # Set outside pixels to white for RGB images

                return image

            def numpy_image_to_torch(image: np.ndarray) -> torch.Tensor:
                """Normalize the image tensor and reorder the dimensions."""
                if image.ndim == 3:
                    image = image.transpose((2, 0, 1))  # HxWxC to CxHxW
                elif image.ndim == 2:
                    image = image[None]  # add channel axis
                else:
                    raise ValueError(f"Not an image: {image.shape}")
                return torch.tensor(image / 255.0, dtype=torch.float)
            
            # Calculate the top-left and bottom-right corner for cropping
            x1 = center_x - crop_width // 2
            y1 = center_y - crop_height // 2
            x2 = x1 + crop_width
            y2 = y1 + crop_height    
            image0 = image0[y1:y2, x1:x2]

            # Apply the bounding box to image0
            image0 = apply_bounding_box(image0, x_min, x_max, y_min, y_max)
            
            # Convert to PyTorch tensor
            image0 = numpy_image_to_torch(image0).to(device)

            # Crop the image
            image1 = image1[y1:y2, x1:x2]

            # Apply the second bounding box to image1
            image1 = apply_bounding_box(image1, x_min_2, x_max_2, y_min_2, y_max_2)

            # Convert cropped image to PyTorch tensor
            image1 = numpy_image_to_torch(image1).to(device)

            extractor = SuperPoint(max_num_keypoints=1000).eval().to(device)
            matcher = LightGlue(features="superpoint").eval().to(device)
            # Extract Features & Match Keypoints**
            feats0 = extractor.extract(image0)
            feats1 = extractor.extract(image1)
            matches01 = matcher({"image0": feats0, "image1": feats1})

            ## Remove batch dimension
            feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]

            # Extract keypoints & matches
            kpts0, kpts1, matches = feats0["keypoints"], feats1["keypoints"], matches01["matches"]
            m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]

            # Define ROI and Check if Keypoints are Within the ROI**
            safety_2 = 3
            print(x_min, x_max, y_min, y_max, x_min_2, x_max_2, y_min_2, y_max_2)
            
            def is_within_roi(kpt, roi_x1, roi_y1, roi_x2, roi_y2):
                """Check if the keypoint is within the ROI."""
                x, y = kpt
                return roi_x1 <= x <= roi_x2 and roi_y1 <= y <= roi_y2

            # Filter Matches Based on ROI**
            filtered_matches = []
            for i, (kpt0, kpt1) in enumerate(zip(m_kpts0.cpu().numpy(), m_kpts1.cpu().numpy())):
                if is_within_roi(kpt0, x_min+safety_2, y_min+safety_2, x_max-safety_2, y_max-safety_2) and is_within_roi(kpt1, x_min_2, y_min_2, x_max_2, y_max_2):
                    filtered_matches.append((matches[i, 0].item(), matches[i, 1].item()))

            print(f"m_kpts0 size: {m_kpts0.shape}")
            print(f"m_kpts1 size: {m_kpts1.shape}")
            print(filtered_matches)
            # Extract Filtered Keypoints**
            filtered_kpts0 = [kpts0[i[0]].cpu().numpy() for i in filtered_matches]
            filtered_kpts1 = [kpts1[i[1]].cpu().numpy() for i in filtered_matches]

            # Convert PyTorch Tensors to NumPy for Visualization**
            def tensor_to_numpy(image):
                """Convert PyTorch tensor (3,H,W) to NumPy (H,W,3)"""
                return (image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

            img0_np = tensor_to_numpy(image0)
            img1_np = tensor_to_numpy(image1)

            def draw_matches_and_publish(image0, image1, kpts0, kpts1, matches, img_pub):
                """Draw matches between two images and publish the annotated image to a ROS 2 topic"""
                
                # Stack images side by side
                combined_image = np.hstack((image0, image1))
                
                # Draw matches between keypoints
                for (x0, y0), (x1, y1) in zip(kpts0, kpts1):
                    # Draw lines between matching keypoints
                    cv2.line(combined_image, (int(x0), int(y0)), (int(x1 + image0.shape[1]), int(y1)), (0, 255, 0), 2)  # green lines
                    # Draw circles at the keypoints
                    cv2.circle(combined_image, (int(x0), int(y0)), 5, (0, 255, 0), -1)  # green circle on image0
                    cv2.circle(combined_image, (int(x1 + image0.shape[1]), int(y1)), 5, (0, 255, 0), -1)  # green circle on image1
                
                # Convert the image to a ROS message
                bridge = CvBridge()
                img_msg = bridge.cv2_to_imgmsg(combined_image, "bgr8")  # Use the appropriate encoding here
                
                # Publish the annotated image
                img_pub.publish(img_msg)

            # Call the function to draw the matches**
            draw_matches_and_publish(img0_np, img1_np, filtered_kpts0, filtered_kpts1, filtered_matches, self.img_pub)

            # Compute Rotation Matrix using Homography**
            pts0 = np.float32(filtered_kpts0)
            pts1 = np.float32(filtered_kpts1)
            affine_matrix, inliers = cv2.estimateAffine2D(pts0, pts1, ransacReprojThreshold=5.0)

            # Calculate the translations for each point pair
            translations = pts1 - pts0  # Shape: (N, 2)

            # Calculate the initial median and standard deviation for x and y
            median_x = np.median(translations[:, 0])
            median_y = np.median(translations[:, 1])
            std_x = np.std(translations[:, 0])
            std_y = np.std(translations[:, 1])

            if std_x == 0 and std_y == 0:
                final_median_x, final_median_y = 0, 0
            else:
                # Filter out points that are above the standard deviation
                filtered_translations = translations[
                    (np.abs(translations[:, 0] - median_x) <= std_x) &
                    (np.abs(translations[:, 1] - median_y) <= std_y)
                ]
                final_median_x = np.median(filtered_translations[:, 0])
                final_median_y = np.median(filtered_translations[:, 1])
            
            # Extract the rotation angle from the affine transformation matrix
            rotation_matrix = affine_matrix[:2, :2]
            angle = np.degrees(np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0]))
            self.get_logger().info(f"Estimated rotation angle: {angle} degrees")
            self.get_logger().info(f"Verschiebung x richtung: {final_median_x}, Verschiebung y-achse {final_median_y}")
            rotation_msg = Float32MultiArray(data = [angle, final_median_x, final_median_y])

            self.results_publisher.publish(rotation_msg)

        except Exception as e:
            msg = Bool()
            msg.data = True
            self.get_logger().error(f"Error in calling_lightning: {e} calling it again")
            self.selfPublisher.publish(msg)

def main(args=None):
    rclpy.init(args=None)
    lightning = Lightning()
    rclpy.spin(lightning)
    rclpy.shutdown()

if __name__ == '__main__':
    main()