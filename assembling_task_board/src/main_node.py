import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose, PoseArray
from std_msgs.msg import String, Bool, Float32MultiArray, Float32
import math
import time
from functions.calculate_camera_pose import estimate_camera_pose, estimate_translation
import numpy as np
import tf_transformations as tf
from sensor_msgs.msg import JointState
import yaml
from collections import deque
import subprocess
from controller_manager_msgs.srv import SwitchController
from geometry_msgs.msg import Quaternion
from tf_transformations import quaternion_from_euler, quaternion_multiply
import csv
import os
import ament_index_python.packages

package_name = 'assembling_task_board'  # Replace with your package name
package_path = ament_index_python.packages.get_package_share_directory(package_name)

class MainNode(Node):

    def __init__(self):
        super().__init__('main_node')

        # Publishers
        self.target_pose_publisher = self.create_publisher(Pose, 'joint_space', 10)
        self.target_joints_publisher = self.create_publisher(JointState, 'target_joints', 10)
        self.target_pose_array_publisher = self.create_publisher(PoseArray, 'task_space', 10)
        self.ask_current_pose = self.create_publisher(Bool, 'ask_current_pose', 10)
        self.publisher_take_image = self.create_publisher(String, 'take_image', 10)
        self.publisher = self.create_publisher(String, 'next_object', 10)
        self.trigger_yolo_publisher = self.create_publisher(String, 'GetPositionPublisher_Assembly', 10)
        self.trigger_edge_detector_publisher = self.create_publisher(Bool, 'calulateExactPosition', 10)
        self.joint_publisher = self.create_publisher(Float32, 'target_wrist_joint', 10)
        self.current_object_pub = self.create_publisher(String, 'current_object', 10)
        self.ending_processImages_pub = self.create_publisher(Bool, '/ending_process_images', 10)
        self.lighning_pub = self.create_publisher(Bool, '/calulateMatchedRotation', 10)
        self.gripper_pub = self.create_publisher(Float32MultiArray, '/gripper_client', 10)
        self.set_speed_scale_pub = self.create_publisher(Float32, '/speed_scale', 10)
        self.twister_pub = self.create_publisher(Float32MultiArray, '/twister_', 10)
        self.control_switcher_publisher = self.create_publisher(String, "/controller_switch_request", 10)

        # Subscriptions
        self.create_subscription(Pose, 'current_pose_publisher', self.current_pose_function, 10)
        self.create_subscription(Pose, 'target_pose_reached', self.target_pose_reached, 10)
        #self.create_subscription(Float32MultiArray, '/Pose_inference_result_Assembly', self.yolo_callback, 10)
        self.create_subscription(Float32MultiArray, '/Pose_inference_result', self.yolo_callback, 10)
        self.create_subscription(Float32MultiArray, '/Lightning_rot_est', self.getLightningRotation_callback, 10)
        #self.create_subscription(Bool, '/gripper_reached', self.gripper_reached_callback, 10)

        self.client = self.create_client(SwitchController, '/controller_manager/switch_controller')
        self.controller = None
    
        # Task board variables
        #self.object_list = ["Waterproof_Male", "Waterproof_Female", "Gear_Large", "Gear_Shaft", "Gear_Medium","Gear_Shaft", "Gear_Small","Gear_Shaft","Ket_Square_12", "Ket_Square_12_Female","USB_Male", "USB_Female","Ket_Square_12", "Ket_Square_12_Female","Ket_Square_8", "Ket_Square_8_Female", "M16_Nut", "M16_Screw"]
        #self.object_list = ["USB_Male", "USB_Female","USB_Male", "USB_Female", "USB_Male", "USB_Female", "USB_Male", "USB_Female", "USB_Male", "USB_Female"]
        #self.object_list = ["Gear_Medium","Gear_Shaft","Gear_Large", "Gear_Shaft", "Gear_Small","Gear_Shaft","M16_Nut", "M16_Screw"]
        self.object_list = ["Ket_Square_12", "USB_Female"]
        self.Task_board_height = 1.033
        self.Object_loader_height = 1.015
        self.image_width = 640
        self.image_height = 480
        self.superior_object = None
        self.driving_to_sup = False
        self.current_object = None
        self.manual_start = False
        self.object_id = 0
        self.current_pose = None
        self.correct_height = None
        self.height_is_correct = False
        self.object_height = None
        self.object_before = None
        self.gripper_open = None
        self.gripper_close = None
        self.manual_offset_x = None
        self.manual_offset_y = None
        self.manual_offset_angle = None
        self.camera_offset_x = None
        self.camera_offset_y = None
        self.camera_offset_angle = None

        # drive to object variables
        self.yolo_finished_before_CNN = False
        self.yolo_finished_after_CNN = False

        self.asked_by_yolo = False
        self.storing_yolo_data = False
        self.estimated_object_pose = None
        self.yolo_data = None
        self.last_yolo_data = None

        self.new_yolo_data = False
        self.new_object = True

        self.vs_tries = 0
        self.got_gripper_callback = False
        self.x_target, self.y_target = None, None
        self.last_call_time = None
        self.asked_by_lightning = False
        self.lightning_finished = False
        
        self.last_small_angle = None  # Store the last small angle value
        self.angle_threshold_counter = 0  # Counter for detecting alternating signs

        self.lightning_close_to_finish = False
        

        # Assembling variables
        self.finished_object = True
        self.rotation = None
        self.calling_gripper_close = False
        self.calling_gripper_open = False
        self.hasObject = False
        self.starting_assembly_traj = False
        self.going_up = False
        self.going_down = True
        self.M16_screw_first = True
        self.Gear_Medium_first = True
        self.Gear_Small_first = True
        
        self.test = False
        self.cutit = False
        # Initialization steps
        self.get_logger().info("Will start other nodes")
        self.callGripper_open()
        msg = String()
        msg.data = "activate:scaled_joint_trajectory_controller;deactivate:forward_position_controller"
        self.control_switcher_publisher.publish(msg)
        self.controller = "joint_trajectory_controller"
        self.pause = False
        #time.sleep(10)
     
        if self.manual_start == False:
            self.current_object = "start_position"
            self.get_logger().info("Drive_to_start function started") 
            self.drive_to_ik()
            
        else:    
            trigger = Bool()
            trigger.data = True
            self.ask_current_pose.publish(trigger)
            
    
    
    def read_meta_data(self):
        # Load the YAML file
        yaml_file_path = os.path.join(package_path, 'Task Board information', 'Object Information', 'Object Information.yaml')

        # Open the YAML file
        with open(yaml_file_path, "r") as file:
            data = yaml.safe_load(file)
        # Access the position data
        specific_height_data = data[self.current_object].get('specific_height', [0])
        rotation_data = data[self.current_object].get('rotation', [0])
        gripper_data = data[self.current_object].get('gripper', [0])
        superior_object = data[self.current_object].get('superior_pos', [])[0]
        bbox_data = data[self.current_object].get('Ground_Truth_BBox', [0])
        manual_offset_data = data[self.current_object].get('Manual_offset', [0])

        if superior_object:
            self.superior_object = superior_object
        else:            
            self.get_logger().error(f"Missing relevant information about object1. Quitting assembly")
            return

        if specific_height_data and len(specific_height_data) >= 2:
            self.specific_height = specific_height_data[0]
            self.specific_height_assembly = specific_height_data[1]
        else:
            self.get_logger().error(f"Missing relevant information about object2. Quitting assembly")
            return
        
        if rotation_data:
            self.rotation = rotation_data[0]
        else:
            rotation_data = None

        if len(gripper_data) == 2:
            self.gripper_open = gripper_data[0]
            self.gripper_close = gripper_data[1]
        else:
            self.get_logger().error(f"Missing relevant information about object3. Quitting assembly")
            return
        
        if len(bbox_data) == 4:
            self.x_target = bbox_data[0]
            self.y_target = bbox_data[1]
        else:
            self.get_logger().error(f"Missing relevant information about object4. Quitting assembly")
            return
        
        if len(manual_offset_data) == 3:
            self.manual_offset_x = manual_offset_data[0]
            self.manual_offset_y = manual_offset_data[1]
            self.manual_offset_angle = manual_offset_data[2]
        elif len(manual_offset_data) == 2:
            self.manual_offset_x = manual_offset_data[0]
            self.manual_offset_y = manual_offset_data[1]
            self.manual_offset_angle = 0
        else:
            self.manual_offset_x = 0
            self.manual_offset_y = 0
            self.manual_offset_angle = 0
        yaml_file_path = os.path.join(package_path, 'Task Board information', 'Camera Parameters', 'Camera.yaml')
        with open(yaml_file_path, "r") as file:
            data = yaml.safe_load(file)

        camera_data = data.get('camera offset', [0])
        if len(camera_data) == 3:
            self.camera_offset_x = camera_data[0]
            self.camera_offset_y = camera_data[1]
            self.camera_offset_angle = camera_data[2]
        
        return True
 
    def read_ik(self, object):
        # Load the YAML file
        yaml_file_path = os.path.join(package_path, 'Task Board information', 'IK solutions','IK solutions.yaml')
        with open(yaml_file_path, "r") as file:
            data = yaml.safe_load(file)

        # Access the position data
        joint_position = data[object].get('joint_positions', [])
        
        print(f"current Object {object}: Joints: {joint_position}")

        return joint_position
      
    def align_robot(self):
        pose = Pose()
        pose.position = self.current_pose.position

        # Set the orientation
        straight_down_quat = tf.quaternion_from_euler(0,np.pi, 0)  # 180 degrees pitch

        # Apply 45-degree rotation around the Z-axis
        rotation_45_z_quat = tf.quaternion_from_euler(0, 0, 0)  # 45 degrees yaw

        # Combine the rotations: quaternion multiplication
        final_quat = tf.quaternion_multiply(rotation_45_z_quat, straight_down_quat)

        # Set the pose orientation
        pose.orientation.x = final_quat[0]
        pose.orientation.y = final_quat[1]
        pose.orientation.z = final_quat[2]
        pose.orientation.w = final_quat[3]

        self.publish_pose(pose)
        
    def drive_to_ik(self):
        speed_scale = Float32(data=1.0)
        self.set_speed_scale_pub.publish(speed_scale)
        if self.driving_to_sup == True:
            self.get_logger().info(f"Driving to superior Position")
            object = self.superior_object
        else:
            self.get_logger().info(f"Driving to {self.current_object}")
            object = self.current_object
        joints = self.read_ik(object)
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]  # Adjust the joint names accordingly
        msg.position = joints  # Use the list of joint positions directly
        self.target_joints_publisher.publish(msg)

    def publish_pose(self, pose: Pose):
        self.get_logger().info(f"Received Pose: x: {pose.position.x}, y: {pose.position.y}, z: {pose.position.z}, "
                                f"xx: {pose.orientation.x}, yy: {pose.orientation.y}, zz: {pose.orientation.z}, "
                                f"ww: {pose.orientation.w}")
        self.target_pose_publisher.publish(pose)

    def request_next_object(self):
        self.vs_tries = 0
        self.estimated_object_pose = None
        self.height_is_correct = False
        self.current_object = self.object_list[self.object_id]
        msg = String()
        msg.data = self.current_object
        self.current_object_pub.publish(msg)
        self.get_logger().info(f"Requesting next Object {self.current_object}")
        if self.object_id == 0:
            self.object_before = "start_position"
        else:
            self.object_before = self.object_list[self.object_id-1]
        self.driving_to_sup = True
        succes = self.read_meta_data()
        if succes != True:
            return
        self.finished_object = False
        self.new_object = True
        self.object_id += 1
        self.vs_tries = 0
        self.CNN_finished = False
        self.lightning_finished = False
        self.Rotation_finished = False
        self.going_down = True
        if self.superior_object == "Object_loader":
            self.callGripper_open()
        else:
            self.drive_to_ik()
  
    def normalize_angle(self, angle):
        """Normalize angle to be within [-pi, pi]."""
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def cartesian_correct_height(self):
        self.get_logger().info(f"Correcting height to height {self.object_height}")
        self.height_is_correct = True
        pose_array = PoseArray()
        new_pose = self.current_pose
        new_pose.position.z = self.object_height 
        # Add current pose to the PoseArray
        pose_array.poses.append(self.current_pose)
        
        pose_array.poses.append(new_pose)
        
        # Set the header for the PoseArray
        pose_array.header.stamp = self.get_clock().now().to_msg()
        pose_array.header.frame_id = "base_link"  # Replace with the appropriate frame_id


        self.asked_by_lightning = True
        
        self.target_pose_array_publisher.publish(pose_array)

    def getLightningRotation_callback(self, msg):
        time.sleep(0.1)
        self.vs_tries += 1
        self.asked_by_lightning = True
        if math.isnan(msg.data[0]) or math.isnan(msg.data[1]):
            self.current_task_distributor()
            return
        if self.superior_object == "Object_loader":
            bottom_height = self.Object_loader_height
        elif self.superior_object == "Task_board":
            bottom_height = self.Task_board_height
        effective_height = self.current_pose.position.z - bottom_height + 0.09

        roll, pitch, yaw = self.quaternion_to_euler(self.current_pose.orientation.x, self.current_pose.orientation.y, self.current_pose.orientation.z, self.current_pose.orientation.w)
        
        # Define thresholds for stopping condition
        angle_threshold = 0.02 # Threshold for angle close to zero
        position_threshold = 0.001  # Threshold for x and y position differences
        if self.rotation:
            speed_scale = Float32(data=1.0)
            self.set_speed_scale_pub.publish(speed_scale)
            angle = msg.data[0] / 180 * np.pi
            self.get_logger().info(f"Current Bauteil: {self.current_object} Angle: {msg.data[0]}")
            new_angle = Float32()
            new_angle.data = angle
            buffer_yaw = yaw - angle
        else:
            angle = 0
            self.last_small_angle = 0
            buffer_yaw = yaw - angle

        if abs(angle) > 0.2:
            if abs(angle) > 1.0:
                self.get_logger().info(f"Zu großer Winkel")
                new_angle.data = 0.0
            self.get_logger().info(f"Angle über 0.1. Keine Translation")
            new_pose = Pose()
            new_pose.position = self.current_pose.position
            new_quat = tf.quaternion_from_euler(roll, pitch, buffer_yaw)

            # Assign new quaternion
            new_pose.orientation.x = new_quat[0]
            new_pose.orientation.y = new_quat[1]
            new_pose.orientation.z = new_quat[2]
            new_pose.orientation.w = new_quat[3]
            self.current_pose = new_pose
            self.process_waypoints(self.current_pose, new_pose)

        else:
            t_x = msg.data[1]
            t_y = msg.data[2]

            self.get_logger().info(f"Manual offset:x {self.manual_offset_x} y: {self.manual_offset_y}")
            X_w, Y_w = estimate_translation(t_x , t_y, effective_height, yaw, self.manual_offset_x, self.manual_offset_y)
            new_pose = Pose()
            new_pose.position = self.current_pose.position
            new_pose.position.x = self.current_pose.position.x + X_w
            new_pose.position.y = self.current_pose.position.y - Y_w

            # Convert updated Euler angles (roll, pitch, new_yaw) back to quaternion
            new_quat = tf.quaternion_from_euler(roll, pitch, buffer_yaw)

            # Assign new quaternion
            new_pose.orientation.x = new_quat[0]
            new_pose.orientation.y = new_quat[1]
            new_pose.orientation.z = new_quat[2]
            new_pose.orientation.w = new_quat[3]

            self.current_pose = new_pose

            self.get_logger().info(f"Verschiebung x richtung real coord: {X_w}, Verschiebung y-achse {Y_w}, t_x {t_x} ")
            
            #Check if the angle and movement are both small enough to stop
            if abs(angle) < angle_threshold and abs(X_w) < position_threshold and abs(Y_w) < position_threshold:
                self.get_logger().info(f"Detected small angle: {angle}")

                if (self.last_small_angle is not None and np.sign(self.last_small_angle) != np.sign(angle)) or not self.rotation:
                    # We have detected one positive and one negative small angle in a row
                    mean_correction = (self.last_small_angle + angle) / 2  # Calculate mean
                    
                    # Adjust the final angle
                    final_angle = angle + mean_correction + self.camera_offset_angle +self.manual_offset_angle
                    self.get_logger().info(f"Final angle adjustment: {final_angle}")
                    self.get_logger().info(f"Yaw: {yaw}, offset_angle {self.camera_offset_angle}")
                    buffer_yaw = yaw - final_angle
                    new_quat = tf.quaternion_from_euler(roll, pitch, buffer_yaw)
                    # Assign new quaternion
                    new_pose.orientation.x = new_quat[0]
                    new_pose.orientation.y = new_quat[1]
                    new_pose.orientation.z = new_quat[2]
                    new_pose.orientation.w = new_quat[3]
                    # Stop the process
                    self.lightning_finished = True  
                    self.asked_by_lightning = False
                    self.starting_assembly_traj = True
                    self.angle_threshold_counter = 0
                    self.last_small_angle = None  # Reset
                    new_pose.position.x += - (self.manual_offset_x + self.camera_offset_x) * math.cos(yaw) + (self.manual_offset_y + self.camera_offset_y) * math.sin(yaw)
                    new_pose.position.y += - (self.manual_offset_x + self.camera_offset_x) * math.sin(yaw) - (self.manual_offset_y + self.camera_offset_y) * math.cos(yaw)

                else:
                    # Store the current angle and wait for the next opposite sign
                    self.last_small_angle = angle
            else:
                self.last_small_angle = None  # Reset if condition is not met
                self.angle_threshold_counter = 0  # Reset counter

            
            self.process_waypoints(self.current_pose, new_pose)

    def current_task_distributor(self):
        self.get_logger().info(f"Current Task distributor")
        if self.pause == True:
            return
        if self.test == True:
            if self.cutit == False:
                self.spiral()
                self.cutit = True
            return
        if self.finished_object:
            self.get_logger().info(f"Finished Target Pose of Object: {self.current_object}")
            self.request_next_object()
        elif self.asked_by_yolo == True:
            msg = String()
            msg.data = self.current_object
            self.trigger_yolo_publisher.publish(msg)
            self.storing_yolo_data = True
            self.get_logger().info(f"Will call YOLO for: {self.current_object}")
        elif self.calling_gripper_open == True:
            self.calling_gripper_open = False
            self.callGripper_open()
        elif self.calling_gripper_close == True:
            self.calling_gripper_close = False
            self.callGripper_close()
        elif self.correct_height == True:
            self.correct_height = False
            self.cartesian_correct_height()
        elif self.asked_by_lightning == True:
            self.asked_by_lightning = False
            trigger = Bool()
            trigger.data = True
            self.lighning_pub.publish(trigger)
            self.get_logger().info("Get Rotation estimation from Lightning")
        elif self.starting_assembly_traj == True:
            self.starting_assembly_traj = False
            self.callAssemblyTrajectory()
        elif self.driving_to_sup == True:
            self.driving_to_sup = False
            self.asked_by_yolo = True
            self.drive_to_ik()

    def target_pose_reached(self, msg: Pose):
        self.current_pose = msg
        self.get_logger().info(f"Current Pose: Position -> x: {msg.position.x}, y: {msg.position.y}, z: {msg.position.z}, "
                               f"Orientation -> x: {msg.orientation.x}, y: {msg.orientation.y}, "
                               f"z: {msg.orientation.z}, w: {msg.orientation.w}")
        self.current_task_distributor()

    def current_pose_function(self, msg: Pose):
        #self.get_logger().info(f"current_pose_function called. Self.asked_by_yolo: {self.asked_by_yolo}")
        self.current_pose = msg
        if self.asked_by_yolo == True:
            #self.get_logger().info(f"Current Pose: Position -> x: {msg.position.x}, y: {msg.position.y}, z: {msg.position.z}, "
                               #f"Orientation -> x: {msg.orientation.x}, y: {msg.orientation.y}, "
                               #f"z: {msg.orientation.z}, w: {msg.orientation.w}")
            self.yolo_servoing()
        elif self.manual_start == True:
            self.manual_start = False
            self.align_robot()
        
    def quaternion_to_euler(self, x, y, z, w):
        roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
        pitch = math.asin(2 * (w * y - z * x))
        yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        return roll, pitch, yaw
    
    def callGripper_close(self):   
        self.get_logger().info(f"Call Gripper close has been called by Sup obj {self.superior_object}")
        gripper_pos = self.gripper_close
        gripper_force = 25
        # Create the message with the position and force values in a list
        msg = Float32MultiArray(data=[gripper_pos, gripper_force])
        self.gripper_pub.publish(msg)
        time.sleep(4)
        self.current_task_distributor()
    
    def callGripper_open(self):
        if self.current_object is None:
            time.sleep(1)
            self.get_logger().info(f"Call Gripper open has been called to start")
            gripper_pos = 65
            gripper_force = 25
            # Create the message with the position and force values in a list
            msg = Float32MultiArray(data=[gripper_pos, gripper_force])
            self.gripper_pub.publish(msg)
            time.sleep(3)
            return   
        gripper_pos = self.gripper_open
        gripper_force = 25
        # Create the message with the position and force values in a list
        msg = Float32MultiArray(data=[gripper_pos, gripper_force])
        self.gripper_pub.publish(msg)

        
        if self.new_object == True:
            time.sleep(0.5)
            self.new_object = False
            self.drive_to_ik()
        else:
            time.sleep(2)
            self.current_task_distributor()

    def safe_csv(self, path):
        return # return to not brake program as path might not be defined
        csv_file = path # Update this path

        # Extract yaw from the quaternion
        _, _, yaw = self.quaternion_to_euler(
            self.current_pose.orientation.x,
            self.current_pose.orientation.y,
            self.current_pose.orientation.z,
            self.current_pose.orientation.w
        )

        # Convert yaw to degrees for better readability (optional)
        yaw_degrees = np.degrees(yaw)

        # Prepare data
        pose_data = [
        self.get_clock().now().to_msg().sec,  # Timestamp
        self.current_pose.position.x,
        self.current_pose.position.y,
        self.current_pose.position.z,
        yaw_degrees,  # Store only yaw
        self.vs_tries
    
    ]


        # Check if file exists to write headers
        file_exists = os.path.isfile(csv_file)

        # Write data to CSV file
        with open(csv_file, mode="a", newline="") as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(["Timestamp", "X", "Y", "Z", "Yaw", "Vs tries"])  # Headers
            writer.writerow(pose_data)  # Write pose data

    def callAssemblyTrajectory(self):
        pose_array = PoseArray()
        pose_array.poses.append(self.current_pose)
        new_pose = Pose()
        new_pose.position = self.current_pose.position
        new_pose.orientation = self.current_pose.orientation

        if self.superior_object == "Object_loader":
            if self.going_down == True:
                self.calling_gripper_close = True
                self.going_down = False
                self.going_up = True
                self.starting_assembly_traj = True
                speed_scale = Float32(data=0.1)
                new_pose.position.z =  new_pose.position.z - self.specific_height_assembly # Adjust Z position (move down by 0.01)

            elif self.going_up == True:
                self.going_down = True
                self.going_up = False
                self.finished_object = True
                self.get_logger().info(f"going upp")
                speed_scale = Float32(data=0.1)
                new_pose.position.z += 0.1  # Adjust Z position (move down by 0.01)

        elif self.superior_object == "Task_board":
            self.new_object = False
            if self.going_down == True:
                self.going_down = False
                self.going_up = True
                self.starting_assembly_traj = True
                self.calling_gripper_open = True

                if self.object_before == "Gear_Large":
                    path = "/home/alex/Auswertung/Gear_Large.csv"
                    self.safe_csv(path)
                    self.Trajectory_Gear_Large()
                elif self.object_before == "Gear_Medium":
                    self.pause = False
                    path = "/home/alex/Auswertung/Gear_Medium.csv"
                    if self.Gear_Medium_first == True:
                        self.safe_csv(path)
                        self.Gear_Medium_first = False
                        self.going_down = True
                        self.going_up = False
                        self.starting_assembly_traj = True
                        self.calling_gripper_open = False
                        self.Trajectory_before_Gear()
                    else: 
                        self.Trajectory_Gear_Medium()

                elif self.object_before == "Gear_Small":
                    self.pause = False
                    path = "/home/alex/Auswertung/Gear_Small.csv"
                    
                    if self.Gear_Small_first == True:
                        self.safe_csv(path)
                        self.Gear_Small_first = False
                        self.going_down = True
                        self.going_up = False
                        self.starting_assembly_traj = True
                        self.calling_gripper_open = False
                        self.Trajectory_before_Gear()
                    else: 
                        self.Trajectory_Gear_Small()
                elif self.current_object == "M16_Screw":
                    
                    if self.M16_screw_first == True:
                        path = "/home/alex/Auswertung/M16_Screw.csv"
                        self.safe_csv(path)
                        self.M16_screw_first = False
                        self.going_down = True
                        self.going_up = False
                        self.starting_assembly_traj = True
                        self.calling_gripper_open = False
                        self.Trajectory_before_M16_Screw()
                    else:
                        self.Trajectory_M16_Screw()
                elif self.current_object == "USB_Female":
                    path = "/home/alex/Auswertung/USB_Female.csv"
                    self.safe_csv(path)
                    self.Trajectory_USB_Female()
                elif self.current_object == "Ket_Square_12_Female":
                    self.pause = False
                    path = "/home/alex/Auswertung/Ket_12_Female.csv"
                    self.safe_csv(path)
                    self.Trajectory_Ket_12_Female()
                elif self.current_object == "Waterproof_Female":
                    self.pause = False
                    path = "/home/alex/Auswertung/Waterproof.csv"
                    self.safe_csv(path)
                    self.Trajectory_Waterproof_Female()
                elif self.current_object == "Ket_Square_8_Female":
                    self.pause = False
                    path = "/home/alex/Auswertung/Ket_8_Female.csv"
                    self.safe_csv(path)
                    self.Trajectory_Ket_8_Female()

                return

            if self.going_up == True:
                self.going_up = False
                self.finished_object = True
                self.get_logger().info(f"going upp")
                speed_scale = Float32(data=0.1)
                new_pose.position.z += 0.1  # Adjust Z position (move down by 0.01)

        self.set_speed_scale_pub.publish(speed_scale)
        pose_array.poses.append(new_pose)
        pose_array.header.stamp = self.get_clock().now().to_msg()
        pose_array.header.frame_id = "base_link" 
        self.target_pose_array_publisher.publish(pose_array)
 
    def extract_yolo_data(self, msg):

        # Determine effective height based on superior object
        if self.superior_object == "Object_loader":
            bottom_height = self.Object_loader_height
        elif self.superior_object == "Task_board":
            bottom_height = self.Task_board_height

        effective_height = self.current_pose.position.z - bottom_height + 0.09

        # Parse YOLO data
        data = msg.data
        num_objects = len(data) // 5  # Each object has (x, y, w, h, prob)
        
        objects = [(data[i * 5], data[i * 5 + 1], data[i * 5 + 2], data[i * 5 + 3], data[i * 5 + 4])
                for i in range(num_objects)]
        last_known_position = self.estimated_object_pose

        if self.current_object == "Gear_Shaft": # if gear_shaft, it needs to be calculated which gear shaft of the three
            gear_shaft_positions = [(x, y, w, h, prob) for x, y, w, h, prob in objects if prob > 0.1]

            if len(gear_shaft_positions) < 3:
                self.get_logger().error("Not enough Gear Shafts detected! Trying based on memory")

                if last_known_position is None:
                    self.get_logger().error("No last known position available! Cannot proceed.")
                    return None

                self.get_logger().info(f"Last known position: {last_known_position}")
                self.get_logger().info(f"Detected Gear Shaft positions: {gear_shaft_positions}")

                # Initialize closest position variables
                closest_position = None
                closest_distance = 0.01  # Threshold for considering a close match


                for position in gear_shaft_positions:
                    # Estimate the actual pose
                    new_known_position = self.estimate_pose(position, effective_height)  # Adjusted position

                    # Compute Euclidean distance
                    dist = np.sqrt((last_known_position[0] - new_known_position[0])**2 + 
                                (last_known_position[1] - new_known_position[1])**2)

                    self.get_logger().info(f"Distance from {last_known_position} to {new_known_position}: {dist:.4f}")

                    if dist < closest_distance:
                        closest_distance = dist
                        closest_position = position

                # If a closest position is found, return it
                if closest_position is not None:
                    self.get_logger().info(f"Using closest detected position: {closest_position}")
                    self.estimated_object_pose = self.estimate_pose(closest_position, effective_height)
                    return closest_position
                else:
                    self.get_logger().error("No detected positions found close to the last known position.")
                    return None

            # Step 1: Calculate pairwise distances
            distances = {}  # Store distances between pairs
            for i in range(len(gear_shaft_positions)):
                for j in range(i + 1, len(gear_shaft_positions)):
                    x1, y1, _, _, _ = gear_shaft_positions[i]
                    x2, y2, _, _, _ = gear_shaft_positions[j]
                    dist = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                    distances[(i, j)] = dist
                    distances[(j, i)] = dist  # Store both ways

            # Step 2: Compute total distance for each shaft (sum of pairwise distances)
            total_distances = [(i, sum(distances[(i, j)] for j in range(len(gear_shaft_positions)) if i != j)) 
                            for i in range(len(gear_shaft_positions))]

            # Sort by total distance (low → middle, high → farthest)
            total_distances.sort(key=lambda x: x[1])  

            # Step 3: Identify gear shafts
            farthest_shaft = total_distances[-1][0]  # Highest total distance (Gear_Large)
            middle_shaft = total_distances[0][0]  # Smallest total distance (Gear_Middle)
            closest_shaft = total_distances[1][0] 

            # Step 4: Select shaft based on `self.object_before`
            if self.object_before == "Gear_Large":
                selected_index = farthest_shaft
            elif self.object_before == "Gear_Middle":
                selected_index = middle_shaft
            elif self.object_before == "Gear_Small":
                selected_index = closest_shaft
            else:
                selected_index = middle_shaft  # Default to middle if unknown

            # Get the selected gear shaft
            selected_gear_shaft = gear_shaft_positions[selected_index]

            new_known_position = self.estimate_pose(selected_gear_shaft, effective_height)  # Adjusted position

            # Compute Euclidean distance
            if last_known_position is None:
                self.get_logger().info(f"No last position")
                dist = 0.0
            else:
                dist = np.sqrt((last_known_position[0] - new_known_position[0])**2 + 
                            (last_known_position[1] - new_known_position[1])**2)
            if dist < 0.01:
                # Estimate pose and log results
                self.estimated_object_pose = new_known_position
                self.get_logger().info(f"Detected Gear Shafts: {gear_shaft_positions}")
                for key, dist in distances.items():
                    self.get_logger().info(f"Distance between {key}: {dist:.4f}")
                self.get_logger().info(f"Total distances: {total_distances}")
                self.get_logger().info(f"Selected Gear Shaft: {selected_gear_shaft} for {self.object_before}")

                return selected_gear_shaft
            else:
                return None

        # If not a Gear Shaft, find the highest probability object
        objects.sort(key=lambda obj: obj[4], reverse=True)

        if self.estimated_object_pose is None:
            self.estimated_object_pose = self.estimate_pose(objects[0], effective_height)
            return objects[0]

        # Find the closest high-probability object
        best_object = objects[0]
        best_distance = 0.05

        for obj in objects:
            x, y, _, _, prob = obj
            new_known_position = self.estimate_pose(obj, effective_height)

            dist = np.sqrt((last_known_position[0] - new_known_position[0]) ** 2 + 
                        (last_known_position[1] - new_known_position[1]) ** 2)

            if np.isclose(prob, objects[0][4], atol=0.3):  #if confidence is in a range of 0.3 to highest confidence score
                if dist < best_distance: # 5cm starting dist
                    best_distance = dist
                    best_object = obj
                    self.estimated_object_pose = new_known_position

        return best_object  # (x, y, w, h, prob)

    def yolo_callback(self, msg: Float32MultiArray):
        
        if self.storing_yolo_data == True:
            self.get_logger().info(f"yolo_callback")
            self.yolo_data = None
            self.yolo_data = self.extract_yolo_data(msg)
            if self.yolo_data is None:
                msg = String()
                msg.data = self.current_object
                self.trigger_yolo_publisher.publish(msg)
                return
            self.get_logger().info(f"yolo data: {self.yolo_data}")
            self.last_yolo_data = self.yolo_data
            self.new_yolo_data = time.time()
            if self.controller == "joint_trajectory_controller":
                    self.get_logger().info(f"trying to switch")
                    msg = String()
                    msg.data = "activate:forward_position_controller;deactivate:scaled_joint_trajectory_controller"
                    self.control_switcher_publisher.publish(msg)
                    self.controller = "forward_position_controller"
                    msg = Bool()
                    msg.data = True
            if self.asked_by_yolo == True:
                msg = Bool()
                msg.data = True
                self.ask_current_pose.publish(msg)

                    
                
        # Further processing (e.g., wait for message, handle pose transformations)
    
    def estimate_pose(self, yolo_data, effective_height):
        # Extract quaternion and convert to Euler angles
        _, _, yaw = self.quaternion_to_euler(self.current_pose.orientation.x,self.current_pose.orientation.y,self.current_pose.orientation.z,self.current_pose.orientation.w)
        
        yolo_data_array = np.array(yolo_data[0:4])
        translation_data = estimate_camera_pose(yolo_data_array, effective_height, yaw, 0, 0, 0)  
        self.get_logger().info(f"offset: {translation_data}")  # Aligned
        norm_translation1_org = -float(translation_data[0])
        norm_translation2_org = float(translation_data[1]) 
        norm_translation1 = norm_translation1_org*np.cos(2*yaw) - norm_translation2_org* np.sin(2*yaw)
        norm_translation2 = norm_translation2_org * np.cos(2*yaw) + norm_translation1_org* np.sin(2*yaw)
        x =self.current_pose.position.x + norm_translation1
        y =self.current_pose.position.y + norm_translation2
        return [x,y]

    def yolo_servoing(self):

        msg = String()
        msg.data = self.current_object
        self.trigger_yolo_publisher.publish(msg)
        if self.yolo_data is None:
            self.yolo_data = self.last_yolo_data

        if self.yolo_data is None:
            self.get_logger().info("YOLO data is none")
            self.current_task_distributor()
            return
        self.last_call_time = time.time() 

        # Determine the effective height
        if self.superior_object == "Object_loader":
            bottom_height = self.Object_loader_height
        elif self.superior_object == "Task_board":
            bottom_height = self.Task_board_height

        # Exit if lightning process is finished
        if self.lightning_finished:
            self.get_logger().info("Lightning finished")
            self.Last_aligning(self.current_pose)
            return
        
        # Convert quaternion to Euler angles and adjust yaw
        _, _, yaw = self.quaternion_to_euler(
            self.current_pose.orientation.x, self.current_pose.orientation.y, 
            self.current_pose.orientation.z, self.current_pose.orientation.w
        )
        yaw = - yaw  # Adjust yaw

        # Get YOLO data and compute errors
        x, y, _, _ = np.array(self.yolo_data)[:4]
        error_x = self.x_target - x
        error_y = self.y_target - y
        #self.get_logger().info(f"Error X: {error_x}, Error Y: {error_y}")

        # Image and camera parameters
        horizontal_fov = 1.2054
        vertical_fov = horizontal_fov * (self.image_height / self.image_width)
        theta = math.radians(15)  # Camera tilt angle

        # Depth and scaling calculations
        error_diff_z = (self.current_pose.position.z - (bottom_height + self.specific_height))
        self.get_logger().info(f"Error_diff_z {error_diff_z}")
        d = (2 * error_diff_z * math.tan(vertical_fov / 2)) / self.image_height
        max_error_xy = max(abs(error_x), abs(error_y))

        # Adjust error_y with scaling factor
        if error_diff_z > 0.01:
            threshold, max_value = 0.01, 0.15
            scaling_factor = min(1, (error_diff_z - threshold) / (max_value - threshold))
            error_y += scaling_factor * (error_diff_z * math.tan(theta) / d)
            error_y = min(480, error_y)
            #self.get_logger().info(f"Adjusted Error Y: {error_y}, Scaling Factor: {scaling_factor}")

        # Estimate depth and construct interaction matrix
        Z = self.current_pose.position.z - (bottom_height) + 0.05
        # simplified image jacobian
        L_s = np.array([
            [-1/Z, 0],
            [0, -1/Z]
        ])

        # Compute the control law
        L_s_pseudo_inv = np.linalg.pinv(L_s)
        error_vector = np.array([error_x, error_y])
        lambda_gain = 0.05
        v_s = -lambda_gain * np.dot(L_s_pseudo_inv, error_vector)

        # Extract velocities
        v_x, v_y = v_s

        # Scale error_z with damping factor
        damp_factor = 1
        if error_diff_z > 0.001:
            if error_diff_z < 0.08:
                damp_factor = 0.01 + ((error_diff_z * 0.99) / 0.1)
                #self.get_logger().info(f"Damping Factor: {damp_factor}")
            error_z = -200 * damp_factor + max_error_xy
            error_z = np.clip(error_z, -150, -1)
        elif error_diff_z < 0:
            error_z = 5  # Move upwards if below target
        else:
            error_z = 0  # No movement

        if time.time() - self.new_yolo_data > 0.3: #if for longer than 0.3 seconds no new YOLO data
            if time.time() - self.new_yolo_data < 2.0:
                error_x_world = -(self.current_pose.position.x - self.estimated_object_pose[0])*7
                error_y_world = -(self.current_pose.position.y - self.estimated_object_pose[1])*7
                #self.get_logger().info(f"Last yolo data over 0.5 seconds. estimated pose: {self.estimated_object_pose[0], self.estimated_object_pose[1]}")
                msg = Float32MultiArray(data=[error_x_world,  error_y_world, error_z*0.01])
                self.twister_pub.publish(msg)
                msg = Bool()
                msg.data = True
                self.ask_current_pose.publish(msg)
            return

        # Transform errors to the world coordinate system
        error_x_world = -(np.cos(yaw) * v_x - np.sin(yaw) * v_y)
        error_y_world = (np.sin(yaw) * v_x + np.cos(yaw) * v_y)

        if (abs(error_x) < 10 and abs(error_y) < 10 and abs(error_diff_z) < 0.005): # threshlod values
            self.asked_by_yolo = False
            self.storing_yolo_data = False
            if self.controller == "forward_position_controller":
                self.get_logger().info("Switching for Rotation")
                msg = String()
                msg.data = "activate:scaled_joint_trajectory_controller;deactivate:forward_position_controller"
                self.control_switcher_publisher.publish(msg)
                self.controller = "joint_trajectory_controller"
                
            if self.height_is_correct == False:
                self.correct_height = True
                self.object_height = bottom_height + self.specific_height
                self.get_logger().info("Going to correct height")

            else:
                self.get_logger().error("Sould not have happened")
                return
            
            self.current_task_distributor()

        else:
            if self.controller == "joint_trajectory_controller":
                self.get_logger().info(f"trying to switch")
                msg = String()
                msg.data = "activate:forward_position_controller;deactivate:scaled_joint_trajectory_controller"
                self.control_switcher_publisher.publish(msg)
                self.controller = "forward_position_controller"
            #self.get_logger().info(f"Error X: {error_x}, Error Y: {error_y}")
            self.get_logger().info(f"Error v_X: {error_x_world}, Error v_Y: {error_y_world} error_z: {error_z*0.01}")
            msg = Float32MultiArray(data=[error_x_world,  error_y_world, error_z*0.012])
            self.twister_pub.publish(msg)
            msg = Bool()
            msg.data = True
            self.ask_current_pose.publish(msg)

    def Trajectory_before_M16_Screw(self):
        speed_scale = Float32(data=1.0)
        self.set_speed_scale_pub.publish(speed_scale)
        self.get_logger().info(f"pre nut")
        new_angle = Float32()
        new_angle.data = -2*np.pi
        self.joint_publisher.publish(new_angle)

    def Trajectory_before_Gear(self): # this has to be used if Task board rotation bigger than 15 degrees
        speed_scale = Float32(data=1.0)
        self.set_speed_scale_pub.publish(speed_scale)
        self.get_logger().info(f"pre nut")
        new_angle = Float32()
        new_angle.data = np.pi/10
        self.joint_publisher.publish(new_angle)

    def Trajectory_M16_Screw(self):
        self.get_logger().info(f"nutting")
        # Set speed scale (slow for precision)
        speed_scale = Float32(data=0.1)
        self.set_speed_scale_pub.publish(speed_scale)

        start_pose = self.current_pose
        start_x = start_pose.position.x
        start_y = start_pose.position.y
        start_z = start_pose.position.z

        pose_array = PoseArray()

        # 1. Move Straight Down by 3.5 cm
        for i in range(10):  
            new_pose = Pose()
            new_pose.position.x = start_x
            new_pose.position.y = start_y
            new_pose.position.z = start_z - 0.0034 * (i + 1)  # Move down 3.5 cm
            new_pose.orientation = start_pose.orientation
            pose_array.poses.append(new_pose)

        # Update start position for next phase
        spiral_start_z = start_z - 0.034  # After moving 3.5 cm down

        # 2. Spiral Down 0.3 cm with 1 mm radius (NO ROTATION)
        radius = 0.0015  # Initial radius
        spiral_turns = 2
        spiral_height = 0.0027
        points_per_turn = 30
        total_spiral_steps = spiral_turns * points_per_turn
        angular_velocity = 2 * np.pi / points_per_turn

        for i in range(total_spiral_steps):
            angle = angular_velocity * i

            # Spiral motion in XY plane
            new_x = start_x + radius * np.cos(angle)
            new_y = start_y + radius * np.sin(angle)
            new_z = spiral_start_z - (spiral_height / total_spiral_steps) * i

            new_pose = Pose()
            new_pose.position.x = new_x
            new_pose.position.y = new_y
            new_pose.position.z = new_z
            new_pose.orientation = start_pose.orientation

            pose_array.poses.append(new_pose)

            radius *= 0.95

        # Update start position after spiral
        rotate_start_z = spiral_start_z - spiral_height # After spiral motion

        # 3. Rotate 2π in Place
        num_rotate_steps = 500 
        half_rots = 1
        total_rotation =  half_rots* np.pi  # 1 full turn
        angle_per_step = total_rotation / num_rotate_steps  
        pitch_per_step = (0.001 *half_rots) / num_rotate_steps  # 2mm per full rotation

        # Get starting quaternion as a list
        current_quat = [start_pose.orientation.x, start_pose.orientation.y, 
                        start_pose.orientation.z, start_pose.orientation.w]

        for i in range(num_rotate_steps):
            new_pose = Pose()
            new_pose.position.x = start_x
            new_pose.position.y = start_y
            new_pose.position.z = rotate_start_z - (pitch_per_step * (i + 1))  # Move down

            # Compute incremental rotation
            delta_angle = -angle_per_step * (i + 1)
            rotation_quat = quaternion_from_euler(0, 0, delta_angle)  # Only yaw (Z-axis)

            # Apply rotation to current orientation
            final_quat = quaternion_multiply(rotation_quat, current_quat)
            new_pose.orientation = Quaternion(x=final_quat[0], y=final_quat[1], 
                                            z=final_quat[2], w=final_quat[3])

            pose_array.poses.append(new_pose)

        # Set header and publish
        pose_array.header.stamp = self.get_clock().now().to_msg()
        pose_array.header.frame_id = "base_link"
        self.target_pose_array_publisher.publish(pose_array)
        self.get_logger().info("Published nut screwing path.")

    def Trajectory_Gear_Large(self):
        self.get_logger().info("Starting spiral_3 movement")

        # Extract yaw angle from the current orientation
        _, _, yaw = self.quaternion_to_euler(
            self.current_pose.orientation.x, self.current_pose.orientation.y, 
            self.current_pose.orientation.z, self.current_pose.orientation.w
        )

        # Set speed scale for precise movements
        speed_scale = Float32(data=0.1)
        self.set_speed_scale_pub.publish(speed_scale)

        # Extract current position
        start_pose = self.current_pose

        delta_y = 0.0295  # 3 cm
        move_x = delta_y * np.sin(yaw)
        move_y = delta_y * np.cos(yaw)

        start_x = start_pose.position.x + move_x
        start_y = start_pose.position.y - move_y
        start_z = start_pose.position.z

        pose_array = PoseArray()
        num_steps = 20  # More steps for a smoother trajectory
        total_downward_movement = 0.0308 # Total downward movement (3.5 cm)
        total_tilt_angle = np.radians(5)  # Total tilt (5 degrees)
        total_x_movement = -0.02  # Total x-direction movement

        for i in range(num_steps):
            step_fraction = (i + 1) / num_steps  # Fraction of total movement

            # Compute incremental downward movement
            new_z = start_z - total_downward_movement * step_fraction  

            # Compute incremental tilt
            step_tilt = total_tilt_angle * step_fraction
            q_orig = (start_pose.orientation.x, start_pose.orientation.y, 
                    start_pose.orientation.z, start_pose.orientation.w)
            q_tilt_offset = quaternion_from_euler(0, step_tilt, 0)  # Tilt around Y-axis
            q_tilt_tuple = quaternion_multiply(q_orig, q_tilt_offset)

            # Convert back to ROS2 Quaternion message
            q_tilt = Quaternion()
            q_tilt.x, q_tilt.y, q_tilt.z, q_tilt.w = q_tilt_tuple

            # Compute incremental movement in X-direction (adjusted for yaw)
            move_x = total_x_movement * np.cos(yaw) * step_fraction
            move_y = total_x_movement * np.sin(yaw) * step_fraction

            # Create new pose
            new_pose = Pose()
            new_pose.position.x = start_x + move_x
            new_pose.position.y = start_y + move_y
            new_pose.position.z = new_z  # Apply downward motion
            new_pose.orientation = q_tilt  # Apply tilt

            pose_array.poses.append(new_pose)

            # Final position after spiral
        final_x = start_x + total_x_movement * np.cos(yaw)
        final_y = start_y + total_x_movement * np.sin(yaw)
        final_z = start_z - total_downward_movement  # Maintain final height
        final_tilt = total_tilt_angle  # Store last tilt value

        # === Move 1 cm in X-direction in steps ===
        move_1cm_steps = 10
        move_x_1cm = 0.02 * np.cos(yaw)
        move_y_1cm = 0.02 * np.sin(yaw)

        for i in range(move_1cm_steps):
            step_fraction = (i + 1) / move_1cm_steps

            move_pose = Pose()
            move_pose.position.x = final_x + move_x_1cm * step_fraction
            move_pose.position.y = final_y + move_y_1cm * step_fraction
            move_pose.position.z = final_z  # Maintain height
            move_pose.orientation = q_tilt  # Maintain last tilt

            pose_array.poses.append(move_pose)

        # === Reset Tilt to 0 Degrees Gradually ===
        total_tilt_angle = np.radians(-5)  # Total tilt (5 degrees)

        # === Reset Tilt to 0 Degrees Gradually ===
        num_steps = 20  # More steps for a smoother transition
        start_pose = move_pose  # Start from the last position
        start_x = move_pose.position.x
        start_y = move_pose.position.y
        start_z = move_pose.position.z

        # Use the original pose's orientation as the starting reference
        # Original orientation (starting pose)
        q_orig = (
            start_pose.orientation.x,
            start_pose.orientation.y,
            start_pose.orientation.z,
            start_pose.orientation.w
        )

        for i in range(num_steps):
            step_fraction = (i + 1) / num_steps  # Fraction of total movement

            # Compute correct tilt for this step
            step_tilt = total_tilt_angle * step_fraction  # Gradually reduce tilt
            q_tilt_offset = quaternion_from_euler(0, step_tilt, 0)  # Create offset

            # Apply tilt relative to the ORIGINAL orientation
            q_tilt_tuple = quaternion_multiply(q_tilt_offset, q_orig) 

            # Convert back to ROS2 Quaternion message
            q_tilt = Quaternion()
            q_tilt.x, q_tilt.y, q_tilt.z, q_tilt.w = q_tilt_tuple

            # Create new pose (position remains the same, only tilt changes)
            new_pose = Pose()
            new_pose.position.x = start_x
            new_pose.position.y = start_y
            new_pose.position.z = start_z
            new_pose.orientation = q_tilt  # Apply smooth tilt reset

            pose_array.poses.append(new_pose)


        # Update start position for next phase
        spiral_start_z = start_z  # After moving 3.5 cm down

        # 2. Spiral Down 0.3 cm with 1 mm radius (NO ROTATION)
        num_spiral_steps = 500  
        spiral_radius = 0.001  # 1 mm radius
        pitch_per_step = 0.02 / num_spiral_steps  # Move down 0.3 cm evenly

        for i in range(num_spiral_steps):
            new_pose = Pose()

            # Compute spiral motion
            angle = (np.pi / num_spiral_steps) * (i + 1)  # Spread points in a circle
            new_pose.position.x = start_x + spiral_radius * np.cos(angle)
            new_pose.position.y = start_y + spiral_radius * np.sin(angle)
            new_pose.position.z = spiral_start_z - (pitch_per_step * (i + 1))  # Move down

            # Maintain original orientation (no rotation)
            new_pose.orientation = q_tilt
            pose_array.poses.append(new_pose)
            spiral_radius *= 0.99
            # Final position after spiral

        # Publish the updated trajectory
        pose_array.header.stamp = self.get_clock().now().to_msg()
        pose_array.header.frame_id = "base_link"
        self.target_pose_array_publisher.publish(pose_array)

        self.get_logger().info("Published smooth spiral screwing path, x-movement, and gradual tilt reset.")

    def Trajectory_Gear_Medium(self):
        self.get_logger().info("Starting spiral_3 movement")
        speed_scale = Float32(data=0.1)
        self.set_speed_scale_pub.publish(speed_scale)

        # Extract yaw angle from the current orientation
        _, _, yaw = self.quaternion_to_euler(
            self.current_pose.orientation.x, self.current_pose.orientation.y, 
            self.current_pose.orientation.z, self.current_pose.orientation.w
        )
        yaw = 3.141

        # Extract current position
        start_pose = self.current_pose

        delta_y = 0.0288  # 3 cm
        move_x = delta_y * np.sin(yaw)
        move_y = delta_y * np.cos(yaw)

        start_x = start_pose.position.x + move_x
        start_y = start_pose.position.y - move_y
        start_z = start_pose.position.z
        # Set speed scale for precise movements

        pose_array = PoseArray()
        num_steps = 20  # More steps for a smoother trajectory
        total_downward_movement = 0.0315 # Total downward movement (3.5 cm)
        total_tilt_angle = np.radians(5)  # Total tilt (5 degrees)
        total_x_movement = -0.02  # Total x-direction movement

        new_pose = Pose()
        new_pose.position.x = start_x 
        new_pose.position.y = start_y 
        new_pose.position.z = start_z
        new_pose.orientation = start_pose.orientation

        pose_array.poses.append(new_pose)

        for i in range(num_steps):
            step_fraction = (i + 1) / num_steps  # Fraction of total movement

            # Compute incremental downward movement
            new_z = start_z - total_downward_movement * step_fraction  

            # Compute incremental tilt
            step_tilt = total_tilt_angle * step_fraction
            q_orig = (start_pose.orientation.x, start_pose.orientation.y, 
                    start_pose.orientation.z, start_pose.orientation.w)
            q_tilt_offset = quaternion_from_euler(0, step_tilt, 0)  # Tilt around Y-axis
            q_tilt_tuple = quaternion_multiply(q_orig, q_tilt_offset)

            # Convert back to ROS2 Quaternion message
            q_tilt = Quaternion()
            q_tilt.x, q_tilt.y, q_tilt.z, q_tilt.w = q_tilt_tuple

            # Compute incremental movement in X-direction (adjusted for yaw)
            move_x = total_x_movement * np.cos(yaw) * step_fraction
            move_y = total_x_movement * np.sin(yaw) * step_fraction

            # Create new pose
            new_pose = Pose()
            new_pose.position.x = start_x + move_x
            new_pose.position.y = start_y + move_y
            new_pose.position.z = new_z  # Apply downward motion
            new_pose.orientation = q_tilt  # Apply tilt

            pose_array.poses.append(new_pose)

            # Final position after spiral
        final_x = start_x + total_x_movement * np.cos(yaw)
        final_y = start_y + total_x_movement * np.sin(yaw)
        final_z = start_z - total_downward_movement  # Maintain final height
        final_tilt = total_tilt_angle  # Store last tilt value

        # === Move 1 cm in X-direction in steps ===
        move_1cm_steps = 10
        move_x_1cm = 0.02 * np.cos(yaw)
        move_y_1cm = 0.02 * np.sin(yaw)

        for i in range(move_1cm_steps):
            step_fraction = (i + 1) / move_1cm_steps

            move_pose = Pose()
            move_pose.position.x = final_x + move_x_1cm * step_fraction
            move_pose.position.y = final_y + move_y_1cm * step_fraction
            move_pose.position.z = final_z  # Maintain height
            move_pose.orientation = q_tilt  # Maintain last tilt

            pose_array.poses.append(move_pose)

        # === Reset Tilt to 0 Degrees Gradually ===
        total_tilt_angle = np.radians(-5)  # Total tilt (5 degrees)

        # === Reset Tilt to 0 Degrees Gradually ===
        num_steps = 20  # More steps for a smoother transition
        start_pose = move_pose  # Start from the last position
        start_x = move_pose.position.x
        start_y = move_pose.position.y
        start_z = move_pose.position.z

        # Use the original pose's orientation as the starting reference
        # Original orientation (starting pose)
        q_orig = (
            start_pose.orientation.x,
            start_pose.orientation.y,
            start_pose.orientation.z,
            start_pose.orientation.w
        )

        for i in range(num_steps):
            step_fraction = (i + 1) / num_steps  # Fraction of total movement

            # Compute correct tilt for this step
            step_tilt = total_tilt_angle * step_fraction  # Gradually reduce tilt
            q_tilt_offset = quaternion_from_euler(0, step_tilt, 0)  # Create offset

            # Apply tilt relative to the ORIGINAL orientation
            q_tilt_tuple = quaternion_multiply(q_tilt_offset, q_orig) 

            # Convert back to ROS2 Quaternion message
            q_tilt = Quaternion()
            q_tilt.x, q_tilt.y, q_tilt.z, q_tilt.w = q_tilt_tuple

            # Create new pose (position remains the same, only tilt changes)
            new_pose = Pose()
            new_pose.position.x = start_x
            new_pose.position.y = start_y
            new_pose.position.z = start_z
            new_pose.orientation = q_tilt  # Apply smooth tilt reset

            pose_array.poses.append(new_pose)


        # Update start position for next phase
        spiral_start_z = start_z  # After moving 3.5 cm down

        # Initialize variables
        num_spiral_steps = 200  # Steps for spiraling motion
        spiral_radius = 0.00  # 1 mm radius
        pitch_per_spiral_step = 0.009 / num_spiral_steps  # Move down 0.8 cm evenly
        spiral_start_z = start_z  # Starting height


        # --- Phase 1: Move Down 0.8 cm (NO Rotation) ---
        for i in range(num_spiral_steps):
            new_pose = Pose()

            # Compute spiral motion
            angle = (2 * np.pi / num_spiral_steps) * (i + 1)  # Spiral movement
            new_pose.position.x = start_x + spiral_radius * np.cos(angle)
            new_pose.position.y = start_y + spiral_radius * np.sin(angle)
            new_pose.position.z = spiral_start_z - (pitch_per_spiral_step * (i + 1))  # Move down

            # Maintain initial tilt orientation (NO rotation)
            new_pose.orientation = q_tilt

            # Append to the list
            pose_array.poses.append(new_pose)

            # Gradually decrease radius for inward spiral
            spiral_radius *= 0.99

        # Update start position for the second phase
        linear_start_z = new_pose.position.z  # End position of the spiral

        # --- Phase 2: Straight Descent 1.2 cm with π Rotation ---
        num_linear_steps = 100  # Steps for linear descent
        pitch_per_linear_step = 0.002 / num_linear_steps  # Move down 1.2 cm evenly

        for i in range(num_linear_steps):
            new_pose = Pose()
            # Move straight down
            new_pose.position.x = start_x
            new_pose.position.y = start_y
            new_pose.position.z = linear_start_z - (pitch_per_linear_step * (i + 1))

            # Compute rotation (π over full descent)
            yaw_angle = -(np.pi / (5 * num_linear_steps)) * (i + 1)  # Linearly interpolate from 0 to π
            rotation_quaternion = tf.quaternion_from_euler(0, 0, yaw_angle)  # (roll, pitch, yaw)

            # Combine tilt orientation with new rotation
            tilt_quaternion = [q_tilt.x, q_tilt.y, q_tilt.z, q_tilt.w]
            combined_quaternion = tf.quaternion_multiply(tilt_quaternion, rotation_quaternion)

            # Set new orientation
            new_pose.orientation.x = combined_quaternion[0]
            new_pose.orientation.y = combined_quaternion[1]
            new_pose.orientation.z = combined_quaternion[2]
            new_pose.orientation.w = combined_quaternion[3]

            # Append to the list
            pose_array.poses.append(new_pose)

        linear_start_z = new_pose.position.z  # End position of the spiral
        num_linear_steps = 20  # Steps for linear descent
        pitch_per_linear_step = 0.009 / num_linear_steps  # Move down 1.2 cm evenly

        for i in range(num_linear_steps):
            new_pose = Pose()
            # Move straight down
            new_pose.position.x = start_x
            new_pose.position.y = start_y
            new_pose.position.z = linear_start_z - (pitch_per_linear_step * (i + 1))

            new_pose.orientation.x = combined_quaternion[0]
            new_pose.orientation.y = combined_quaternion[1]
            new_pose.orientation.z = combined_quaternion[2]
            new_pose.orientation.w = combined_quaternion[3]

            # Append to the list
            pose_array.poses.append(new_pose)

        # Publish the updated trajectory
        pose_array.header.stamp = self.get_clock().now().to_msg()
        pose_array.header.frame_id = "base_link"
        self.target_pose_array_publisher.publish(pose_array)


        self.get_logger().info("Published smooth two-phase screwing path: Spiral down (0.8 cm), then rotate (π) while descending (1.2 cm).")

    def Trajectory_Gear_Small(self):
        self.get_logger().info("Starting spiral_3 movement")

        # Extract yaw angle from the current orientation
        _, _, yaw = self.quaternion_to_euler(
            self.current_pose.orientation.x, self.current_pose.orientation.y, 
            self.current_pose.orientation.z, self.current_pose.orientation.w
        )
        yaw = 3.141
        # Set speed scale for precise movements
        speed_scale = Float32(data=0.1)
        self.set_speed_scale_pub.publish(speed_scale)

        # Extract current position
        start_pose = self.current_pose

        delta_y = 0.03  # 3 cm
        move_x = delta_y * np.sin(yaw)
        move_y = delta_y * np.cos(yaw)

        start_x = start_pose.position.x + move_x
        start_y = start_pose.position.y - move_y
        start_z = start_pose.position.z

        pose_array = PoseArray()
        num_steps = 20  # More steps for a smoother trajectory
        total_downward_movement = 0.0315 # Total downward movement (3.5 cm)
        total_tilt_angle = np.radians(5)  # Total tilt (5 degrees)
        total_x_movement = -0.01  # Total x-direction movement

        for i in range(num_steps):
            step_fraction = (i + 1) / num_steps  # Fraction of total movement

            # Compute incremental downward movement
            new_z = start_z - total_downward_movement * step_fraction  

            # Compute incremental tilt
            step_tilt = total_tilt_angle * step_fraction
            q_orig = (start_pose.orientation.x, start_pose.orientation.y, 
                    start_pose.orientation.z, start_pose.orientation.w)
            q_tilt_offset = quaternion_from_euler(0, step_tilt, 0)  # Tilt around Y-axis
            q_tilt_tuple = quaternion_multiply(q_orig, q_tilt_offset)

            # Convert back to ROS2 Quaternion message
            q_tilt = Quaternion()
            q_tilt.x, q_tilt.y, q_tilt.z, q_tilt.w = q_tilt_tuple

            # Compute incremental movement in X-direction (adjusted for yaw)
            move_x = total_x_movement * np.cos(yaw) * step_fraction
            move_y = total_x_movement * np.sin(yaw) * step_fraction

            # Create new pose
            new_pose = Pose()
            new_pose.position.x = start_x + move_x
            new_pose.position.y = start_y + move_y
            new_pose.position.z = new_z  # Apply downward motion
            new_pose.orientation = q_tilt  # Apply tilt

            pose_array.poses.append(new_pose)

            # Final position after spiral
        final_x = start_x + total_x_movement * np.cos(yaw)
        final_y = start_y + total_x_movement * np.sin(yaw)
        final_z = start_z - total_downward_movement  # Maintain final height
        final_tilt = total_tilt_angle  # Store last tilt value

        # === Move 1 cm in X-direction in steps ===
        move_1cm_steps = 10
        move_x_1cm = 0.01 * np.cos(yaw)
        move_y_1cm = 0.01 * np.sin(yaw)

        for i in range(move_1cm_steps):
            step_fraction = (i + 1) / move_1cm_steps

            move_pose = Pose()
            move_pose.position.x = final_x + move_x_1cm * step_fraction
            move_pose.position.y = final_y + move_y_1cm * step_fraction
            move_pose.position.z = final_z  # Maintain height
            move_pose.orientation = q_tilt  # Maintain last tilt

            pose_array.poses.append(move_pose)

        # === Reset Tilt to 0 Degrees Gradually ===
        total_tilt_angle = np.radians(-5)  # Total tilt (5 degrees)

        # === Reset Tilt to 0 Degrees Gradually ===
        num_steps = 20  # More steps for a smoother transition
        start_pose = move_pose  # Start from the last position
        start_x = move_pose.position.x
        start_y = move_pose.position.y
        start_z = move_pose.position.z

        # Use the original pose's orientation as the starting reference
        # Original orientation (starting pose)
        q_orig = (
            start_pose.orientation.x,
            start_pose.orientation.y,
            start_pose.orientation.z,
            start_pose.orientation.w
        )

        for i in range(num_steps):
            step_fraction = (i + 1) / num_steps  # Fraction of total movement

            # Compute correct tilt for this step
            step_tilt = total_tilt_angle * step_fraction  # Gradually reduce tilt
            q_tilt_offset = quaternion_from_euler(0, step_tilt, 0)  # Create offset

            # Apply tilt relative to the ORIGINAL orientation
            q_tilt_tuple = quaternion_multiply(q_tilt_offset, q_orig) 

            # Convert back to ROS2 Quaternion message
            q_tilt = Quaternion()
            q_tilt.x, q_tilt.y, q_tilt.z, q_tilt.w = q_tilt_tuple

            # Create new pose (position remains the same, only tilt changes)
            new_pose = Pose()
            new_pose.position.x = start_x
            new_pose.position.y = start_y
            new_pose.position.z = start_z
            new_pose.orientation = q_tilt  # Apply smooth tilt reset

            pose_array.poses.append(new_pose)


        # Update start position for next phase
        spiral_start_z = start_z  # After moving 3.5 cm down

        # Initialize variables
        num_spiral_steps = 200  # Steps for spiraling motion
        spiral_radius = 0.0  # 1 mm radius
        pitch_per_spiral_step = 0.0085 / num_spiral_steps  # Move down 0.8 cm evenly
        spiral_start_z = start_z  # Starting height


        # --- Phase 1: Spiral Down 0.8 cm (NO Rotation) ---
        for i in range(num_spiral_steps):
            new_pose = Pose()

            # Compute spiral motion
            angle = (2 * np.pi / num_spiral_steps) * (i + 1)  # Spiral movement
            new_pose.position.x = start_x + spiral_radius * np.cos(angle)
            new_pose.position.y = start_y + spiral_radius * np.sin(angle)
            new_pose.position.z = spiral_start_z - (pitch_per_spiral_step * (i + 1))  # Move down

            # Maintain initial tilt orientation (NO rotation)
            new_pose.orientation = q_tilt

            # Append to the list
            pose_array.poses.append(new_pose)

            # Gradually decrease radius for inward spiral
            spiral_radius *= 0.99

        # Update start position for the second phase
        linear_start_z = new_pose.position.z  # End position of the spiral

        # --- Phase 2: Straight Descent 1.2 cm with π Rotation ---
        num_linear_steps = 100  # Steps for linear descent
        pitch_per_linear_step = 0.002 / num_linear_steps  # Move down 1.2 cm evenly

        for i in range(num_linear_steps):
            new_pose = Pose()
            # Move straight down
            new_pose.position.x = start_x
            new_pose.position.y = start_y
            new_pose.position.z = linear_start_z - (pitch_per_linear_step * (i + 1))

            # Compute rotation (π over full descent)
            yaw_angle = -(np.pi / (5* num_linear_steps)) * (i + 1)  # Linearly interpolate from 0 to π
            rotation_quaternion = tf.quaternion_from_euler(0, 0, yaw_angle)  # (roll, pitch, yaw)

            # Combine tilt orientation with new rotation
            tilt_quaternion = [q_tilt.x, q_tilt.y, q_tilt.z, q_tilt.w]
            combined_quaternion = tf.quaternion_multiply(tilt_quaternion, rotation_quaternion)

            # Set new orientation
            new_pose.orientation.x = combined_quaternion[0]
            new_pose.orientation.y = combined_quaternion[1]
            new_pose.orientation.z = combined_quaternion[2]
            new_pose.orientation.w = combined_quaternion[3]

            # Append to the list
            pose_array.poses.append(new_pose)

        linear_start_z = new_pose.position.z  # End position of the spiral
        num_linear_steps = 20  # Steps for linear descent
        pitch_per_linear_step = 0.009 / num_linear_steps  # Move down 1.2 cm evenly

        for i in range(num_linear_steps):
            new_pose = Pose()
            # Move straight down
            new_pose.position.x = start_x
            new_pose.position.y = start_y
            new_pose.position.z = linear_start_z - (pitch_per_linear_step * (i + 1))

            new_pose.orientation.x = combined_quaternion[0]
            new_pose.orientation.y = combined_quaternion[1]
            new_pose.orientation.z = combined_quaternion[2]
            new_pose.orientation.w = combined_quaternion[3]

            # Append to the list
            pose_array.poses.append(new_pose)


        # Publish the updated trajectory
        pose_array.header.stamp = self.get_clock().now().to_msg()
        pose_array.header.frame_id = "base_link"
        self.target_pose_array_publisher.publish(pose_array)

        self.get_logger().info("Published smooth two-phase screwing path: Spiral down (0.8 cm), then rotate (π) while descending (1.2 cm).")

    def Trajectory_USB_Female(self):
        self.get_logger().info("Starting USB")

        # Extract yaw angle from the current orientation
        _, _, yaw = self.quaternion_to_euler(
            self.current_pose.orientation.x, self.current_pose.orientation.y, 
            self.current_pose.orientation.z, self.current_pose.orientation.w
        )

        # Set speed scale for precise movements
        speed_scale = Float32(data=0.01)
        self.set_speed_scale_pub.publish(speed_scale)

        # Extract current position
        start_pose = self.current_pose
        start_x = start_pose.position.x
        start_y = start_pose.position.y
        start_z = start_pose.position.z

        pose_array = PoseArray()
        num_steps = 20  # More steps for a smoother trajectory
        total_downward_movement = 0.033 # Total downward movement (3.5 cm)
        total_tilt_angle = np.radians(0)  # Total tilt (5 degrees)
        total_x_movement = -0.000  # Total x-direction movement

        for i in range(num_steps):
            step_fraction = (i + 1) / num_steps  # Fraction of total movement

            # Compute incremental downward movement
            new_z = start_z - total_downward_movement * step_fraction  

            # Compute incremental tilt
            step_tilt = total_tilt_angle * step_fraction
            q_orig = (start_pose.orientation.x, start_pose.orientation.y, 
                    start_pose.orientation.z, start_pose.orientation.w)
            q_tilt_offset = quaternion_from_euler(0, step_tilt, 0)  # Tilt around Y-axis
            q_tilt_tuple = quaternion_multiply(q_orig, q_tilt_offset)

            # Convert back to ROS2 Quaternion message
            q_tilt = Quaternion()
            q_tilt.x, q_tilt.y, q_tilt.z, q_tilt.w = q_tilt_tuple

            # Compute incremental movement in X-direction (adjusted for yaw)
            move_x = total_x_movement * np.cos(yaw) * step_fraction
            move_y = total_x_movement * np.sin(yaw) * step_fraction

            # Create new pose
            new_pose = Pose()
            new_pose.position.x = start_x + move_x
            new_pose.position.y = start_y + move_y
            new_pose.position.z = new_z  # Apply downward motion
            new_pose.orientation = q_tilt  # Apply tilt

            pose_array.poses.append(new_pose)

        
        pose_array.header.stamp = self.get_clock().now().to_msg()
        pose_array.header.frame_id = "base_link"
        self.target_pose_array_publisher.publish(pose_array)

        self.get_logger().info("Published USB")

    def Trajectory_Ket_12_Female(self):
        self.get_logger().info("Starting precise screw trajectory")

        # Extract yaw angle
        _, _, yaw = self.quaternion_to_euler(
            self.current_pose.orientation.x, self.current_pose.orientation.y, 
            self.current_pose.orientation.z, self.current_pose.orientation.w
        )

        # Set slow movement
        self.set_speed_scale_pub.publish(Float32(data=0.1))

        start_pose = self.current_pose
        start_x, start_y, start_z = start_pose.position.x, start_pose.position.y, start_pose.position.z

        pose_array = PoseArray()

        # === Phase 1: Tilted Descent with Y-Movement ===
        num_steps = 21
        downward_z = 0.018  # 3.2 cm down
        total_tilt = np.radians(-10)  # 5-degree tilt
        total_y_move = np.tan(total_tilt) * 0.05 + 0.006 # Move in Y-direction

        for i in range(num_steps):
            fraction = (i + 1) / num_steps
            new_z = start_z - downward_z * fraction
            step_tilt = total_tilt * fraction

            q_tilt_tuple = quaternion_multiply(
                (start_pose.orientation.x, start_pose.orientation.y, start_pose.orientation.z, start_pose.orientation.w),
                quaternion_from_euler(step_tilt, 0, 0)  # Negative step tilt here
            )

            move_x = total_y_move * np.sin(yaw) * fraction
            move_y = total_y_move * np.cos(yaw) * fraction

            new_pose = Pose()
            new_pose.position.x = start_x + move_x
            new_pose.position.y = start_y - move_y
            new_pose.position.z = new_z

            # Correct Quaternion assignment
            new_pose.orientation = Quaternion(
                x=q_tilt_tuple[0], 
                y=q_tilt_tuple[1], 
                z=q_tilt_tuple[2], 
                w=q_tilt_tuple[3]
            )

            pose_array.poses.append(new_pose)
        

        final_x, final_y, final_z = new_pose.position.x, new_pose.position.y, new_pose.position.z
        final_tilt = new_pose.orientation

        # === Phase 2: Pure Y-Movement ===
        num_steps = 50
        move_x_1cm = -(0.0055) * np.sin(yaw)
        move_y_1cm = -(0.0055) * np.cos(yaw)
        total_descent = 0.007 # 2 mm descent

                        # Publish trajectory

        for i in range(num_steps):
            fraction = (i + 1) / num_steps
            move_pose = Pose()
            move_pose.position.x = final_x + move_x_1cm * fraction
            move_pose.position.y = final_y - move_y_1cm * fraction
            move_pose.position.z = final_z - (total_descent * fraction)  # Gradual descent
            move_pose.orientation = final_tilt
            pose_array.poses.append(move_pose)



        # === Phase 3: Reverse Tilt to Neutral ===
        num_steps = 20
        reverse_tilt = np.radians(10)
        start_pose = move_pose
        downward_z = 0.005  # 3.2 cm dow
        total_y_move = 0.002 + (np.tan(reverse_tilt)) * 0.05 # Move in Y-direction

        for i in range(num_steps):
            fraction = (i + 1) / num_steps
            step_tilt = reverse_tilt * fraction
            new_z = start_pose.position.z - downward_z * fraction
            move_x = total_y_move * np.sin(yaw) * fraction
            move_y = total_y_move * np.cos(yaw) * fraction

            q_tilt_tuple = quaternion_multiply(
                (start_pose.orientation.x, start_pose.orientation.y, start_pose.orientation.z, start_pose.orientation.w),  quaternion_from_euler(step_tilt, 0, 0)
            )

            neutral_pose = Pose()
            neutral_pose.position.x = start_pose.position.x + move_x
            neutral_pose.position.y = start_pose.position.y - move_y
            neutral_pose.position.z = new_z
            neutral_pose.orientation = Quaternion(
                x=q_tilt_tuple[0], 
                y=q_tilt_tuple[1], 
                z=q_tilt_tuple[2], 
                w=q_tilt_tuple[3]
            )

            pose_array.poses.append(neutral_pose)

        


        pose_array.header.stamp = self.get_clock().now().to_msg()
        pose_array.header.frame_id = "base_link"
        self.target_pose_array_publisher.publish(pose_array)

        # === Phase 4: Straight Descent ===
        num_steps = 200
        linear_z = 0.005

        for i in range(num_steps):
            descent_pose = Pose()
            descent_pose.position.x = neutral_pose.position.x
            descent_pose.position.y = neutral_pose.position.y
            descent_pose.position.z = neutral_pose.position.z - (linear_z / num_steps) * (i + 1)
            descent_pose.orientation = neutral_pose.orientation
            pose_array.poses.append(descent_pose)

    def Trajectory_Waterproof_Female(self):
        # Extract yaw angle from the current orientation
        _, _, yaw = self.quaternion_to_euler(
            self.current_pose.orientation.x, self.current_pose.orientation.y, 
            self.current_pose.orientation.z, self.current_pose.orientation.w
        )

        # Set speed scale for precise movements
        speed_scale = Float32(data=0.1)
        self.set_speed_scale_pub.publish(speed_scale)

        # Extract current position
        start_pose = self.current_pose
        start_x = start_pose.position.x
        start_y = start_pose.position.y
        start_z = start_pose.position.z

        pose_array = PoseArray()

        # === Phase 1: Move in Y-Direction Based on Yaw ===
        num_steps_y = 20  # More steps for smooth motion
        total_movement = 0.044  # Move 4.4 cm along robot's Y-direction

        # Compute movement components in X and Y based on yaw
        move_x_total = total_movement * np.sin(yaw)
        move_y_total = total_movement * np.cos(yaw)

        for i in range(num_steps_y):
            step_fraction = (i + 1) / num_steps_y  # Fraction of total movement
            move_pose = Pose()
            move_pose.position.x = start_x + move_x_total * step_fraction  # Adjusted X movement
            move_pose.position.y = start_y - move_y_total * step_fraction  # Adjusted Y movement
            move_pose.position.z = start_z  # Maintain height
            move_pose.orientation = start_pose.orientation  # Keep original orientation
            pose_array.poses.append(move_pose)

        # Update start position after Y-movement
        start_x = move_pose.position.x
        start_y = move_pose.position.y
        start_z = move_pose.position.z

        # === Phase 1: Tilted Descent with Y-Movement ===
        num_steps = 50
        downward_z = 0.06  # 3.2 cm down
        total_tilt = np.radians(-10)  # 5-degree tilt
        total_x_move = np.tan(total_tilt) * 0.03 + 0.002 # Move in Y-direction

        for i in range(num_steps):
            fraction = (i + 1) / num_steps
            new_z = start_z - downward_z * fraction
            step_tilt = total_tilt * fraction

            q_tilt_tuple = quaternion_multiply(
                (start_pose.orientation.x, start_pose.orientation.y, start_pose.orientation.z, start_pose.orientation.w),
                quaternion_from_euler(0, step_tilt,0)  # Negative step tilt here
            )

            move_x = total_x_move * np.cos(yaw) * fraction
            move_y = total_x_move * np.sin(yaw) * fraction

            new_pose = Pose()
            new_pose.position.x = start_x - move_x
            new_pose.position.y = start_y - move_y
            new_pose.position.z = new_z

            # Correct Quaternion assignment
            new_pose.orientation = Quaternion(
                x=q_tilt_tuple[0], 
                y=q_tilt_tuple[1], 
                z=q_tilt_tuple[2], 
                w=q_tilt_tuple[3]
            )

            pose_array.poses.append(new_pose)
        final_x, final_y, final_z = new_pose.position.x, new_pose.position.y, new_pose.position.z
        final_tilt = new_pose.orientation

        # === Phase 2: Pure Y-Movement ===
        num_steps = 50
        x_movement = -0.0105
        move_x_1cm = (x_movement) * np.cos(yaw)
        move_y_1cm = (x_movement) * np.sin(yaw)
        total_descent = 0.001 # 2 mm descent

        for i in range(num_steps):
            fraction = (i + 1) / num_steps
            move_pose = Pose()
            move_pose.position.x = final_x - move_x_1cm * fraction
            move_pose.position.y = final_y - move_y_1cm * fraction
            move_pose.position.z = final_z - (total_descent * fraction)  # Gradual descent
            move_pose.orientation = final_tilt
            pose_array.poses.append(move_pose)

        # === Phase 3: Reverse Tilt to Neutral ===
        num_steps = 20
        reverse_tilt = np.radians(10)
        start_pose = move_pose
        downward_z = 0.007  # 3.2 cm dow
        total_x_move = 0.007 + (np.tan(reverse_tilt)) * 0.03 # Move in Y-direction

        for i in range(num_steps):
            fraction = (i + 1) / num_steps
            step_tilt = reverse_tilt * fraction
            new_z = start_pose.position.z - downward_z * fraction
            move_x = total_x_move * np.cos(yaw) * fraction
            move_y = total_x_move * np.sin(yaw) * fraction

            q_tilt_tuple = quaternion_multiply(
                (start_pose.orientation.x, start_pose.orientation.y, start_pose.orientation.z, start_pose.orientation.w),  quaternion_from_euler(0, step_tilt, 0)
            )

            neutral_pose = Pose()
            neutral_pose.position.x = start_pose.position.x + move_x
            neutral_pose.position.y = start_pose.position.y - move_y
            neutral_pose.position.z = new_z
            neutral_pose.orientation = Quaternion(
                x=q_tilt_tuple[0], 
                y=q_tilt_tuple[1], 
                z=q_tilt_tuple[2], 
                w=q_tilt_tuple[3]
            )

            pose_array.poses.append(neutral_pose)

        # === Phase 4: Straight Descent ===
        num_steps = 20
        linear_z = 0.005

        for i in range(num_steps):
            descent_pose = Pose()
            descent_pose.position.x = neutral_pose.position.x
            descent_pose.position.y = neutral_pose.position.y
            descent_pose.position.z = neutral_pose.position.z - (linear_z / num_steps) * (i + 1)
            descent_pose.orientation = neutral_pose.orientation
            pose_array.poses.append(descent_pose)


        # Publish the updated trajectory
        pose_array.header.stamp = self.get_clock().now().to_msg()
        pose_array.header.frame_id = "base_link"
        self.target_pose_array_publisher.publish(pose_array)

        self.get_logger().info("Published smooth two-phase screwing path: Move in Y (4.05 cm) → Spiral down (7 cm).")

    def Trajectory_Ket_8_Female(self):
        self.get_logger().info("Starting precise screw trajectory")

        # Extract yaw angle
        _, _, yaw = self.quaternion_to_euler(
            self.current_pose.orientation.x, self.current_pose.orientation.y, 
            self.current_pose.orientation.z, self.current_pose.orientation.w
        )

        # Set slow movement
        self.set_speed_scale_pub.publish(Float32(data=0.1))

        start_pose = self.current_pose
        start_x, start_y, start_z = start_pose.position.x, start_pose.position.y, start_pose.position.z

        pose_array = PoseArray()

        # === Phase 1: Tilted Descent with Y-Movement ===
        num_steps = 20
        downward_z = 0.0165  # 3.2 cm down
        total_tilt = np.radians(-10)  # 5-degree tilt
        total_y_move = np.tan(total_tilt) * 0.05 + 0.012  # Move in Y-direction

        for i in range(num_steps):
            fraction = (i + 1) / num_steps
            new_z = start_z - downward_z * fraction
            step_tilt = total_tilt * fraction

            q_tilt_tuple = quaternion_multiply(
                (start_pose.orientation.x, start_pose.orientation.y, start_pose.orientation.z, start_pose.orientation.w),
                quaternion_from_euler(step_tilt, 0, 0)  # Negative step tilt here
            )

            move_x = total_y_move * np.sin(yaw) * fraction
            move_y = total_y_move * np.cos(yaw) * fraction

            new_pose = Pose()
            new_pose.position.x = start_x + move_x
            new_pose.position.y = start_y - move_y
            new_pose.position.z = new_z

            # Correct Quaternion assignment
            new_pose.orientation = Quaternion(
                x=q_tilt_tuple[0], 
                y=q_tilt_tuple[1], 
                z=q_tilt_tuple[2], 
                w=q_tilt_tuple[3]
            )


            pose_array.poses.append(new_pose)



        final_x, final_y, final_z = new_pose.position.x, new_pose.position.y, new_pose.position.z
        final_tilt = new_pose.orientation

        # === Phase 2: Pure Y-Movement ===
        num_steps = 50
        move_x_1cm = -(0.006) * np.sin(yaw)
        move_y_1cm = -(0.006) * np.cos(yaw)
        total_descent = 0.005  # 2 mm descent

        for i in range(num_steps):
            fraction = (i + 1) / num_steps
            move_pose = Pose()
            move_pose.position.x = final_x + move_x_1cm * fraction
            move_pose.position.y = final_y - move_y_1cm * fraction
            move_pose.position.z = final_z - (total_descent * fraction)  # Gradual descent
            move_pose.orientation = final_tilt
            pose_array.poses.append(move_pose)



        # === Phase 3: Reverse Tilt to Neutral ===
        num_steps = 20
        reverse_tilt = np.radians(10)
        start_pose = move_pose
        downward_z = 0.00  # 3.2 cm dow
        total_y_move = 0.005 + (np.tan(reverse_tilt)) * 0.05 # Move in Y-direction

        for i in range(num_steps):
            fraction = (i + 1) / num_steps
            step_tilt = reverse_tilt * fraction
            new_z = start_pose.position.z - downward_z * fraction
            move_x = total_y_move * np.sin(yaw) * fraction
            move_y = total_y_move * np.cos(yaw) * fraction

            q_tilt_tuple = quaternion_multiply(
                (start_pose.orientation.x, start_pose.orientation.y, start_pose.orientation.z, start_pose.orientation.w),  quaternion_from_euler(step_tilt, 0, 0)
            )

            neutral_pose = Pose()
            neutral_pose.position.x = start_pose.position.x + move_x
            neutral_pose.position.y = start_pose.position.y - move_y
            neutral_pose.position.z = new_z
            neutral_pose.orientation = Quaternion(
                x=q_tilt_tuple[0], 
                y=q_tilt_tuple[1], 
                z=q_tilt_tuple[2], 
                w=q_tilt_tuple[3]
            )

            pose_array.poses.append(neutral_pose)




        # === Phase 4: Straight Descent ===
        num_steps = 20
        linear_z = 0.005

        for i in range(num_steps):
            descent_pose = Pose()
            descent_pose.position.x = neutral_pose.position.x
            descent_pose.position.y = neutral_pose.position.y
            descent_pose.position.z = neutral_pose.position.z - (linear_z / num_steps) * (i + 1)
            descent_pose.orientation = neutral_pose.orientation
            pose_array.poses.append(descent_pose)

        

        self.get_logger().info("Published precise screwing path: Tilted descent → Y-shift")      
        # Publish trajectory
        pose_array.header.stamp = self.get_clock().now().to_msg()
        pose_array.header.frame_id = "base_link"
        self.target_pose_array_publisher.publish(pose_array)

    def process_waypoints(self, current_pose, new_pose):
        """Stub function for processing waypoints."""
        # Initialize PoseArray
        pose_array = PoseArray()
        pose_array.poses.append(current_pose)
        pose_array.poses.append(new_pose)
        pose_array.header.stamp = self.get_clock().now().to_msg()
        pose_array.header.frame_id = "base_link"  # Replace with the appropriate frame_id
        # Publish the PoseArray
        self.target_pose_array_publisher.publish(pose_array)
        
def main(args=None):
    rclpy.init(args=args)
    main_node = MainNode()
    rclpy.spin(main_node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
