import cv2
import numpy as np

def estimate_camera_pose(yolo_data, Z_c, yaw, offset_x, offset_y, offset_z):
    # Funktion teilweise unnötig (solvePnP hier unnötig). Muss überarbeitet werden. 
    # Camera parameters
    Z_c = (Z_c - 0.04)/np.cos(np.radians(15)) 
    print(f"Höhe: {Z_c}")
    
    horizontal_fov = 1.2054  # Horizontal FOV in radians
    image_width = 640  # Image width in pixels
    image_height = 480  # Image height in pixels

    # Compute focal length in pixels
    f = image_width / (2 * np.tan(horizontal_fov / 2))

    # Principal point (center of the image)
    u0 = image_width / 2.0
    v0 = image_height / 2.0

    # Camera intrinsic matrix
    camera_matrix = np.array([[f, 0, u0],
                               [0, f, v0],
                               [0, 0, 1]], dtype=np.float64)

    # Distortion coefficients (assuming no distortion)
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    x_center, y_center, width, height = yolo_data
    # Calculate the corner points of the bounding box in image coordinates
    x_min = x_center - width / 2
    x_max = x_center + width / 2
    y_min = y_center - height / 2
    y_max = y_center + height / 2

    # Create image points: top-left, top-right, bottom-left, bottom-right
    image_points = np.array([
        [x_min, y_min],
        [x_max, y_min],
        [x_min, y_max],
        [x_max, y_max]
    ], dtype=np.float64)

    # Calculate center point coordinates of the bounding box
    center_x = np.mean(image_points[:, 0])
    center_y = np.mean(image_points[:, 1])

    # Convert 2D image points to 3D world points
    object_points = []
    for point in image_points:
        # Convert image coordinates to normalized coordinates
        x = (point[0] - center_x) / f
        y = (point[1] - center_y) / f

        # Convert normalized coordinates to world coordinates
        xZc = x * Z_c
        yZc = y * Z_c
        object_points.append([xZc, yZc, 0])

    object_points = np.array(object_points, dtype=np.float64)

    # Solve for rotation and translation vectors
    success, rvec, tvec = cv2.solvePnP(object_points, 
                                       image_points, 
                                       camera_matrix, 
                                       dist_coeffs)

    if success:
        # Convert rotation vector to rotation matrix
        R, _ = cv2.Rodrigues(rvec)
        yaw_cam = yaw
        yaw = -yaw  # Transform yaw to align with the robot coordinate system

        # Tilt adjustment (fixed tilt)
        tilt_angle = np.radians(-15)
        tilt_matrix = np.array([[1, 0, 0],
                                [0, np.cos(tilt_angle), -np.sin(tilt_angle)],
                                [0, np.sin(tilt_angle), np.cos(tilt_angle)]])

        # Apply yaw (robot frame rotation around z-axis)
        yaw_matrix_robot = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                                    [np.sin(yaw), np.cos(yaw), 0],
                                    [0, 0, 1]])

        # Separate camera yaw relative to the world
        yaw_matrix_cam = np.array([[np.cos(yaw_cam), -np.sin(yaw_cam), 0],
                                [np.sin(yaw_cam), np.cos(yaw_cam), 0],
                                [0, 0, 1]])

        # First apply yaw to robot frame
        R_yawed = yaw_matrix_robot @ R

        # Then apply tilt (camera tilt relative to end effector)
        R_final = tilt_matrix @ R_yawed

        # Define offset in robot's frame when yaw = 0
        offset_in_robot = np.array([offset_x, -0.045 + offset_y, offset_z], dtype=np.float64).reshape((3, 1))

        # Correctly transform offset to world frame using camera's yaw matrix
        offset_in_world = yaw_matrix_cam @ offset_in_robot


        # Compute camera position

        camera_position = -R_final.T @ tvec  # Compute initial camera position

        # Apply the mounting offset
        camera_position += offset_in_world
        return camera_position
    

def estimate_translation(t_x, t_y, Z_c, yaw, off_x, off_y):
    print(yaw)
    yaw = -yaw  # Inverting yaw as per your original function
    Z_c = (Z_c - 0.055) / np.cos(np.radians(15))  
    
    # Camera parameters
    horizontal_fov = 1.2054  # Horizontal FOV in radians
    image_width = 640  # Image width in pixels
    image_height = 480  # Image height in pixels

    # Compute focal length in pixels
    f = image_width / (2 * np.tan(horizontal_fov / 2))

    # Convert pixel translation (t_x, t_y) to normalized image coordinates
    delta_u = t_x  # Pixel shift along x-axis
    delta_v = t_y  # Pixel shift along y-axis

    # Convert pixel displacement to real-world displacement in camera frame
    X_c = (delta_u / f) * Z_c
    Y_c = (delta_v / f) * Z_c  # Y_c is in camera coordinates

    # Transform from camera coordinates to world coordinates using yaw
    X_w = np.cos(yaw) * X_c - np.sin(yaw) * Y_c
    Y_w = np.sin(yaw) * X_c + np.cos(yaw) * Y_c

    
    return X_w, Y_w