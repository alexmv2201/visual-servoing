import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'assembling_task_board'

def get_all_files(base_folder):
    return [
            (
                os.path.join('share', package_name, base_folder, os.path.relpath(os.path.dirname(f), base_folder)),
                [f]
            )
            for f in glob(os.path.join(base_folder, '**'), recursive=True)
            if os.path.isfile(f)
        ]

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
   install_requires=[
        'setuptools', 
        'torch', 
        'opencv-python', 
        'cv_bridge',  # Add other dependencies here if needed
        'ultralytics',
        'tf2_geometry_msgs',  # Add tf2_geometry_msgs for transforms between ROS messages
        'tf2_py',  # Optional for Python bindings to tf2

    ],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/combine_launch.py']),  # Include launch files
        ('share/' + package_name + '/launch', ['launch/combine_launch_sim.py']),  # Include launch files
        ('share/' + package_name + '/functions', ['functions/calculate_camera_pose.py']),  # Include CNN folder
    ]+ get_all_files('Task Board information'),
    zip_safe=True,
    maintainer='alex',
    maintainer_email='alex@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            "lightGlue_node = src.lightGlue_node:main",
            "calculatingOffset_node = src.calculatingOffset_node:main",
            "driveToInput_node = src.drive_to_input_pose_node:main",
            "takeImages_node = src.take_images_node:main",
            "main_node = src.main_node:main",
            "simulatingPathes_node = src.simulatingPathes_node:main",
            "gripper_node = src.gripper_node:main",
            "watchPathes_node = src.watchPathes_node:main", 
            "twistPublisher_node = src.twistPublisher_node:main", 
            "controlSwitcher_node = src.controlSwitcher_node:main",
            "controlSwitcher_sim_node = src.controlSwitcher_sim_node:main"
        ],
    },
)