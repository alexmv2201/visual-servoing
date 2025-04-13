cmake_minimum_required(VERSION 3.8)
project(trajectory_planning)

# Add compile options
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Specify C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


# Find dependencies
find_package(ament_cmake REQUIRED)

find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)         # Add std_msgs package
find_package(geometry_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(moveit_ros_planning_interface REQUIRED)
find_package(moveit_core REQUIRED)
find_package(moveit_ros_planning REQUIRED)
find_package(moveit_ros_move_group REQUIRED)
find_package(OpenCV REQUIRED)

# Include directories
include_directories(
  include
  ${rclcpp_INCLUDE_DIRS}
  ${moveit_ros_planning_interface_INCLUDE_DIRS}
  ${moveit_core_INCLUDE_DIRS}
  ${moveit_ros_planning_INCLUDE_DIRS}
  ${moveit_ros_move_group_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${std_msgs_INCLUDE_DIRS}               # Include std_msgs directory
  ${rosidl_default_generators_INCLUDE_DIRS}
)


# Declare executables

add_executable(trajectory_planning_node src/trajectory_planning_node)

add_executable(trajectory_planning_sim_node src/trajectory_planning_sim_node)

# Set compile features

target_compile_features(trajectory_planning_node PUBLIC cxx_std_17)
target_compile_features(trajectory_planning_sim_node PUBLIC cxx_std_17)

ament_target_dependencies(trajectory_planning_node

  "rclcpp"
  "geometry_msgs"
  "std_msgs"                             # Link std_msgs dependency
  "moveit_ros_planning_interface"
  "moveit_core"
  "moveit_ros_planning"
  "moveit_ros_move_group")

ament_target_dependencies(trajectory_planning_sim_node

  "rclcpp"
  "geometry_msgs"
  "std_msgs"                             # Link std_msgs dependency
  "moveit_ros_planning_interface"
  "moveit_core"
  "moveit_ros_planning"
  "moveit_ros_move_group")

# Install targets
install(TARGETS trajectory_planning_node trajectory_planning_sim_node
  DESTINATION lib/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
