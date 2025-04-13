#include <memory>
#include <rclcpp/rclcpp.hpp>
#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/planning_interface/planning_interface.h>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <sstream>
#include <string>
#include <moveit/robot_model_loader/robot_model_loader.h>
#include <moveit/robot_model/robot_model.h>
#include <moveit/robot_state/robot_state.h>
#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/float32.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>
#include <geometry_msgs/msg/pose_array.hpp>
#include <moveit/planning_scene_interface/planning_scene_interface.h>
#include <shape_msgs/msg/solid_primitive.hpp>
#include <geometric_shapes/solid_primitive_dims.h>
#include <moveit/robot_state/conversions.h>
#include <chrono>
#include "sensor_msgs/msg/joint_state.hpp"

using std::placeholders::_1;
using namespace std::chrono_literals;

class Trajectory_Planning_Node : public rclcpp::Node
{
public:
  explicit Trajectory_Planning_Node(const rclcpp::NodeOptions &options);

private:
  double speed_scale;

  moveit::planning_interface::MoveGroupInterfacePtr move_group_interface_;
  rclcpp::Node::SharedPtr node_;
  rclcpp::Executor::SharedPtr executor_;
  std::thread executor_thread_;

  // Publishers
  rclcpp::Publisher<geometry_msgs::msg::Pose>::SharedPtr publisher_;
  rclcpp::Publisher<geometry_msgs::msg::Pose>::SharedPtr pose_publisher_;
  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr ik_solution_publisher_;

  // Subscriptions
  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr tcp_subscriber;
  rclcpp::Subscription<geometry_msgs::msg::Pose>::SharedPtr pose_subscription_;
  rclcpp::Subscription<geometry_msgs::msg::PoseArray>::SharedPtr pose_array_subscription_;
  rclcpp::Subscription<std_msgs::msg::Float32MultiArray>::SharedPtr array_subscription_;
  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr request_current_pose_subscription_;
  rclcpp::Subscription<std_msgs::msg::Float32>::SharedPtr joint_subscription_;
  rclcpp::Subscription<std_msgs::msg::Float32>::SharedPtr speed_scale_sub_;
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr all_joint_subscription_;

  // Callbacks
  void task_space_callback(const geometry_msgs::msg::PoseArray::SharedPtr waypoint_array_msg);
  void joint_space_callback(const geometry_msgs::msg::Pose::SharedPtr pose);
  void publish_pose(const std_msgs::msg::Bool::SharedPtr msg);
  void target_wrist_joint_callback(const std_msgs::msg::Float32::SharedPtr msg);
  void set_speed_scale_callback(const std_msgs::msg::Float32::SharedPtr msg);
  void target_joints_callback(const sensor_msgs::msg::JointState::SharedPtr msg);
  void addCollisionObjects(moveit::planning_interface::PlanningSceneInterface &planning_scene_interface);
};

Trajectory_Planning_Node::Trajectory_Planning_Node(const rclcpp::NodeOptions &options)
  : Node("trajectory_planning_node", options),
    node_(std::make_shared<rclcpp::Node>(
      "trajectory_planning_node_shared",
      rclcpp::NodeOptions().allow_undeclared_parameters(true))),
    executor_(std::make_shared<rclcpp::executors::SingleThreadedExecutor>())
{
  auto const logger = rclcpp::get_logger("hello_moveit");

  // Sync parameters with MoveGroup
  auto parameters_client = std::make_shared<rclcpp::SyncParametersClient>(node_, "/move_group");
  while (!parameters_client->wait_for_service(1s)) {
    if (!rclcpp::ok()) {
      RCLCPP_ERROR(node_->get_logger(), "Interrupted while waiting for the service. Exiting.");
      rclcpp::shutdown();
    }
    RCLCPP_INFO(node_->get_logger(), "Service not available, waiting again...");
  }
  speed_scale =1.0;
  auto parameter_list = parameters_client->list_parameters({"robot_description_kinematics"}, 10);
  auto parameters = parameters_client->get_parameters(parameter_list.names);
  node_->set_parameters(parameters);

  // Setup MoveGroupInterface
  auto mgi_options = moveit::planning_interface::MoveGroupInterface::Options("ur_manipulator");
  move_group_interface_ = std::make_shared<moveit::planning_interface::MoveGroupInterface>(node_, mgi_options);

  // Start executor thread
  executor_->add_node(node_);
  executor_thread_ = std::thread([this]() { this->executor_->spin(); });

  // Subscriptions
  pose_subscription_ = this->create_subscription<geometry_msgs::msg::Pose>(
    "joint_space", 10, std::bind(&Trajectory_Planning_Node::joint_space_callback, this, _1));

  joint_subscription_ = this->create_subscription<std_msgs::msg::Float32>(
    "target_wrist_joint", 10, std::bind(&Trajectory_Planning_Node::target_wrist_joint_callback, this, _1));

  speed_scale_sub_ = this->create_subscription<std_msgs::msg::Float32>(
    "speed_scale", 10, std::bind(&Trajectory_Planning_Node::set_speed_scale_callback, this, _1));

  all_joint_subscription_ = this->create_subscription<sensor_msgs::msg::JointState>(
    "target_joints", 10, std::bind(&Trajectory_Planning_Node::target_joints_callback, this, _1));

  pose_array_subscription_ = this->create_subscription<geometry_msgs::msg::PoseArray>(
    "task_space", 10, std::bind(&Trajectory_Planning_Node::task_space_callback, this, _1));

  request_current_pose_subscription_ = this->create_subscription<std_msgs::msg::Bool>(
    "ask_current_pose", 10, std::bind(&Trajectory_Planning_Node::publish_pose, this, _1));

  // Publishers
  publisher_ = this->create_publisher<geometry_msgs::msg::Pose>("target_pose_reached", 10);
  pose_publisher_ = this->create_publisher<geometry_msgs::msg::Pose>("current_pose_publisher", 10);
  ik_solution_publisher_ = this->create_publisher<sensor_msgs::msg::JointState>("/ik_solutions", 10);

  // Add collision objects
  moveit::planning_interface::PlanningSceneInterface planning_scene_interface;
  addCollisionObjects(planning_scene_interface);
}

void Trajectory_Planning_Node::addCollisionObjects(moveit::planning_interface::PlanningSceneInterface& planning_scene_interface) {
    // Define the table
    moveit_msgs::msg::CollisionObject table;
    table.id = "table";
    table.header.frame_id = "world";
  
    shape_msgs::msg::SolidPrimitive table_primitive;
    table_primitive.type = shape_msgs::msg::SolidPrimitive::BOX;
    table_primitive.dimensions.resize(3);
    table_primitive.dimensions[shape_msgs::msg::SolidPrimitive::BOX_X] = 0.96; // Length
    table_primitive.dimensions[shape_msgs::msg::SolidPrimitive::BOX_Y] = 0.76; // Width
    table_primitive.dimensions[shape_msgs::msg::SolidPrimitive::BOX_Z] = 1.0; // Height
  
    geometry_msgs::msg::Pose table_pose;
    table_pose.position.x = 0.347;
    table_pose.position.y = 0.272;
    table_pose.position.z = 0.5;
    table_pose.orientation.w = 1.0;
  
    table.primitives.push_back(table_primitive);
    table.primitive_poses.push_back(table_pose);
    table.operation = table.ADD;
  
    // Define the Task Board
    moveit_msgs::msg::CollisionObject task_board;
    task_board.id = "Task Board";
    task_board.header.frame_id = "world";

    shape_msgs::msg::SolidPrimitive task_board_primitive;
    task_board_primitive.type = shape_msgs::msg::SolidPrimitive::BOX;
    task_board_primitive.dimensions.resize(3);
    task_board_primitive.dimensions[shape_msgs::msg::SolidPrimitive::BOX_X] = 0.35; // Length
    task_board_primitive.dimensions[shape_msgs::msg::SolidPrimitive::BOX_Y] = 0.35; // Width
    task_board_primitive.dimensions[shape_msgs::msg::SolidPrimitive::BOX_Z] = 0.03; // Height

    geometry_msgs::msg::Pose task_board_pose;
    task_board_pose.position.x = 0.5;
    task_board_pose.position.y = 0.3;
    task_board_pose.position.z = 1.015;
    task_board_pose.orientation.w = 1.0;

    task_board.primitives.push_back(task_board_primitive);
    task_board.primitive_poses.push_back(task_board_pose);
    task_board.operation = task_board.ADD;

    // Define the wall
    moveit_msgs::msg::CollisionObject wall;
    wall.id = "wall";
    wall.header.frame_id = "world";

    shape_msgs::msg::SolidPrimitive wall_primitive;
    wall_primitive.type = shape_msgs::msg::SolidPrimitive::BOX;
    wall_primitive.dimensions.resize(3);
    wall_primitive.dimensions[shape_msgs::msg::SolidPrimitive::BOX_X] = 0.953;    // Thickness
    wall_primitive.dimensions[shape_msgs::msg::SolidPrimitive::BOX_Y] = 0.1;  // Width (same as table)
    wall_primitive.dimensions[shape_msgs::msg::SolidPrimitive::BOX_Z] = 2.0;  // Height (2 meters tall)

    geometry_msgs::msg::Pose wall_pose;
    wall_pose.position.x = 0.3435;  // Adjust to place next to the table
    wall_pose.position.y = -0.158;
    wall_pose.position.z = 1.0;  // Wall's center at 1 meter height
    wall_pose.orientation.w = 1.0;

    wall.primitives.push_back(wall_primitive);
    wall.primitive_poses.push_back(wall_pose);
    wall.operation = wall.ADD;

    // Define the roof above the wall
    moveit_msgs::msg::CollisionObject roof;
    roof.id = "roof";
    roof.header.frame_id = "world";

    shape_msgs::msg::SolidPrimitive roof_primitive;
    roof_primitive.type = shape_msgs::msg::SolidPrimitive::BOX;
    roof_primitive.dimensions.resize(3);
    roof_primitive.dimensions[shape_msgs::msg::SolidPrimitive::BOX_X] = 0.953;  // Length (same as wall)
    roof_primitive.dimensions[shape_msgs::msg::SolidPrimitive::BOX_Y] = 0.6;  // Width (a bit wider to cover the wall)
    roof_primitive.dimensions[shape_msgs::msg::SolidPrimitive::BOX_Z] = 0.1;  // Thickness

    geometry_msgs::msg::Pose roof_pose;
    roof_pose.position.x = 0.3435;  // Same x position as the wall
    roof_pose.position.y = 0.1;
    roof_pose.position.z = 2.05;  // Positioned just above the wall
    roof_pose.orientation.w = 1.0;

    roof.primitives.push_back(roof_primitive);
    roof.primitive_poses.push_back(roof_pose);
    roof.operation = roof.ADD;

    // Apply all objects to the planning scene
    planning_scene_interface.applyCollisionObjects({table, task_board, wall, roof});
}

void Trajectory_Planning_Node::set_speed_scale_callback(const std_msgs::msg::Float32::SharedPtr msg)
{
    speed_scale = msg->data;
    RCLCPP_INFO(this->get_logger(), "Recieved new Speed scale %f", speed_scale);
}

void Trajectory_Planning_Node::target_wrist_joint_callback(const std_msgs::msg::Float32::SharedPtr msg)
{
  // Extract the target wrist joint angle (in radians)
  float target_angle = msg->data;

  // Log the target angle
  RCLCPP_INFO(this->get_logger(), "Received target wrist joint angle: %f", target_angle);

  // Get the current joint values (if needed)
  std::vector<double> current_joint_values = move_group_interface_->getCurrentJointValues();

  // Assuming the wrist joint is the last joint in the vector (adjust based on your robot)
  current_joint_values.back() += target_angle;

  // Set the new joint target
  double velocity_scaling = speed_scale;  // 50% of max velocity
  double acceleration_scaling = 0.1;  // 50% of max acceleration

  move_group_interface_->setMaxVelocityScalingFactor(velocity_scaling);
  move_group_interface_->setMaxAccelerationScalingFactor(acceleration_scaling);
  move_group_interface_->setJointValueTarget(current_joint_values);

  // Plan and execute the motion to the new joint position
  moveit::planning_interface::MoveGroupInterface::Plan plan;
  move_group_interface_->plan(plan);
  moveit::core::MoveItErrorCode execute_status = move_group_interface_->execute(plan);
  move_group_interface_->execute(plan);

  bool move_success = (execute_status == moveit::core::MoveItErrorCode::SUCCESS);

  if (move_success){
     geometry_msgs::msg::PoseStamped current_pose_stamped = move_group_interface_->getCurrentPose();
      geometry_msgs::msg::Pose current_pose = current_pose_stamped.pose;
      publisher_->publish(current_pose);
  }
  
}

void Trajectory_Planning_Node::target_joints_callback(const sensor_msgs::msg::JointState::SharedPtr msg)
{
    int i = 0;
    while (rclcpp::ok())  // Continue as long as the node is active
    {
        i += 1;
        if (i == 5) {
            break;
        }
        // Extract joint positions from the incoming message
        std::vector<double> joint_positions = msg->position;

        // Set the joint target for motion planning
        move_group_interface_->setJointValueTarget(joint_positions);

        // Get the joint model group for better clarity in logging
        const moveit::core::JointModelGroup* joint_model_group = move_group_interface_->getCurrentState()->getJointModelGroup("ur_manipulator"); // Replace "arm" with your actual group name

        // Log the joint configuration
        const std::vector<std::string>& joint_names = joint_model_group->getVariableNames();
        RCLCPP_INFO(this->get_logger(), "Setting joint target:");
        for (size_t i = 0; i < joint_positions.size(); ++i) {
            RCLCPP_INFO(this->get_logger(), "%s: %f", joint_names[i].c_str(), joint_positions[i]);
        }

        // Plan the motion
        move_group_interface_->setPlanningTime(3.0);
        double velocity_scaling = speed_scale;  // 50% of max velocity
        move_group_interface_->setMaxVelocityScalingFactor(velocity_scaling);
        move_group_interface_->setMaxAccelerationScalingFactor(0.3);
        moveit::planning_interface::MoveGroupInterface::Plan plan;
        bool success = (move_group_interface_->plan(plan) == moveit::core::MoveItErrorCode::SUCCESS);

        if (success)
        {
            const auto& trajectory = plan.trajectory.joint_trajectory;
            if (!trajectory.points.empty()) {
                const auto& last_waypoint = trajectory.points.back();
                std::vector<double> last_joint_positions = last_waypoint.positions;

                bool match = true;
                for (size_t i = 0; i < joint_positions.size(); ++i) {
                    if (std::abs(joint_positions[i] - last_joint_positions[i]) > 1e-3) { // Tolerance of 0.001 radians
                        match = false;
                        RCLCPP_WARN(this->get_logger(), "Mismatch at Joint %zu: Target=%f, Plan=%f", i, joint_positions[i], last_joint_positions[i]);
                    }
                }

                if (match) {
                    RCLCPP_INFO(this->get_logger(), "Last waypoint matches joint target.");
                } else {
                    success = false;
                    RCLCPP_WARN(this->get_logger(), "Last waypoint does NOT match joint target.");
                }
            }

            if (success)
            {
                RCLCPP_INFO(this->get_logger(), "Plan found, executing.");
                moveit::core::MoveItErrorCode execute_status = move_group_interface_->move();
                if (execute_status == moveit::core::MoveItErrorCode::SUCCESS)
                {
                    // Publish the joint state after execution
                    auto joint_state_msg = sensor_msgs::msg::JointState();
                    joint_state_msg.header.stamp = this->get_clock()->now();
                    joint_state_msg.name = joint_names; // Replace with actual joint names if needed
                    joint_state_msg.position = move_group_interface_->getCurrentJointValues(); // Populate with current joint positions

                    // Publish the JointState message
                    ik_solution_publisher_->publish(joint_state_msg);

                    // Get and publish current pose
                    geometry_msgs::msg::PoseStamped current_pose_stamped = move_group_interface_->getCurrentPose();
                    geometry_msgs::msg::Pose current_pose = current_pose_stamped.pose;
                    publisher_->publish(current_pose);
                    break;

                    RCLCPP_INFO(this->get_logger(), "Motion executed successfully.");
                }
                else
                {
                    RCLCPP_ERROR(this->get_logger(), "Execution failed. Retrying...");
                }
            }
            else
            {
                RCLCPP_ERROR(this->get_logger(), "Failed to plan. Retrying...");
            }
        }
        else
        {
            RCLCPP_ERROR(this->get_logger(), "Failed to plan. Retrying...");
        }
    }
}

void Trajectory_Planning_Node::joint_space_callback(const geometry_msgs::msg::Pose::SharedPtr pose_)
{    
    int i = 0;
    geometry_msgs::msg::Pose pose = *pose_;
    while (rclcpp::ok())  // Continue as long as the node is active
    {
        i += 1;
        if (i == 5) {
            break;
        }
        move_group_interface_->setStartStateToCurrentState();
        auto const logger = this->get_logger();
        RCLCPP_INFO(
            this->get_logger(),
            "Received Pose: x: '%f', y: '%f', z: '%f', xx: '%f', yy: '%f', zz: '%f', ww: '%f'",
            pose.position.x, pose.position.y, pose.position.z,
            pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w);

        // Retrieve the current state of the robot
        moveit::core::RobotStatePtr current_state = move_group_interface_->getCurrentState(10);
        const moveit::core::JointModelGroup* joint_model_group =
        current_state->getJointModelGroup(move_group_interface_->getName());

        RCLCPP_INFO(logger, "Trying to find IK.");
        // Solve Inverse Kinematics
        std::vector<double> joint_values;
        bool ik_success = current_state->setFromIK(joint_model_group, pose, 10);

        if (ik_success)
        {
            // Get the joint names for better clarity
            const std::vector<std::string>& joint_names = joint_model_group->getVariableNames();
            current_state->copyJointGroupPositions(joint_model_group, joint_values);
            RCLCPP_INFO(logger, "Found Joint configuration:");
            for (size_t i = 0; i < joint_values.size(); ++i) {
                RCLCPP_INFO(logger, "%s: %f", joint_names[i].c_str(), joint_values[i]);
            }

            RCLCPP_INFO(logger, "IK solution found, checking for collisions.");

            // Set the joint target and check for collisions
            move_group_interface_->setJointValueTarget(joint_values);

            // Plan the motion
            moveit::planning_interface::MoveGroupInterface::Plan plan;

            move_group_interface_->setPlanningTime(3.0);
            double velocity_scaling = speed_scale;  // 50% of max velocity
            move_group_interface_->setMaxVelocityScalingFactor(velocity_scaling);
            move_group_interface_->setMaxAccelerationScalingFactor(0.1);

            bool success = (move_group_interface_->plan(plan) == moveit::core::MoveItErrorCode::SUCCESS);

            if (success)
            {
                const auto& trajectory = plan.trajectory.joint_trajectory;
                if (!trajectory.points.empty()) {
                    const auto& last_waypoint = trajectory.points.back();
                    std::vector<double> last_joint_positions = last_waypoint.positions;

                    bool match = true;
                    for (size_t i = 0; i < joint_values.size(); ++i) {
                        if (std::abs(joint_values[i] - last_joint_positions[i]) > 1e-3) { // Tolerance of 0.001 radians
                            match = false;
                            RCLCPP_WARN(logger, "Mismatch at Joint %zu: IK=%f, Plan=%f", i, joint_values[i], last_joint_positions[i]);
                        }
                    }

                    if (match) {
                        RCLCPP_INFO(logger, "Last waypoint matches IK solution.");
                    } else {
                        success = false;
                        RCLCPP_WARN(logger, "Last waypoint does NOT match IK solution.");
                    }
                }
            }

            if (success)
            {
                RCLCPP_INFO(logger, "Plan found, executing.");
                moveit::core::MoveItErrorCode execute_status = move_group_interface_->move();
                if (execute_status == moveit::core::MoveItErrorCode::SUCCESS)
                {
                auto joint_state_msg = sensor_msgs::msg::JointState();
                    // Get current joint values using MoveGroupInterface
                    std::vector<double> joint_values = move_group_interface_->getCurrentJointValues();
                    
                    // Set header and joint names (you should replace with actual joint names)
                    joint_state_msg.header.stamp = this->get_clock()->now();
                    joint_state_msg.name = {"joint1", "joint2", "joint3", "joint4", "joint5", "joint6"}; // Replace with actual joint names
                    joint_state_msg.position = joint_values; // Populate with current joint positions

                    // Publish the JointState message
                    ik_solution_publisher_->publish(joint_state_msg);

                    geometry_msgs::msg::PoseStamped current_pose_stamped = move_group_interface_->getCurrentPose();
                    geometry_msgs::msg::Pose current_pose = current_pose_stamped.pose;
                    publisher_->publish(current_pose);
                    RCLCPP_INFO(logger, "Motion executed successfully.");
                    break;  // Exit loop on success
                }
                else
                {
                    RCLCPP_ERROR(logger, "Execution failed. Retrying...");
                }
            }
            else
            {
                RCLCPP_ERROR(logger, "Failed to plan. Retrying...");
            }
        }
        else
        {
            RCLCPP_ERROR(logger, "IK solution not found, retrying...");
        }
    }
}

void Trajectory_Planning_Node::publish_pose(const std_msgs::msg::Bool::SharedPtr msg)

{
    // Get current pose as PoseStamped
    geometry_msgs::msg::PoseStamped current_pose_stamped = move_group_interface_->getCurrentPose();

    // Extract Pose from PoseStamped
    geometry_msgs::msg::Pose current_pose = current_pose_stamped.pose;
    
    // Publish the Pose
    std::string current_ee_link = move_group_interface_->getEndEffectorLink();
    
    pose_publisher_->publish(current_pose);
}

void Trajectory_Planning_Node::task_space_callback(const geometry_msgs::msg::PoseArray::SharedPtr waypoint_array_msg)
{
    RCLCPP_INFO(this->get_logger(), "Cartesian Path Function called");

    // Convert PoseArray to vector of waypoints
    const std::vector<geometry_msgs::msg::Pose>& waypoints = waypoint_array_msg->poses;

    // Set velocity and acceleration scaling factors
    move_group_interface_->setMaxVelocityScalingFactor(speed_scale);
    move_group_interface_->setMaxAccelerationScalingFactor(0.2);

    // Plan Cartesian path
    moveit_msgs::msg::RobotTrajectory trajectory;
    const double eef_step = 0.005;
    const double jump_threshold = 30.0;
    const bool avoid_collisions = true;

    double fraction = move_group_interface_->computeCartesianPath(waypoints, eef_step, jump_threshold, trajectory, avoid_collisions);

    if (fraction >= 0.8)
    {
        RCLCPP_INFO(this->get_logger(), "Cartesian path computed successfully. Fraction: %f", fraction);

        moveit::planning_interface::MoveGroupInterface::Plan cartesian_plan;
        cartesian_plan.trajectory = trajectory;

        if (move_group_interface_->execute(cartesian_plan) == moveit::core::MoveItErrorCode::SUCCESS)
        {
            RCLCPP_INFO(this->get_logger(), "Cartesian path executed successfully.");
            publisher_->publish(move_group_interface_->getCurrentPose().pose);
        }
        else
        {
            RCLCPP_ERROR(this->get_logger(), "Failed to execute Cartesian path.");
        }
    }
    else
    {
        RCLCPP_ERROR(this->get_logger(), "Failed to compute Cartesian path. Fraction: %f", fraction);
    }
}

int main(int argc, char *argv[])
{
  rclcpp::init(argc, argv);
  // Create the move group node with options to automatically declare parameters from overrides
  rclcpp::NodeOptions node_options;
  node_options.automatically_declare_parameters_from_overrides(true);
   auto const logger = rclcpp::get_logger("hello_moveit");
 

  // Create and spin the Trajectory_Planning_Node
  auto move_group_node = std::make_shared<Trajectory_Planning_Node>(node_options);
  rclcpp::spin(move_group_node);

  // Shutdown ROS
  rclcpp::shutdown();
  return 0;
}
