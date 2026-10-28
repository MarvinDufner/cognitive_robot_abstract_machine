from datetime import timedelta

from giskardpy.middleware.ros2 import rospy
from giskardpy.middleware.ros2.server_config import ExecutionMode, GiskardServerConfig
from giskardpy.middleware.ros2.giskard import Giskard
from giskardpy.middleware.ros2.scripts.iai_robots.tracy.configs import (
    WorldWithTracyConfig,
    TracyVelocityInterface,
)
from giskardpy.middleware.ros2.utils.utils import load_xacro
from giskardpy.qp.qp_controller_config import QPControllerConfig
from rclpy import Parameter
from semantic_digital_twin.robots.tracy import Tracy


def main():
    rospy.init_node("giskard")
    rospy.get_node().declare_parameters(
        namespace="", parameters=[("robot_description", Parameter.Type.STRING)]
    )
    robot_description = rospy.get_node().get_parameter_or("robot_description").value
    if robot_description is None:
        robot_description = load_xacro(Tracy.get_ros_file_path())
    giskard = Giskard(
        world_config=WorldWithTracyConfig(urdf=robot_description),
        robot_interface_config=TracyVelocityInterface(),
        server_config=GiskardServerConfig(execution_mode=ExecutionMode.CLOSED_LOOP),
        qp_controller_config=QPControllerConfig(
            # reproduces the braking of the former prediction_horizon=30
            target_frequency=80,
            braking_time=timedelta(seconds=0.36),
        ),
    )
    giskard.live()


if __name__ == "__main__":
    main()
