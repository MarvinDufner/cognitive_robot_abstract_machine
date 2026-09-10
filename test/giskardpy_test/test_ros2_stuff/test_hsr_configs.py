from giskardpy.middleware.ros2.giskard import Giskard
from giskardpy.middleware.ros2.input_synchronization import WrenchSynchronizer
from giskardpy.middleware.ros2.scripts.iai_robots.hsr.configs import (
    HSRVelocityInterface,
    WorldWithHSRConfig,
)
from giskardpy.middleware.ros2.server_config import ExecutionMode, GiskardServerConfig
from giskardpy.middleware.ros2.utils.utils import load_xacro
from giskardpy.qp.qp_controller_config import QPControllerConfig
from semantic_digital_twin.robots.hsrb import HSRB
from semantic_digital_twin.robots.robot_parts import ForceTorqueSensor


def hsr_giskard() -> Giskard:
    return Giskard(
        world_config=WorldWithHSRConfig(urdf=load_xacro(HSRB.get_ros_file_path())),
        robot_interface_config=HSRVelocityInterface(),
        server_config=GiskardServerConfig(execution_mode=ExecutionMode.CLOSED_LOOP),
        qp_controller_config=QPControllerConfig(target_frequency=40),
    )


def wrench_synchronizers(synchronizers) -> list:
    return [
        synchronizer
        for synchronizer in synchronizers
        if isinstance(synchronizer, WrenchSynchronizer)
    ]


# %% following the sensor without being told to


def test_bringing_the_robot_up_follows_its_force_torque_sensor(init_rospy):
    """The sensor carries its own topic, so nothing in the robot's interface config has
    to name it: bringing the robot up is enough for the wrench to arrive."""
    giskard = hsr_giskard()

    giskard.setup()

    sensor = giskard.executor.context.world.get_semantic_annotations_by_type(
        ForceTorqueSensor
    )[0]
    followed = wrench_synchronizers(giskard.motion_server.control_loop.inputs.synchronizers)
    assert [synchronizer.topic_name for synchronizer in followed] == [
        sensor.wrench_topic
    ]
    assert followed[0].sensor is sensor


def test_the_control_loop_reads_the_wrench(init_rospy):
    """The motion state chart is ticked by the control loop, so a reading that only
    reached the motion server would never be seen by a task that presses."""
    giskard = hsr_giskard()

    giskard.setup()

    assert wrench_synchronizers(giskard.motion_server.control_loop.inputs.synchronizers)
    assert wrench_synchronizers(giskard.motion_server.inputs.synchronizers)
