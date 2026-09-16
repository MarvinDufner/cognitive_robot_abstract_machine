from dataclasses import dataclass

import numpy as np
import pytest
from typing_extensions import List

from semantic_digital_twin.datastructures.joint_state import JointState
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.exceptions import (
    NoForceTorqueSensorForFrameError,
    NoForceTorqueSensorForTipError,
)
from semantic_digital_twin.robots.hsrb import HSRBWristForceTorqueSensor
from semantic_digital_twin.robots.robot_parts import ForceTorqueSensor
from semantic_digital_twin.robots.tracy import (
    TracyLeftForceTorqueSensor,
    TracyRightForceTorqueSensor,
    TracyWrenchService,
    TracyWrenchTopic,
)
from semantic_digital_twin.world_description.world_entity import (
    KinematicStructureEntity,
)

WRIST_SENSOR_FRAME = "wrist_ft_sensor_frame"
TOOL_FRAME = "hand_gripper_tool_frame"
LEFT_TOOL_FRAME = "l_gripper_tool_frame"
RIGHT_TOOL_FRAME = "r_gripper_tool_frame"
LEFT_DRIVER_WRENCH_FRAME = "left_tool0"
"""The frame the left arm's driver stamps its wrench messages with."""


@dataclass(eq=False)
class SensorClosestToTip(ForceTorqueSensor):
    """
    Minimal concrete force/torque sensor, to place a second sensor on a chain that
    already carries one.
    """

    def setup_hardware_interfaces(self):
        pass

    def setup_joint_states(self) -> List[JointState]:
        return []

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ):
        raise NotImplementedError


def test_the_hsrb_annotates_a_sensor_at_its_wrist(hsr_world_copy):
    sensors = hsr_world_copy.get_semantic_annotations_by_type(ForceTorqueSensor)

    assert len(sensors) == 1
    assert isinstance(sensors[0], HSRBWristForceTorqueSensor)
    assert sensors[0].root.name.name == WRIST_SENSOR_FRAME


def test_the_sensor_is_found_from_the_tip_it_measures_for(hsr_world_copy):
    tool = hsr_world_copy.get_body_by_name(TOOL_FRAME)

    sensor = ForceTorqueSensor.for_tip(hsr_world_copy, tool)

    assert sensor.root.name.name == WRIST_SENSOR_FRAME


def test_the_sensor_closest_to_the_tip_wins(hsr_world_copy):
    tool = hsr_world_copy.get_body_by_name(TOOL_FRAME)
    higher_up = SensorClosestToTip(
        name=PrefixedName("higher_up", prefix="test"),
        root=hsr_world_copy.get_body_by_name("arm_lift_link"),
        _world=hsr_world_copy,
    )
    with hsr_world_copy.modify_world():
        hsr_world_copy.add_semantic_annotations([higher_up])

    sensor = ForceTorqueSensor.for_tip(hsr_world_copy, tool)

    assert sensor.root.name.name == WRIST_SENSOR_FRAME


def test_a_tip_off_the_sensor_chain_has_no_sensor(hsr_world_copy):
    off_chain = hsr_world_copy.get_body_by_name("head_tilt_link")

    with pytest.raises(NoForceTorqueSensorForTipError):
        ForceTorqueSensor.for_tip(hsr_world_copy, off_chain)


def test_the_sensor_is_found_from_the_frame_it_is_rooted_at(hsr_world_copy):
    frame = hsr_world_copy.get_body_by_name(WRIST_SENSOR_FRAME)

    assert ForceTorqueSensor.with_root(hsr_world_copy, frame).root is frame


def test_a_frame_without_a_sensor_is_reported(hsr_world_copy):
    with pytest.raises(NoForceTorqueSensorForFrameError):
        ForceTorqueSensor.with_root(
            hsr_world_copy, hsr_world_copy.get_body_by_name("base_footprint")
        )


def test_an_idle_sensor_is_distinguishable_from_one_measuring_zero(hsr_world_copy):
    sensor = ForceTorqueSensor.with_root(
        hsr_world_copy, hsr_world_copy.get_body_by_name(WRIST_SENSOR_FRAME)
    )

    assert not sensor.has_received_wrench

    sensor.write_wrench(np.zeros(3), np.zeros(3))

    assert sensor.has_received_wrench


def test_a_written_wrench_reaches_the_symbols(hsr_world_copy):
    sensor = ForceTorqueSensor.with_root(
        hsr_world_copy, hsr_world_copy.get_body_by_name(WRIST_SENSOR_FRAME)
    )

    sensor.write_wrench(np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0]))

    np.testing.assert_allclose(sensor.force.evaluate().flatten()[:3], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(sensor.torque.evaluate().flatten()[:3], [4.0, 5.0, 6.0])


def test_writing_a_wrench_overwrites_the_previous_one(hsr_world_copy):
    sensor = ForceTorqueSensor.with_root(
        hsr_world_copy, hsr_world_copy.get_body_by_name(WRIST_SENSOR_FRAME)
    )

    sensor.write_wrench(np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0]))
    sensor.write_wrench(np.array([-1.0, 0.0, 9.5]), np.zeros(3))

    np.testing.assert_allclose(sensor.force.evaluate().flatten()[:3], [-1.0, 0.0, 9.5])
    np.testing.assert_allclose(sensor.torque.evaluate().flatten()[:3], np.zeros(3))


def test_the_wrench_holds_exactly_the_six_axes(hsr_world_copy):
    """
    The admittance step compiles against this group, so its size must not depend on
    anything else in the world.
    """
    sensor = ForceTorqueSensor.with_root(
        hsr_world_copy, hsr_world_copy.get_body_by_name(WRIST_SENSOR_FRAME)
    )

    for _ in range(3):
        sensor.write_wrench(np.ones(3), np.ones(3))

    assert len(sensor.wrench_data.variables) == 6
    assert sensor.wrench_data.data.shape == (6,)


def test_the_wrench_is_expressed_in_the_sensor_frame(hsr_world_copy):
    sensor = ForceTorqueSensor.with_root(
        hsr_world_copy, hsr_world_copy.get_body_by_name(WRIST_SENSOR_FRAME)
    )

    assert sensor.force.reference_frame is sensor.root
    assert sensor.torque.reference_frame is sensor.root


# %% sensors bolted beside the chain


def test_each_arm_of_a_two_armed_robot_finds_its_own_sensor(tracy_world):
    """
    Tracy's wrist sensors hang off their wrist rather than carrying the hand, so
    resolving one means following the fixed attachments and not just the chain.
    """
    left = ForceTorqueSensor.for_tip(
        tracy_world, tracy_world.get_body_by_name(LEFT_TOOL_FRAME)
    )
    right = ForceTorqueSensor.for_tip(
        tracy_world, tracy_world.get_body_by_name(RIGHT_TOOL_FRAME)
    )

    assert isinstance(left, TracyLeftForceTorqueSensor)
    assert isinstance(right, TracyRightForceTorqueSensor)


def test_the_two_arms_of_a_two_armed_robot_read_their_own_topic():
    """
    One topic for both arms would feed each admittance the other hand's contact.
    """
    assert TracyLeftForceTorqueSensor.wrench_topic is TracyWrenchTopic.LEFT_COMPENSATED
    assert (
        TracyRightForceTorqueSensor.wrench_topic is TracyWrenchTopic.RIGHT_COMPENSATED
    )


def test_each_arm_is_zeroed_through_its_own_compensation(tracy_world):
    """
    Each arm needs its own compensation node, so zeroing one must not zero the other.
    """
    assert TracyLeftForceTorqueSensor.retare_service is TracyWrenchService.LEFT
    assert TracyRightForceTorqueSensor.retare_service is TracyWrenchService.RIGHT


def test_a_sensor_nothing_compensates_has_nothing_to_be_zeroed_through(hsr_world_copy):
    """
    The default says so, rather than naming a service no node offers.
    """
    sensor = ForceTorqueSensor.with_root(
        hsr_world_copy, hsr_world_copy.get_body_by_name(WRIST_SENSOR_FRAME)
    )

    assert sensor.retare_service is None


def test_the_sensor_sits_in_the_frame_its_readings_are_stamped_with(tracy_world):
    """
    A reading is written axis by axis, so the sensor has to be rooted where the driver
    measures. The wrist's own ``ft_frame`` sits at the same place but half a turn around,
    which would silently flip two of the three axes.
    """
    sensor = ForceTorqueSensor.for_tip(
        tracy_world, tracy_world.get_body_by_name(LEFT_TOOL_FRAME)
    )

    assert sensor.root is tracy_world.get_body_by_name(LEFT_DRIVER_WRENCH_FRAME)

    ft_frame = tracy_world.get_body_by_name("left_ft_frame")
    np.testing.assert_allclose(
        tracy_world.compute_forward_kinematics_np(sensor.root, ft_frame)[:3, :3],
        np.diag([1.0, -1.0, -1.0]),
        atol=1e-9,
    )
