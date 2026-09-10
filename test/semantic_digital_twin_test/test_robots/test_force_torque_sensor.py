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
from semantic_digital_twin.robots.hsrb import HSRB, HSRBWristForceTorqueSensor
from semantic_digital_twin.robots.robot_parts import ForceTorqueSensor
from semantic_digital_twin.world_description.world_entity import (
    KinematicStructureEntity,
)

WRIST_SENSOR_FRAME = "wrist_ft_sensor_frame"
TOOL_FRAME = "hand_gripper_tool_frame"


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
