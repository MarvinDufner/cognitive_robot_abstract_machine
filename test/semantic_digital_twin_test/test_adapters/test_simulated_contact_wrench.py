"""
Tests for writing the wrench a physics simulation presses a tool with into the
force/torque sensor that measures for it.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

import numpy as np
import pytest
from typing_extensions import List

from ...pytest_environment import runs_in_continuous_integration

from semantic_digital_twin.adapters.multi_sim import MujocoSim
from semantic_digital_twin.adapters.sensors.force_torque import (
    PhysicallySimulatedContactWrench,
)
from semantic_digital_twin.datastructures.joint_state import JointState
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.robot_parts import ForceTorqueSensor
from semantic_digital_twin.spatial_types.spatial_types import (
    HomogeneousTransformationMatrix,
    Vector3,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    FixedConnection,
    PrismaticConnection,
)
from semantic_digital_twin.world_description.degree_of_freedom import (
    DegreeOfFreedom,
    DegreeOfFreedomLimits,
)
from semantic_digital_twin.world_description.geometry import Box, Scale
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body

pytestmark = pytest.mark.skipif(
    not runs_in_continuous_integration(), reason="MuJoCo tests only run in CI"
)

# %% a tool that can be lowered onto a floor


@dataclass(eq=False)
class SensorWithoutHardware(ForceTorqueSensor):
    """
    Minimal concrete force/torque sensor, to be written into without a robot.
    """

    def setup_hardware_interfaces(self):
        pass

    def setup_joint_states(self) -> List[JointState]:
        return []

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(cls, robot_root):
        raise NotImplementedError


@dataclass
class LoweredToolWorld:
    """
    A world where one tool slides down onto a floor, and the handles a test needs.
    """

    world: World
    """
    The world itself.
    """

    tool: Body
    """
    The body that comes to rest on the floor.
    """

    sensor: ForceTorqueSensor
    """
    The sensor measuring for the tool, rooted beside it and turned half a turn about its
    x axis, as a sensor flange is against the frame a driver stamps.
    """

    simulation: MujocoSim
    """
    The physics the tool slides in.
    """


TOOL_WIDTH = 0.1
"""Side length of the tool's cube, in m."""

SENSOR_OFFSET = 0.1
"""How far beside the tool the sensor sits, along the tool's x axis, in m."""

DROP_HEIGHT = 0.3
"""Where the tool starts above the floor, in m."""

SETTLING_TIME = timedelta(seconds=1.5)
"""How long the tool needs to land and come to rest."""


@pytest.fixture
def lowered_tool() -> LoweredToolWorld:
    world = World()
    with world.modify_world():
        floor = Body(
            name=PrefixedName("floor"),
            collision=ShapeCollection([Box(scale=Scale(2.0, 2.0, 0.1))]),
        )
        world.add_body(floor)
        tool = Body(
            name=PrefixedName("tool"),
            collision=ShapeCollection(
                [Box(scale=Scale(TOOL_WIDTH, TOOL_WIDTH, TOOL_WIDTH))]
            ),
        )
        degree_of_freedom = DegreeOfFreedom(
            name=PrefixedName("lift"),
            limits=DegreeOfFreedomLimits.from_position_range_and_speed(
                lower_position=-1.0, upper_position=1.0
            ),
        )
        world.add_degree_of_freedom(degree_of_freedom)
        world.add_connection(
            PrismaticConnection(
                name=PrefixedName("lift"),
                parent=floor,
                child=tool,
                raw_dof=degree_of_freedom,
                axis=Vector3.Z(reference_frame=tool),
                parent_T_connection_expression=(
                    HomogeneousTransformationMatrix.from_xyz_rpy(
                        z=0.05 + TOOL_WIDTH / 2 + DROP_HEIGHT
                    )
                ),
            )
        )
        sensor_frame = Body(name=PrefixedName("sensor_frame"))
        world.add_connection(
            FixedConnection(
                parent=tool,
                child=sensor_frame,
                parent_T_connection_expression=(
                    HomogeneousTransformationMatrix.from_xyz_rpy(
                        x=SENSOR_OFFSET, roll=np.pi
                    )
                ),
            )
        )
        sensor = SensorWithoutHardware(
            name=PrefixedName("sensor", prefix="test"), root=sensor_frame
        )
        world.add_semantic_annotations([sensor])
    simulation = MujocoSim(world=world, headless=True)
    simulation.start_stepped_simulation()
    yield LoweredToolWorld(world=world, tool=tool, sensor=sensor, simulation=simulation)
    simulation.stop_simulation()


# %% what the sensor is left holding


def test_a_tool_in_the_air_is_pressed_with_nothing(lowered_tool):
    """
    A reading of zero is still a reading: a sensor that has been read tells a plan that
    it is out of contact rather than that it is not working.
    """
    reading = PhysicallySimulatedContactWrench(
        world=lowered_tool.world,
        tool=lowered_tool.tool,
        simulation=lowered_tool.simulation,
    )

    announced = reading.apply()

    assert not announced
    assert lowered_tool.sensor.has_received_wrench
    np.testing.assert_allclose(lowered_tool.sensor.measured_force, np.zeros(3))


def test_a_landed_tool_is_pressed_along_the_sensors_own_axis(lowered_tool):
    """
    The floor pushes up, and the sensor is mounted upside down, so a sensor reporting the
    push as its own +z would have the press driving the tool further into the floor.
    """
    reading = PhysicallySimulatedContactWrench(
        world=lowered_tool.world,
        tool=lowered_tool.tool,
        simulation=lowered_tool.simulation,
    )
    lowered_tool.simulation.step_simulation(SETTLING_TIME)

    reading.apply()

    force = lowered_tool.sensor.measured_force
    assert force[2] < 0.0
    np.testing.assert_allclose(force[:2], np.zeros(2), atol=1e-6)


def test_the_torque_is_the_push_about_the_sensors_own_origin(lowered_tool):
    """
    The sensor sits beside the tool, so the push it feels is a torque, which is what
    tells a plan how the tool is loaded rather than only how hard.
    """
    reading = PhysicallySimulatedContactWrench(
        world=lowered_tool.world,
        tool=lowered_tool.tool,
        simulation=lowered_tool.simulation,
    )
    lowered_tool.simulation.step_simulation(SETTLING_TIME)

    reading.apply()

    sensor_P_contact = np.array([-SENSOR_OFFSET, 0.0, TOOL_WIDTH / 2.0])
    np.testing.assert_allclose(
        lowered_tool.sensor.wrench_data.data[3:],
        np.cross(sensor_P_contact, lowered_tool.sensor.measured_force),
        atol=1e-3,
    )
