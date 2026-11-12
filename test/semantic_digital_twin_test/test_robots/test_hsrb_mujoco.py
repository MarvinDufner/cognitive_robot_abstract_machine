"""
Tests for simulating the HSRB physically in MuJoCo.
"""

from __future__ import annotations

from datetime import timedelta

import pytest

from ...pytest_environment import runs_in_continuous_integration

from semantic_digital_twin.adapters.multi_sim import MujocoBuilder, MujocoSim
from semantic_digital_twin.datastructures.definitions import StaticJointState
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.hsrb import HSRB, HSRBJoint
from semantic_digital_twin.specifications.robots import RobotSpecification
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import (
    Body,
    GravityCompensation,
    PositionServo,
)


@pytest.fixture
def spawned_hsrb() -> HSRB:
    world = World()
    with world.modify_world():
        world.add_kinematic_structure_entity(Body(name=PrefixedName("floor")))
    return RobotSpecification(HSRB).spawn(world)


# %% what holds the joints


def test_every_joint_a_controller_commands_is_driven_by_one_servo(
    spawned_hsrb, tmp_path
):
    """
    A degree of freedom with no servo is written straight into MuJoCo and then left to
    gravity, so the arm falls to the floor while the controller believes its commands
    are holding it up.
    """
    world = spawned_hsrb._world
    commanded = {world.get_connection_by_name(joint).raw_dof for joint in HSRBJoint}

    driven = [
        degree_of_freedom
        for actuator in world.actuators
        for degree_of_freedom in actuator.dofs
    ]

    assert commanded <= set(driven)
    assert len(driven) == len(set(driven))
    # Only the joints a controller commands: the base is servoed on velocity instead,
    # and the drive is not one of them.
    for actuator in world.actuators:
        if commanded.intersection(actuator.dofs):
            assert isinstance(actuator, PositionServo)
    builder = MujocoBuilder()
    builder.build_world(world=world, file_path=str(tmp_path / "scene.xml"))
    assert {degree_of_freedom.name.name for degree_of_freedom in commanded} <= {
        actuator.target for actuator in builder.spec.actuators
    }


def test_the_arm_is_held_by_the_gains_its_description_limits(spawned_hsrb):
    """
    The servos may not ask for more force than the robot's own description allows, and
    the lift, which carries the whole arm and takes a press straight down its axis, is
    the stiffest of them.
    """
    world = spawned_hsrb._world
    arm = spawned_hsrb.mobile_base.torso.arm
    lift = world.get_connection_by_name(HSRBJoint.ARM_LIFT)
    flex = world.get_connection_by_name(HSRBJoint.ARM_FLEX)

    [lift_servo] = [
        actuator for actuator in world.actuators if lift.raw_dof in actuator.dofs
    ]
    [flex_servo] = [
        actuator for actuator in world.actuators if flex.raw_dof in actuator.dofs
    ]

    assert flex_servo.gains == arm.drive_of_each_joint[HSRBJoint.ARM_FLEX].gains
    assert lift_servo.gains.stiffness > flex_servo.gains.stiffness
    assert lift_servo.gains.torque_limit == 300.0


def test_the_servoed_parts_carry_their_own_weight(spawned_hsrb):
    """
    A servo holds its joint against what the robot works on; the weight of the links is
    the simulation's to carry, as it is on a robot whose motors are already holding it.
    """
    torso = spawned_hsrb.mobile_base.torso
    parts = [torso, torso.arm, torso.arm.end_effector, torso.neck]

    for part in parts:
        for body in part.bodies:
            assert body.get_simulator_property_of_type(GravityCompensation) == (
                GravityCompensation(fraction=1.0)
            )


@pytest.mark.skipif(
    not runs_in_continuous_integration(), reason="MuJoCo tests only run in CI"
)
def test_the_servos_hold_the_parked_arm_where_the_world_has_it(spawned_hsrb):
    """
    End to end: an unheld arm sags out of the pose the world says it is in, and a tool
    held in its hand ends up on the floor.
    """
    world = spawned_hsrb._world
    arm = spawned_hsrb.mobile_base.torso.arm
    arm.get_joint_state_by_type(StaticJointState.PARK).apply_to(world)
    world.notify_state_change()
    tool_frame = arm.end_effector.tool_frame
    parked = world.compute_forward_kinematics_np(world.root, tool_frame)[:3, 3]

    simulation = MujocoSim(world=world, headless=True)
    simulation.start_stepped_simulation()
    try:
        simulation.step_simulation(timedelta(seconds=1))
        simulated = simulation.simulator.get_body_position(
            body_name=tool_frame.name.name
        ).result
    finally:
        simulation.stop_simulation()

    assert list(simulated) == pytest.approx(list(parked), abs=0.01)
