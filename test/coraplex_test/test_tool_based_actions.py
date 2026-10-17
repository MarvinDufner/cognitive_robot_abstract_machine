from dataclasses import dataclass
from datetime import timedelta

import numpy as np
import pytest
from giskardpy.motion_statechart.goals.collision_avoidance import (
    UpdateTemporaryCollisionRules,
)
from scipy.spatial.transform import Rotation

from coraplex.datastructures.enums import (
    CuttingTechnique,
    PouringSide,
    WipingTechnique,
)
from coraplex.exceptions import WipingTargetMissing
from coraplex.plans.failures import MotionMadeNoProgress
from giskardpy.motion_statechart.exceptions import NoProgressError
from giskardpy.motion_statechart.graph_node import EndMotion
from giskardpy.motion_statechart.monitors.progress_monitors import StillProgressing
from coraplex.plans.factories import sequential
from coraplex.plans.plan_node import MotionNode
from coraplex.robot_plans.actions.composite.tool_based import (
    CuttingAction,
    MixingAction,
    PouringAction,
    WipingAction,
)
from coraplex.robot_plans.motions.gripper import MoveTCPWaypointsAlignedMotion
from giskardpy.motion_statechart.graph_node import Goal
from giskardpy.motion_statechart.tasks.admittance_tasks import (
    AdmittanceCartesianTrajectory,
)
from krrood.ormatic.data_access_objects.helper import to_dao
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    PouringCup,
    CuttingKnife,
    Sponge,
    Whisk,
)
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix, Vector3
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world_description.connections import FixedConnection
from semantic_digital_twin.world_description.geometry import Box, Scale
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body


def _add_box_body(world, name, size, position):
    shape_collection = ShapeCollection([Box(scale=Scale(*size))])
    body = Body(
        name=PrefixedName(name), collision=shape_collection, visual=shape_collection
    )
    with world.modify_world():
        world.add_kinematic_structure_entity(body)
        world.add_connection(
            FixedConnection(
                parent=world.root,
                child=body,
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    *position, reference_frame=world.root
                ),
            )
        )
    return body


@pytest.fixture
def tool_action_world(pr2_apartment_context):
    world, robot, context = pr2_apartment_context
    container = _add_box_body(
        world, "tool_test_container", (0.2, 0.2, 0.1), (2.4, 2.2, 1.0)
    )
    tool_body = _add_box_body(
        world, "tool_test_tool", (0.04, 0.04, 0.2), (1.0, 1.0, 1.0)
    )
    return world, robot, context, container, tool_body


def _expanded_aligned_motions(action, context):
    sequential([action], context)
    action.expand()
    return [
        node.designator
        for node in action.plan.all_nodes
        if isinstance(node, MotionNode)
        and isinstance(node.designator, MoveTCPWaypointsAlignedMotion)
    ]


def test_mixing_action_expands_to_aligned_motion(tool_action_world):
    world, robot, context, container, tool_body = tool_action_world
    whisk = Whisk(root=tool_body)

    action = MixingAction(container=container, arm=context.robot.right_arm, tool=whisk)
    motions = _expanded_aligned_motions(action, context)

    assert len(motions) == 1
    motion = motions[0]
    assert len(motion.waypoints) > 0
    assert len(motion.alignment_pairs) == 1
    assert motion.tip == whisk.get_tool_frame()


def test_cutting_action_pointer_stride_reduces_waypoints(tool_action_world):
    world, robot, context, container, tool_body = tool_action_world
    knife = CuttingKnife(root=tool_body)

    dense_action = CuttingAction(
        object_to_cut=container,
        arm=context.robot.right_arm,
        tool=knife,
        technique=CuttingTechnique.SLICE,
    )
    strided_action = CuttingAction(
        object_to_cut=container,
        arm=context.robot.right_arm,
        tool=knife,
        technique=CuttingTechnique.SLICE,
        pointer_stride=10,
    )

    dense_motion = _expanded_aligned_motions(dense_action, context)[0]
    strided_motion = _expanded_aligned_motions(strided_action, context)[0]

    assert len(dense_motion.waypoints) > 0
    assert len(strided_motion.waypoints) == pytest.approx(
        len(dense_motion.waypoints) / 10, abs=1
    )
    assert len(dense_motion.alignment_pairs) == 2


def _collision_rule_nodes(motion_chart):
    return [
        node
        for node in motion_chart.nodes
        if isinstance(node, UpdateTemporaryCollisionRules)
    ]


def test_aligned_motion_collision_rules_follow_allow_gripper_collision(
    tool_action_world,
):
    world, robot, context, container, tool_body = tool_action_world
    whisk = Whisk(root=tool_body)

    action = MixingAction(container=container, arm=context.robot.right_arm, tool=whisk)
    motion = _expanded_aligned_motions(action, context)[0]

    assert motion.allow_gripper_collision is True
    assert len(_collision_rule_nodes(motion._motion_chart)) == 1

    motion.allow_gripper_collision = False
    assert len(_collision_rule_nodes(motion._motion_chart)) == 0


def test_wiping_action_requires_container_or_target_pose(tool_action_world):
    world, robot, context, container, tool_body = tool_action_world
    sponge = Sponge(root=tool_body)

    with pytest.raises(WipingTargetMissing):
        WipingAction(arm=context.robot.right_arm, tool=sponge)


def test_wiping_action_around_target_pose(tool_action_world):
    world, robot, context, container, tool_body = tool_action_world
    sponge = Sponge(root=tool_body)

    action = WipingAction(
        arm=context.robot.right_arm,
        tool=sponge,
        target_pose=Pose.from_xyz_rpy(x=2.4, y=2.2, z=1.0, reference_frame=world.root),
    )
    motions = _expanded_aligned_motions(action, context)

    assert len(motions) == 1
    assert len(motions[0].waypoints) > 0
    assert len(motions[0].alignment_pairs) == 1


def test_pouring_action_poses_tilt_and_mirror(tool_action_world):
    world, robot, context, container, tool_body = tool_action_world
    cup = PouringCup(root=tool_body)

    right_action = PouringAction(
        target_container=container,
        source_container=cup,
        arm=context.robot.right_arm,
    )
    sequential([right_action], context)
    right_pre_pose, right_pour_pose = right_action._pour_poses()

    pre_rotation = Rotation.from_quat(
        [float(value) for value in right_pre_pose.quaternion.to_np()]
    )
    pour_rotation = Rotation.from_quat(
        [float(value) for value in right_pour_pose.quaternion.to_np()]
    )
    tilt_magnitude = (pre_rotation.inv() * pour_rotation).magnitude()
    assert tilt_magnitude == pytest.approx(right_action.tilt_angle, abs=1e-6)
    assert float(right_pre_pose.x) == pytest.approx(float(right_pour_pose.x))
    assert float(right_pre_pose.y) == pytest.approx(float(right_pour_pose.y))

    left_action = PouringAction(
        target_container=container,
        source_container=cup,
        arm=context.robot.right_arm,
        pour_side=PouringSide.LEFT,
    )
    sequential([left_action], context)
    left_pre_pose, _ = left_action._pour_poses()

    container_position = np.array(
        [
            float(container.global_pose.x),
            float(container.global_pose.y),
        ]
    )
    right_offset = (
        np.array([float(right_pre_pose.x), float(right_pre_pose.y)])
        - container_position
    )
    left_offset = (
        np.array([float(left_pre_pose.x), float(left_pre_pose.y)]) - container_position
    )
    np.testing.assert_allclose(left_offset, -right_offset, atol=1e-9)


@pytest.mark.parametrize(
    "arm_of, side",
    [
        (lambda robot: robot.right_arm, PouringSide.RIGHT),
        (lambda robot: robot.left_arm, PouringSide.LEFT),
    ],
    ids=["right-arm", "left-arm"],
)
def test_pouring_pours_to_the_side_of_its_arm_unless_told_otherwise(
    tool_action_world, arm_of, side
):
    world, robot, context, container, tool_body = tool_action_world
    action = PouringAction(
        target_container=container,
        source_container=PouringCup(root=tool_body),
        arm=arm_of(context.robot),
    )
    sequential([action], context)

    assert action._effective_pour_side() is side


@dataclass
class _GivingUpSubplan:
    """
    A subplan whose motion stops approaching its goal.
    """

    def perform(self) -> None:
        raise MotionMadeNoProgress(
            NoProgressError(
                progress_monitor=StillProgressing(
                    monitored_node=EndMotion(), timeout=timedelta(seconds=3)
                )
            )
        )


@pytest.mark.parametrize("reached_final_waypoint", [True, False])
def test_wiping_accepts_a_motion_that_gave_up_only_at_its_last_waypoint(
    tool_action_world, monkeypatch, reached_final_waypoint
):
    """
    A wipe that stops making progress once the sponge is at its last waypoint has done
    its job; one that stops anywhere else has not.
    """
    world, robot, context, container, tool_body = tool_action_world
    action = WipingAction(
        arm=context.robot.right_arm,
        tool=Sponge(root=tool_body),
        target_pose=Pose.from_xyz_rpy(x=2.4, y=2.2, z=1.0, reference_frame=world.root),
    )
    sequential([action], context)
    monkeypatch.setattr(
        WipingAction, "add_subplan", lambda self, root: _GivingUpSubplan()
    )
    monkeypatch.setattr(
        WipingAction,
        "_tool_reached_final_waypoint",
        lambda self: reached_final_waypoint,
    )

    if reached_final_waypoint:
        action._perform_plan()
    else:
        with pytest.raises(MotionMadeNoProgress):
            action._perform_plan()


def _attach_box_to_gripper(world, robot, name, size, mount_z):
    shape_collection = ShapeCollection([Box(scale=Scale(*size))])
    body = Body(
        name=PrefixedName(name), collision=shape_collection, visual=shape_collection
    )
    tool_frame = robot.right_arm.end_effector.tool_frame
    with world.modify_world():
        world.add_kinematic_structure_entity(body)
        world.add_connection(
            FixedConnection(
                parent=tool_frame,
                child=body,
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    z=mount_z, reference_frame=tool_frame
                ),
            )
        )
    return body


def test_pouring_action_pour_point_lands_on_target_container_center(
    tool_action_world,
):
    world, robot, context, container, tool_body = tool_action_world
    held_source = _attach_box_to_gripper(
        world, robot, "held_pour_source", (0.04, 0.04, 0.2), -0.08
    )
    cup = PouringCup(root=held_source)

    action = PouringAction(
        target_container=container, source_container=cup, arm=context.robot.right_arm
    )
    sequential([action], context)
    _, pour_pose = action._pour_poses()

    tool_frame = context.robot.right_arm.end_effector.tool_frame
    tool_frame_T_source = world.compute_forward_kinematics_np(tool_frame, held_source)
    mouth_in_tool_frame = tool_frame_T_source @ np.array([0.0, 0.0, 0.1, 1.0])
    mouth_in_world = pour_pose.homogeneous_matrix.to_np() @ mouth_in_tool_frame

    assert mouth_in_world[0] == pytest.approx(float(container.global_pose.x), abs=1e-6)
    assert mouth_in_world[1] == pytest.approx(float(container.global_pose.y), abs=1e-6)


def test_mixing_action_orm_roundtrip(tool_action_world, coraplex_testing_session):
    world, robot, context, container, tool_body = tool_action_world
    whisk = Whisk(root=tool_body)

    action = MixingAction(container=container, arm=context.robot.right_arm, tool=whisk)
    sequential([action], context)
    action.expand()

    dao = to_dao(action)
    coraplex_testing_session.add(dao)
    coraplex_testing_session.commit()

    assert dao.database_id is not None


# %% covering a patch rather than a lane


def test_a_patch_with_no_width_is_wiped_in_a_single_lane(tool_action_world):
    """
    The default keeps the earlier behaviour: back and forth over one line.
    """
    world, robot, context, container, tool_body = tool_action_world
    sponge = Sponge(root=tool_body)

    action = WipingAction(
        arm=robot.right_arm,
        tool=sponge,
        target_pose=Pose.from_xyz_rpy(x=2.4, y=2.2, z=1.0, reference_frame=world.root),
        technique=WipingTechnique.SPREAD,
        length=0.3,
    )
    motions = _expanded_aligned_motions(action, context)
    waypoints = np.array(
        [point.to_np()[:3].flatten() for point in motions[0].waypoints]
    )

    assert np.ptp(waypoints[:, 0]) == pytest.approx(0.3, abs=0.01)
    assert np.ptp(waypoints[:, 1]) == pytest.approx(0.0, abs=1e-6)


def test_a_patch_with_a_width_is_wiped_in_lanes_across_it(tool_action_world):
    world, robot, context, container, tool_body = tool_action_world
    sponge = Sponge(root=tool_body)
    length, width = 0.3, 0.2

    action = WipingAction(
        arm=robot.right_arm,
        tool=sponge,
        target_pose=Pose.from_xyz_rpy(x=2.4, y=2.2, z=1.0, reference_frame=world.root),
        technique=WipingTechnique.SPREAD,
        length=length,
        width=width,
    )
    motions = _expanded_aligned_motions(action, context)
    waypoints = np.array(
        [point.to_np()[:3].flatten() for point in motions[0].waypoints]
    )

    assert np.ptp(waypoints[:, 0]) == pytest.approx(length, abs=0.01)
    assert np.ptp(waypoints[:, 1]) == pytest.approx(width, abs=0.01)


def test_the_lanes_are_closer_together_than_the_tool_is_wide(tool_action_world):
    """
    Lanes a tool width apart or more would leave unwiped strips between them.
    """
    world, robot, context, container, tool_body = tool_action_world
    sponge = Sponge(root=tool_body)
    width = 0.2
    action = WipingAction(
        arm=robot.right_arm,
        tool=sponge,
        target_pose=Pose.from_xyz_rpy(x=2.4, y=2.2, z=1.0, reference_frame=world.root),
        technique=WipingTechnique.SPREAD,
        length=0.3,
        width=width,
    )
    tool_width = (
        tool_body.collision.as_bounding_box_collection_in_frame(tool_body)
        .bounding_box()
        .width
    )

    assert width / (action._lanes() - 1) <= tool_width


def test_a_patch_narrower_than_the_tool_still_has_two_lanes(tool_action_world):
    """
    A raster needs a lane to start and a lane to end on.
    """
    world, robot, context, container, tool_body = tool_action_world
    sponge = Sponge(root=tool_body)

    action = WipingAction(
        arm=robot.right_arm,
        tool=sponge,
        target_pose=Pose.from_xyz_rpy(x=2.4, y=2.2, z=1.0, reference_frame=world.root),
        technique=WipingTechnique.SPREAD,
        length=0.3,
        width=0.001,
    )

    assert action._lanes() == 2


# %% pressing while wiping


def pressing_wipe(world, robot, sponge, **arguments) -> WipingAction:
    """
    :return: A wipe that presses onto a patch at a known height.
    """
    return WipingAction(
        arm=robot.right_arm,
        tool=sponge,
        target_pose=Pose.from_xyz_rpy(x=2.4, y=2.2, z=1.0, reference_frame=world.root),
        technique=WipingTechnique.SPREAD,
        length=0.3,
        desired_force=Vector3(z=10.0),
        **arguments,
    )


def test_the_pressing_path_puts_the_tools_face_on_the_surface(tool_action_world):
    """
    A path that puts the frame itself on the surface starts the press a tool thickness
    inside it, which the admittance then has to climb back out of.
    """
    world, robot, context, container, tool_body = tool_action_world
    sponge = Sponge(root=tool_body)
    surface_height = 1.0

    action = pressing_wipe(world, robot, sponge)
    motions = _expanded_aligned_motions(action, context)

    heights = [point.to_np().flatten()[2] for point in motions[0].waypoints]
    assert heights == pytest.approx(
        [surface_height + sponge.contact_face_distance] * len(heights)
    )


def test_a_press_tracks_ahead_of_its_own_offset_when_asked_to(tool_action_world):
    """
    The arm reaches a commanded point later than it is commanded, so a press that is
    given the offset as it stands always acts on a pose the arm has left behind.
    """
    world, robot, context, container, tool_body = tool_action_world
    sponge = Sponge(root=tool_body)
    lead_time = 0.04

    action = pressing_wipe(world, robot, sponge, admittance_lead_time=lead_time)
    motions = _expanded_aligned_motions(action, context)
    trajectory = next(
        node
        for group in motions[0].motion_chart.nodes
        if isinstance(group, Goal)
        for node in group.nodes
        if isinstance(node, AdmittanceCartesianTrajectory)
    )

    assert trajectory.lead_time == lead_time


def test_a_press_yields_only_as_far_as_it_is_allowed_to_travel(tool_action_world):
    """
    The tool feels friction as it travels, and a press free to yield to that drifts
    against its own direction of travel until it stalls.
    """
    world, robot, context, container, tool_body = tool_action_world
    sponge = Sponge(root=tool_body)
    travel = Vector3(x=0.0, y=0.0, z=0.03)

    action = pressing_wipe(world, robot, sponge, admittance_travel=travel)
    motions = _expanded_aligned_motions(action, context)
    trajectory = next(
        node
        for group in motions[0].motion_chart.nodes
        if isinstance(group, Goal)
        for node in group.nodes
        if isinstance(node, AdmittanceCartesianTrajectory)
    )

    assert trajectory.maximum_offset is travel
