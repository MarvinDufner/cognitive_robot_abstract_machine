import numpy as np
import pytest
import rclpy

from krrood.entity_query_language.entity import (
    exists,
    get_false_statements,
    evaluate_condition,
)
from pycram.datastructures.enums import Arms, ApproachDirection, VerticalAlignment
from pycram.datastructures.grasp import GraspDescription
from pycram.failures import ConditionNotSatisfied, PlanFailure
from pycram.language import SequentialPlan
from pycram.motion_executor import simulated_robot
from pycram.robot_plans import PickUpAction, PickUpActionDescription
from semantic_digital_twin.adapters.ros.visualization.viz_marker import (
    VizMarkerPublisher,
)
from semantic_digital_twin.reasoning.predicates import reachable
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world_description.world_entity import Body


def test_get_bound_variables(immutable_model_world):
    world, view, context = immutable_model_world

    pick_action = PickUpAction(
        world.get_body_by_name("milk.stl"),
        Arms.LEFT,
        GraspDescription(
            ApproachDirection.FRONT,
            VerticalAlignment.NoAlignment,
            view.left_arm.manipulator,
        ),
    )

    bound_variables = pick_action._create_variables(bound=True)

    assert len(bound_variables) == 3
    assert list(bound_variables.keys()) == [
        pick_action.object_designator,
        pick_action.arm,
        pick_action.grasp_description,
    ]
    assert list(bound_variables[pick_action.arm]._domain_) == [Arms.LEFT]
    assert bound_variables[pick_action.arm]._type_ == Arms
    assert list(bound_variables[pick_action.object_designator]._domain_) == [
        world.get_body_by_name("milk.stl")
    ]
    assert bound_variables[pick_action.object_designator]._type_ == Body


def test_get_unbound_variables(immutable_model_world):
    world, view, context = immutable_model_world

    pick_action = PickUpAction(
        kse := world.get_body_by_name("milk.stl"),
        arm := Arms.LEFT,
        grasp_desc := GraspDescription(
            ApproachDirection.FRONT,
            VerticalAlignment.NoAlignment,
            view.left_arm.manipulator,
        ),
    )
    SequentialPlan(context, pick_action)

    unbound_variables = pick_action._create_variables(bound=False)

    assert len(unbound_variables) == 3
    assert list(unbound_variables[arm]._domain_) == [Arms.LEFT, Arms.RIGHT, Arms.BOTH]
    assert len(list(unbound_variables[grasp_desc]._domain_)) == 12
    assert len(list(unbound_variables[kse]._domain_)) == 1


def test_pick_up_pre_conditions(mutable_model_world):
    world, view, context = mutable_model_world

    pick_action = PickUpAction(
        world.get_body_by_name("milk.stl"),
        Arms.LEFT,
        GraspDescription(
            ApproachDirection.FRONT,
            VerticalAlignment.NoAlignment,
            view.left_arm.manipulator,
        ),
    )

    plan = SequentialPlan(context, pick_action)

    pre_condition = pick_action.pre_condition()
    post_condition = pick_action.post_condition()

    assert evaluate_condition(pre_condition) == False

    false_statements = get_false_statements(pre_condition)

    assert len(false_statements) == 1
    assert false_statements[0]._name_ == "reachable"

    with pytest.raises(ConditionNotSatisfied):
        pick_action.evaluate_pre_condition()

    view.root.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
        1.8, 2, 0
    )

    assert evaluate_condition(pre_condition) == True

    with simulated_robot:
        plan.perform()

    assert evaluate_condition(pre_condition) == False
    assert evaluate_condition(post_condition) == True
    assert pick_action.evaluate_post_condition() == True


def test_pick_up_pre_condition_find_parameter(immutable_model_world):
    world, view, context = immutable_model_world
    node = rclpy.create_node("test")
    VizMarkerPublisher(world, node).with_tf_publisher()

    pick_action = PickUpAction(
        world.get_body_by_name("milk.stl"),
        Arms.RIGHT,
        GraspDescription(
            ApproachDirection.FRONT,
            VerticalAlignment.NoAlignment,
            view.right_arm.manipulator,
        ),
    )

    plan = SequentialPlan(context, pick_action)

    view.root.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
        1.8,
        1.4,
        0,
    )

    with pytest.raises(ConditionNotSatisfied):
        pick_action.evaluate_pre_condition()

    possible = list(pick_action.find_possible_parameter())
    assert len(possible) == 24
    assert Arms.RIGHT not in [p["arm"] for p in possible]
    assert list(possible[0].keys()) == ["object_designator", "arm", "grasp_description"]
    assert len(set([p["object_designator"] for p in possible])) == 1


def test_pick_up_post_condition(mutable_model_world):
    world, view, context = mutable_model_world
    pick_action = PickUpAction(
        world.get_body_by_name("milk.stl"),
        Arms.LEFT,
        GraspDescription(
            ApproachDirection.FRONT,
            VerticalAlignment.NoAlignment,
            view.left_arm.manipulator,
        ),
    )
    view.root.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
        1.8, 2, 0
    )

    plan = SequentialPlan(context, pick_action)

    assert pick_action.evaluate_pre_condition()

    with simulated_robot:
        plan.perform()

    assert world.get_body_by_name(
        "milk.stl"
    ) in world.get_kinematic_structure_entities_of_branch(
        view.left_arm.manipulator.tool_frame
    )

    assert pick_action.evaluate_post_condition()


def test_context_evaluate_condition(mutable_model_world):
    world, view, context = mutable_model_world

    pick_action = PickUpAction(
        world.get_body_by_name("milk.stl"),
        Arms.LEFT,
        GraspDescription(
            ApproachDirection.FRONT,
            VerticalAlignment.NoAlignment,
            view.left_arm.manipulator,
        ),
    )
    # Make action impossible
    view.root.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
        1.0, 2, 0
    )

    plan = SequentialPlan(context, pick_action)
    with pytest.raises(ConditionNotSatisfied):
        with simulated_robot:
            plan.perform()

    context.evaluate_conditions = False

    with pytest.raises(TimeoutError):
        with simulated_robot:
            plan.perform()
