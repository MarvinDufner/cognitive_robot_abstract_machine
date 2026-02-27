from dataclasses import dataclass

from typing_extensions import List

from pycram.datastructures.dataclasses import Context
from pycram.datastructures.enums import ApproachDirection, VerticalAlignment, Arms
from pycram.datastructures.grasp import GraspDescription
from pycram.datastructures.pose import PoseStamped
from pycram.language import SequentialPlan
from pycram.parameter_inference import (
    ParameterIdentifier,
    T,
    PlanDomain,
    ValueDomainSpecification,
)
from pycram.parameter_rules.condition_parameterizer import ConditionParameterizer
from pycram.parameter_rules.default_type_domains import (
    EnumDomainSpecification,
    GraspDomainSpecification,
    load_default_domains,
)
from pycram.robot_plans import (
    PickUpActionDescription,
    MoveTorsoActionDescription,
    ParkArmsActionDescription,
    PlaceActionDescription,
)
from semantic_digital_twin.datastructures.definitions import TorsoState
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix


def test_infer_arm_param(immutable_model_world):
    world, view, context = immutable_model_world

    pick_action = PickUpActionDescription(
        world.get_body_by_name("milk.stl"),
        ...,
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

    plan.parameter_infeerer.add_domains(
        e_domain := EnumDomainSpecification(Arms),
        g_domain := GraspDomainSpecification(
            GraspDescription, view.left_arm.manipulator
        ),
    )

    plan.parameter_infeerer.add_infer_system(ConditionParameterizer())

    bindings = list(plan.parameter_infeerer.parameterize(pick_action))[0]

    assert len(bindings) == 3
    assert list(bindings.keys()) == [
        "object_designator",
        "arm",
        "grasp_description",
    ]

    assert bindings["arm"] in [Arms.LEFT, Arms.RIGHT, Arms.BOTH]
    assert bindings["grasp_description"].approach_direction in [
        ApproachDirection.FRONT,
        ApproachDirection.BACK,
        ApproachDirection.LEFT,
        ApproachDirection.RIGHT,
    ]
    assert bindings["object_designator"] == world.get_body_by_name("milk.stl")
