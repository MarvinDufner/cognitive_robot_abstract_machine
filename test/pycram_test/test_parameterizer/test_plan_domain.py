from pycram.datastructures.enums import ApproachDirection, VerticalAlignment, Arms
from pycram.datastructures.grasp import GraspDescription
from pycram.language import SequentialPlan
from pycram.parameter_inference import ValueDomainSpecification
from pycram.parameter_rules.default_type_domains import (
    EnumDomainSpecification,
    GraspDomainSpecification,
)
from pycram.robot_plans import PickUpActionDescription


def test_plan_domain(immutable_model_world):
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

    plan = SequentialPlan(context, pick_action)

    plan.parameter_infeerer.add_domains(
        e_domain := EnumDomainSpecification(Arms),
        g_domain := GraspDomainSpecification(
            GraspDescription, view.left_arm.manipulator
        ),
    )

    assert len(plan.parameter_infeerer.plan_domain.domain_specifications) == 2
    assert len(plan.parameter_infeerer.plan_domain.designator_domains) == 1
    assert plan.parameter_infeerer.plan_domain.get_domain_for_type(Arms) == e_domain
    assert (
        plan.parameter_infeerer.plan_domain.get_domain_for_type(GraspDescription)
        == g_domain
    )

    pick_domain = plan.parameter_infeerer.plan_domain.designator_domains[pick_action]

    assert len(pick_domain.parameter_domains) == 3
    assert len(pick_domain.kwargs) == 3
    assert [type(d) for d in pick_domain.parameter_domains.values()] == [
        ValueDomainSpecification,
        EnumDomainSpecification,
        GraspDomainSpecification,
    ]


def test_plan_domain_no_domain(immutable_model_world):
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

    plan = SequentialPlan(context, pick_action)
    plan.parameter_infeerer.plan_domain.create_plan_domain()

    assert len(plan.parameter_infeerer.plan_domain.designator_domains) == 1
    assert len(plan.parameter_infeerer.plan_domain.domain_specifications) == 0

    pick_domain = plan.parameter_infeerer.plan_domain.designator_domains[pick_action]

    assert len(pick_domain.parameter_domains) == 3
    assert [type(d) for d in pick_domain.parameter_domains.values()] == [
        ValueDomainSpecification,
        ValueDomainSpecification,
        ValueDomainSpecification,
    ]
