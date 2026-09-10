import numpy as np

from semantic_digital_twin.robots.hsrb import HSRB
from semantic_digital_twin.robots.pr2 import PR2

SAMPLES_COARSE = 3
SAMPLES_FINE = 5


def test_the_arm_reaches_a_plausible_distance(hsr_world_copy):
    arm = hsr_world_copy.get_semantic_annotations_by_type(HSRB)[0].get_arms()[0]

    reach = arm.maximum_reach(SAMPLES_COARSE)

    assert 0.3 < reach < 1.5


def test_denser_sampling_never_reports_a_shorter_reach(hsr_world_copy):
    """
    The sweep is a maximum over sampled configurations, so refining the grid can only
    find configurations at least as far out.
    """
    arm = hsr_world_copy.get_semantic_annotations_by_type(HSRB)[0].get_arms()[0]

    assert arm.maximum_reach(SAMPLES_FINE) >= arm.maximum_reach(SAMPLES_COARSE)


def test_measuring_the_reach_leaves_the_world_as_it_was(hsr_world_copy):
    arm = hsr_world_copy.get_semantic_annotations_by_type(HSRB)[0].get_arms()[0]
    positions_before = hsr_world_copy.state.positions.copy()

    arm.maximum_reach(SAMPLES_COARSE)

    np.testing.assert_allclose(hsr_world_copy.state.positions, positions_before)


def test_every_arm_of_a_two_armed_robot_is_measured_separately(pr2_world_copy):
    """
    The two arms are not mirror images -- their elbow and wrist limits are identical
    rather than negated -- so each one is swept on its own.
    """
    left, right = pr2_world_copy.get_semantic_annotations_by_type(PR2)[0].get_arms()[:2]

    assert 0.5 < left.maximum_reach(SAMPLES_COARSE) < 1.5
    assert 0.5 < right.maximum_reach(SAMPLES_COARSE) < 1.5


def test_the_footprint_radius_covers_the_wider_horizontal_side(hsr_world_copy):
    base = hsr_world_copy.get_semantic_annotations_by_type(HSRB)[0].mobile_base

    box = base.bounding_box

    assert base.footprint_radius == max(box.depth, box.width) / 2.0
