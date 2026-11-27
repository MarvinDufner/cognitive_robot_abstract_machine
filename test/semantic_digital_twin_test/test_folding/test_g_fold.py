"""
Tests for g-folds of layered cloth.
"""

from __future__ import annotations

import numpy
import pytest

from semantic_digital_twin.folding.g_fold import GFold, LayeredCloth
from semantic_digital_twin.folding.geometry import FoldSegment, Polygon

# %% a towel


@pytest.fixture
def towel() -> LayeredCloth:
    """
    :return: A towel 0.6 m wide and 0.4 m long, lying with a corner at the origin.
    """
    return LayeredCloth(
        layers=[
            Polygon(
                vertices=numpy.array([[0.0, 0.0], [0.6, 0.0], [0.6, 0.4], [0.0, 0.4]])
            )
        ]
    )


@pytest.fixture
def fold_in_half() -> FoldSegment:
    """
    :return: The segment across the middle of the towel that folds its far half onto
        its near half.
    """
    return FoldSegment.through_points(
        first=numpy.array([-0.1, 0.2]),
        second=numpy.array([0.7, 0.2]),
        folded_point=numpy.array([0.3, 0.4]),
    )


def test_folding_a_towel_in_half_lays_its_far_half_onto_its_near_half(
    towel: LayeredCloth, fold_in_half: FoldSegment
) -> None:
    """
    The fold leaves two layers, each the near half of the towel.
    """
    folded = towel.folded(segment=fold_in_half)

    assert [layer.area for layer in folded.layers] == pytest.approx([0.12, 0.12])
    assert all(
        numpy.allclose(numpy.sort(layer.vertices[:, 1]), [0.0, 0.0, 0.2, 0.2])
        for layer in folded.layers
    )


def test_a_towel_folded_in_half_is_held_at_its_far_corners(
    towel: LayeredCloth, fold_in_half: FoldSegment
) -> None:
    """
    Only the two far corners would slide down while the far half hangs, so a gripper
    each holds them and lays them onto the near corners.
    """
    grasps = GFold(cloth=towel, segment=fold_in_half).grasps(gripper_radius=0.05)

    assert sorted(map(tuple, (grasp.grasp_position for grasp in grasps))) == [
        (0.0, 0.4),
        (0.6, 0.4),
    ]
    assert sorted(map(tuple, (grasp.placement_position for grasp in grasps))) == [
        pytest.approx((0.0, 0.0)),
        pytest.approx((0.6, 0.0)),
    ]


def test_a_gripper_wide_enough_takes_both_far_corners_at_once(
    towel: LayeredCloth, fold_in_half: FoldSegment
) -> None:
    """
    A gripper whose reach spans the far edge holds both corners from its middle.
    """
    [grasp] = GFold(cloth=towel, segment=fold_in_half).grasps(gripper_radius=0.3)

    assert numpy.allclose(grasp.grasp_position, [0.3, 0.4])
    assert len(grasp.held_vertices) == 2


def test_coinciding_corners_of_stacked_layers_share_a_grasp(
    towel: LayeredCloth, fold_in_half: FoldSegment
) -> None:
    """
    Folding the halved towel again lifts two layers whose corners lie on each other; one
    gripper takes both layers at each pair.
    """
    halved = towel.folded(segment=fold_in_half)
    across = FoldSegment.through_points(
        first=numpy.array([0.3, -0.1]),
        second=numpy.array([0.3, 0.3]),
        folded_point=numpy.array([0.0, 0.1]),
    )

    grasps = GFold(cloth=halved, segment=across).grasps(gripper_radius=0.05)

    assert sorted(map(tuple, (grasp.grasp_position for grasp in grasps))) == [
        (0.0, 0.0),
        (0.0, 0.2),
    ]
    assert [len(grasp.held_vertices) for grasp in grasps] == [2, 2]


def test_the_tent_path_peaks_above_the_segment_at_the_grasp_distance(
    towel: LayeredCloth, fold_in_half: FoldSegment
) -> None:
    """
    The gripper rises as high as it is far from the segment, right above the segment,
    and comes down again mirrored.
    """
    path = GFold(cloth=towel, segment=fold_in_half).tent_path(
        grasp_position=numpy.array([0.6, 0.4])
    )

    peak = path[numpy.argmax(path[:, 2])]
    assert numpy.allclose(peak, [0.6, 0.2, 0.2])
    assert numpy.allclose(path[[0, -1]], [[0.6, 0.4, 0.0], [0.6, 0.0, 0.0]])
