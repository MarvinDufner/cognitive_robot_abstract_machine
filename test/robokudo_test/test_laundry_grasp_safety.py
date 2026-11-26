"""
Tests for keeping the laundry grasp on the laundry and off the basket: the fingers stay
clear of the basket's walls, noisy or thin depth cannot win the highest point, and
surfaces curving down more sharply than cloth are not grasped.
"""

from __future__ import annotations

# %% Imports
import dataclasses

import numpy as np
import pytest

from robokudo.annotators.exceptions import GripperOpeningMissing
from robokudo.annotators.laundry_grasp_pose import LaundryGraspPoseAnnotator
from robokudo.cas import CAS
from robokudo.types.camera import RegisteredDepthImage
from robokudo.types.cv import Point2D
from robokudo.types.laundry import BasketFootprint
from robokudo.types.scene import ObjectHypothesis

from .dataset.laundry_basket_on_floor import LaundryBasketScene
from .test_cloth_unfolding import update_in_pipeline
from .test_laundry_pickup import (
    basket,
    cas,
    grasp_annotator,
    laundry,
    laundry_grasps,
    scene,
)

pytestmark = pytest.mark.usefixtures("annotator_output_buffer", "laundry")

# %% the basket's footprint


def test_the_footprint_measures_distances_to_its_outline() -> None:
    """
    Points inside the outline lie at a positive distance from it, points outside at a
    negative one.
    """
    corners = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]]
    )
    footprint = BasketFootprint.from_points(points=corners)

    distances = footprint.distances_inside(
        points=np.array([[0.5, 0.5], [0.9, 0.5], [1.5, 0.5]])
    )

    assert np.allclose(distances, [0.5, 0.1, -0.5])


# %% keeping the fingers clear of the walls


def test_the_grasp_keeps_half_the_gripper_opening_from_the_walls(
    scene: LaundryBasketScene,
    cas: CAS,
    basket: ObjectHypothesis,
    grasp_annotator: LaundryGraspPoseAnnotator,
) -> None:
    """
    Laundry peaking next to a wall is grasped where the open fingers still clear it.
    """
    apex = Point2D(x=125, y=100)
    cas.depth_image[:] = dataclasses.replace(scene, peak_pixel=apex).depth_image()

    update_in_pipeline(grasp_annotator, cas)

    [grasp] = laundry_grasps(basket)
    depth = RegisteredDepthImage.from_cas(cas)
    rows, columns = np.nonzero(
        basket.roi.full_image_mask((scene.image_size, scene.image_size))
    )
    right_wall = np.nanmax(depth.world_points(np.column_stack([columns, rows]))[:, 0])
    clearance = grasp_annotator.descriptor.parameters.gripper_maximum_opening / 2
    assert grasp.pixel.x < apex.x
    assert grasp.grasp_pose.to_np()[0, 3] <= right_wall - clearance


def test_without_the_gripper_opening_the_walls_cannot_be_kept_clear(cas: CAS) -> None:
    """
    The opening comes from the robot in use, so it has to be configured.
    """
    with pytest.raises(GripperOpeningMissing):
        update_in_pipeline(LaundryGraspPoseAnnotator(), cas)


# %% finding a top wide enough to be laundry


def test_a_single_noisy_pixel_above_the_laundry_is_not_grasped(
    scene: LaundryBasketScene,
    cas: CAS,
    basket: ObjectHypothesis,
    grasp_annotator: LaundryGraspPoseAnnotator,
) -> None:
    """
    A lone pixel measured higher than the laundry around it does not win the grasp.
    """
    cas.depth_image[110, 80] = scene.rim_depth

    update_in_pipeline(grasp_annotator, cas)

    [grasp] = laundry_grasps(basket)
    assert grasp.pixel == scene.peak_pixel


def test_a_narrow_raised_edge_is_not_grasped(
    scene: LaundryBasketScene,
    cas: CAS,
    basket: ObjectHypothesis,
    grasp_annotator: LaundryGraspPoseAnnotator,
) -> None:
    """
    A strip only a few pixels wide that stands sharply above the laundry, as a wall's
    edge seen through the laundry's mask does, does not win the grasp.
    """
    cas.depth_image[70:120, 75:79] = scene.rim_depth + 100

    update_in_pipeline(grasp_annotator, cas)

    [grasp] = laundry_grasps(basket)
    assert grasp.pixel == scene.peak_pixel


# %% refusing surfaces sharper than cloth


def test_laundry_curving_down_more_sharply_than_allowed_is_not_grasped(
    scene: LaundryBasketScene,
    cas: CAS,
    basket: ObjectHypothesis,
    grasp_annotator: LaundryGraspPoseAnnotator,
) -> None:
    """
    A top that drops off more sharply than a cloth fold offers no grasp.
    """
    grasp_annotator.descriptor.parameters.maximum_curvature = 1.0

    update_in_pipeline(grasp_annotator, cas)

    assert laundry_grasps(basket) == []


def test_laundry_curving_down_as_gently_as_cloth_is_grasped(
    scene: LaundryBasketScene,
    cas: CAS,
    basket: ObjectHypothesis,
    grasp_annotator: LaundryGraspPoseAnnotator,
) -> None:
    """
    The same top is grasped within the default limit on how sharply cloth curves.
    """
    update_in_pipeline(grasp_annotator, cas)

    [grasp] = laundry_grasps(basket)
    assert grasp.pixel == scene.peak_pixel
