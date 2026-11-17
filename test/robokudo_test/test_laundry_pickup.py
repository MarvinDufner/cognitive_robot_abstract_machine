"""
Tests for perceiving where to pick laundry up from a basket: the point cloud of the
detected basket, the grasp on the highest point of its laundry, and recording that grasp
on the basket in the semantic digital twin.
"""

from __future__ import annotations

import numpy as np
import open3d
import pytest

import robokudo.world
from robokudo.annotators.hypothesis_point_cloud import HypothesisPointCloudAnnotator
from robokudo.annotators.exceptions import LaundryBasketNotAnnotated
from robokudo.annotators.laundry_grasp_connector import LaundryGraspConnector
from robokudo.annotators.laundry_grasp_pose import LaundryGraspPoseAnnotator
from robokudo.cas import CAS
from robokudo.types.annotation import Classification
from robokudo.types.camera import RegisteredDepthImage
from robokudo.types.laundry import LaundryBasketClassName, LaundryGraspAnnotation
from robokudo.types.scene import ObjectHypothesis
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    LaundryBasket,
)
from semantic_digital_twin.world import World

from .dataset.laundry_basket_on_floor import LaundryBasketScene
from .test_cloth_unfolding import update_in_pipeline

pytestmark = pytest.mark.usefixtures("annotator_output_buffer")

# %% fixtures


@pytest.fixture
def scene(perception_world: World) -> LaundryBasketScene:
    """
    :return: A laundry basket seen from one metre above, posed in RoboKudo's world.
    """
    return LaundryBasketScene(world_frame=perception_world.root)


@pytest.fixture
def cas(scene: LaundryBasketScene) -> CAS:
    """
    :return: A CAS holding the scene's camera frame.
    """
    return scene.cas()


@pytest.fixture
def basket(scene: LaundryBasketScene, cas: CAS) -> ObjectHypothesis:
    """
    :return: The basket as the open-vocabulary detector segments and classifies it.
    """
    hypothesis = ObjectHypothesis()
    hypothesis.roi.roi = scene.basket_region
    hypothesis.roi.mask = scene.basket_mask()
    hypothesis.annotations.append(
        Classification(classname=LaundryBasketClassName.LAUNDRY_BASKET, confidence=0.9)
    )
    cas.annotations.append(hypothesis)
    return hypothesis


def laundry_grasps(hypothesis: ObjectHypothesis) -> list[LaundryGraspAnnotation]:
    """
    :return: The laundry grasps the hypothesis is annotated with.
    """
    return CAS.filter_by_type(LaundryGraspAnnotation, hypothesis.annotations)


# %% the point cloud of a detected object


def test_a_segmented_object_gets_the_point_cloud_its_mask_covers(
    scene: LaundryBasketScene, cas: CAS, basket: ObjectHypothesis
) -> None:
    """
    Every depth measurement inside the mask becomes a point of the object.
    """
    update_in_pipeline(HypothesisPointCloudAnnotator(), cas)

    assert len(basket.points.points) == scene.basket_region.width * (
        scene.basket_region.height
    )


def test_an_object_that_has_a_point_cloud_keeps_it(
    cas: CAS, basket: ObjectHypothesis
) -> None:
    """
    A point cloud the detector already built is left as it is.
    """
    points = open3d.geometry.PointCloud()
    basket.points = points

    update_in_pipeline(HypothesisPointCloudAnnotator(), cas)

    assert basket.points is points


def test_an_object_without_a_mask_gets_no_point_cloud(cas: CAS) -> None:
    """
    Without a mask it is unclear which measurements belong to the object.
    """
    hypothesis = ObjectHypothesis()
    cas.annotations.append(hypothesis)

    update_in_pipeline(HypothesisPointCloudAnnotator(), cas)

    assert hypothesis.points is None


# %% the grasp on the laundry


def test_the_laundry_is_grasped_at_its_highest_point_not_at_the_rim(
    scene: LaundryBasketScene, cas: CAS, basket: ObjectHypothesis
) -> None:
    """
    The rim stands higher than the laundry, yet the grasp is on the laundry's peak.
    """
    update_in_pipeline(LaundryGraspPoseAnnotator(), cas)

    [grasp] = laundry_grasps(basket)
    assert grasp.pixel == scene.peak_pixel
    expected_position = RegisteredDepthImage.from_cas(cas).world_point(scene.peak_pixel)
    assert np.allclose(grasp.grasp_pose.to_np()[:3, 3], expected_position.to_np()[:3])


@pytest.mark.parametrize("classname", list(LaundryBasketClassName))
def test_a_basket_detected_by_any_of_its_class_names_offers_a_laundry_grasp(
    cas: CAS, basket: ObjectHypothesis, classname: LaundryBasketClassName
) -> None:
    """
    A laundry basket is searched for laundry whichever of its class names detected it.
    """
    basket.annotations = [Classification(classname=classname, confidence=0.9)]

    update_in_pipeline(LaundryGraspPoseAnnotator(), cas)

    assert len(laundry_grasps(basket)) == 1


def test_the_laundry_is_approached_from_above_with_fingers_across_the_view(
    cas: CAS, basket: ObjectHypothesis
) -> None:
    """
    The gripper comes straight down and closes along the image's horizontal axis.
    """
    update_in_pipeline(LaundryGraspPoseAnnotator(), cas)

    [grasp] = laundry_grasps(basket)
    rotation = grasp.grasp_pose.to_np()[:3, :3]
    camera_x_axis = cas.camera_observation.world_T_camera.to_np()[:3, 0]
    assert np.allclose(rotation[:, 0], [0.0, 0.0, -1.0])
    assert np.allclose(rotation[:, 1], camera_x_axis)


def test_no_grasp_is_found_where_the_camera_measured_no_depth_in_the_basket(
    scene: LaundryBasketScene, cas: CAS, basket: ObjectHypothesis
) -> None:
    """
    A basket whose inside the camera did not measure offers no grasp.
    """
    left, top, right, bottom = scene.basket_region.get_corner_points()
    cas.depth_image[top:bottom, left:right] = 0

    update_in_pipeline(LaundryGraspPoseAnnotator(), cas)

    assert laundry_grasps(basket) == []


def test_an_object_that_is_no_laundry_basket_offers_no_laundry_grasp(
    cas: CAS, basket: ObjectHypothesis
) -> None:
    """
    Only laundry baskets are searched for laundry.
    """
    basket.annotations = [Classification(classname="chair", confidence=0.9)]

    update_in_pipeline(LaundryGraspPoseAnnotator(), cas)

    assert laundry_grasps(basket) == []


# %% recording the grasp on the laundry basket in the world


@pytest.fixture
def laundry_basket(
    cas: CAS, basket: ObjectHypothesis, perception_world: World
) -> LaundryBasket:
    """
    :return: The laundry basket annotation of the body the world connector made of the
        basket.
    """
    robokudo.world.add_object_hypothesis_as_belief_state(basket, cas)
    with perception_world.modify_world():
        laundry_basket = LaundryBasket(
            root=robokudo.world.find_believed_body(basket),
            class_label=LaundryBasketClassName.LAUNDRY_BASKET,
        )
        perception_world.add_semantic_annotation(laundry_basket)
    return laundry_basket


def test_the_laundry_basket_records_the_grasp_perception_found(
    cas: CAS, basket: ObjectHypothesis, laundry_basket: LaundryBasket
) -> None:
    """
    The grasp on the laundry is recorded on the laundry basket.
    """
    update_in_pipeline(LaundryGraspPoseAnnotator(), cas)
    [grasp] = laundry_grasps(basket)

    update_in_pipeline(LaundryGraspConnector(), cas)

    assert laundry_basket.laundry_grasp_pose is grasp.grasp_pose


def test_a_frame_without_a_grasp_keeps_the_grasp_found_before(
    cas: CAS, basket: ObjectHypothesis, laundry_basket: LaundryBasket
) -> None:
    """
    Between queries the basket is only tracked, so the last grasp found stays recorded.
    """
    update_in_pipeline(LaundryGraspPoseAnnotator(), cas)
    [grasp] = laundry_grasps(basket)
    update_in_pipeline(LaundryGraspConnector(), cas)
    basket.annotations.remove(grasp)

    update_in_pipeline(LaundryGraspConnector(), cas)

    assert laundry_basket.laundry_grasp_pose is grasp.grasp_pose


def test_a_grasp_on_a_basket_not_annotated_as_laundry_basket_is_refused(
    cas: CAS, basket: ObjectHypothesis
) -> None:
    """
    A grasp has nowhere to be recorded when the basket's body is no laundry basket.
    """
    robokudo.world.add_object_hypothesis_as_belief_state(basket, cas)
    update_in_pipeline(LaundryGraspPoseAnnotator(), cas)

    with pytest.raises(LaundryBasketNotAnnotated):
        update_in_pipeline(LaundryGraspConnector(), cas)
