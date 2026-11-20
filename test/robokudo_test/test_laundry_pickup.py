"""
Tests for perceiving where to pick laundry up from a basket: the point cloud of the
detected basket, the grasp on the highest point of its laundry, and recording that grasp
on the basket in the semantic digital twin.
"""

from __future__ import annotations

import cv2
import numpy as np
import open3d
import pytest

import robokudo.world
from robokudo.annotators.hypothesis_point_cloud import HypothesisPointCloudAnnotator
from robokudo.annotators.exceptions import FloorPlaneMissing, LaundryBasketNotAnnotated
from robokudo.annotators.laundry_grasp_connector import LaundryGraspConnector
from robokudo.annotators.laundry_grasp_pose import LaundryGraspPoseAnnotator
from robokudo.cas import CAS, CASViews
from robokudo.types.annotation import Classification, Plane
from robokudo.types.camera import RegisteredDepthImage
from robokudo.types.cv import Rect
from robokudo.types.laundry import (
    LaundryBasketClassName,
    LaundryClassName,
    LaundryGraspAnnotation,
)
from robokudo.types.scene import ObjectHypothesis
from semantic_digital_twin.robots.hsrb import HSRBGripper
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
def grasp_annotator() -> LaundryGraspPoseAnnotator:
    """
    :return: A laundry grasp annotator for the HSRB's hand.
    """
    descriptor = LaundryGraspPoseAnnotator.Descriptor()
    descriptor.parameters.gripper_maximum_opening = HSRBGripper.maximum_opening_width
    return LaundryGraspPoseAnnotator(descriptor=descriptor)


@pytest.fixture
def cas(scene: LaundryBasketScene) -> CAS:
    """
    :return: A CAS holding the scene's camera frame and the floor plane detected in it.
    """
    cas = scene.cas()
    cas.annotations.append(scene.floor_plane())
    return cas


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


def detected_laundry(cas: CAS, region: Rect, mask: np.ndarray) -> ObjectHypothesis:
    """
    :return: Laundry as the open-vocabulary detector segments and classifies it, in the
        CAS.
    """
    hypothesis = ObjectHypothesis()
    hypothesis.roi.roi = region
    hypothesis.roi.mask = mask
    hypothesis.annotations.append(
        Classification(classname=LaundryClassName.CLOTHES, confidence=0.5)
    )
    cas.annotations.append(hypothesis)
    return hypothesis


@pytest.fixture
def laundry(scene: LaundryBasketScene, cas: CAS) -> ObjectHypothesis:
    """
    :return: The laundry inside the basket, as the detector segments it.
    """
    return detected_laundry(cas, scene.laundry_region(), scene.laundry_mask())


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

    assert len(basket.points.points) == np.count_nonzero(scene.basket_mask())


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


@pytest.mark.usefixtures("laundry")
def test_the_laundry_is_grasped_at_its_highest_point_not_at_the_rim(
    scene: LaundryBasketScene,
    cas: CAS,
    basket: ObjectHypothesis,
    grasp_annotator: LaundryGraspPoseAnnotator,
) -> None:
    """
    The rim stands higher than the laundry, yet the grasp is on the laundry's peak.
    """
    update_in_pipeline(grasp_annotator, cas)

    [grasp] = laundry_grasps(basket)
    assert grasp.pixel == scene.peak_pixel
    expected_position = RegisteredDepthImage.from_cas(cas).world_point(scene.peak_pixel)
    assert np.allclose(grasp.grasp_pose.to_np()[:3, 3], expected_position.to_np()[:3])


@pytest.mark.usefixtures("laundry")
def test_the_grasp_records_the_height_of_the_laundry_above_the_floor(
    scene: LaundryBasketScene,
    cas: CAS,
    basket: ObjectHypothesis,
    grasp_annotator: LaundryGraspPoseAnnotator,
) -> None:
    """
    The laundry's height is measured from the floor, not from the world's origin.
    """
    update_in_pipeline(grasp_annotator, cas)

    [grasp] = laundry_grasps(basket)
    assert np.isclose(
        grasp.height_above_floor,
        (scene.table_depth - scene.peak_depth)
        / RegisteredDepthImage.millimetres_per_metre,
    )


def test_the_laundry_farthest_from_a_tilted_floor_is_grasped(
    scene: LaundryBasketScene,
    cas: CAS,
    basket: ObjectHypothesis,
    laundry: ObjectHypothesis,
    grasp_annotator: LaundryGraspPoseAnnotator,
) -> None:
    """
    Where the floor is tilted against the world, the laundry is highest where it is
    farthest from the floor, not where it is highest in the world, as far toward that
    side as the open fingers still clear the wall.
    """
    [floor] = CAS.filter_by_type(Plane, cas.annotations)
    floor.model = [np.sqrt(0.5), 0.0, np.sqrt(0.5), -1.0]

    update_in_pipeline(grasp_annotator, cas)

    [grasp] = laundry_grasps(basket)
    depth = RegisteredDepthImage.from_cas(cas)
    basket_rows, basket_columns = np.nonzero(
        basket.roi.full_image_mask((scene.image_size, scene.image_size))
    )
    left_wall = np.nanmin(
        depth.world_points(np.column_stack([basket_columns, basket_rows]))[:, 0]
    )
    clearance = grasp_annotator.descriptor.parameters.gripper_maximum_opening / 2
    left, _, right, _ = laundry.roi.roi.get_corner_points()
    row_columns = np.arange(left, right)
    row_points = depth.world_points(
        np.column_stack([row_columns, np.full_like(row_columns, grasp.pixel.y)])
    )
    first_clear_column = row_columns[row_points[:, 0] >= left_wall + clearance][0]
    assert first_clear_column <= grasp.pixel.x < scene.peak_pixel.x - scene.peak_radius


@pytest.mark.usefixtures("laundry")
def test_without_a_detected_floor_the_laundry_has_no_height(
    cas: CAS, basket: ObjectHypothesis, grasp_annotator: LaundryGraspPoseAnnotator
) -> None:
    """
    Heights are measured from the floor, so plane detection has to find it first.
    """
    [floor] = CAS.filter_by_type(Plane, cas.annotations)
    cas.annotations.remove(floor)

    with pytest.raises(FloorPlaneMissing):
        update_in_pipeline(grasp_annotator, cas)


@pytest.mark.usefixtures("basket")
def test_the_lowest_laundry_is_shown_in_the_coldest_color_of_the_heatmap(
    scene: LaundryBasketScene,
    cas: CAS,
    laundry: ObjectHypothesis,
    grasp_annotator: LaundryGraspPoseAnnotator,
) -> None:
    """
    The laundry is colored by its height above the floor, from its lowest to its highest
    point.
    """
    left, top, right, bottom = laundry.roi.roi.get_corner_points()

    update_in_pipeline(grasp_annotator, cas)

    coldest_color = cv2.applyColorMap(
        np.zeros((1, 1), dtype=np.uint8), grasp_annotator.height_colormap
    )[0, 0]
    image = grasp_annotator.get_annotator_output_struct().image
    assert (image[bottom - 1, left] == coldest_color).all()


def test_a_laundry_mask_bleeding_onto_the_rim_is_grasped_at_its_peak(
    scene: LaundryBasketScene,
    cas: CAS,
    basket: ObjectHypothesis,
    grasp_annotator: LaundryGraspPoseAnnotator,
) -> None:
    """
    Segmentation that spills a little over the laundry's edge onto the higher rim does
    not move the grasp onto the rim.
    """
    left, top, right, bottom = scene.laundry_region().get_corner_points()
    bleeding_region = Rect.from_corner_points(left - 2, top - 2, right + 2, bottom + 2)
    detected_laundry(
        cas,
        bleeding_region,
        np.full(
            (bleeding_region.height, bleeding_region.width),
            np.iinfo(np.uint8).max,
            dtype=np.uint8,
        ),
    )

    update_in_pipeline(grasp_annotator, cas)

    [grasp] = laundry_grasps(basket)
    assert grasp.pixel == scene.peak_pixel


def test_laundry_reaching_onto_the_rim_is_not_grasped_there(
    scene: LaundryBasketScene,
    cas: CAS,
    basket: ObjectHypothesis,
    grasp_annotator: LaundryGraspPoseAnnotator,
) -> None:
    """
    Laundry does not hang over the rim, so laundry segmented onto the rim, as high as
    the rim and above the laundry's peak inside, is not grasped there.
    """
    left, top, right, bottom = scene.laundry_region().get_corner_points()
    basket_right = scene.basket_region.get_corner_points()[2]
    hanging_region = Rect.from_corner_points(left, top, basket_right, bottom)
    detected_laundry(
        cas,
        hanging_region,
        np.full(
            (hanging_region.height, hanging_region.width),
            np.iinfo(np.uint8).max,
            dtype=np.uint8,
        ),
    )

    update_in_pipeline(grasp_annotator, cas)

    [grasp] = laundry_grasps(basket)
    assert grasp.pixel.x < right
    assert cas.depth_image[grasp.pixel.y, grasp.pixel.x] != scene.rim_depth


def test_a_basket_without_detected_laundry_offers_no_grasp(
    cas: CAS, basket: ObjectHypothesis, grasp_annotator: LaundryGraspPoseAnnotator
) -> None:
    """
    Only detected laundry is grasped, never the basket itself.
    """
    update_in_pipeline(grasp_annotator, cas)

    assert laundry_grasps(basket) == []


def test_laundry_outside_the_basket_is_not_grasped_from_it(
    scene: LaundryBasketScene,
    cas: CAS,
    basket: ObjectHypothesis,
    grasp_annotator: LaundryGraspPoseAnnotator,
) -> None:
    """
    Laundry lying beside the basket does not belong to it.
    """
    image_size = scene.image_size
    floor_region = Rect.from_corner_points(
        image_size - 40, image_size - 40, image_size - 10, image_size - 10
    )
    detected_laundry(
        cas,
        floor_region,
        np.full(
            (floor_region.height, floor_region.width),
            np.iinfo(np.uint8).max,
            dtype=np.uint8,
        ),
    )

    update_in_pipeline(grasp_annotator, cas)

    assert laundry_grasps(basket) == []


@pytest.mark.usefixtures("laundry")
@pytest.mark.parametrize("classname", list(LaundryBasketClassName))
def test_a_basket_detected_by_any_of_its_class_names_offers_a_laundry_grasp(
    cas: CAS,
    basket: ObjectHypothesis,
    classname: LaundryBasketClassName,
    grasp_annotator: LaundryGraspPoseAnnotator,
) -> None:
    """
    A laundry basket is searched for laundry whichever of its class names detected it.
    """
    basket.annotations = [Classification(classname=classname, confidence=0.9)]

    update_in_pipeline(grasp_annotator, cas)

    assert len(laundry_grasps(basket)) == 1


@pytest.mark.usefixtures("basket")
@pytest.mark.parametrize("classname", list(LaundryClassName))
def test_laundry_detected_by_any_of_its_class_names_is_grasped(
    scene: LaundryBasketScene,
    cas: CAS,
    basket: ObjectHypothesis,
    laundry: ObjectHypothesis,
    classname: LaundryClassName,
    grasp_annotator: LaundryGraspPoseAnnotator,
) -> None:
    """
    The laundry in a basket is grasped whichever of its class names detected it.
    """
    laundry.annotations = [Classification(classname=classname, confidence=0.5)]

    update_in_pipeline(grasp_annotator, cas)

    [grasp] = laundry_grasps(basket)
    assert grasp.pixel == scene.peak_pixel


@pytest.mark.usefixtures("laundry")
def test_the_laundry_is_approached_from_above_with_fingers_across_the_view(
    cas: CAS, basket: ObjectHypothesis, grasp_annotator: LaundryGraspPoseAnnotator
) -> None:
    """
    The gripper comes straight down and closes along the image's horizontal axis.
    """
    update_in_pipeline(grasp_annotator, cas)

    [grasp] = laundry_grasps(basket)
    rotation = grasp.grasp_pose.to_np()[:3, :3]
    camera_x_axis = cas.camera_observation.world_T_camera.to_np()[:3, 0]
    assert np.allclose(rotation[:, 0], [0.0, 0.0, -1.0])
    assert np.allclose(rotation[:, 1], camera_x_axis)


@pytest.mark.usefixtures("laundry")
@pytest.mark.parametrize(
    "across_ridge_axis", [0, 1], ids=["ridge_down_columns", "ridge_along_rows"]
)
def test_the_fingers_close_across_a_ridge_of_the_laundry(
    scene: LaundryBasketScene,
    cas: CAS,
    basket: ObjectHypothesis,
    across_ridge_axis: int,
    grasp_annotator: LaundryGraspPoseAnnotator,
) -> None:
    """
    Where the laundry forms a ridge at its highest point, the gripper still comes
    straight down, but its fingers close across the ridge so that they pinch it.
    """
    rows, columns = np.mgrid[: scene.image_size, : scene.image_size]
    column_offsets = np.abs(columns - scene.peak_pixel.x)
    row_offsets = np.abs(rows - scene.peak_pixel.y)
    distance_from_ridge, distance_along_ridge = (
        (column_offsets, row_offsets)
        if across_ridge_axis == 0
        else (row_offsets, column_offsets)
    )
    ridge_half_width, ridge_height, ridge_length = 8, 40, 100
    ridge = np.clip(1.0 - distance_from_ridge / ridge_half_width, 0.0, None) * (
        1.0 - distance_along_ridge / ridge_length
    )
    region = scene.laundry_region()
    left, top, right, bottom = region.get_corner_points()
    cas.depth_image[top:bottom, left:right] = np.round(
        scene.laundry_depth - ridge_height * ridge[top:bottom, left:right]
    ).astype(np.uint16)

    update_in_pipeline(grasp_annotator, cas)

    [grasp] = laundry_grasps(basket)
    rotation = grasp.grasp_pose.to_np()[:3, :3]
    camera_axis_across_ridge = cas.camera_observation.world_T_camera.to_np()[
        :3, across_ridge_axis
    ]
    assert grasp.pixel == scene.peak_pixel
    assert np.allclose(rotation[:, 0], [0.0, 0.0, -1.0])
    assert np.isclose(abs(rotation[:, 1] @ camera_axis_across_ridge), 1.0, atol=1e-3)


@pytest.mark.usefixtures("laundry")
def test_no_grasp_is_found_where_the_camera_measured_no_depth_in_the_basket(
    scene: LaundryBasketScene,
    cas: CAS,
    basket: ObjectHypothesis,
    grasp_annotator: LaundryGraspPoseAnnotator,
) -> None:
    """
    Laundry the camera did not measure offers no grasp.
    """
    left, top, right, bottom = scene.basket_region.get_corner_points()
    cas.depth_image[top:bottom, left:right] = 0

    update_in_pipeline(grasp_annotator, cas)

    assert laundry_grasps(basket) == []


@pytest.mark.usefixtures("laundry")
def test_an_object_that_is_no_laundry_basket_offers_no_laundry_grasp(
    cas: CAS, basket: ObjectHypothesis, grasp_annotator: LaundryGraspPoseAnnotator
) -> None:
    """
    Only laundry baskets are searched for laundry.
    """
    basket.annotations = [Classification(classname="chair", confidence=0.9)]

    update_in_pipeline(grasp_annotator, cas)

    assert laundry_grasps(basket) == []


# %% showing the grasp


def test_the_grasp_marker_is_red() -> None:
    """
    The grasp is marked in pure red, in the image and in the point cloud.
    """
    assert np.allclose(
        LaundryGraspPoseAnnotator.grasp_marker_color.to_rgb(), (1.0, 0.0, 0.0)
    )


@pytest.mark.usefixtures("laundry")
def test_the_grasp_is_shown_as_a_dot_in_the_image(
    scene: LaundryBasketScene,
    cas: CAS,
    basket: ObjectHypothesis,
    grasp_annotator: LaundryGraspPoseAnnotator,
) -> None:
    """
    The pixel of the grasp is drawn in the grasp marker's color.
    """
    update_in_pipeline(grasp_annotator, cas)

    red, green, blue = grasp_annotator.grasp_marker_color.to_rgb()
    marker_bgr = np.round(np.array([blue, green, red]) * np.iinfo(np.uint8).max)
    image = grasp_annotator.get_annotator_output_struct().image
    assert (image[scene.peak_pixel.y, scene.peak_pixel.x] == marker_bgr).all()


@pytest.mark.usefixtures("laundry")
def test_the_grasp_is_shown_as_an_arrow_in_the_point_cloud(
    cas: CAS,
    basket: ObjectHypothesis,
    grasp_annotator: LaundryGraspPoseAnnotator,
) -> None:
    """
    An arrow in the grasp marker's color points along the grasp's approach, its tip
    where the grasp takes hold.
    """
    cas.set(CASViews.CLOUD, open3d.geometry.PointCloud())

    update_in_pipeline(grasp_annotator, cas)

    [grasp] = laundry_grasps(basket)
    [arrow] = [
        shown["geometry"]
        for shown in grasp_annotator.get_annotator_output_struct().geometries
        if isinstance(shown["geometry"], open3d.geometry.TriangleMesh)
    ]
    camera_T_grasp = (
        np.linalg.inv(cas.camera_observation.world_T_camera.to_np())
        @ grasp.grasp_pose.to_np()
    )
    approach, grasp_point = camera_T_grasp[:3, 0], camera_T_grasp[:3, 3]
    vertices = np.asarray(arrow.vertices)
    tip = vertices[np.argmax(vertices @ approach)]
    axis = tip - vertices.mean(axis=0)
    assert np.allclose(tip, grasp_point)
    assert np.allclose(axis / np.linalg.norm(axis), approach)
    assert np.allclose(
        np.asarray(arrow.vertex_colors), grasp_annotator.grasp_marker_color.to_rgb()
    )


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


@pytest.mark.usefixtures("laundry")
def test_the_laundry_basket_records_the_grasp_perception_found(
    cas: CAS,
    basket: ObjectHypothesis,
    laundry_basket: LaundryBasket,
    grasp_annotator: LaundryGraspPoseAnnotator,
) -> None:
    """
    The grasp on the laundry is recorded on the laundry basket.
    """
    update_in_pipeline(grasp_annotator, cas)
    [grasp] = laundry_grasps(basket)

    update_in_pipeline(LaundryGraspConnector(), cas)

    assert laundry_basket.laundry_grasp_pose is grasp.grasp_pose


@pytest.mark.usefixtures("laundry")
def test_a_frame_without_a_grasp_keeps_the_grasp_found_before(
    cas: CAS,
    basket: ObjectHypothesis,
    laundry_basket: LaundryBasket,
    grasp_annotator: LaundryGraspPoseAnnotator,
) -> None:
    """
    Between queries the basket is only tracked, so the last grasp found stays recorded.
    """
    update_in_pipeline(grasp_annotator, cas)
    [grasp] = laundry_grasps(basket)
    update_in_pipeline(LaundryGraspConnector(), cas)
    basket.annotations.remove(grasp)

    update_in_pipeline(LaundryGraspConnector(), cas)

    assert laundry_basket.laundry_grasp_pose is grasp.grasp_pose


@pytest.mark.usefixtures("laundry")
def test_a_grasp_on_a_basket_not_annotated_as_laundry_basket_is_refused(
    cas: CAS, basket: ObjectHypothesis, grasp_annotator: LaundryGraspPoseAnnotator
) -> None:
    """
    A grasp has nowhere to be recorded when the basket's body is no laundry basket.
    """
    robokudo.world.add_object_hypothesis_as_belief_state(basket, cas)
    update_in_pipeline(grasp_annotator, cas)

    with pytest.raises(LaundryBasketNotAnnotated):
        update_in_pipeline(LaundryGraspConnector(), cas)
