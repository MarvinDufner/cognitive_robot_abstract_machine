"""
Tests for holding the laundry grasp steady across the frames of a query stream: it
follows a new spot only once the grasp keeps returning there, and forgets the grasp once
the laundry stays without one.
"""

from __future__ import annotations

# %% Imports
import numpy as np
import pytest
from krrood_spatial_math.spatial_types import Pose
from semantic_digital_twin.world import World
from typing_extensions import Callable

from robokudo.annotators.steady_laundry_grasp import SteadyLaundryGraspAnnotator
from robokudo.behaviours.query_stream import QueryStream
from robokudo.cas import CAS
from robokudo.types.annotation import Classification
from robokudo.types.cv import Point2D
from robokudo.types.laundry import (
    LaundryBasketClassName,
    LaundryGraspAnnotation,
    SteadyLaundryGrasp,
    SteadyLaundryGraspAnnotation,
)
from robokudo.types.scene import ObjectHypothesis

from .test_cloth_unfolding import update_in_pipeline

pytestmark = pytest.mark.usefixtures("annotator_output_buffer")

# %% Fixtures


@pytest.fixture
def grasp_at(perception_world: World) -> Callable[..., LaundryGraspAnnotation]:
    """
    :return: A factory of top-down grasps at a given x in the world frame.
    """

    def make(x: float, height: float = 0.2) -> LaundryGraspAnnotation:
        """
        :return: A grasp at the given x and height above the floor.
        """
        return LaundryGraspAnnotation(
            pixel=Point2D(x=0, y=0),
            height_above_floor=height,
            grasp_pose=Pose.from_xyz_rpy(
                x=x, z=height, pitch=np.pi / 2, reference_frame=perception_world.root
            ),
        )

    return make


def follow_all(
    steady: SteadyLaundryGrasp, grasps: list[LaundryGraspAnnotation | None]
) -> LaundryGraspAnnotation | None:
    """
    :return: The steady grasp after following the frames' grasps in order.
    """
    for grasp in grasps:
        steady.follow(grasp=grasp)
    return steady.steady_grasp


def cas_with_grasps(grasps: list[LaundryGraspAnnotation]) -> CAS:
    """
    :return: A CAS with one detected basket per grasp.
    """
    cas = CAS()
    for grasp in grasps:
        basket = ObjectHypothesis()
        basket.annotations.append(
            Classification(
                classname=LaundryBasketClassName.LAUNDRY_BASKET, confidence=0.9
            )
        )
        basket.annotations.append(grasp)
        cas.annotations.append(basket)
    return cas


def steady_grasps_in(cas: CAS) -> list[LaundryGraspAnnotation]:
    """
    :return: The steady grasps the CAS is annotated with.
    """
    return [
        annotation.grasp
        for annotation in CAS.filter_by_type(
            SteadyLaundryGraspAnnotation, cas.annotations
        )
    ]


# %% following the grasp


def test_the_first_grasp_is_held_at_once(grasp_at) -> None:
    """
    The stream publishes from its first grasp on.
    """
    first = grasp_at(0.0)

    assert follow_all(SteadyLaundryGrasp(), [first]) is first


def test_a_grasp_jumping_elsewhere_for_one_frame_is_not_followed(grasp_at) -> None:
    """
    A single frame finding another top does not move the grasp.
    """
    held = grasp_at(0.0)

    steady = follow_all(SteadyLaundryGrasp(), [held, grasp_at(0.2), grasp_at(0.0)])

    assert steady.position.tolist() == held.position.tolist()


def test_a_grasp_staying_elsewhere_is_followed(grasp_at) -> None:
    """
    Once the grasp keeps returning to another spot for the confirmation frames, it moves
    there.
    """
    steady = SteadyLaundryGrasp()
    elsewhere = [grasp_at(0.2) for _ in range(steady.confirmation_frames)]

    moved = follow_all(steady, [grasp_at(0.0), *elsewhere])

    assert moved.position.tolist() == elsewhere[-1].position.tolist()


def test_the_steady_grasp_is_the_most_central_of_the_recent_grasps_at_its_spot(
    grasp_at,
) -> None:
    """
    Grasps wandering within the spot are evened out by taking the one in the middle.
    """
    steady = SteadyLaundryGrasp()
    near = [grasp_at(0.0), grasp_at(steady.stay_radius / 2), grasp_at(0.002)]

    assert follow_all(steady, near) is near[2]


def test_a_few_frames_without_a_grasp_keep_it(grasp_at) -> None:
    """
    The grasp outlasts frames in which no laundry was found.
    """
    steady = SteadyLaundryGrasp()
    held = grasp_at(0.0)

    kept = follow_all(steady, [held, *[None] * (steady.confirmation_frames - 1)])

    assert kept is held


def test_staying_without_a_grasp_forgets_it(grasp_at) -> None:
    """
    Once no grasp was found for the confirmation frames, none is held, so that a grasp
    does not outlive the laundry.
    """
    steady = SteadyLaundryGrasp()

    forgotten = follow_all(
        steady, [grasp_at(0.0), *[None] * steady.confirmation_frames]
    )

    assert forgotten is None


# %% the annotator


def test_the_highest_grasp_of_the_frame_is_held(grasp_at) -> None:
    """
    Of several baskets, the grasp on the laundry lying highest is held.
    """
    highest = grasp_at(0.5, height=0.4)
    cas = cas_with_grasps([grasp_at(0.0), highest])

    update_in_pipeline(SteadyLaundryGraspAnnotator(query_stream=QueryStream()), cas)

    assert steady_grasps_in(cas) == [highest]


def test_a_new_stream_forgets_the_grasp_of_the_previous_one(grasp_at) -> None:
    """
    A grasp held in one stream does not hold back the grasp of the next.
    """
    query_stream = QueryStream(started_streams=1)
    annotator = SteadyLaundryGraspAnnotator(query_stream=query_stream)
    update_in_pipeline(annotator, cas_with_grasps([grasp_at(0.0)]))
    query_stream.started_streams += 1
    new_grasp = grasp_at(0.3)
    cas = cas_with_grasps([new_grasp])
    annotator.get_parent_pipeline().cas = cas

    annotator.update()

    assert steady_grasps_in(cas) == [new_grasp]
