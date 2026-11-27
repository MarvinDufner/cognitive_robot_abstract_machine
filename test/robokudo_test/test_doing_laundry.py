"""
Tests for the doing-laundry analysis engine reading every frame for all its subtrees and
routing each of its queries to the subtree that answers it.
"""

from __future__ import annotations

# %% Imports
import copy
from unittest.mock import MagicMock

import numpy as np
import open3d
import pytest
from rclpy.node import Node
from robokudo_msgs.action import Query

from robokudo.annotators.cluster_pose_bb import ClusterPoseBBAnnotator
from robokudo.annotators.collection_reader import CollectionReaderAnnotator
from robokudo.annotators.core import BaseAnnotator
from robokudo.annotators.hypothesis_point_cloud import HypothesisPointCloudAnnotator
from robokudo.annotators.image_preprocessor import ImagePreprocessorAnnotator
from robokudo.annotators.laundry_grasp_pose import LaundryGraspPoseAnnotator
from robokudo.annotators.laundry_grasp_publisher import LaundryGraspPublisher
from robokudo.annotators.query import GenerateQueryResult
from robokudo.annotators.steady_laundry_grasp import SteadyLaundryGraspAnnotator
from robokudo.cas import CAS
from robokudo.annotators.open_vocabulary_object_detection_annotator import (
    OpenVocabularyObjectDetectionAnnotator,
    Owlv2ForObjectDetection,
    Owlv2Processor,
)
from robokudo.behaviours.query_stream import ContinueQueryStream, StartQueryStream
from robokudo.descriptors.analysis_engines.doing_laundry import (
    AnalysisEngine,
    DoingLaundryQuery,
)
from robokudo.pipeline import Pipeline
from robokudo.tree_components.query_based_task_scheduler import QueryBasedScheduler
from robokudo.types.annotation import PoseAnnotation
from robokudo.types.laundry import LaundryBasketClassName
from robokudo.types.scene import ObjectHypothesis
from semantic_digital_twin.robots.hsrb import HSRBGripper

from .test_cloth_unfolding import update_in_pipeline
from .test_laundry_pickup import basket, cas, scene  # noqa: F401 - shared fixtures

# %% Fixtures


@pytest.fixture
def pipeline(node: Node, monkeypatch: pytest.MonkeyPatch) -> Pipeline:
    """
    :return: The pipeline of the engine, built without loading the detection and
        segmentation models.
    """
    monkeypatch.setattr(Owlv2Processor, "from_pretrained", MagicMock())
    monkeypatch.setattr(Owlv2ForObjectDetection, "from_pretrained", MagicMock())
    monkeypatch.setattr(
        "robokudo.annotators.open_vocabulary_object_detection_annotator.SegmentAnythingModel",
        MagicMock(),
    )
    return AnalysisEngine().implementation()


@pytest.fixture
def scheduler(pipeline: Pipeline) -> QueryBasedScheduler:
    """
    :return: The query scheduler of the engine.
    """
    (scheduler,) = [
        behaviour
        for behaviour in pipeline.iterate()
        if isinstance(behaviour, QueryBasedScheduler)
    ]
    return scheduler


# %% Reading the camera


def test_the_pipeline_reads_and_preprocesses_every_frame_for_all_subtrees(
    pipeline: Pipeline, scheduler: QueryBasedScheduler
) -> None:
    """
    Each pipeline iteration reads one frame, so that its outputs are shown once the
    subtree of the query is done with it.
    """
    pipeline_child_types = {type(child) for child in pipeline.children}
    subtree_types = {
        type(behaviour)
        for subtree in scheduler.tasks.values()
        for behaviour in subtree.iterate()
    }

    assert {CollectionReaderAnnotator, ImagePreprocessorAnnotator} <= (
        pipeline_child_types
    )
    assert not {CollectionReaderAnnotator, ImagePreprocessorAnnotator} & subtree_types


# %% Routing


def test_a_tshirt_pile_query_streams_the_grasp(
    pipeline: Pipeline, scheduler: QueryBasedScheduler
) -> None:
    """
    The T-shirt pile subtree finds the grasp, holds it steady across the stream and
    publishes it, starting the stream the pipeline continues with in its next iteration.
    """
    route = scheduler.filter_fn(Query.Goal(query=DoingLaundryQuery.TSHIRT_PILE))

    subtree_behaviours = list(scheduler.tasks[route].iterate())
    (start,) = [
        behaviour
        for behaviour in subtree_behaviours
        if isinstance(behaviour, StartQueryStream)
    ]
    (continuation,) = [
        behaviour
        for behaviour in pipeline.iterate()
        if isinstance(behaviour, ContinueQueryStream)
    ]
    (steady,) = [
        behaviour
        for behaviour in subtree_behaviours
        if isinstance(behaviour, SteadyLaundryGraspAnnotator)
    ]
    assert {LaundryGraspPoseAnnotator, LaundryGraspPublisher} <= {
        type(behaviour) for behaviour in subtree_behaviours
    }
    assert start.query_stream is continuation.query_stream
    assert steady.query_stream is continuation.query_stream


def test_the_tshirt_pile_grasp_keeps_the_hsrb_hand_clear_of_the_walls(
    scheduler: QueryBasedScheduler,
) -> None:
    """
    The grasp keeps the open fingers of the robot in use, the HSRB, clear of the
    basket's walls.
    """
    route = scheduler.filter_fn(Query.Goal(query=DoingLaundryQuery.TSHIRT_PILE))

    (grasp_annotator,) = [
        behaviour
        for behaviour in scheduler.tasks[route].iterate()
        if isinstance(behaviour, LaundryGraspPoseAnnotator)
    ]
    assert (
        grasp_annotator.descriptor.parameters.gripper_maximum_opening
        == HSRBGripper.maximum_opening_width
    )


def test_a_find_basket_query_answers_once_with_the_pose_of_the_basket(
    scheduler: QueryBasedScheduler,
) -> None:
    """
    The find-basket subtree detects only laundry baskets, estimates their pose and
    answers the query with it, without streaming.
    """
    route = scheduler.filter_fn(Query.Goal(query=DoingLaundryQuery.FIND_BASKET))

    subtree_behaviours = list(scheduler.tasks[route].iterate())
    (detection,) = [
        behaviour
        for behaviour in subtree_behaviours
        if isinstance(behaviour, OpenVocabularyObjectDetectionAnnotator)
    ]
    subtree_types = {type(behaviour) for behaviour in subtree_behaviours}
    assert detection.descriptor.parameters.classes == list(LaundryBasketClassName)
    assert {ClusterPoseBBAnnotator, GenerateQueryResult} <= subtree_types
    assert StartQueryStream not in subtree_types


@pytest.fixture
def basket_pose_annotators(scheduler: QueryBasedScheduler) -> list[BaseAnnotator]:
    """
    :return: The annotators of the find-basket subtree between its detector and its
        answer, taken out of the subtree.
    """
    route = scheduler.filter_fn(Query.Goal(query=DoingLaundryQuery.FIND_BASKET))
    subtree = scheduler.tasks[route]
    annotators = [
        behaviour
        for behaviour in subtree.children
        if not isinstance(
            behaviour, (OpenVocabularyObjectDetectionAnnotator, GenerateQueryResult)
        )
    ]
    for annotator in annotators:
        subtree.remove_child(annotator)
    return annotators


def estimate_basket_pose(
    annotators: list[BaseAnnotator], cas: CAS, smear_depth_behind_the_rim: bool = False
) -> PoseAnnotation:
    """
    Run the annotators of the find-basket subtree on a CAS holding the detected basket,
    as the engine runs them on a preprocessed frame.

    :param smear_depth_behind_the_rim: Whether to add points behind the basket's rim to
        its point cloud once it was built, as a depth camera measures them where the rim
        meets the floor far behind it.
    :return: The pose the annotators estimate for the basket.
    """
    update_in_pipeline(ImagePreprocessorAnnotator("ImagePreprocessor"), cas)
    (basket,) = cas.filter_annotations_by_type(ObjectHypothesis)
    for annotator in annotators:
        update_in_pipeline(annotator, cas)
        annotator.parent.remove_child(annotator)
        if smear_depth_behind_the_rim and isinstance(
            annotator, HypothesisPointCloudAnnotator
        ):
            points = np.asarray(basket.points.points)
            rim = points[np.argsort(points[:, 2])[:50]]
            behind_the_rim = rim + np.outer(np.linspace(0.1, 0.5, len(rim)), [0, 0, 1])
            basket.points.points.extend(open3d.utility.Vector3dVector(behind_the_rim))
    (pose,) = CAS.filter_by_type(PoseAnnotation, basket.annotations)
    return pose


@pytest.mark.usefixtures("annotator_output_buffer", "basket")
def test_a_basket_the_find_basket_subtree_only_segmented_gets_a_pose(
    basket_pose_annotators: list[BaseAnnotator], cas: CAS
) -> None:
    """
    The detector only segments the basket, so the find-basket subtree has to find the
    points of the basket itself before it can estimate the pose it answers with.
    """
    assert isinstance(estimate_basket_pose(basket_pose_annotators, cas), PoseAnnotation)


@pytest.mark.usefixtures("annotator_output_buffer", "basket")
def test_depth_smeared_behind_the_rim_does_not_move_the_basket(
    basket_pose_annotators: list[BaseAnnotator], cas: CAS
) -> None:
    """
    The points a depth camera measures between the basket's rim and the floor far behind
    it do not stretch the basket away from the camera.
    """
    expected = estimate_basket_pose(basket_pose_annotators, copy.deepcopy(cas))

    pose = estimate_basket_pose(
        basket_pose_annotators, cas, smear_depth_behind_the_rim=True
    )

    assert pose.translation == pytest.approx(expected.translation, abs=0.01)


def test_every_query_of_the_challenge_has_a_subtree(
    scheduler: QueryBasedScheduler,
) -> None:
    """
    Each query name routes to a subtree of its own.
    """
    routes = {
        scheduler.filter_fn(Query.Goal(query=query_name))
        for query_name in DoingLaundryQuery
    }

    assert routes == set(scheduler.tasks)


def test_an_unknown_query_has_no_subtree(scheduler: QueryBasedScheduler) -> None:
    """
    A query the challenge does not know is left without a subtree, which aborts it.
    """
    route = scheduler.filter_fn(Query.Goal(query="dance"))

    assert route not in scheduler.tasks
