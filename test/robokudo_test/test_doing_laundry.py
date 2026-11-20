"""
Tests for the doing-laundry analysis engine reading every frame for all its subtrees and
routing each of its queries to the subtree that answers it.
"""

from __future__ import annotations

# %% Imports
from unittest.mock import MagicMock

import pytest
from rclpy.node import Node
from robokudo_msgs.action import Query

from robokudo.annotators.collection_reader import CollectionReaderAnnotator
from robokudo.annotators.image_preprocessor import ImagePreprocessorAnnotator
from robokudo.annotators.laundry_grasp_pose import LaundryGraspPoseAnnotator
from robokudo.annotators.laundry_grasp_publisher import LaundryGraspPublisher
from robokudo.annotators.steady_laundry_grasp import SteadyLaundryGraspAnnotator
from robokudo.annotators.open_vocabulary_object_detection_annotator import (
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
from semantic_digital_twin.robots.hsrb import HSRBGripper

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
