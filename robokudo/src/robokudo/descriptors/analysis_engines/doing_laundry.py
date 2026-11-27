"""
Analysis engine for the doing-laundry challenge, answering each of its queries with the
subtree for it, from the HSRB's head camera.
"""

from __future__ import annotations

# %% Imports
from enum import StrEnum

from py_trees.composites import Selector, Sequence

from robokudo.analysis_engine import AnalysisEngineInterface
from robokudo.annotators.collection_reader import CollectionReaderAnnotator
from robokudo.annotators.image_preprocessor import ImagePreprocessorAnnotator
from robokudo.annotators.query import QueryAnnotator
from robokudo.behaviours.query_stream import ContinueQueryStream, QueryStream
from robokudo.descriptors.analysis_engines.subtree_find_basket import (
    Subtree as FindBasketSubtree,
)
from robokudo.descriptors.analysis_engines.subtree_tshirt_pile import (
    Subtree as TShirtPileSubtree,
)
from robokudo.descriptors.factories.cr_descriptor_factory import (
    CollectionReaderDescriptorFactory,
)
from robokudo.idioms import pipeline_init
from robokudo.pipeline import Pipeline
from robokudo.tree_components.query_based_task_scheduler import QueryBasedScheduler
from semantic_digital_twin.robots.hsrb import HSRBGripper

# %% queries


class DoingLaundryQuery(StrEnum):
    """
    The names of the queries of the doing-laundry challenge.
    """

    TSHIRT_PILE = "tshirt-pile"
    """
    Publish the grasp from above on the highest point of the T-shirt pile in a laundry
    basket, until the next query arrives or the query is canceled.
    """

    FIND_BASKET = "find-basket"
    """
    Answer once with the pose of each laundry basket in view.
    """


# %% analysis engine


class AnalysisEngine(AnalysisEngineInterface):
    """
    The perception of the doing-laundry challenge: every iteration reads one frame and
    runs the subtree named by the current query's name on it.
    """

    def name(self) -> str:
        """
        :return: ``doing_laundry``, the name the engine is started by.
        """
        return "doing_laundry"

    def implementation(self) -> Pipeline:
        """
        :return: A pipeline that takes up the next query, or keeps the current one while
            it streams, reads and preprocesses a frame and runs the query's subtree on
            it, aborting a query whose name it does not know.
        """
        query_stream = QueryStream()

        next_query = Sequence("Next Query", memory=True)
        next_query.add_children([pipeline_init(), QueryAnnotator()])
        current_query = Selector("Current Query", memory=True)
        current_query.add_children(
            [ContinueQueryStream(query_stream=query_stream), next_query]
        )

        task_scheduling = Sequence("Task Scheduling", memory=True)
        task_scheduling.add_child(
            QueryBasedScheduler(
                tasks={
                    DoingLaundryQuery.TSHIRT_PILE: TShirtPileSubtree(
                        query_stream=query_stream,
                        gripper_maximum_opening=HSRBGripper.maximum_opening_width,
                    ).implementation(),
                    DoingLaundryQuery.FIND_BASKET: FindBasketSubtree().implementation(),
                },
                filter_fn=lambda query: query.query,
            )
        )

        pipeline = Pipeline("DoingLaundryPipeline")
        pipeline.add_children(
            [
                current_query,
                CollectionReaderAnnotator(
                    descriptor=CollectionReaderDescriptorFactory.create_descriptor(
                        "hsr_ros2"
                    )
                ),
                ImagePreprocessorAnnotator("ImagePreprocessor"),
                task_scheduling,
            ]
        )
        return pipeline
