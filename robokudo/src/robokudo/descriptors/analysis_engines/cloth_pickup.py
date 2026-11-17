"""
Analysis engine perceiving where to pick laundry up from a laundry basket, from the
HSRB's head camera.
"""

from __future__ import annotations

from py_trees.composites import Sequence
from py_trees.decorators import FailureIsSuccess, RunningIsFailure

from robokudo.analysis_engine import AnalysisEngineInterface
from robokudo.annotators.cluster_pose_bb import ClusterPoseBBAnnotator
from robokudo.annotators.collection_reader import CollectionReaderAnnotator
from robokudo.annotators.hypothesis_point_cloud import HypothesisPointCloudAnnotator
from robokudo.annotators.image_preprocessor import ImagePreprocessorAnnotator
from robokudo.annotators.laundry_grasp_connector import LaundryGraspConnector
from robokudo.annotators.laundry_grasp_pose import LaundryGraspPoseAnnotator
from robokudo.annotators.open_vocabulary_object_detection_annotator import (
    OpenVocabularyObjectDetectionAnnotator,
)
from robokudo.annotators.query import GenerateQueryResult, QueryAnnotator
from robokudo.annotators.semantic_annotation_connector import (
    SemanticAnnotationConnector,
)
from robokudo.annotators.semantic_world_connector import SemanticDigitalTwinConnector
from robokudo.descriptors.factories.cr_descriptor_factory import (
    CollectionReaderDescriptorFactory,
)
from robokudo.idioms import pipeline_init
from robokudo.pipeline import Pipeline
from robokudo.types.laundry import LaundryBasketClassName
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    LaundryBasket,
)

# %% analysis engine


class AnalysisEngine(AnalysisEngineInterface):
    """
    Perceives where to pick laundry up from the laundry baskets in view, recording each
    basket and the grasp on its laundry as a :class:`LaundryBasket` in the semantic
    digital twin.
    """

    def name(self) -> str:
        """
        :return: ``cloth_pickup``, the name the engine is started by.
        """
        return "cloth_pickup"

    def implementation(self) -> Pipeline:
        """
        :return: A pipeline that tracks the baskets every frame and, only when a query
            arrives, finds the highest point of the laundry in each basket, records a
            top-down grasp on it and answers the query.
        """
        detection_descriptor = OpenVocabularyObjectDetectionAnnotator.Descriptor()
        detection_descriptor.parameters.classes = list(LaundryBasketClassName)
        detection_descriptor.parameters.precision_mode = True

        annotation_descriptor = SemanticAnnotationConnector.Descriptor()
        annotation_descriptor.parameters.semantic_annotation_types = {
            classname: LaundryBasket for classname in LaundryBasketClassName
        }

        laundry_grasp_on_query = Sequence("Laundry Grasp On Query", memory=True)
        laundry_grasp_on_query.add_children(
            [
                RunningIsFailure(name="Query Received", child=QueryAnnotator()),
                LaundryGraspPoseAnnotator(),
                LaundryGraspConnector(),
                GenerateQueryResult(),
            ]
        )

        pipeline = Pipeline("ClothPickupPipeline")
        pipeline.add_children(
            [
                pipeline_init(),
                CollectionReaderAnnotator(
                    descriptor=CollectionReaderDescriptorFactory.create_descriptor(
                        "hsr_ros2"
                    )
                ),
                ImagePreprocessorAnnotator("ImagePreprocessor"),
                OpenVocabularyObjectDetectionAnnotator(
                    name="LaundryBasketDetection", descriptor=detection_descriptor
                ),
                HypothesisPointCloudAnnotator(),
                ClusterPoseBBAnnotator(),
                SemanticDigitalTwinConnector(),
                SemanticAnnotationConnector(descriptor=annotation_descriptor),
                FailureIsSuccess(
                    name="Laundry Grasp If Queried", child=laundry_grasp_on_query
                ),
            ]
        )
        return pipeline
