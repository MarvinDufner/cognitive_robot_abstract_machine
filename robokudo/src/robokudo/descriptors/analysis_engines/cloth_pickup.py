"""
Analysis engine perceiving where to pick laundry up from a laundry basket, from the
HSRB's head camera.
"""

from __future__ import annotations

from py_trees.composites import Sequence
from py_trees.decorators import FailureIsSuccess, RunningIsFailure

from robokudo.analysis_engine import AnalysisEngineInterface
from robokudo.annotators.cloth_height_coloring import ClothHeightColoringAnnotator
from robokudo.annotators.cluster_pose_bb import ClusterPoseBBAnnotator
from robokudo.annotators.collection_reader import CollectionReaderAnnotator
from robokudo.annotators.hypothesis_point_cloud import HypothesisPointCloudAnnotator
from robokudo.annotators.image_preprocessor import ImagePreprocessorAnnotator
from robokudo.annotators.laundry_grasp_connector import LaundryGraspConnector
from robokudo.annotators.laundry_grasp_pose import LaundryGraspPoseAnnotator
from robokudo.annotators.plane import PlaneAnnotator
from robokudo.annotators.open_vocabulary_object_detection_annotator import (
    OpenVocabularyObjectDetectionAnnotator,
)
from robokudo.annotators.query import GenerateQueryResult, QueryAnnotator
from robokudo.annotators.scene_freeze import (
    FreezeSceneOnRequest,
    HoldFrozenScene,
    SceneFreeze,
)
from robokudo.annotators.semantic_annotation_connector import (
    SemanticAnnotationConnector,
)
from robokudo.annotators.semantic_world_connector import SemanticDigitalTwinConnector
from robokudo.descriptors.factories.cr_descriptor_factory import (
    CollectionReaderDescriptorFactory,
)
from robokudo.idioms import pipeline_init
from robokudo.pipeline import Pipeline
from robokudo.types.laundry import LaundryBasketClassName, LaundryClassName
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Cloth,
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
        :return: A pipeline that tracks the baskets every frame and, on a query, records
            a top-down grasp on the point of each basket's laundry highest above the
            floor and answers;
            after a query named ``freeze`` it holds that scene until the next query.
        """
        detection_descriptor = OpenVocabularyObjectDetectionAnnotator.Descriptor()
        detection_descriptor.parameters.classes = [
            *LaundryBasketClassName,
            *LaundryClassName,
        ]
        detection_descriptor.parameters.precision_mode = True

        annotation_descriptor = SemanticAnnotationConnector.Descriptor()
        annotation_descriptor.parameters.semantic_annotation_types = {
            **{classname: LaundryBasket for classname in LaundryBasketClassName},
            **{classname: Cloth for classname in LaundryClassName},
        }

        scene_freeze = SceneFreeze()

        laundry_grasp_on_query = Sequence("Laundry Grasp On Query", memory=True)
        laundry_grasp_on_query.add_children(
            [
                RunningIsFailure(name="Query Received", child=QueryAnnotator()),
                PlaneAnnotator(name="FloorPlaneDetection"),
                LaundryGraspPoseAnnotator(),
                LaundryGraspConnector(),
                GenerateQueryResult(),
                FreezeSceneOnRequest(scene_freeze=scene_freeze),
            ]
        )

        pipeline = Pipeline("ClothPickupPipeline")
        pipeline.add_children(
            [
                pipeline_init(),
                HoldFrozenScene(scene_freeze=scene_freeze),
                CollectionReaderAnnotator(
                    descriptor=CollectionReaderDescriptorFactory.create_descriptor(
                        "hsr_ros2"
                    )
                ),
                ImagePreprocessorAnnotator("ImagePreprocessor"),
                OpenVocabularyObjectDetectionAnnotator(
                    name="LaundryBasketDetection", descriptor=detection_descriptor
                ),
                ClothHeightColoringAnnotator(),
                ClusterPoseBBAnnotator(),
                SemanticDigitalTwinConnector(),
                SemanticAnnotationConnector(descriptor=annotation_descriptor),
            ]
        )
        return pipeline
