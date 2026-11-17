"""
Annotates the bodies the world connector made of detected objects with the semantic
annotation their detected class stands for.
"""

from __future__ import annotations

from py_trees.common import Status

from robokudo import world
from robokudo.annotators.core import BaseAnnotator
from robokudo.cas import CAS
from robokudo.types.annotation import Classification
from robokudo.types.scene import ObjectHypothesis
from semantic_digital_twin.semantic_annotations.mixins import (
    HasRootBody,
    IsPerceivable,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import Body

# %% annotator


class SemanticAnnotationConnector(BaseAnnotator):
    """
    Annotates the body of each believed object whose detected class has a semantic
    annotation type, once; a perceivable annotation records the class as its label.

    Runs after :class:`~robokudo.annotators.semantic_world_connector.SemanticDigitalTwinConnector`.
    """

    class Descriptor(BaseAnnotator.Descriptor):
        """
        Descriptor for the SemanticAnnotationConnector.
        """

        class Parameters:
            """
            Parameters for the SemanticAnnotationConnector.
            """

            def __init__(self) -> None:
                self.semantic_annotation_types: dict[str, type[HasRootBody]] = {}
                """
                The semantic annotation type each detected class name stands for.
                """

        parameters = Parameters()

    def __init__(
        self,
        name: str = "SemanticAnnotationConnector",
        descriptor: SemanticAnnotationConnector.Descriptor | None = None,
    ) -> None:
        super().__init__(name, descriptor)

    def update(self) -> Status:
        """
        Annotate the body of each believed object whose class has an annotation type.
        """
        perception_world = world.world_instance()
        semantic_annotation_types = self.descriptor.parameters.semantic_annotation_types
        for hypothesis in self.get_cas().filter_annotations_by_type(ObjectHypothesis):
            body = world.find_believed_body(hypothesis)
            if body is None:
                continue
            for classification in CAS.filter_by_type(
                Classification, hypothesis.annotations
            ):
                if classification.classname not in semantic_annotation_types:
                    continue
                self.add_missing_semantic_annotation(
                    body=body,
                    semantic_annotation_type=semantic_annotation_types[
                        classification.classname
                    ],
                    class_label=classification.classname,
                    perception_world=perception_world,
                )
        return Status.SUCCESS

    @staticmethod
    def add_missing_semantic_annotation(
        *,
        body: Body,
        semantic_annotation_type: type[HasRootBody],
        class_label: str,
        perception_world: World,
    ) -> None:
        """
        Annotate the body with the type, unless it already has an annotation of it.

        :param body: The body the world connector made of a detected object.
        :param semantic_annotation_type: What the detected object is.
        :param class_label: The class the object was detected as.
        :param perception_world: RoboKudo's world.
        """
        if world.find_semantic_annotation(body, semantic_annotation_type) is not None:
            return
        if issubclass(semantic_annotation_type, IsPerceivable):
            semantic_annotation = semantic_annotation_type(
                root=body, class_label=class_label
            )
        else:
            semantic_annotation = semantic_annotation_type(root=body)
        with perception_world.modify_world():
            perception_world.add_semantic_annotation(semantic_annotation)
