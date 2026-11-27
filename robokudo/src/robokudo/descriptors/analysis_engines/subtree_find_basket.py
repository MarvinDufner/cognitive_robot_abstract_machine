"""
Subtree that finds the laundry baskets in view and answers its query with their poses.
"""

from __future__ import annotations

# %% Imports
from dataclasses import dataclass

from py_trees.composites import Sequence

from robokudo.analysis_engine import SubtreeInterface
from robokudo.annotators.cluster_pose_bb import ClusterPoseBBAnnotator
from robokudo.annotators.open_vocabulary_object_detection_annotator import (
    OpenVocabularyObjectDetectionAnnotator,
)
from robokudo.annotators.query import GenerateQueryResult
from robokudo.types.laundry import LaundryBasketClassName

# %% subtree


@dataclass(kw_only=True)
class Subtree(SubtreeInterface):
    """
    In a single frame, detects the laundry baskets, estimates the pose of each from its
    points and answers the query with them.
    """

    def name(self) -> str:
        """
        :return: ``find_basket``, the name of the subtree.
        """
        return "find_basket"

    def implementation(self) -> Sequence:
        """
        :return: The sequence that detects the baskets, estimates their poses and
            answers the query.
        """
        detection_descriptor = OpenVocabularyObjectDetectionAnnotator.Descriptor()
        detection_descriptor.parameters.classes = list(LaundryBasketClassName)
        detection_descriptor.parameters.precision_mode = True

        sequence = Sequence("Find Basket", memory=True)
        sequence.add_children(
            [
                OpenVocabularyObjectDetectionAnnotator(
                    name="BasketDetection", descriptor=detection_descriptor
                ),
                ClusterPoseBBAnnotator(),
                GenerateQueryResult(),
            ]
        )
        return sequence
