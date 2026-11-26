"""
Subtree that tracks the pile of T-shirts in a laundry basket and publishes where to
grasp it from above.
"""

from __future__ import annotations

# %% Imports
from dataclasses import dataclass

from py_trees.composites import Sequence

from robokudo.analysis_engine import SubtreeInterface
from robokudo.annotators.laundry_grasp_pose import LaundryGraspPoseAnnotator
from robokudo.annotators.laundry_grasp_publisher import LaundryGraspPublisher
from robokudo.annotators.open_vocabulary_object_detection_annotator import (
    OpenVocabularyObjectDetectionAnnotator,
)
from robokudo.annotators.plane import PlaneAnnotator
from robokudo.annotators.steady_laundry_grasp import SteadyLaundryGraspAnnotator
from robokudo.behaviours.query_stream import QueryStream, StartQueryStream
from robokudo.types.laundry import LaundryBasketClassName, LaundryClassName

# %% subtree


@dataclass(kw_only=True)
class Subtree(SubtreeInterface):
    """
    Frame after frame of its query, publishes a top-down grasp in the world frame on the
    highest top of the laundry in a detected basket, held steady across the frames,
    leaving out the basket's depth points and keeping the fingers clear of its walls.
    """

    query_stream: QueryStream
    """
    The stream the pipeline continues the query with in its next iteration.
    """

    gripper_maximum_opening: float
    """
    The widest the gripper of the robot in use opens, in metres.
    """

    def name(self) -> str:
        """
        :return: ``tshirt_pile``, the name of the subtree.
        """
        return "tshirt_pile"

    def implementation(self) -> Sequence:
        """
        :return: The sequence that starts the stream, detects the baskets and their
            laundry, and finds, steadies and publishes the grasp in the current frame.
        """
        detection_descriptor = OpenVocabularyObjectDetectionAnnotator.Descriptor()
        detection_descriptor.parameters.classes = [
            *LaundryBasketClassName,
            *LaundryClassName,
        ]
        detection_descriptor.parameters.precision_mode = True

        grasp_descriptor = LaundryGraspPoseAnnotator.Descriptor()
        grasp_descriptor.parameters.gripper_maximum_opening = (
            self.gripper_maximum_opening
        )

        sequence = Sequence("T-shirt Pile", memory=True)
        sequence.add_children(
            [
                StartQueryStream(query_stream=self.query_stream),
                OpenVocabularyObjectDetectionAnnotator(
                    name="LaundryBasketDetection", descriptor=detection_descriptor
                ),
                PlaneAnnotator(name="FloorPlaneDetection"),
                LaundryGraspPoseAnnotator(descriptor=grasp_descriptor),
                SteadyLaundryGraspAnnotator(query_stream=self.query_stream),
                LaundryGraspPublisher(),
            ]
        )
        return sequence
