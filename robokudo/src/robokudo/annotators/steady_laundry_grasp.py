"""
Holds the laundry grasp steady across the frames of a query stream, so that a robot is
not sent back and forth between tops of nearly the same height.
"""

from __future__ import annotations

# %% Imports
from py_trees.common import Status

from robokudo.annotators.core import BaseAnnotator
from robokudo.behaviours.query_stream import QueryStream
from robokudo.cas import CAS
from robokudo.types.laundry import (
    LaundryGraspAnnotation,
    SteadyLaundryGrasp,
    SteadyLaundryGraspAnnotation,
)
from robokudo.types.scene import ObjectHypothesis

# %% annotator


class SteadyLaundryGraspAnnotator(BaseAnnotator):
    """
    Follows the grasp on the laundry lying highest in each frame with a
    :class:`SteadyLaundryGrasp`, starting over with each stream, and annotates the CAS
    with the steady grasp while one is held.
    """

    class Descriptor(BaseAnnotator.Descriptor):
        """
        Configuration of the steady grasp.
        """

        class Parameters:
            """
            Parameters of the steady grasp, as :class:`SteadyLaundryGrasp` describes
            them.
            """

            def __init__(self) -> None:
                """
                Set the defaults of :class:`SteadyLaundryGrasp`.
                """
                defaults = SteadyLaundryGrasp()
                self.stay_radius: float = defaults.stay_radius
                """
                How far, in metres, a grasp may lie from the held one to count as the
                same spot.
                """
                self.confirmation_frames: int = defaults.confirmation_frames
                """
                For how many frames in a row a grasp has to stay elsewhere, or be
                missing, before the held grasp follows.
                """
                self.history_length: int = defaults.history_length
                """
                How many of the latest grasps at the held spot are kept to choose from.
                """

        parameters = Parameters()

    def __init__(
        self,
        *,
        query_stream: QueryStream,
        name: str = "SteadyLaundryGraspAnnotator",
        descriptor: SteadyLaundryGraspAnnotator.Descriptor | None = None,
    ) -> None:
        """
        :param query_stream: The stream whose frames the grasp is held across.
        :param name: The name of the node in the tree.
        :param descriptor: The configuration of the steady grasp.
        """
        super().__init__(name, descriptor)
        self.query_stream = query_stream
        """
        The stream whose frames the grasp is held across.
        """
        self.stream_number = query_stream.started_streams
        """
        The count of started streams when the steady grasp was last started over.
        """
        self.steady_grasp = self.new_steady_grasp()
        """
        The grasp held across the frames of the current stream.
        """

    def new_steady_grasp(self) -> SteadyLaundryGrasp:
        """
        :return: A steady grasp holding nothing yet, configured by the descriptor.
        """
        parameters = self.descriptor.parameters
        return SteadyLaundryGrasp(
            stay_radius=parameters.stay_radius,
            confirmation_frames=parameters.confirmation_frames,
            history_length=parameters.history_length,
        )

    def update(self) -> Status:
        """
        Follow this frame's highest grasp and annotate the CAS with the steady grasp.
        """
        if self.query_stream.started_streams != self.stream_number:
            self.stream_number = self.query_stream.started_streams
            self.steady_grasp = self.new_steady_grasp()
        cas = self.get_cas()
        grasps = [
            grasp
            for hypothesis in cas.filter_annotations_by_type(ObjectHypothesis)
            for grasp in CAS.filter_by_type(
                LaundryGraspAnnotation, hypothesis.annotations
            )
        ]
        self.steady_grasp.follow(
            grasp=max(grasps, key=lambda grasp: grasp.height_above_floor, default=None)
        )
        steady_grasp = self.steady_grasp.steady_grasp
        if steady_grasp is None:
            self.feedback_message = "No grasp held"
            return Status.SUCCESS
        cas.annotations.append(
            SteadyLaundryGraspAnnotation(source=self.name, grasp=steady_grasp)
        )
        self.feedback_message = "Holding the grasp steady"
        return Status.SUCCESS
