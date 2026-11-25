"""
Grasp poses of the object hypotheses, predicted on their point clouds.
"""

from __future__ import annotations

# %% Imports
from timeit import default_timer

import numpy as np
from graspgenx._constants import MIN_OBJECT_POINT_COUNT
from py_trees.common import Status

from robokudo.annotators.core import BaseAnnotator
from robokudo.annotators.exceptions import MissingGraspPredictionGripper
from robokudo.cas import CAS
from robokudo.grasp_prediction import (
    GraspGenXGripper,
    GraspPosePredictor,
    PredictedGrasp,
)
from robokudo.types.annotation import GraspAnnotation
from robokudo.types.scene import ObjectHypothesis
from robokudo.utils.transform import get_quaternion_from_rotation_matrix
from semantic_digital_twin.semantic_annotations.mixins import IsGraspable

# %% grasp pose annotation


class GraspPoseAnnotator(BaseAnnotator):
    """
    Annotates every object hypothesis with the grasps predicted on its points that keep
    the gripper clear of the rest of the scene, the most confident first.

    A hypothesis matched to a body of the world also hands the grasps to every
    :class:`~semantic_digital_twin.semantic_annotations.mixins.IsGraspable` rooted at
    that body, as its perceived grasps.
    """

    class Descriptor(BaseAnnotator.Descriptor):
        class Parameters:
            def __init__(self) -> None:
                self.gripper: GraspGenXGripper | None = None
                """
                The gripper to predict the grasps of.
                """

                self.maximum_grasps: int = 10
                """
                How many of the most confident grasps a hypothesis is annotated with.
                """

        parameters = Parameters()

    def __init__(
        self,
        name: str = "GraspPoseAnnotator",
        descriptor: GraspPoseAnnotator.Descriptor | None = None,
    ) -> None:
        """
        :param name: Name of this annotator in the tree.
        :param descriptor: Parameters of this annotator.
        :raises MissingGraspPredictionGripper: If the descriptor names no gripper.
        """
        super().__init__(name, descriptor)
        if self.descriptor.parameters.gripper is None:
            raise MissingGraspPredictionGripper()
        self.predictor = GraspPosePredictor(gripper=self.descriptor.parameters.gripper)

    def update(self) -> Status:
        """
        Predict and annotate the grasps of every object hypothesis with enough points,
        clear of the rest of the CAS's point cloud.

        :return: :attr:`Status.SUCCESS`.
        """
        start = default_timer()
        cas = self.get_cas()
        scene_points = None if cas.cloud is None else np.asarray(cas.cloud.points)
        for hypothesis in cas.filter_annotations_by_type(ObjectHypothesis):
            if (
                hypothesis.points is None
                or len(hypothesis.points.points) < MIN_OBJECT_POINT_COUNT
            ):
                continue
            grasps = self.predictor.predict(
                points=np.asarray(hypothesis.points.points), scene_points=scene_points
            )[: self.descriptor.parameters.maximum_grasps]
            hypothesis.annotations.extend(
                self._grasp_annotation(grasp=grasp) for grasp in grasps
            )
            self._hand_to_graspables(hypothesis=hypothesis, grasps=grasps, cas=cas)
        self.feedback_message = f"Processing took {default_timer() - start:.2f}s"
        return Status.SUCCESS

    def _grasp_annotation(self, *, grasp: PredictedGrasp) -> GraspAnnotation:
        """
        :param grasp: A grasp predicted on a hypothesis' points, in the camera frame.
        :return: The annotation of that grasp.
        """
        return GraspAnnotation(
            source=self.get_class_name(),
            translation=list(grasp.points_T_grasp[:3, 3]),
            rotation=list(
                get_quaternion_from_rotation_matrix(grasp.points_T_grasp[:3, :3])
            ),
            confidence=grasp.confidence,
        )

    @staticmethod
    def _hand_to_graspables(
        *, hypothesis: ObjectHypothesis, grasps: list[PredictedGrasp], cas: CAS
    ) -> None:
        """
        Store ``grasps`` as the perceived grasps of every graspable rooted at the body
        ``hypothesis`` was matched to.

        :param hypothesis: The hypothesis the grasps were predicted for.
        :param grasps: The grasps, in the camera frame.
        :param cas: The CAS holding the camera's pose in the world of that body.
        """
        body = hypothesis.object_knowledge
        if body is None:
            return
        graspables = [
            annotation
            for annotation in body._world.get_semantic_annotations_by_type(IsGraspable)
            if annotation.root is body
        ]
        if not graspables:
            return
        world_T_camera = cas.require_camera_observation().world_T_camera_or_raise()
        body_T_camera = (
            body._world.compute_forward_kinematics_np(
                body, world_T_camera.reference_frame
            )
            @ world_T_camera.to_np()
        )
        for graspable in graspables:
            graspable.perceived_grasps = [
                grasp.to_perceived_grasp(body=body, body_T_points=body_T_camera)
                for grasp in grasps
            ]
