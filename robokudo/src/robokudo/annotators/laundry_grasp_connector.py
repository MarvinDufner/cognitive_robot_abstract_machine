"""
Records the grasps on the laundry in detected baskets on their laundry baskets in the
semantic digital twin.
"""

from __future__ import annotations

from py_trees.common import Status

from robokudo import world
from robokudo.annotators.core import BaseAnnotator
from robokudo.annotators.exceptions import LaundryBasketNotAnnotated
from robokudo.cas import CAS
from robokudo.types.laundry import LaundryGraspAnnotation
from robokudo.types.scene import ObjectHypothesis
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    LaundryBasket,
)

# %% annotator


class LaundryGraspConnector(BaseAnnotator):
    """
    Records the latest grasp on the laundry of each believed basket on the
    :class:`LaundryBasket` annotating the basket's body.
    """

    def __init__(
        self,
        name: str = "LaundryGraspConnector",
        descriptor: BaseAnnotator.Descriptor | None = None,
    ) -> None:
        super().__init__(name, descriptor)

    def update(self) -> Status:
        """
        Record the grasp found on each basket's laundry on its laundry basket.

        :raises LaundryBasketNotAnnotated: If a basket with a grasp has a body that is
            not annotated as a laundry basket.
        """
        perception_world = world.world_instance()
        for hypothesis in self.get_cas().filter_annotations_by_type(ObjectHypothesis):
            grasps = CAS.filter_by_type(LaundryGraspAnnotation, hypothesis.annotations)
            body = world.find_believed_body(hypothesis)
            if len(grasps) == 0 or body is None:
                continue
            laundry_basket = world.find_semantic_annotation(body, LaundryBasket)
            if laundry_basket is None:
                raise LaundryBasketNotAnnotated(body=body)
            with perception_world.modify_world():
                laundry_basket.update_laundry_grasp_pose(grasps[-1].grasp_pose)
        return Status.SUCCESS
