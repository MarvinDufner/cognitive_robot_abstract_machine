"""
Tests for annotating the bodies the world connector made of detected objects with the
semantic annotation their detected class stands for.
"""

from __future__ import annotations

import pytest

import robokudo.world
from robokudo.annotators.semantic_annotation_connector import (
    SemanticAnnotationConnector,
)
from robokudo.cas import CAS
from robokudo.types.annotation import Classification
from robokudo.types.scene import ObjectHypothesis
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Bottle,
    LaundryBasket,
)
from semantic_digital_twin.world import World

from .dataset.laundry_basket_on_floor import LaundryBasketScene
from .test_cloth_unfolding import update_in_pipeline

pytestmark = pytest.mark.usefixtures("annotator_output_buffer")

# %% fixtures


@pytest.fixture
def cas(perception_world: World) -> CAS:
    """
    :return: A CAS holding a camera frame posed in RoboKudo's world.
    """
    return LaundryBasketScene(world_frame=perception_world.root).cas()


def bottle_and_basket_connector() -> SemanticAnnotationConnector:
    """
    :return: A new connector that knows bottles and laundry baskets.
    """
    descriptor = SemanticAnnotationConnector.Descriptor()
    descriptor.parameters.semantic_annotation_types = {
        "bottle": Bottle,
        "laundry basket": LaundryBasket,
    }
    return SemanticAnnotationConnector(descriptor=descriptor)


def detected_object(cas: CAS, classname: str) -> ObjectHypothesis:
    """
    :return: An object detected as the class, in the CAS.
    """
    hypothesis = ObjectHypothesis()
    hypothesis.annotations.append(Classification(classname=classname, confidence=0.9))
    cas.annotations.append(hypothesis)
    return hypothesis


def believed_object(cas: CAS, classname: str) -> ObjectHypothesis:
    """
    :return: An object detected as the class, after the world connector made a body
        of it.
    """
    hypothesis = detected_object(cas, classname)
    robokudo.world.add_object_hypothesis_as_belief_state(hypothesis, cas)
    return hypothesis


# %% annotating believed objects


def test_a_believed_object_gets_the_annotation_its_class_stands_for(
    cas: CAS, perception_world: World
) -> None:
    """
    The body of an object detected as a bottle is annotated as a bottle.
    """
    hypothesis = believed_object(cas, "bottle")

    update_in_pipeline(bottle_and_basket_connector(), cas)

    [bottle] = perception_world.get_semantic_annotations_by_type(Bottle)
    assert bottle.root is robokudo.world.find_believed_body(hypothesis)


def test_a_perceivable_annotation_records_the_class_it_was_detected_as(
    cas: CAS, perception_world: World
) -> None:
    """
    The detected class becomes the class label of a perceivable annotation.
    """
    believed_object(cas, "laundry basket")

    update_in_pipeline(bottle_and_basket_connector(), cas)

    [laundry_basket] = perception_world.get_semantic_annotations_by_type(LaundryBasket)
    assert laundry_basket.class_label == "laundry basket"


def test_an_object_is_annotated_only_once_over_several_frames(
    cas: CAS, perception_world: World
) -> None:
    """
    Seeing the same object again keeps its one annotation.
    """
    believed_object(cas, "bottle")

    update_in_pipeline(bottle_and_basket_connector(), cas)
    update_in_pipeline(bottle_and_basket_connector(), cas)

    assert len(perception_world.get_semantic_annotations_by_type(Bottle)) == 1


def test_an_object_of_a_class_without_annotation_stays_unannotated(
    cas: CAS, perception_world: World
) -> None:
    """
    A class the connector does not know adds no annotation.
    """
    believed_object(cas, "chair")

    update_in_pipeline(bottle_and_basket_connector(), cas)

    assert perception_world.semantic_annotations == []


def test_an_object_the_world_connector_has_not_seen_stays_unannotated(
    cas: CAS, perception_world: World
) -> None:
    """
    Without a body in the world there is nothing to annotate.
    """
    detected_object(cas, "bottle")

    update_in_pipeline(bottle_and_basket_connector(), cas)

    assert perception_world.semantic_annotations == []
