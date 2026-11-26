"""
Tests for publishing the steady grasp on the laundry in a basket as a homogeneous
transformation matrix in the world frame, and offering it as the active query's result.
"""

from __future__ import annotations

# %% Imports
import numpy as np
import pytest
import rclpy
from krrood_spatial_math.spatial_types import Pose
from py_trees.blackboard import Blackboard
from py_trees.common import Status
from rclpy.node import Node
from semantic_digital_twin.adapters.ros.msg_converter import SemDTToRos2Converter
from semantic_digital_twin.world import World
from std_msgs.msg import Float64MultiArray

from robokudo.annotators.laundry_grasp_publisher import LaundryGraspPublisher
from robokudo.cas import CAS
from robokudo.identifier import BBIdentifier
from robokudo.types.annotation import Classification
from robokudo.types.cv import Point2D
from robokudo.types.laundry import (
    LaundryBasketClassName,
    LaundryGraspAnnotation,
    SteadyLaundryGraspAnnotation,
)
from robokudo.types.scene import ObjectHypothesis

from .test_cloth_unfolding import update_in_pipeline

pytestmark = pytest.mark.usefixtures("annotator_output_buffer")

# %% Fixtures


@pytest.fixture
def publisher(node: Node) -> LaundryGraspPublisher:
    """
    :return: A publisher whose ROS publisher is set up on the test node.
    """
    Blackboard().set(BBIdentifier.QUERY_INTERIM_RESULT, None)
    annotator = LaundryGraspPublisher()
    annotator.setup()
    return annotator


@pytest.fixture
def received_matrices(
    node: Node, publisher: LaundryGraspPublisher
) -> list[Float64MultiArray]:
    """
    :return: The messages received on the publisher's topic, filled while spinning.
    """
    messages: list[Float64MultiArray] = []
    node.create_subscription(
        Float64MultiArray,
        publisher.descriptor.parameters.topic_name,
        messages.append,
        10,
    )
    return messages


def laundry_grasp(*, world: World, x: float, height: float) -> LaundryGraspAnnotation:
    """
    :return: A top-down grasp at the given x and height in the world frame.
    """
    return LaundryGraspAnnotation(
        pixel=Point2D(x=10, y=20),
        height_above_floor=height,
        grasp_pose=Pose.from_xyz_rpy(
            x=x, y=0.5, z=height, pitch=np.pi / 2, yaw=0.3, reference_frame=world.root
        ),
    )


@pytest.fixture
def steady_grasp(perception_world: World) -> LaundryGraspAnnotation:
    """
    :return: The grasp held steady across the stream's frames.
    """
    return laundry_grasp(world=perception_world, x=-1.0, height=0.3)


@pytest.fixture
def cas_with_steady_grasp(
    perception_world: World, steady_grasp: LaundryGraspAnnotation
) -> CAS:
    """
    :return: A CAS with the steady grasp and a basket whose grasp in this frame lies
        elsewhere and higher.
    """
    cas = CAS()
    basket = ObjectHypothesis()
    basket.annotations.append(
        Classification(classname=LaundryBasketClassName.LAUNDRY_BASKET, confidence=0.9)
    )
    basket.annotations.append(laundry_grasp(world=perception_world, x=1.0, height=0.5))
    cas.annotations.append(basket)
    cas.annotations.append(SteadyLaundryGraspAnnotation(grasp=steady_grasp))
    return cas


def spin_until_received(
    *, node: Node, messages: list[Float64MultiArray], timeout_seconds: float = 5.0
) -> None:
    """
    Spin the node until a message arrives.
    """
    for _ in range(int(timeout_seconds / 0.1)):
        if messages:
            return
        rclpy.spin_once(node, timeout_sec=0.1)


# %% Publishing


def test_the_steady_grasp_is_published_as_its_matrix_in_the_world_frame(
    node: Node,
    publisher: LaundryGraspPublisher,
    received_matrices: list[Float64MultiArray],
    cas_with_steady_grasp: CAS,
    steady_grasp: LaundryGraspAnnotation,
) -> None:
    """
    The message holds the rotation and translation of the steady grasp, not of this
    frame's grasp, row by row as its layout describes.
    """
    status = update_in_pipeline(publisher, cas_with_steady_grasp)
    spin_until_received(node=node, messages=received_matrices)

    (message,) = received_matrices
    shape = tuple(dimension.size for dimension in message.layout.dim)
    assert status == Status.SUCCESS
    np.testing.assert_allclose(
        np.array(message.data).reshape(shape), steady_grasp.grasp_pose.to_np()
    )


def test_the_steady_grasp_is_offered_as_the_active_query_result(
    publisher: LaundryGraspPublisher,
    cas_with_steady_grasp: CAS,
    steady_grasp: LaundryGraspAnnotation,
) -> None:
    """
    A newer query ends the active goal with the grasp published last.
    """
    update_in_pipeline(publisher, cas_with_steady_grasp)

    (designator,) = Blackboard().get(BBIdentifier.QUERY_INTERIM_RESULT).res
    assert designator.pose == [SemDTToRos2Converter.convert(steady_grasp.grasp_pose)]


def test_a_frame_without_a_steady_grasp_publishes_nothing(
    node: Node,
    publisher: LaundryGraspPublisher,
    received_matrices: list[Float64MultiArray],
    cas_with_steady_grasp: CAS,
) -> None:
    """
    Until a grasp is held steady, the frame neither publishes nor offers a result.
    """
    cas_with_steady_grasp.annotations = [
        annotation
        for annotation in cas_with_steady_grasp.annotations
        if not isinstance(annotation, SteadyLaundryGraspAnnotation)
    ]

    status = update_in_pipeline(publisher, cas_with_steady_grasp)
    spin_until_received(node=node, messages=received_matrices, timeout_seconds=0.5)

    assert status == Status.SUCCESS
    assert received_matrices == []
    assert Blackboard().get(BBIdentifier.QUERY_INTERIM_RESULT) is None
