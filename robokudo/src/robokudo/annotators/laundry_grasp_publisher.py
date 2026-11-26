"""
Publishes the grasp on the laundry in a basket, so that a robot can pick it up while
perception keeps tracking it.
"""

from __future__ import annotations

# %% Imports
from enum import StrEnum

import numpy as np
from py_trees.common import Status
from rclpy.publisher import Publisher
from robokudo_msgs.action import Query
from robokudo_msgs.msg import ObjectDesignator
from semantic_digital_twin.adapters.ros.msg_converter import SemDTToRos2Converter
from semantic_digital_twin.adapters.ros.node_registry import ROSNodeRegistry
from std_msgs.msg import Float64MultiArray, MultiArrayDimension, MultiArrayLayout
from typing_extensions import Any

from robokudo.annotators.core import BaseAnnotator
from robokudo.cas import CAS
from robokudo.types.laundry import SteadyLaundryGraspAnnotation
from robokudo.utils.query import QueryHandler

# %% matrix layout


class MatrixAxis(StrEnum):
    """
    The axes of a matrix laid out in a :class:`Float64MultiArray`, outermost first.
    """

    ROWS = "rows"
    """
    The rows of the matrix.
    """

    COLUMNS = "columns"
    """
    The columns of a row.
    """


# %% annotator


class LaundryGraspPublisher(BaseAnnotator):
    """
    Publishes the laundry grasp held steady across the stream's frames as its 4x4
    homogeneous transformation matrix in the world frame, row by row, and offers it as
    the result of the active query.
    """

    class Descriptor(BaseAnnotator.Descriptor):
        """
        Configuration of the publisher.
        """

        class Parameters:
            """
            Parameters of the publisher.
            """

            def __init__(self) -> None:
                """
                Set the default parameters of the publisher.
                """
                self.topic_name: str = "/basket_tshirt_grasp_pose"
                """
                The topic the grasp matrix is published on.
                """

        parameters = Parameters()

    def __init__(
        self,
        name: str = "LaundryGraspPublisher",
        descriptor: LaundryGraspPublisher.Descriptor | None = None,
    ) -> None:
        """
        :param name: The name of the node in the tree.
        :param descriptor: The configuration of the publisher.
        """
        super().__init__(name, descriptor)
        self.publisher: Publisher | None = None
        """
        The ROS publisher of the grasp matrix, created when the annotator is set up.
        """

    def setup(self, **kwargs: Any) -> bool:
        """
        Create the publisher of the grasp matrix on RoboKudo's ROS node.
        """
        self.publisher = (
            ROSNodeRegistry()
            .get()
            .create_publisher(
                Float64MultiArray, self.descriptor.parameters.topic_name, 10
            )
        )
        return True

    def update(self) -> Status:
        """
        Publish the steady grasp and offer it as the query's result, if one is held.
        """
        steady_grasps = CAS.filter_by_type(
            SteadyLaundryGraspAnnotation, self.get_cas().annotations
        )
        if len(steady_grasps) == 0:
            self.feedback_message = "No steady grasp to publish"
            return Status.SUCCESS
        steady_grasp = steady_grasps[-1].grasp
        matrix = steady_grasp.grasp_pose.to_np()
        rows, columns = matrix.shape
        self.publisher.publish(
            Float64MultiArray(
                layout=MultiArrayLayout(
                    dim=[
                        MultiArrayDimension(
                            label=MatrixAxis.ROWS, size=rows, stride=rows * columns
                        ),
                        MultiArrayDimension(
                            label=MatrixAxis.COLUMNS, size=columns, stride=columns
                        ),
                    ]
                ),
                data=matrix.astype(np.float64).flatten().tolist(),
            )
        )
        QueryHandler.send_interim_result(
            result=Query.Result(
                res=[
                    ObjectDesignator(
                        pose=[SemDTToRos2Converter.convert(steady_grasp.grasp_pose)]
                    )
                ]
            )
        )
        self.feedback_message = f"Published the grasp {steady_grasp.height_above_floor:.3f} m above the floor"
        return Status.SUCCESS
