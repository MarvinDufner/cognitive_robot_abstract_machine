"""
Query handling annotator for RoboKudo.

This module provides an annotator that handles queries from external ROS nodes.
It supports:

* Spawning an action server for query handling
* Type-agnostic query processing
* Asynchronous query response
* Integration with ROS action system
* CAS annotation with query data

The module is used for:

* External system integration
* Query-based perception
* Interactive perception tasks
* Asynchronous data exchange
"""

import logging
import queue
import time
from enum import StrEnum
from threading import Event

import rclpy
from geometry_msgs.msg import PoseStamped
from py_trees.blackboard import Blackboard
from py_trees.common import Status
from rclpy.action import CancelResponse, GoalResponse
from rclpy.action.server import ActionServer, ServerGoalHandle
from rclpy.node import Node
from robokudo_msgs.action import Query
from robokudo_msgs.msg import ObjectDesignator
from typing_extensions import Any

from robokudo.annotators.core import BaseAnnotator
from robokudo.cas import CASViews
from robokudo.defs import LOGGING_IDENTIFIER_QUERY, PACKAGE_NAME
from robokudo.identifier import BBIdentifier
from robokudo.types.annotation import (
    BoundingBox3DAnnotation,
    Classification,
    Cuboid,
    Cylinder,
    LocationAnnotation,
    PoseAnnotation,
    PositionAnnotation,
    SemanticColor,
    Shape,
    Sphere,
    StampedPoseAnnotation,
    StampedPositionAnnotation,
)
from robokudo.types.cv import BoundingBox3D
from robokudo.types.scene import ObjectHypothesis
from robokudo.utils.annotation_conversion import (
    BoundingBox3DForShapeSizeConverter,
    Classification2ODConverter,
    Cuboid2ODConverter,
    Cylinder2ODConverter,
    Location2ODConverter,
    Pose2ODConverter,
    Position2ODConverter,
    SemanticColor2ODConverter,
    Shape2ODConverter,
    Sphere2ODConverter,
    StampedPose2ODConverter,
    StampedPosition2ODConverter,
)
from robokudo.utils.error_handling import (
    clear_blackboard_exception,
    get_blackboard_exception,
    has_blackboard_exception,
)
from robokudo.utils.query import ObjectHypothesisQueryMatcher, QueryHandler


class QueryAnnotator(BaseAnnotator):
    """
    Handle external queries through ROS action server.

    This Annotator spawns an Action Server that listens for Queries from external ROS
    nodes. It will then annotate the CAS and put the Query into CASViews.QUERY. The
    Annotator and the Actionserver are type-agnostic, which means that you are not bound
    to a specific type of query. You can pass these from your AE to this QueryAnnotator.
    """

    def __init__(self, name: str = "QueryAnnotator") -> None:
        """
        Initialize the query annotator.

        :param name: Annotator name
        """
        super().__init__(name=name)

        self.feedback_instance = Query.Feedback()
        self.result_instance = Query.Result()

        self.action_server = None
        """
        Action server placeholder.
        """

    def setup(self, **kwargs: Any) -> None:
        """
        Ensure that the Query Server is spawned early on, directly after PPT creation.
        """
        self.initialise()

    def initialise(self) -> None:
        """
        Initialize query handling.

        Sets up the action server if not already initialized. Stores server instance on
        blackboard for access by other nodes.
        """
        self.rk_logger.debug(f"{self.__class__.__name__}.initialise()")
        blackboard = Blackboard()
        if not blackboard.exists(BBIdentifier.QUERY_SERVER):
            query_action_server = QueryActionServer(name="query")
            blackboard.set(BBIdentifier.QUERY_SERVER, query_action_server)

        blackboard.set(BBIdentifier.QUERY_SERVER_IN_PIPELINE, True)

        self.action_server = blackboard.get(BBIdentifier.QUERY_SERVER)

    def update(self) -> Status:
        """
        Process new queries and update CAS.

        Checks for new queries from action server and updates CAS if found. Provides
        feedback about query status.

        :return: SUCCESS if query processed, RUNNING if waiting
        """
        self.rk_logger.debug(f"{self.__class__.__name__}.update()")

        assert self.action_server is not None, (
            "Action server should be initialized by now."
        )
        query = self.action_server.new_query
        self.rk_logger.debug(f"self.action_server.new_query: {query}")

        if query:
            self.feedback_message = f"Query: {query}"
            self.get_cas().set(CASViews.QUERY, query)
            self.action_server.start_processing()
            # self.publish_feedback()

            return Status.SUCCESS

        self.feedback_message = "Waiting for query"
        return Status.RUNNING


class QueryFeedback(BaseAnnotator):
    """
    A test class which simply generates a fixed-string feedback.
    """

    def __init__(self, name: str = "QueryFeedback", feedback_str: str = "") -> None:
        """
        Initialize query feedback generator.

        :param name: Annotator name
        :param feedback_str: Feedback string to send to the client
        """
        super().__init__(name=name)

        self.feedback_str = feedback_str
        """
        Feedback string to send to the client.
        """

    def update(self) -> Status:
        self.rk_logger.debug(f"{self.__class__.__name__}.update()")
        QueryHandler.send_feedback_str(self.feedback_str)

        return Status.SUCCESS


class QueryFeedbackAndCount(BaseAnnotator):
    """
    A test class which simply counts up until a fixed number.

    Until this number is reached, a pre-defined status is returned.
    """

    def __init__(
        self,
        name: str = "QueryFeedback",
        count_until: int = 20,
        return_code: Status = Status.RUNNING,
    ) -> None:
        """
        Initialize query feedback generator.

        :param name: Annotator name
        :param count_until: Number until which to count before stopping
        :param return_code: The return code to return while still counting
        """
        super().__init__(name=name)

        self.i = 0
        """
        The current count.
        """
        self.count_until = count_until
        """
        The number until which to count.
        """
        self.return_code = return_code
        """
        The return code to return while still counting.
        """

    def update(self) -> Status:
        self.rk_logger.debug(f"{self.__class__.__name__}.update()")
        QueryHandler.send_feedback_str(f"Count: {self.i}")
        self.i = self.i + 1
        if self.i > self.count_until:
            self.i = 0
            return Status.SUCCESS
        else:
            return self.return_code


class QueryReply(BaseAnnotator):
    """
    A test class which simply generates an empty Query Answer to check if the Action
    server can reply properly.

    Create a single, empty Object Designator that will be sent to the caller.
    """

    def __init__(self, name: str = "QueryReply"):
        """
        Initialize query reply generator.

        :param name: Annotator name
        """
        super().__init__(name=name)

    def initialise(self) -> None:
        """
        Initialize reply generator.
        """
        self.rk_logger.debug(f"{self.__class__.__name__}.initialise()")

    def update(self) -> Status:
        """
        Generate test query response.

        Creates an empty ObjectDesignator with a test pose and adds it to blackboard.

        :return: SUCCESS after generating response
        """
        self.rk_logger.debug(f"{self.__class__.__name__}.update()")
        result = Query.Result()
        od = ObjectDesignator()

        pose_stamped = PoseStamped()

        # Explicitly cast to float
        pose_stamped.pose.position.x = float(1)
        pose_stamped.pose.position.y = float(2)
        pose_stamped.pose.position.z = float(3)

        pose_stamped.pose.orientation.x = float(0)
        pose_stamped.pose.orientation.y = float(0)
        pose_stamped.pose.orientation.z = float(0)
        pose_stamped.pose.orientation.w = float(1)

        od.pose.append(pose_stamped)
        result.res = [od]

        QueryHandler.send_answer(result)

        return Status.SUCCESS


class GenerateQueryResult(BaseAnnotator):
    """
    This class reads in the annotations done by the previous Annotators and generates
    Object Designators from them.

    These will be placed into the Blackboard so that a running Query Action Server can
    pick the information up and send it as a query reply.
    """

    class Descriptor(BaseAnnotator.Descriptor):
        class Parameters:
            def __init__(self) -> None:
                self.filter_by_query: bool = False
                """Only return ObjectHypotheses matching requested query attributes."""

        parameters = Parameters()

    def __init__(
        self, name: str = "GenerateQueryResult", descriptor: Descriptor | None = None
    ) -> None:
        """Initialize query result generator.

        :param name: Annotator name
        :param descriptor: Annotator configuration descriptor
        """
        self.rk_logger = logging.getLogger(PACKAGE_NAME)

        self.color_converter = SemanticColor2ODConverter()
        self.class_converter = Classification2ODConverter()
        self.position_converter = Position2ODConverter()
        self.stamped_position_converter = StampedPosition2ODConverter()
        self.pose_converter = Pose2ODConverter()
        self.stamped_pose_converter = StampedPose2ODConverter()
        self.shape_converter = Shape2ODConverter()
        self.cuboid_converter = Cuboid2ODConverter()
        self.cylinder_converter = Cylinder2ODConverter()
        self.sphere_converter = Sphere2ODConverter()
        self.location_converter = Location2ODConverter()
        self.bb_size_converter = BoundingBox3DForShapeSizeConverter()

        self.type_converter = {
            SemanticColor: self.color_converter,
            Classification: self.class_converter,
            PoseAnnotation: self.pose_converter,
            StampedPoseAnnotation: self.stamped_pose_converter,
            PositionAnnotation: self.position_converter,
            StampedPositionAnnotation: self.stamped_position_converter,
            Shape: self.shape_converter,
            Cuboid: self.cuboid_converter,
            Cylinder: self.cylinder_converter,
            Sphere: self.sphere_converter,
            LocationAnnotation: self.location_converter,
            BoundingBox3D: self.bb_size_converter,
            BoundingBox3DAnnotation: self.bb_size_converter,
        }
        self.query_matcher = ObjectHypothesisQueryMatcher()

        super().__init__(name=name, descriptor=descriptor)

    def update(self) -> Status:
        """
        Generate query result from current CAS annotations.

        For each ObjectHypothesis in CAS:
        * Creates ObjectDesignator
        * Adds color information if available
        * Adds classification if available
        * Adds pose information if available
        * Packages into query result

        :return: SUCCESS after generating result
        """
        if QueryHandler.preempt_requested():
            QueryHandler.acknowledge_preempt_request()
            self.rk_logger.warning("Acknowledge preempt")
            return Status.FAILURE

        cas = self.get_cas()
        annotations = cas.annotations
        object_hypotheses_count = 0
        skipped_object_hypotheses_count = 0
        query_result = []
        result = Query.Result()
        requested_object = None
        if self.descriptor.parameters.filter_by_query and cas.contains(CASViews.QUERY):
            query = cas.get(CASViews.QUERY)
            requested_object = query.obj

        for annotation in annotations:
            if not isinstance(annotation, ObjectHypothesis):
                continue

            if requested_object is not None and not self.query_matcher.matches(
                annotation, requested_object
            ):
                skipped_object_hypotheses_count += 1
                continue

            object_designator = ObjectDesignator()

            for oh_annotation in annotation.annotations:
                converter = self.type_converter.get(type(oh_annotation), None)
                if converter is None:
                    self.rk_logger.warning(
                        f"no converter available for annotation type {type(oh_annotation)}, skipping annotation."
                    )
                    continue
                if not converter.can_convert(oh_annotation):
                    self.rk_logger.warning(
                        f"converter for {type(oh_annotation)} available but cannot convert, skipping annotation."
                    )
                    continue

                converter.convert(oh_annotation, cas, object_designator)
                self.rk_logger.info(
                    f"converted {type(oh_annotation)} on OH {object_hypotheses_count}"
                )

            query_result.append(object_designator)
            object_hypotheses_count += 1

        result.res = query_result
        QueryHandler.send_answer(result)

        self.feedback_message = (
            f"Send result for {object_hypotheses_count} object hypotheses "
            f"({skipped_object_hypotheses_count} filtered by query)"
        )
        return Status.SUCCESS


class QueryResultText(StrEnum):
    """
    The text result of a goal the pipeline did not answer.
    """

    CANCELED = "Canceled"
    """
    The client canceled the goal and the pipeline acknowledged it.
    """

    ABORTED = "Aborted"
    """
    The pipeline raised an error while processing the goal.
    """

    SUPERSEDED = "Superseded by a newer query"
    """
    A newer query arrived before the pipeline offered any result for the goal.
    """


class QueryActionServer(Node):
    """
    ROS action server for handling perception queries.

    Action server that listens for queries and executes them by checking blackboard for
    results generated by QueryAnnotator and QueryReply. A query that arrives while
    another is processed supersedes it.
    """

    def __init__(
        self,
        name: str,
        feedback_instance: Query.Feedback | None = None,
        result_instance: Query.Result | None = None,
        action_type: type = Query,
    ) -> None:
        """
        :param name: The name of the action, served in the ``robokudo`` namespace.
        :param feedback_instance: The feedback message to fill, a new one if ``None``.
        :param result_instance: The result message to fill, a new one if ``None``.
        :param action_type: The action the server serves.
        """
        super().__init__(name, namespace="robokudo")
        self._action_name = name
        """
        Name of the ROS action.
        """
        self._as = ActionServer(
            self,
            action_type,
            self._action_name,
            execute_callback=self.execute_cb,
            goal_callback=self.goal_cb,
            cancel_callback=self.cancel_cb,
        )
        """
        Action server instance.
        """
        self.feedback_instance: Query.Feedback = feedback_instance or Query.Feedback()
        """
        Latest feedback message.
        """
        self.result_instance: Query.Result = result_instance or Query.Result()
        """
        Latest result message.
        """
        self.new_query: Query.Goal | None = None
        """
        Latest received query.
        """
        self.query: Query.Goal | None = None
        """
        Currently processing query.
        """
        self.active_goal: ServerGoalHandle | None = None
        """
        The goal whose query is the current one.
        """
        self.superseded_results: dict[bytes, Query.Result | None] = {}
        """
        The result each superseded goal ends with, by goal ID, ``None`` if it streamed
        none.
        """
        self.reset_bookkeeping_vars()

        self.query_processed_event: Event = Event()
        """
        Event to signal query processing completion.
        """
        self.logger = logging.getLogger(LOGGING_IDENTIFIER_QUERY)

    def reset_bookkeeping_vars(self) -> None:
        """
        Reset internal state variables.

        Clears query state and blackboard variables.
        """
        self.query = None
        self.new_query = None
        self.active_goal = None
        Blackboard().set(BBIdentifier.QUERY_ANSWER, None)
        Blackboard().set(BBIdentifier.QUERY_INTERIM_RESULT, None)
        Blackboard().set(BBIdentifier.QUERY_FEEDBACK, queue.Queue())
        Blackboard().set(BBIdentifier.QUERY_PREEMPT_REQUESTED, False)
        Blackboard().set(BBIdentifier.QUERY_PREEMPT_ACK, False)

    def goal_cb(self, goal_request: Query.Goal) -> GoalResponse:
        self.logger.info(f"Received new goal: {goal_request}")
        return GoalResponse.ACCEPT

    def cancel_cb(self, goal_handle: ServerGoalHandle) -> CancelResponse:
        self.logger.info(f"Received cancel request:{goal_handle}")
        Blackboard().set(BBIdentifier.QUERY_PREEMPT_REQUESTED, True)
        return CancelResponse.ACCEPT

    def start_processing(self) -> None:
        """
        Start processing new query.

        Tell the ActionServer that we are now starting the execution and it can start
        the monitoring/response process.
        """
        self.logger.info("start_processing called, setting new_query to None.")
        self.new_query = None

    def is_active(self) -> bool:
        """
        Check if query is being processed.

        :return: True if the pipeline took up the current query and has not finished
            it, False otherwise, also while a query waits to be taken up.
        """
        return self.query is not None and self.new_query is None

    def supersede_active_goal(self) -> None:
        """
        Set aside the result the active goal, if any, ends with, which is the result the
        pipeline offered it last.
        """
        if self.active_goal is None:
            return
        blackboard = Blackboard()
        self.superseded_results[bytes(self.active_goal.goal_id.uuid)] = blackboard.get(
            BBIdentifier.QUERY_INTERIM_RESULT
        )
        blackboard.set(BBIdentifier.QUERY_INTERIM_RESULT, None)

    def end_superseded_goal(self, *, goal_handle: ServerGoalHandle) -> Query.Result:
        """
        Succeed a superseded goal with the result set aside for it, or abort it if the
        pipeline offered it none.

        :param goal_handle: The superseded goal.
        :return: The result the goal ends with.
        """
        result = self.superseded_results.pop(bytes(goal_handle.goal_id.uuid), None)
        if result is None:
            goal_handle.abort()
            return Query.Result(text_result=QueryResultText.SUPERSEDED)
        goal_handle.succeed()
        return result

    async def execute_cb(self, goal_handle: ServerGoalHandle) -> Query.Result | None:
        """
        Action server execution callback.

        Makes the goal's query the current one, superseding the active goal, and ends
        the goal once the pipeline answers, cancels or aborts it, or a newer query
        supersedes it.

        :param goal_handle: Query goal from client
        :return: The result the goal ends with.
        """
        self.logger.info(f"Received query: {goal_handle.request}")
        self.supersede_active_goal()
        self.active_goal = goal_handle
        self.new_query = goal_handle.request
        self.query = goal_handle.request

        feedback_queue = Blackboard().get(BBIdentifier.QUERY_FEEDBACK)

        while rclpy.ok():
            time.sleep(1.0 / 50.0)

            if goal_handle is not self.active_goal:
                self.logger.info("Goal superseded by a newer query.")
                return self.end_superseded_goal(goal_handle=goal_handle)

            # At least one node in the Tree has to acknowledge the preempt request. This allows the tree
            # to properly shutdown.
            preempt_acknowledge = Blackboard().get(BBIdentifier.QUERY_PREEMPT_ACK)
            if goal_handle.is_cancel_requested and preempt_acknowledge:
                self.logger.info("Goal cancel acknowledged by PPT.")
                goal_handle.canceled()
                self.reset_bookkeeping_vars()
                return Query.Result(text_result=QueryResultText.CANCELED)

            if has_blackboard_exception():
                exception_text = str(get_blackboard_exception())
                self.logger.error(f"Aborting due to error: {exception_text}")
                clear_blackboard_exception()
                goal_handle.abort()
                self.reset_bookkeeping_vars()
                return Query.Result(text_result=QueryResultText.ABORTED)

            if not feedback_queue.empty():
                feedback_msg = feedback_queue.get_nowait()
                if isinstance(feedback_msg, Query.Feedback):
                    self.feedback_instance.feedback = feedback_msg.feedback
                    goal_handle.publish_feedback(self.feedback_instance)
                    self.logger.info(
                        f"Published feedback: {self.feedback_instance.feedback}"
                    )

            answer = Blackboard().get(BBIdentifier.QUERY_ANSWER)
            if answer is not None:
                goal_handle.succeed()
                self.logger.info(f"Send: {answer}")
                self.reset_bookkeeping_vars()
                return answer

        goal_handle.abort()
        return None

    def destroy(self) -> None:
        """
        Destroy the action server and its ROS node.
        """
        self._as.destroy()
        super().destroy_node()
