import os
from copy import deepcopy
from dataclasses import field
from enum import Enum
from typing import Optional

import numpy as np

from giskardpy.data_types.suturo_types import GraspTypes
from giskardpy.data_types.suturo_types import ObjectTypes, TakePoseTypes
from giskardpy.middleware import get_middleware
from giskardpy.model.links import (
    BoxGeometry,
    LinkGeometry,
    SphereGeometry,
    CylinderGeometry,
)
from giskardpy.motion_statechart.context import BuildContext
from giskardpy.motion_statechart.goals.cartesian_goals import (
    CartesianPosition,
    CartesianOrientation,
)
from giskardpy.motion_statechart.graph_node import Goal, NodeArtifacts
from giskardpy.motion_statechart.tasks.align_planes import AlignPlanes
from giskardpy.motion_statechart.tasks.joint_tasks import JointPositionList, JointState
from krrood.symbolic_math.symbolic_math import trinary_logic_and

from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types import Vector3
from semantic_digital_twin.world_description.geometry import Shape

if "GITHUB_WORKFLOW" not in os.environ:
    pass


class ContextActionModes(Enum):
    grasping = "grasping"
    placing = "placing"
    pouring = "pouring"
    door_opening = "door-opening"


class ObjectGoal(Goal):
    """
    Inherit from this class if the goal tries to get the object by name from the world
    """

    def get_object_by_name(self, context: BuildContext, object_name: str):
        try:
            get_middleware().loginfo("trying to get objects with name")

            object_link = context.world.get_body_by_name(object_name)
            object_collisions = object_link.collision
            if len(object_collisions) == 0:
                object_geometry = BoxGeometry(
                    link_T_geometry=np.eye(4), depth=0, width=0, height=0, color=None
                )
            else:
                object_geometry: Shape = object_collisions[0]

            goal_pose = context.world.compute_forward_kinematics(
                context.world.get_kinematic_structure_entity_by_name("map"),
                context.world.get_body_by_name(object_name),
            )

            get_middleware().loginfo(f"goal_pose by name: {goal_pose}")

            # Declare instance of geometry
            if isinstance(object_geometry, BoxGeometry):
                object_type = "box"
                object_geometry: BoxGeometry = object_geometry
                # FIXME use expression instead of vector3, unless its really a vector
                object_size = cas.Vector3.from_xyz(
                    x=object_geometry.width,
                    y=object_geometry.depth,
                    z=object_geometry.height,
                )

            elif isinstance(object_geometry, CylinderGeometry):
                object_type = "cylinder"
                object_geometry: CylinderGeometry = object_geometry
                object_size = cas.Vector3.from_xyz(
                    x=object_geometry.radius,
                    y=object_geometry.radius,
                    z=object_geometry.height,
                )

            elif isinstance(object_geometry, SphereGeometry):
                object_type = "sphere"
                object_geometry: SphereGeometry = object_geometry
                object_size = cas.Vector3.from_xyz(
                    x=object_geometry.radius,
                    y=object_geometry.radius,
                    z=object_geometry.radius,
                )

            else:
                raise Exception("Not supported geometry")

            get_middleware().loginfo(f"Got geometry: {object_type}")
            return goal_pose, object_size

        except:
            get_middleware().loginfo("Could not get geometry from name")
            return None


class Reaching(ObjectGoal):
    def __init__(
        self,
        root_link: PrefixName,
        tip_link: PrefixName,
        grasp: str,
        align: str,
        name: str = None,
        object_name: Optional[str] = None,
        object_shape: Optional[str] = None,
        goal_pose: Optional[cas.TransMatrix] = None,
        object_size: Optional[cas.Vector3] = None,
        velocity: float = 0.2,
        weight: float = WEIGHT_ABOVE_CA,
    ):
        """
        Concludes Reaching type goals.
        Executes them depending on the given context action.
        Context is a dictionary in an action is given as well as situational parameters.
        All available context Messages are found in the Enum 'ContextTypes'

        :param grasp: Direction from which object is being grasped from
        :param align: String which decides whether HSR tries to vertically align with the object before reaching
        :param name: name of the executed goal, in this case Reaching
        :param object_name: Name of the object to use. Optional as long as goal_pose and object_size are filled instead
        :param object_shape: Shape of the object to manipulate. Edit object size when having a sphere or cylinder
        :param goal_pose: Goal pose for the object. Alternative if no object name is given.
        :param object_size: Given object size. Alternative if no object name is given.
        :param root_link: Current root Link
        :param tip_link: Current tip link
        :param velocity: Desired velocity of this goal
        :param weight: weight of this goal
        """
        if name is None:
            name = "Reaching"

        super().__init__(name=name)

        if root_link is None:
            root_link = god_map.world.groups[
                god_map.world.robot_name
            ].root_link.name.short_name
        if tip_link is None:
            tip_link = self.gripper_tool_frame

        self.grasp = grasp
        self.align = align
        self.object_name = object_name
        self.object_shape = object_shape
        self.root_link_name = root_link
        self.tip_link_name = tip_link
        self.velocity = velocity
        self.weight = weight
        self.offsets = cas.Vector3.from_xyz(0, 0, 0)
        self.careful = False
        self.object_in_world = goal_pose is None

        # Get object geometry from name
        if goal_pose is None:
            self.goal_pose, self.object_size = self.get_object_by_name(self.object_name)
            self.reference_frame = self.object_name

        else:
            try:
                god_map.world.search_for_link_name(goal_pose.reference_frame)
                self.goal_pose = goal_pose
            except:
                get_middleware().logwarn(
                    f"Couldn't find {goal_pose.reference_frame}. Searching in tf."
                )
                self.goal_pose = god_map.world.compute_fk(
                    god_map.world.search_for_link_name("map"), goal_pose
                )

            self.object_size = object_size
            self.reference_frame = "base_footprint"
            get_middleware().logwarn(f"Warning: Object not in giskard world")

        if self.object_shape == "sphere" or self.object_shape == "cylinder":
            self.offsets = cas.Vector3.from_xyz(
                x=self.object_size.x, y=self.object_size.y, z=self.object_size.z
            )

        # TODO: fine tune and add correct object names
        # FIXME: add Parameter instead of hard-coded values
        elif self.object_name == "Plate":
            self.offsets = cas.Vector3.from_xyz(
                x=-(self.object_size.x / 2) + 0.03,
                y=self.object_size.y,
                z=self.object_size.z,
            )

        elif self.object_name == "Bowl":
            self.offsets = cas.Vector3.from_xyz(
                x=-(self.object_size.x / 2) + 0.15,
                y=self.object_size.y,
                z=self.object_size.z,
            )

        elif self.object_name == "Cutlery":
            self.offsets = cas.Vector3.from_xyz(
                x=-(self.object_size.x / 2) + 0.02,
                y=self.object_size.y,
                z=self.object_size.z,
            )

        # TODO: Test Tray on real robot
        elif self.object_name == ObjectTypes.OT_Tray.value:
            self.offsets = cas.Vector3.from_xyz(
                x=-(self.object_size.x / 3),
                y=self.object_size.y,
                z=self.object_size.z - 0.05,
            )

        else:
            if self.object_in_world:
                self.offsets = cas.Vector3.from_xyz(
                    -self.object_size.x / 2,
                    self.object_size.y / 2,
                    self.object_size.z / 2,
                )
            else:
                self.offsets = cas.Vector3.from_xyz(
                    max(min(0.08, self.object_size.x.to_np() / 2), 0.05), 0, 0
                )

        if all(self.grasp != member.value for member in GraspTypes):
            raise Exception(f"Unknown grasp value: {grasp}")

        go = GraspObject(
            goal_pose=self.goal_pose,
            reference_frame_alignment=self.reference_frame,
            offsets=self.offsets,
            grasp=self.grasp,
            align=self.align,
            root_link=self.root_link_name,
            tip_link=self.tip_link_name,
            velocity=self.velocity,
            weight=self.weight,
            name="GraspObject Reaching",
        )
        go.start_condition = self.start_condition
        go.pause_condition = self.pause_condition
        go.end_condition = self.end_condition
        self.add_goal(go)


class GraspObject(ObjectGoal):
    def __init__(
        self,
        goal_pose: cas.TransMatrix,
        align: str,
        grasp: str,
        offsets: cas.Vector3 = cas.Vector3.from_xyz(0, 0, 0),
        name: Optional[str] = None,
        reference_frame_alignment: Optional[str] = None,
        root_link: Optional[PrefixName] = None,
        tip_link: Optional[PrefixName] = None,
        velocity: float = 0.2,
        weight: float = WEIGHT_ABOVE_CA,
    ):
        """
        Concludes Reaching type goals.
        Executes them depending on the given context action.
        Context is a dictionary in an action is given as well as situational parameters.
        All available context Messages are found in the Enum 'ContextTypes'

        :param goal_pose: Goal pose for the object.
        :param align: States if the gripper should be rotated and in which "direction"
        :param offsets: Optional parameter to pass a specific offset in x, y or z direction
        :param grasp: The Direction from with an object should be grasped
        :param name: Name of the executed goal, in this case GraspObject
        :param reference_frame_alignment: Reference frame to align with. Is usually either an object link or 'base_footprint'
        :param root_link: Current root Link
        :param tip_link: Current tip link
        :param velocity: Desired velocity of this goal
        :param weight: weight of this goal
        """
        if name is None:
            name = "GraspObject"

        super().__init__(name=name)
        self.goal_pose = goal_pose

        self.offsets = offsets
        self.grasp = grasp
        self.align = align

        if reference_frame_alignment is None:
            reference_frame_alignment = "base_footprint"

        if root_link is None:
            root_link = god_map.world.groups[god_map.world.robot_name].root_link.name

        if tip_link is None:
            tip_link = self.gripper_tool_frame

        self.reference_link = god_map.world.search_for_link_name(
            reference_frame_alignment
        )
        self.root_link = god_map.world.search_for_link_name(root_link)
        self.tip_link = god_map.world.search_for_link_name(tip_link)

        self.velocity = velocity
        self.weight = weight

        self.goal_frontal_axis = cas.Vector3()
        self.goal_frontal_axis.reference_frame = self.reference_link

        self.tip_frontal_axis = cas.Vector3()
        self.tip_frontal_axis.reference_frame = self.tip_link

        self.goal_vertical_axis = cas.Vector3()
        self.goal_vertical_axis.reference_frame = self.reference_link

        self.tip_vertical_axis = cas.Vector3()
        self.tip_vertical_axis.reference_frame = self.tip_link

        root_goal_point = self.goal_pose.to_position()

        self.goal_point = god_map.world.transform(self.reference_link, root_goal_point)

        if self.grasp == GraspTypes.ABOVE.value:
            self.goal_vertical_axis.x = self.standard_forward.x
            self.goal_vertical_axis.y = self.standard_forward.y
            self.goal_vertical_axis.z = self.standard_forward.z
            v = multiply_vector(self.standard_up, -1)
            self.goal_frontal_axis.x = v.x
            self.goal_frontal_axis.y = v.y
            self.goal_frontal_axis.z = v.z
            self.goal_point.z += self.offsets.z

        elif self.grasp == GraspTypes.BELOW.value:
            v = multiply_vector(self.standard_forward, -1)
            self.goal_vertical_axis.x = v.x
            self.goal_vertical_axis.y = v.y
            self.goal_vertical_axis.z = v.z
            self.goal_frontal_axis.x = self.standard_up.x
            self.goal_frontal_axis.y = self.standard_up.y
            self.goal_frontal_axis.z = self.standard_up.z
            self.goal_point.z -= self.offsets.z

        elif self.grasp == GraspTypes.FRONT.value:
            # v = cas.Vector3.from_xyz(0, 2, 0)
            self.goal_vertical_axis.x = self.standard_up.x
            self.goal_vertical_axis.y = self.standard_up.y  # v.y
            self.goal_vertical_axis.z = self.standard_up.z
            self.goal_frontal_axis.x = self.standard_forward.x  # self.standard_up.x
            self.goal_frontal_axis.y = self.standard_forward.y  # self.standard_up.y
            self.goal_frontal_axis.z = self.standard_forward.z  # self.standard_up.z

            self.goal_point.z -= 0.01
        # self.goal_point.x += self.offsets.x

        elif self.grasp == GraspTypes.LEFT.value:
            self.goal_vertical_axis.x = self.standard_up.x
            self.goal_vertical_axis.y = self.standard_up.y
            self.goal_vertical_axis.z = self.standard_up.z
            self.goal_frontal_axis.x = self.gripper_left.x
            self.goal_frontal_axis.y = self.gripper_left.y
            self.goal_frontal_axis.z = self.gripper_left.z

        elif self.grasp == GraspTypes.RIGHT.value:
            v = multiply_vector(self.gripper_left, -1)
            self.goal_vertical_axis.x = self.standard_up.x
            self.goal_vertical_axis.y = self.standard_up.y
            self.goal_vertical_axis.z = self.standard_up.z
            self.goal_frontal_axis.x = v.x
            self.goal_frontal_axis.y = v.y
            self.goal_frontal_axis.z = v.z

        if self.align == "vertical":
            self.tip_vertical_axis.x = self.gripper_left.x
            self.tip_vertical_axis.y = self.gripper_left.y
            self.tip_vertical_axis.z = self.gripper_left.z

        else:
            self.tip_vertical_axis.x = self.gripper_up.x
            self.tip_vertical_axis.y = self.gripper_up.y
            self.tip_vertical_axis.z = self.gripper_up.z

        self.tip_frontal_axis.x = self.gripper_forward.x
        self.tip_frontal_axis.y = self.gripper_forward.y
        self.tip_frontal_axis.z = self.gripper_forward.z

        # Position
        cp = CartesianPosition(
            root_link=self.root_link,
            tip_link=self.tip_link,
            goal_point=self.goal_point,
            reference_velocity=self.velocity,
            weight=self.weight,
        )
        cp.start_condition = self.start_condition
        cp.pause_condition = self.pause_condition
        cp.end_condition = self.end_condition
        self.add_task(cp)

        # Align vertical
        apv = AlignPlanes(
            name="APlanesVertical",
            root_link=self.root_link,
            tip_link=self.tip_link,
            goal_normal=self.goal_vertical_axis,
            tip_normal=self.tip_vertical_axis,
            reference_velocity=self.velocity,
            weight=self.weight,
        )
        apv.start_condition = self.start_condition
        apv.pause_condition = self.pause_condition
        apv.end_condition = self.end_condition
        self.add_task(apv)

        # Align frontal
        aph = AlignPlanes(
            name="APlanesFrontal",
            root_link=self.root_link,
            tip_link=self.tip_link,
            goal_normal=self.goal_frontal_axis,
            tip_normal=self.tip_frontal_axis,
            reference_velocity=self.velocity,
            weight=self.weight,
        )
        aph.start_condition = self.start_condition
        aph.pause_condition = self.pause_condition
        aph.end_condition = self.end_condition
        self.add_task(aph)

        kr = KeepRotationGoal(tip_link="base_footprint", weight=self.weight)
        kr.start_condition = self.start_condition
        kr.pause_condition = self.pause_condition
        kr.end_condition = self.end_condition
        self.add_constraints_of_goal(kr)


class VerticalMotion(ObjectGoal):
    def __init__(
        self,
        action: str = None,
        name: str = None,
        distance: float = 0.02,
        root_link: Optional[str] = None,
        tip_link: Optional[str] = None,
        velocity: float = 0.2,
        weight: float = WEIGHT_ABOVE_CA,
    ):
        """
        Move the tip link vertical according to the given context.

        :param action: Action to take.
        :param name: Name of the goal, in this case VerticalMotion.
        :param distance: Optional parameter to adjust the distance to move.
        :param root_link: Current root Link
        :param tip_link: Current tip link
        :param velocity: Desired velocity of this goal
        :param weight: weight of this goal
        """
        if name is None:
            name = "VerticalMotion"

        super().__init__(name=name)

        if root_link is None:
            root_link = "base_footprint"
        if tip_link is None:
            tip_link = self.gripper_tool_frame
        self.distance = distance
        self.root_link = god_map.world.search_for_link_name(root_link)
        self.tip_link = god_map.world.search_for_link_name(tip_link)
        self.velocity = velocity
        self.weight = weight
        self.base_footprint = god_map.world.search_for_link_name("base_footprint")
        self.action = action

        start_point_tip = cas.TransMatrix()
        start_point_tip.reference_frame = self.tip_link
        goal_point_base = god_map.world.transform(self.base_footprint, start_point_tip)

        up = ContextActionModes.grasping.value in self.action
        down = ContextActionModes.placing.value in self.action
        if up:
            goal_point_base.z += self.distance
        elif down:
            goal_point_base.z -= self.distance
        else:
            get_middleware().logwarn("no direction given")

        krg = KeepRotationGoal(tip_link=self.tip_link.short_name, weight=self.weight)
        krg.start_condition = self.start_condition
        krg.pause_condition = self.pause_condition
        krg.end_condition = self.end_condition
        self.add_constraints_of_goal(krg)

        goal_point_tip = god_map.world.transform(self.tip_link, goal_point_base)
        self.goal_point = deepcopy(goal_point_tip)
        # self.root_T_tip_start = god_map.world.compute_fk_np(self.root_link, self.tip_link)
        # self.start_tip_T_current_tip = np.eye(4)

        # start_tip_T_current_tip = w.TransMatrix(self.get_parameter_as_symbolic_expression('start_tip_T_current_tip'))
        root_T_tip = god_map.world.compose_fk_expression(self.root_link, self.tip_link)

        # t_T_g = w.TransMatrix(self.goal_point)
        # r_T_tip_eval = w.TransMatrix(god_map.evaluate_expr(root_T_tip))

        # root_T_goal = r_T_tip_eval.dot(start_tip_T_current_tip).dot(t_T_g)

        root_T_goal = god_map.world.transform(self.root_link, self.goal_point)

        r_P_g = root_T_goal.to_position()
        r_P_c = root_T_tip.to_position()

        task = Task(name="VerticalMotion")
        self.add_task(task)

        task.add_point_goal_constraints(
            frame_P_goal=r_P_g,
            frame_P_current=r_P_c,
            reference_velocity=self.velocity,
            weight=self.weight,
        )


class Retracting(ObjectGoal):
    def __init__(
        self,
        name: str = None,
        distance: float = 0.3,
        reference_frame: Optional[str] = None,
        root_link: Optional[str] = None,
        tip_link: Optional[str] = None,
        velocity: float = 0.2,
        threshold: float = 0.01,
        weight: float = WEIGHT_ABOVE_CA,
    ):
        """
        Retract the tip link from the current position by the given distance.
        The exact direction is based on the given reference frame.

        :param name: Name of the goal, in this case Retracting.
        :param distance: Optional parameter to adjust the distance to move.
        :param reference_frame: Reference axis from which should be retracted. Is usually 'base_footprint' or 'hand_palm_link'
        :param root_link: Current root Link
        :param tip_link: Current tip link
        :param velocity: Desired velocity of this goal
        :param weight: weight of this goal

        """
        if name is None:
            name = "Retracting"

        super().__init__(name=name)

        if reference_frame is None:
            reference_frame = "base_footprint"
        if root_link is None:
            root_link = god_map.world.groups[god_map.world.robot_name].root_link.name
        if tip_link is None:
            tip_link = self.gripper_tool_frame
        self.distance = distance
        self.reference_frame = god_map.world.search_for_link_name(reference_frame)
        self.root_link = god_map.world.search_for_link_name(root_link)
        self.tip_link = god_map.world.search_for_link_name(tip_link)
        self.velocity = velocity
        self.weight = weight
        self.hand_frames = [self.gripper_tool_frame, "hand_palm_link"]

        tip_P_start = cas.TransMatrix()
        tip_P_start.reference_frame = self.tip_link
        reference_P_start = god_map.world.transform(self.reference_frame, tip_P_start)

        if self.reference_frame.short_name in self.hand_frames:
            reference_P_start.z -= self.distance
        else:
            reference_P_start.x -= self.distance

        self.goal_point = god_map.world.transform(self.tip_link, reference_P_start)
        # self.root_T_tip_start = god_map.world.compute_fk_np(self.root_link, self.tip_link)
        # self.start_tip_T_current_tip = np.eye(4)

        krg = KeepRotationGoal(
            tip_link="base_footprint", weight=self.weight, name="Retracting KRG"
        )
        krg.start_condition = self.start_condition
        krg.pause_condition = self.pause_condition
        krg.end_condition = self.end_condition
        self.add_goal(krg)

        if "base" not in self.tip_link.short_name:
            krg2 = KeepRotationGoal(
                tip_link=self.tip_link.short_name,
                weight=self.weight,
                name="Tip Link KRG",
            )
            krg2.start_condition = self.start_condition
            krg2.pause_condition = self.pause_condition
            krg2.end_condition = self.end_condition
            self.add_goal(krg2)

        task = Task(name="Retracting")
        self.add_task(task)

        # start_tip_T_current_tip = w.TransMatrix(self.get_parameter_as_symbolic_expression('start_tip_T_current_tip'))
        root_T_tip = god_map.world.compose_fk_expression(self.root_link, self.tip_link)

        # t_T_g = w.TransMatrix(self.goal_point)
        # r_T_tip_eval = w.TransMatrix(god_map.evaluate_expr(root_T_tip))

        # root_T_goal = r_T_tip_eval.dot(start_tip_T_current_tip).dot(t_T_g)

        root_T_goal = god_map.world.transform(self.root_link, self.goal_point)

        r_P_g = root_T_goal.to_position()
        r_P_c = root_T_tip.to_position()

        task.add_point_goal_constraints(
            frame_P_goal=r_P_g,
            frame_P_current=r_P_c,
            reference_velocity=self.velocity,
            weight=self.weight,
        )

        distance = cas.euclidean_distance(r_P_g, r_P_c)
        self.observation_expression = cas.less_equal(distance, threshold)


class Placing(ObjectGoal):

    def __init__(
        self,
        goal_pose: cas.TransMatrix,
        align: str,
        grasp: str,
        name: str = None,
        root_link: Optional[str] = None,
        tip_link: Optional[str] = None,
        velocity: float = 0.02,
        weight: float = WEIGHT_ABOVE_CA,
    ):
        """
        Place an object. Use monitor_placing in python_interface.py in
         case of using the force-/torque-sensor to place objects.

        :param goal_pose: Goal pose for the object.
        :param align: States if the gripper should be rotated and in which "direction"
        :param name: Name of the goal, in this case Placing
        :param root_link: Current root Link
        :param tip_link: Current tip link
        :param velocity: Desired velocity of this goal
        :param weight: weight of this goal
        """
        if name is None:
            name = "Placing"

        self.goal_pose = goal_pose
        self.velocity = velocity
        self.weight = weight
        self.align = align
        self.grasp = grasp

        # self.from_above = check_context_element('from_above', ContextFromAbove, context)

        super().__init__(name=name)

        if root_link is None:
            root_link = "base_footprint"

        if tip_link is None:
            tip_link = self.gripper_tool_frame

        self.root_link = god_map.world.search_for_link_name(root_link)
        self.tip_link = god_map.world.search_for_link_name(tip_link)

        go = GraspObject(
            goal_pose=self.goal_pose,
            align=self.align,
            grasp=self.grasp,
            root_link=self.root_link.short_name,
            tip_link=self.tip_link.short_name,
            velocity=self.velocity,
            weight=self.weight,
        )
        go.start_condition = self.start_condition
        go.pause_condition = self.pause_condition
        go.end_condition = self.pause_condition
        self.add_goal(go)

    # might need to be removed in the future, as soon as the old interface isn't in use anymore

    # def recovery_modifier(self) -> Dict:
    #     joint_states = {'arm_lift_joint': 0.03}
    #
    #     return joint_states


class TakePose(Goal):
    """
    Get into a predefined pose with a given keyword.
    Used to get into complete poses. To move only specific joints use 'JointPositionList'

    :param pose_keyword: Keyword for the given poses
    """

    pose_keyword: str = field(kw_only=True)
    "Keyword for the given poses"
    name: str = field(default=None, kw_only=True)

    def expand(self, context: BuildContext) -> None:
        if self.name is None:
            self.name = f"TakePose-{self.pose_keyword}"

        if self.pose_keyword == TakePoseTypes.PARK.value:
            head_pan_joint = 0.0
            head_tilt_joint = 0.0
            arm_lift_joint = 0.0
            arm_flex_joint = 0.0
            arm_roll_joint = -1.5
            wrist_flex_joint = -1.5
            wrist_roll_joint = 0.0

        elif self.pose_keyword == TakePoseTypes.PARK_LEFT.value:
            head_pan_joint = 0.0
            head_tilt_joint = 0.0
            arm_lift_joint = 0.0
            arm_flex_joint = 0.0
            arm_roll_joint = 1.5
            wrist_flex_joint = -1.5
            wrist_roll_joint = 0.0

        elif self.pose_keyword == TakePoseTypes.PERCEIVE.value:
            head_pan_joint = 0.0
            head_tilt_joint = -0.65
            arm_lift_joint = 0.25
            arm_flex_joint = 0.0
            arm_roll_joint = 1.5
            wrist_flex_joint = -1.5
            wrist_roll_joint = 0.0

        elif self.pose_keyword == TakePoseTypes.ASSISTANCE.value:
            head_pan_joint = 0.0
            head_tilt_joint = 0.0
            arm_lift_joint = 0.0
            arm_flex_joint = 0.0
            arm_roll_joint = -1.5
            wrist_flex_joint = -1.5
            wrist_roll_joint = 1.6

        elif self.pose_keyword == TakePoseTypes.PRE_ALIGN_HEIGHT.value:
            head_pan_joint = 0.0
            head_tilt_joint = 0.0
            arm_lift_joint = 0.0
            arm_flex_joint = 0.0
            arm_roll_joint = 0.0
            wrist_flex_joint = -1.5
            wrist_roll_joint = 0.0

        elif self.pose_keyword == TakePoseTypes.CARRY.value:
            head_pan_joint = 0.0
            head_tilt_joint = -0.65
            arm_lift_joint = 0.0
            arm_flex_joint = -0.43
            arm_roll_joint = 0.0
            wrist_flex_joint = -1.17
            wrist_roll_joint = -1.62

        elif self.pose_keyword == TakePoseTypes.TEST.value:
            head_pan_joint = 0.0
            head_tilt_joint = 0.0
            arm_lift_joint = 0.38
            arm_flex_joint = -1.44
            arm_roll_joint = 0.0
            wrist_flex_joint = -0.19
            wrist_roll_joint = 0.0

        elif self.pose_keyword == TakePoseTypes.PRE_TRAY.value:
            head_pan_joint = 0.0
            head_tilt_joint = 0.0
            arm_lift_joint = 0.6
            arm_flex_joint = -1.44
            arm_roll_joint = 0.0
            wrist_flex_joint = -1.22
            wrist_roll_joint = np.pi / 2

        else:
            get_middleware().loginfo(f"{self.pose_keyword} is not a valid pose")
            return

        joint_states = {
            "head_pan_joint": head_pan_joint,
            "head_tilt_joint": head_tilt_joint,
            "arm_lift_joint": arm_lift_joint,
            "arm_flex_joint": arm_flex_joint,
            "arm_roll_joint": arm_roll_joint,
            "wrist_flex_joint": wrist_flex_joint,
            "wrist_roll_joint": wrist_roll_joint,
        }

        joint_states = JointState.from_str_dict(joint_states, context.world)
        self.add_node(JointPositionList(goal_state=joint_states, name="TakePoseJPL"))

    def build(self, context: BuildContext) -> NodeArtifacts:
        return NodeArtifacts(
            observation=trinary_logic_and(
                *[node.observation_variable for node in self.nodes]
            )
        )


# class Mixing(Goal):
#     def __init__(self,
#                  name=None,
#                  mixing_time: float = 20,
#                  weight: float = WEIGHT_ABOVE_CA):
#         """
#         Simple Mixing motion.
#
#         :param mixing_time: States how long this goal should be executed.
#         :param weight: weight of this goal
#         """
#         if name is None:
#             name = 'Mixing'
#         super().__init__(name=name)
#
#         self.weight = weight
#
#         target_speed = 1
#
#         self.add_constraints_of_goal(JointRotationGoalContinuous(joint_name='wrist_roll_joint',
#                                                                  joint_center=0.0,
#                                                                  joint_range=0.9,
#                                                                  trajectory_length=mixing_time,
#                                                                  target_speed=target_speed))
#
#         self.add_constraints_of_goal(JointRotationGoalContinuous(joint_name='wrist_flex_joint',
#                                                                  joint_center=-1.3,
#                                                                  joint_range=0.2,
#                                                                  trajectory_length=mixing_time,
#                                                                  target_speed=target_speed))
#
#         self.add_constraints_of_goal(JointRotationGoalContinuous(joint_name='arm_roll_joint',
#                                                                  joint_center=0.0,
#                                                                  joint_range=0.1,
#                                                                  trajectory_length=mixing_time,
#                                                                  target_speed=target_speed))


class JointRotationGoalContinuous(Goal):
    def __init__(
        self,
        joint_name: str,
        joint_center: float,
        joint_range: float,
        name: str = None,
        trajectory_length: float = 20,
        target_speed: float = 1,
        period_length: float = 1.0,
    ):
        """
        Rotate a joint continuously around a center. The execution time and speed is variable.

        :param joint_name: joint name that should be rotated
        :param joint_center: Center of the rotation point
        :param joint_range: Range of the rotational movement. Note that this is calculated + and - joint_center.
        :param trajectory_length: length of this goal in seconds.
        :param target_speed: execution speed of this goal. Adjust when the trajectory is not executed right
        :param period_length: length of the period that should be executed. Adjust when the trajectory is not executed right.
        """
        if name is None:
            name = "JointRotationGoalContinuous"
        super().__init__(name=name)
        self.joint = god_map.world.search_for_joint_name(joint_name)
        self.target_speed = target_speed
        self.trajectory_length = trajectory_length
        self.joint_center = joint_center
        self.joint_range = joint_range
        self.period_length = period_length

    def make_constraints(self):
        time = self.traj_time_in_seconds()
        joint_position = self.get_joint_position_symbol(self.joint)

        joint_goal = self.joint_center + (
            w.cos(time * np.pi * self.period_length) * self.joint_range
        )

        god_map.debug_expression_manager.add_debug_expression(
            f"{self.joint.short_name}_goal", joint_goal
        )
        god_map.debug_expression_manager.add_debug_expression(
            f"{self.joint.short_name}_position", joint_position
        )

        self.add_position_constraint(
            expr_current=joint_position,
            expr_goal=joint_goal,
            reference_velocity=self.target_speed,
            weight=w.if_greater(time, self.trajectory_length, 0, WEIGHT_ABOVE_CA),
            name=self.joint.short_name,
        )


class KeepRotationGoal(Goal):
    def __init__(
        self, tip_link: str, name: str = None, weight: float = WEIGHT_ABOVE_CA
    ):
        """
        Use this if a specific link should not rotate during a goal execution. Typically used for the hand.

        :param tip_link: link that shall keep its rotation
        :param weight: weight of this goal
        """
        if name is None:
            name = "KeepRotationGoal"

        super().__init__(name=name)

        self.tip_link = god_map.world.search_for_link_name(tip_link)
        self.weight = weight

        tip_orientation = cas.RotationMatrix()
        tip_orientation.reference_frame = self.tip_link

        co = CartesianOrientation(
            root_link=PrefixName("map"),
            tip_link=self.tip_link,
            goal_orientation=tip_orientation,
            weight=self.weight,
            name=f"CartesianOrientation of {self.name}",
        )
        self.add_task(co)


def check_context_element(name: str, context_type, context):
    if name in context:
        if isinstance(context[name], context_type):
            return context[name].content
        else:
            return context[name]


def multiply_vector(vec: Vector3, number: int):
    return Vector3(
        x=vec.x * number,
        y=vec.y * number,
        z=vec.z * number,
        reference_frame=vec.reference_frame,
    )
