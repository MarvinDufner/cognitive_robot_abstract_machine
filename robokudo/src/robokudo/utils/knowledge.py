"""Legacy object-knowledge utilities for Robokudo.

This module provides utility functions for working with object knowledge bases
and transforming object representations. It supports:

* Transform matrix generation from object knowledge
* Bounding box calculations
* Object knowledge base loading
* Oriented bounding box generation for objects and their parts

The module integrates with:
* Open3D for geometric operations
* NumPy for numerical computations
* Legacy object knowledge definitions
"""

from __future__ import annotations

import numpy
import open3d as o3d
from typing_extensions import TYPE_CHECKING, Tuple, Dict

from robokudo.utils.module_loader import ModuleLoader
from robokudo.utils.o3d_helper import get_obb_from_size_and_transform
from robokudo.utils.transform import (
    get_rotation_matrix_from_euler_angles,
    get_quaternion_from_rotation_matrix,
    get_transform_matrix_from_q,
)

if TYPE_CHECKING:
    import numpy.typing as npt
    from ..world_descriptor import ObjectKnowledge, BaseWorldDescriptor
    from ..annotators.core import BaseAnnotator


def get_quaternion_from_rotation_information(
    ok: ObjectKnowledge,
) -> Tuple[float, float, float, float]:
    """
    Return a quaternion based on the rotation in ObjectKnowledge, and its type of rotation.
    If the pose type is EULER, then it will be converted to Quaternion beforehand.

    :param ok: Object knowledge containing pose information
    :return: [x, y, z, w] rotation quaternion
    """
    if ok.pose_type == ok.PoseType.EULER:
        rot_matrix = get_rotation_matrix_from_euler_angles(
            ok.orientation_x, ok.orientation_x, ok.orientation_z
        )
        return tuple(get_quaternion_from_rotation_matrix(rot_matrix))
    elif ok.pose_type == ok.PoseType.QUATERNION:
        # Interpret values directly as a quaternion
        return ok.orientation_x, ok.orientation_y, ok.orientation_z, ok.orientation_w

    raise Exception("Unknown pose type")


def get_transform_matrix_from_object_knowledge(
    ok: ObjectKnowledge,
) -> npt.NDArray:
    """Extract transform matrix from object knowledge.

    Creates a 4x4 transformation matrix from the object's position and orientation.

    :param ok: Object knowledge containing pose information
    :return: 4x4 transformation matrix
    """
    quaternion = get_quaternion_from_rotation_information(ok)

    return get_transform_matrix_from_q(
        quaternion=quaternion, translation=[ok.position_x, ok.position_y, ok.position_z]
    )


def get_bb_size_from_object_knowledge(
    ok: ObjectKnowledge,
) -> npt.NDArray:
    """Extract bounding box dimensions from object knowledge.

    :param ok: Object knowledge containing size information
    :return: Array of [x_size, y_size, z_size]
    """
    return numpy.array([ok.x_size, ok.y_size, ok.z_size])


def load_world_descriptor(
    annotator: BaseAnnotator,
) -> BaseWorldDescriptor:
    """Load world descriptor from annotator parameters.

    :param annotator: Annotator containing world descriptor parameters
    :return: Loaded world descriptor
    """
    loader = ModuleLoader()
    return loader.load_world_descriptor(
        annotator.descriptor.parameters.world_descriptor_ros_package,
        annotator.descriptor.parameters.world_descriptor_name,
    )


def get_obb_for_object_and_transform(
    object_knowledge: ObjectKnowledge,
    transform_matrix: npt.NDArray,
) -> o3d.geometry.OrientedBoundingBox:
    """Get an OBB only for THIS object and not any descendants.

    :param object_knowledge:
    :param transform_matrix: Transformation matrix describing the position of the object.
                      This is usually used for objects that have been perceived and do not have fixed
                      Position/Orientation information.
    :return: A OrientedBoundingbox
    """
    bb_size = numpy.array(
        [object_knowledge.x_size, object_knowledge.y_size, object_knowledge.z_size]
    )
    return get_obb_from_size_and_transform(bb_size, transform_matrix)


def get_obb_for_child_object_and_transform(
    object_knowledge: ObjectKnowledge,
    parent_transform: npt.NDArray,
) -> o3d.geometry.OrientedBoundingBox:
    """Create oriented bounding box for child object.

    Creates an oriented bounding box for a child object by combining its local
    transform with the parent's transform.

    :param object_knowledge: Knowledge about the child object
    :param parent_transform: 4x4 transformation matrix of parent object
    :return: Oriented bounding box for child object
    """
    object_transform = get_transform_matrix_from_object_knowledge(object_knowledge)
    object_transform = parent_transform @ object_transform
    object_bb_size = get_bb_size_from_object_knowledge(object_knowledge)
    object_bb = get_obb_from_size_and_transform(object_bb_size, object_transform)
    object_bb.color = [1.0, 0, 0]

    return object_bb


def get_obbs_for_object_and_childs(
    object_knowledge: ObjectKnowledge,
    transform_matrix: npt.NDArray,
) -> Dict[str, o3d.geometry.OrientedBoundingBox]:
    """Create oriented bounding boxes for object and all children.

    Creates oriented bounding boxes for:
    * Main object
    * Component parts
    * Feature parts

    :param object_knowledge: Knowledge about main object and its parts
    :param transform_matrix: 4x4 transformation matrix for main object
    :return: Dictionary mapping object names to bounding boxes
    """
    obbs = {
        object_knowledge.name: get_obb_for_object_and_transform(
            object_knowledge, transform_matrix
        )
    }

    for component in object_knowledge.components:
        obbs[component.name] = get_obb_for_child_object_and_transform(
            component, transform_matrix
        )

    for feature in object_knowledge.features:
        obbs[feature.name] = get_obb_for_child_object_and_transform(
            feature, transform_matrix
        )

    return obbs
