"""Semantic map utilities for Robokudo.

This module provides functions for working with semantic map regions and their
transformations to different coordinate frames. It supports:

* Converting semantic map regions to oriented bounding boxes
* Transforming regions between world and camera coordinates
* Applying arbitrary transforms to regions

"""

raise RuntimeError(
    "utils/semantic_map.py has been replaced by utils/region.py and world descriptors."
)

from __future__ import annotations

import numpy as np
import open3d as o3d
from typing_extensions import TYPE_CHECKING

from . import transform

if TYPE_CHECKING:
    from robokudo.semantic_map import SemanticMapEntry
    import numpy.typing as npt


def get_obb_from_semantic_map_region(
    region: SemanticMapEntry,
) -> o3d.geometry.OrientedBoundingBox:
    """Create oriented bounding box from semantic map region.

    Instantiate an Open3D OrientedBoundingBox that represents a Semantic Map region.
    It can be used to crop pointclouds. It doesn't carry any frame information, so
    make sure to translate/rotate it to the desired coordinate frame if necessary.

    .. note::
       The returned box does not carry frame information. Transform it to the
       desired coordinate frame if needed.

    :param region: A semantic map entry that defines the region of interest in 3D
    :return: The bounding box with the pose and extent defined by region
    """
    obb = o3d.geometry.OrientedBoundingBox(
        center=np.array([region.position_x, region.position_y, region.position_z]),
        R=transform.get_rotation_matrix_from_q(
            np.array(
                [
                    region.orientation_x,
                    region.orientation_y,
                    region.orientation_z,
                    region.orientation_w,
                ]
            )
        ),
        extent=np.array([region.x_size, region.y_size, region.z_size]),
    )
    return obb


def get_obb_from_semantic_map_region_in_cam_coordinates(
    region: SemanticMapEntry,
    world_frame_name: str,
    world_to_cam_transform_matrix: npt.NDArray,
) -> o3d.geometry.OrientedBoundingBox:
    """Transform semantic map region to camera coordinates.

    Creates oriented bounding box and transforms it to camera frame if needed.

    :param region: Semantic map region definition
    :param world_frame_name: Name of world coordinate frame
    :param world_to_cam_transform_matrix: 4x4 world-to-camera transform
    :return: Oriented bounding box in camera coordinates

    .. note::
       If region is not in world frame, assumes region frame matches camera frame.
       TODO: Add proper SemDT/TF-based transform for other frames.
    """
    # If the Semantic Map Region is in camera coordinates, we can keep the OBB as-is.
    # Otherwise, check if and how we shall transform it
    if region.frame_id == world_frame_name:
        obb = get_obb_from_semantic_map_region(region)
        # Use the cam to world transform in the CAS.
        # The benefit is, that this transform can also be recorded for our stored percepts in mongo
        # and therefore allows using the region filter even without having live tf data
        obb.rotate(
            transform.get_rotation_from_transform_matrix(world_to_cam_transform_matrix),
            center=(0, 0, 0),
        )
        obb.translate(
            transform.get_translation_from_transform_matrix(
                world_to_cam_transform_matrix
            )
        )

    else:
        # TODO Handle non-empty frames by using SDT/TF to transform them properly
        obb = get_obb_from_semantic_map_region(region)

    return obb


def get_obb_from_semantic_map_region_with_transform_matrix(
    region: SemanticMapEntry, transform_matrix: npt.NDArray
) -> o3d.geometry.OrientedBoundingBox:
    """Transform semantic map region by arbitrary transform.

    Creates oriented bounding box and applies provided transform.

    :param region: Semantic map region definition
    :param transform_matrix: 4x4 transform matrix to apply
    :return: Transformed oriented bounding box
    """
    obb = get_obb_from_semantic_map_region(region)
    obb.rotate(
        transform.get_rotation_from_transform_matrix(transform_matrix),
        center=(0, 0, 0),
    )
    obb.translate(transform.get_translation_from_transform_matrix(transform_matrix))

    return obb
