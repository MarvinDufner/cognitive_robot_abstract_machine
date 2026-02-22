"""
General methods to access the current World.
Reasoning about alternate world states is done in the corresponding Annotators.
"""
import sys

import semantic_digital_twin.world
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.world_description.connections import Connection6DoF

# Module-level singleton-like variables
this = sys.modules[__name__]
this.world = None


def world_instance() -> semantic_digital_twin.world.World:
    """
    A singleton-like World instance.

    :return: The world state which is the current belief state.
    :rtype: semantic_digital_twin.world.World
    """
    if this.world is None:
        this.world = semantic_digital_twin.world.World()

        # Setup of this world is currently the responsibility of the other nodes, loaded URDF
        # and/or camera interface.

    return this.world


def world_has_body_by_name(world: semantic_digital_twin.world.World, body_name: str):
    bodies = world.get_bodies_by_name(name=body_name)
    return len(bodies) > 0


# def add_dummy_frame_if_non_existent(frame_name: str) -> None:
#     if not world_has_body_by_name(world=world_instance(), body_name=frame_name):
#         with world_instance().modify_world():
#             world_instance().add_body(
#                 semantic_digital_twin.world.Body(name=PrefixedName(name=frame_name)))


def setup_world_for_camera_frame(world_frame: str, camera_frame: str):
    world_exists = world_has_body_by_name(world=world_instance(), body_name=world_frame)
    camera_exists = world_has_body_by_name(world=world_instance(), body_name=camera_frame)

    if world_exists and camera_exists:
        return

    if not world_exists and not camera_exists:
        with world_instance().modify_world():
            world_body = semantic_digital_twin.world.Body(name=PrefixedName(name=world_frame))
            camera_body = semantic_digital_twin.world.Body(name=PrefixedName(name=camera_frame))
            world_c_camera = Connection6DoF.create_with_dofs(parent=world_body, child=camera_body,
                                                             world=world_instance())
            world_instance().add_connection(world_c_camera)

        return

    raise AssertionError(f"This method can currently only be called when neither the world or camera frame exist. "
                         f"Existence of camera frame: {camera_exists}, world frame: {world_exists}.")
