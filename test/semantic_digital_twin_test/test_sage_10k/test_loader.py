import numpy as np

from semantic_digital_twin.adapters.ros.visualization.viz_marker import (
    VizMarkerPublisher,
)
from semantic_digital_twin.adapters.sage_10k_dataset.loader import Sage10kDatasetLoader


def test_loader(rclpy_node):
    loader = Sage10kDatasetLoader()
    scene = loader.create_scene(
        scene_url="https://huggingface.co/datasets/nvidia/SAGE-10k/resolve/main/scenes/20251213_020526_layout_84b703fb.zip",
    )

    world = scene.create_world()
    pub = VizMarkerPublisher(
        _world=world,
        node=rclpy_node,
    )
    pub.with_tf_publisher()

    # check that the positions of the objects in the scene match
    for room in scene.rooms:
        for obj in room.objects:
            body = world.get_body_by_name(obj.id)
            global_position = body.global_pose.to_position()
            assert np.isclose(global_position.x, obj.position.x)
            assert np.isclose(global_position.y, obj.position.y)
            assert np.isclose(global_position.z, obj.position.z)
