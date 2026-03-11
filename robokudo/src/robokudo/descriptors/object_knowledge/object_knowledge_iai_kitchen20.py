from dataclasses import dataclass

from robokudo.object_knowledge_base import (
    BaseObjectKnowledgeBase,
    ObjectKnowledge,
    PredefinedObject,
)
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world_description.connections import Connection6DoF
from semantic_digital_twin.world_description.geometry import Box, Scale, Color
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body


@dataclass
class ObjectKnowledgeBase(BaseObjectKnowledgeBase):
    def __init__(self) -> None:
        super().__init__()
        root = self.world.root

        foobar1_shape = Box(
            scale=Scale(0.20, 0.20, 0.20), color=Color(0.1, 0.2, 0.8, 1.0)
        )
        foobar1_body = Body(
            name=PrefixedName(name="cereal"),
            visual=ShapeCollection([foobar1_shape]),
            collision=ShapeCollection([foobar1_shape]),
        )

        with self.world.modify_world():
            result_world_C_foobar1 = Connection6DoF.create_with_dofs(
                parent=root, child=foobar1_body, world=self.world
            )
            self.world.add_connection(result_world_C_foobar1)
            self.world.add_semantic_annotation(PredefinedObject(body=foobar1_body))

        # Set origins in a separate modification block so FK is compiled first
        with self.world.modify_world():
            result_world_C_foobar1.origin = (
                HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=-1.3, y=1.0, z=1.1, reference_frame=root
                )
            )
