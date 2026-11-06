"""
Cloth a physical simulation deforms, and the garments it is made into.

A physics engine reads a :class:`ClothSheet` off the world as a simulator property; the
world's kinematic structure holds no cloth.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib.resources import files
from pathlib import Path

import trimesh

from krrood.patterns.path_enumeration import PathEnumeration
from krrood_spatial_math.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.mixin import SimulatorAdditionalProperty
from semantic_digital_twin.world_description.geometry import Color

# %% cloth sheets


@dataclass
class ClothElasticity:
    """
    How stiffly a cloth's material resists bending, as the elastic constants of a thin
    shell.
    """

    young_modulus: float = 3e5
    """
    Young's modulus of the material, in pascals.
    """

    poisson_ratio: float = 0.0
    """
    Poisson's ratio of the material.
    """


@dataclass
class ClothSheet(SimulatorAdditionalProperty):
    """
    A piece of cloth a physical simulation deforms: a sheet of triangles whose edges keep
    their length, which bends elastically and collides as a shell of its thickness.
    """

    name: str
    """
    The name the simulation knows the cloth by.
    """

    surface: trimesh.Trimesh
    """
    The cloth's triangles as they lie when the simulation starts, with their vertices in
    the world frame.
    """

    mass: float
    """
    Mass of the whole cloth, in kilograms.
    """

    thickness: float = 8e-3
    """
    Thickness of the cloth, in metres: it collides as a shell this thick, and its bending
    stiffness grows with it.
    """

    elasticity: ClothElasticity = field(default_factory=ClothElasticity)
    """
    How stiffly the cloth resists bending.
    """

    edge_damping: float = 0.0
    """
    How strongly the cloth's edges damp the speed at which they stretch and shrink.
    """

    color: Color = field(default_factory=Color)
    """
    The color the cloth is drawn in.
    """


# %% garments


class GarmentFile(PathEnumeration):
    """
    The garments under the semantic_digital_twin resources, each a mesh of its panels
    lying flat as the garment was designed, the panels sharing the vertices of their
    seams.
    """

    T_SHIRT = (
        Path(files("semantic_digital_twin")).parent.parent
        / "resources"
        / "obj"
        / "t_shirt.obj"
    )
    """
    A short-sleeved T-shirt of two layers, open at the hem, the neck and the cuffs, with
    its neckline at the origin, its hem along the x-axis and its sleeves along the
    y-axis. It is RGBench's green T-shirt, see ``t_shirt_ATTRIBUTION.md`` next to it.
    """

    def create_surface(
        self, world_T_garment: HomogeneousTransformationMatrix
    ) -> trimesh.Trimesh:
        """
        :param world_T_garment: Where the garment lies.
        :return: The garment's panels sewn into one sheet of triangles, with its
            vertices in the world frame.
        """
        surface = trimesh.load(str(self), force="mesh", process=False)
        # a vertex on a seam is read once for each panel, since each panel has its own
        # texture coordinates there
        surface.merge_vertices(merge_tex=True, merge_norm=True)
        return surface.apply_transform(world_T_garment.to_np())
