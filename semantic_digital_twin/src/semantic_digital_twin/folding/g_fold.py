"""
G-folds of cloth lying flat on a table (van den Berg et al., *Gravity-Based Robotic
Cloth Folding*, WAFR 2010): grippers lift the cloth on one side of a segment and lay it
down mirrored, holding the points that would otherwise slide while it hangs.

The cloth does not stretch, resist bending, or slide over the table or itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy

from semantic_digital_twin.folding.geometry import (
    FoldSegment,
    PlaneTransform,
    Polygon,
)

# %% layered cloth


@dataclass(kw_only=True)
class LayeredCloth:
    """
    Cloth lying flat on a table in layers, as folding leaves it.
    """

    layers: list[Polygon]
    """
    The outline of each flat piece of the cloth, the bottom one first.
    """

    tolerance: float = 1e-6
    """
    How far from a fold segment's line a point may lie to count as lying on it, in
    metres.
    """

    def lifted_layers(self, *, segment: FoldSegment) -> list[Polygon]:
        """
        :param segment: The segment the cloth is folded over.
        :return: The pieces of the layers that folding over the segment lifts, the
            bottom one first.
        """
        return [
            piece
            for layer in self.layers
            for piece in layer.cut(segment=segment, tolerance=self.tolerance)
            if self._fold_lifts(piece=piece, segment=segment)
        ]

    def folded(self, *, segment: FoldSegment) -> LayeredCloth:
        """
        :param segment: The segment to fold the cloth over.
        :return: The cloth once folded over the segment: the lifted pieces lie mirrored
            on top, their former top one lowest, and the rest lies where it was.
        """
        staying = []
        lifted = []
        for layer in self.layers:
            for piece in layer.cut(segment=segment, tolerance=self.tolerance):
                if self._fold_lifts(piece=piece, segment=segment):
                    lifted.append(piece)
                else:
                    staying.append(piece)
        return LayeredCloth(
            layers=staying
            + [piece.reflected(segment=segment) for piece in reversed(lifted)],
            tolerance=self.tolerance,
        )

    def _fold_lifts(self, *, piece: Polygon, segment: FoldSegment) -> bool:
        """
        :param piece: A piece of a layer, cut along the segment.
        :param segment: The segment the cloth is folded over.
        :return: Whether folding over the segment lifts the piece: it lies left of the
            segment's line and an edge of it runs along the segment.
        """
        return bool(
            numpy.all(segment.signed_distances(piece.vertices) >= -self.tolerance)
            and piece.shared_edge_length(segment=segment, tolerance=self.tolerance)
            > self.tolerance
        )

    def folded_points(
        self, *, points: numpy.ndarray, segment: FoldSegment
    ) -> numpy.ndarray:
        """
        :param points: Points on the cloth, of shape (n, 2).
        :param segment: The segment to fold the cloth over.
        :return: Where each point lies once the cloth is folded over the segment.
        """
        lifted = numpy.zeros(len(points), dtype=bool)
        for piece in self.lifted_layers(segment=segment):
            lifted |= piece.contains(points, 1e3 * self.tolerance)
        return numpy.where(lifted[:, None], segment.reflect(points), points)

    def contains(self, points: numpy.ndarray, tolerance: float = 0.0) -> numpy.ndarray:
        """
        :param points: Points of shape (n, 2).
        :param tolerance: As for :meth:`Polygon.contains`.
        :return: Whether each point lies on any layer of the cloth.
        """
        covered = numpy.zeros(len(points), dtype=bool)
        for layer in self.layers:
            covered |= layer.contains(points, tolerance)
        return covered

    def transformed(self, transform: PlaneTransform) -> LayeredCloth:
        """
        :param transform: Carries the frame the cloth is given in onto another.
        :return: The same cloth in the other frame.
        """
        return LayeredCloth(
            layers=[layer.transformed(transform) for layer in self.layers],
            tolerance=self.tolerance,
        )


# %% g-folds


@dataclass(kw_only=True)
class FoldGrasp:
    """
    Where one gripper takes the cloth for a g-fold, which vertices of the cloth's
    outline it holds there, and the path it carries them along.
    """

    held_vertices: numpy.ndarray
    """
    The vertices the gripper holds, of shape (k, 2).
    """

    path: numpy.ndarray
    """
    The points the gripper passes through, of shape (n, 3), from where it takes the
    cloth to where it lays it down; their third coordinate is the height above the
    cloth's starting level.
    """

    @property
    def grasp_position(self) -> numpy.ndarray:
        """
        Where the gripper takes the cloth, on the table's plane.
        """
        return self.path[0, :2]

    @property
    def placement_position(self) -> numpy.ndarray:
        """
        Where the gripper lays the cloth down, on the table's plane.
        """
        return self.path[-1, :2]


@dataclass(kw_only=True)
class GFold:
    """
    A fold of layered cloth over a segment, made as a g-fold.
    """

    cloth: LayeredCloth
    """
    The cloth before the fold.
    """

    segment: FoldSegment
    """
    The segment the cloth is folded over.
    """

    tolerance: float = 5e-3
    """
    How close two points may lie to count as the same, and a point to the segment's line
    to count as lying on it, in metres.
    """

    apex_height_ratio: float = 1.0
    """
    How high each gripper path rises, as a share of its grasp's distance from the
    segment's line.

    At one, the lifted cloth hangs taut all the way; below one, it hangs slack.
    """

    @cached_property
    def lifted_layers(self) -> list[Polygon]:
        """
        The layers of the cloth that the fold lifts, the bottom one first.
        """
        return self.cloth.lifted_layers(segment=self.segment)

    @cached_property
    def result(self) -> LayeredCloth:
        """
        The cloth once folded.
        """
        return self.cloth.folded(segment=self.segment)

    @cached_property
    def vertices_to_hold(self) -> numpy.ndarray:
        """
        The vertices of the lifted cloth that have to be held, of shape (m, 2), for none
        of it to slide while it hangs from them: the free-hanging vertices of each
        lifted layer off the segment's line, except those another lifted layer holds up
        because they rest on its cloth between the vertices it has to have held itself.
        """
        free_hanging = []
        for layer in self.lifted_layers:
            vertices = layer.vertices[
                layer.free_hanging_vertices(upward=self.segment.normal)
            ]
            free_hanging.append(
                vertices[self.segment.signed_distances(vertices) > self.tolerance]
            )
        held = [
            vertices[
                ~self._rest_on_other_layer(
                    vertices=vertices, layer_index=index, free_hanging=free_hanging
                )
            ]
            for index, vertices in enumerate(free_hanging)
        ]
        return numpy.concatenate(held) if held else numpy.empty((0, 2))

    def _rest_on_other_layer(
        self,
        *,
        vertices: numpy.ndarray,
        layer_index: int,
        free_hanging: list[numpy.ndarray],
    ) -> numpy.ndarray:
        """
        :param vertices: Free-hanging vertices of one lifted layer.
        :param layer_index: The index of that layer among the lifted layers.
        :param free_hanging: The free-hanging vertices of every lifted layer.
        :return: Whether each vertex lies on the cloth of another lifted layer, away
            from that layer's own free-hanging vertices.
        """
        resting = numpy.zeros(len(vertices), dtype=bool)
        for index, layer in enumerate(self.lifted_layers):
            if index == layer_index or len(vertices) == 0:
                continue
            on_layer = layer.contains(vertices, self.tolerance)
            if len(free_hanging[index]) > 0:
                on_layer &= (
                    numpy.linalg.norm(
                        vertices[:, None] - free_hanging[index][None], axis=2
                    ).min(axis=1)
                    > self.tolerance
                )
            resting |= on_layer
        return resting

    def grasps(
        self, *, gripper_radius: float, path_spacing: float = 0.02
    ) -> list[FoldGrasp]:
        """
        :param gripper_radius: How far from its centre a gripper holds cloth, in
            metres.
        :param path_spacing: How far apart the points of each path lie at most, in
            metres.
        :return: As few grasps as hold every vertex in :attr:`vertices_to_hold`, each
            gripper taking every vertex within its radius.
        """
        remaining = self.vertices_to_hold
        grasps = []
        while len(remaining) > 0:
            pairs = (remaining[:, None] + remaining[None]) / 2
            centres = pairs.reshape(-1, 2)
            covered = (
                numpy.linalg.norm(centres[:, None] - remaining[None], axis=2)
                <= gripper_radius + self.tolerance
            )
            best = int(numpy.argmax(covered.sum(axis=1)))
            grasps.append(
                FoldGrasp(
                    held_vertices=remaining[covered[best]],
                    path=self.tent_path(
                        grasp_position=centres[best], spacing=path_spacing
                    ),
                )
            )
            remaining = remaining[~covered[best]]
        return grasps

    def tent_path(
        self, *, grasp_position: numpy.ndarray, spacing: float = 0.02
    ) -> numpy.ndarray:
        """
        :param grasp_position: Where the gripper takes the cloth, on the table's plane.
        :param spacing: How far apart the points of the path lie at most, in metres.
        :return: The points of shape (n, 3) the gripper passes through, in the plane
            perpendicular to the segment: up to :attr:`apex_height_ratio` times its
            distance from the segment's line, right above that line, and down again to
            the mirror image of where it started. The third coordinate is the height
            above the starting level.
        """
        distance = float(self.segment.signed_distances(grasp_position))
        foot = grasp_position - distance * self.segment.normal
        point_count = (
            2 * max(1, int(numpy.ceil(numpy.sqrt(2) * distance / spacing))) + 1
        )
        across = numpy.linspace(distance, -distance, point_count)
        return numpy.column_stack(
            [
                foot + numpy.multiply.outer(across, self.segment.normal),
                self.apex_height_ratio * (distance - numpy.abs(across)),
            ]
        )
