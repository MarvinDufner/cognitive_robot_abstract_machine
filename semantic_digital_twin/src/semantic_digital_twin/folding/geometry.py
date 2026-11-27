"""
Plane geometry of cloth lying flat on a table: its outline, the segments it is folded
over, and where a cloth lies on the table.

Points are rows of numpy arrays on the table's plane, in metres.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy
from matplotlib.path import Path as MatplotlibPath

# %% placement on the table


@dataclass(kw_only=True)
class PlaneTransform:
    """
    A turn, an even scaling and a shift of the table's plane, which carries a cloth's
    own frame onto the table.
    """

    translation: numpy.ndarray
    """
    Where the origin of the cloth's frame lies on the table's plane.
    """

    rotation_angle: float = 0.0
    """
    How far the cloth's frame is turned counter-clockwise, in radians.
    """

    scale: float = 1.0
    """
    How much larger the cloth is on the table than in its own frame.
    """

    @property
    def rotation(self) -> numpy.ndarray:
        """
        The 2x2 matrix turning a vector by :attr:`rotation_angle`.
        """
        cosine, sine = numpy.cos(self.rotation_angle), numpy.sin(self.rotation_angle)
        return numpy.array([[cosine, -sine], [sine, cosine]])

    def apply(self, points: numpy.ndarray) -> numpy.ndarray:
        """
        :param points: Points in the cloth's frame, of shape (n, 2) or (2,).
        :return: The same points on the table's plane.
        """
        return self.scale * points @ self.rotation.T + self.translation

    def apply_inverse(self, points: numpy.ndarray) -> numpy.ndarray:
        """
        :param points: Points on the table's plane, of shape (n, 2) or (2,).
        :return: The same points in the cloth's frame.
        """
        return (points - self.translation) @ self.rotation / self.scale


# %% fold segments


@dataclass(kw_only=True)
class FoldSegment:
    """
    A directed line segment on the table's plane.

    Folding over it lays the cloth left of its line that hangs on it, the pieces whose
    edge runs along the segment, onto the cloth right of its line.
    """

    start: numpy.ndarray
    """
    Where the segment starts.
    """

    end: numpy.ndarray
    """
    Where the segment ends.
    """

    @classmethod
    def through_points(
        cls, *, first: numpy.ndarray, second: numpy.ndarray, folded_point: numpy.ndarray
    ) -> FoldSegment:
        """
        :param first: One end of the segment.
        :param second: The other end of the segment.
        :param folded_point: A point off the segment's line on the side that is folded
            over.
        :return: The segment between both points, directed to fold the side
            ``folded_point`` lies on.
        """
        segment = cls(start=first, end=second)
        if segment.signed_distances(folded_point) < 0:
            return cls(start=second, end=first)
        return segment

    @property
    def length(self) -> float:
        """
        The length of the segment, in metres.
        """
        return float(numpy.linalg.norm(self.end - self.start))

    @property
    def direction(self) -> numpy.ndarray:
        """
        The unit vector the segment runs along.
        """
        return (self.end - self.start) / self.length

    @property
    def normal(self) -> numpy.ndarray:
        """
        The unit vector perpendicular to the segment that points to its left, away from
        it on the side that is folded over.
        """
        x, y = self.direction
        return numpy.array([-y, x])

    def signed_distances(self, points: numpy.ndarray) -> numpy.ndarray:
        """
        :param points: Points of shape (n, 2) or (2,).
        :return: How far each point lies left of the segment's line, negative for points
            right of it.
        """
        return (points - self.start) @ self.normal

    def foot_distances(self, points: numpy.ndarray) -> numpy.ndarray:
        """
        :param points: Points of shape (n, 2) or (2,).
        :return: How far from :attr:`start` towards :attr:`end` the foot of each point
            on the segment's line lies, in metres.
        """
        return (points - self.start) @ self.direction

    def reflect(self, points: numpy.ndarray) -> numpy.ndarray:
        """
        :param points: Points of shape (n, 2) or (2,).
        :return: The mirror image of each point across the segment's line.
        """
        return points - 2 * numpy.multiply.outer(
            self.signed_distances(points), self.normal
        )

    def transformed(self, transform: PlaneTransform) -> FoldSegment:
        """
        :param transform: Carries the frame the segment is given in onto another.
        :return: The same segment in the other frame.
        """
        return FoldSegment(
            start=transform.apply(self.start), end=transform.apply(self.end)
        )


# %% outlines


@dataclass(kw_only=True)
class Polygon:
    """
    A simple polygon on the table's plane, such as the outline of a layer of cloth.
    """

    vertices: numpy.ndarray
    """
    The corners of the polygon in order, of shape (n, 2); counter-clockwise once the
    polygon is made.
    """

    def __post_init__(self) -> None:
        """
        Turn the vertices counter-clockwise if they run clockwise.
        """
        if self.signed_area < 0:
            self.vertices = self.vertices[::-1].copy()

    @property
    def signed_area(self) -> float:
        """
        The area of the polygon, in square metres, negative while its vertices run
        clockwise.
        """
        x, y = self.vertices.T
        return 0.5 * float(
            numpy.dot(x, numpy.roll(y, -1)) - numpy.dot(numpy.roll(x, -1), y)
        )

    @property
    def area(self) -> float:
        """
        The area of the polygon, in square metres.
        """
        return abs(self.signed_area)

    def contains(self, points: numpy.ndarray, tolerance: float = 0.0) -> numpy.ndarray:
        """
        :param points: Points of shape (n, 2).
        :param tolerance: How far outside the polygon a point may lie to still count as
            inside it, in metres; negative to only count points that far inside it.
        :return: Whether each point lies inside the polygon.
        """
        return MatplotlibPath(self.vertices).contains_points(points, radius=tolerance)

    def cut(self, *, segment: FoldSegment, tolerance: float = 1e-9) -> list[Polygon]:
        """
        :param segment: The segment to cut along.
        :param tolerance: How far from the segment a point may lie to count as lying on
            it, in metres.
        :return: The pieces the polygon falls into when cut along every stretch of the
            segment that runs through its inside; the polygon alone if there is none.
        """
        points = self._boundary_with_cut_points(segment=segment, tolerance=tolerance)
        distances = segment.signed_distances(points)
        positions = segment.foot_distances(points)
        on_segment = numpy.flatnonzero(
            (numpy.abs(distances) <= tolerance)
            & (positions >= -tolerance)
            & (positions <= segment.length + tolerance)
        )
        on_segment = on_segment[numpy.argsort(positions[on_segment])]
        outgoing = {node: [(node + 1) % len(points)] for node in range(len(points))}
        for lower, upper in zip(on_segment, on_segment[1:]):
            middle = (points[lower] + points[upper]) / 2
            if (
                positions[upper] - positions[lower] > tolerance
                and self.contains(middle[None], -tolerance)[0]
            ):
                outgoing[lower].append(upper)
                outgoing[upper].append(lower)
        return self._faces(points=points, outgoing=outgoing, tolerance=tolerance)

    def _boundary_with_cut_points(
        self, *, segment: FoldSegment, tolerance: float
    ) -> numpy.ndarray:
        """
        :param segment: The segment to cut along.
        :param tolerance: How far from the segment's line a vertex may lie to count as
            lying on it.
        :return: The vertices in order, with the points added where an edge crosses the
            segment.
        """
        distances = segment.signed_distances(self.vertices)
        points = []
        for index, vertex in enumerate(self.vertices):
            following = (index + 1) % len(self.vertices)
            points.append(vertex)
            if (
                distances[index] * distances[following] >= 0
                or min(abs(distances[index]), abs(distances[following])) <= tolerance
            ):
                continue
            share = distances[index] / (distances[index] - distances[following])
            crossing = vertex + share * (self.vertices[following] - vertex)
            if (
                -tolerance
                <= segment.foot_distances(crossing)
                <= segment.length + tolerance
            ):
                points.append(crossing)
        return numpy.array(points)

    @staticmethod
    def _faces(
        *, points: numpy.ndarray, outgoing: dict[int, list[int]], tolerance: float
    ) -> list[Polygon]:
        """
        :param points: The corners of a polygon with points added along cuts.
        :param outgoing: The points each point is joined to, along the boundary
            counter-clockwise and along the cuts both ways.
        :param tolerance: The smallest side length a piece has to have.
        :return: The pieces the edges enclose, each traced with its inside on the left,
            turning at each point into the first edge clockwise from the one it came
            along.
        """
        traversed = set()
        pieces = []
        for first_edge in [(node, end) for node in outgoing for end in outgoing[node]]:
            corners = []
            start, end = first_edge
            while (start, end) not in traversed:
                traversed.add((start, end))
                corners.append(start)
                incoming = points[end] - points[start]
                backward_angle = numpy.arctan2(-incoming[1], -incoming[0])
                candidates = [node for node in outgoing[end] if node != start]
                start, end = (
                    end,
                    min(
                        candidates or outgoing[end],
                        key=lambda node: (
                            (
                                backward_angle
                                - numpy.arctan2(*(points[node] - points[end])[::-1])
                            )
                            % (2 * numpy.pi)
                            or 2 * numpy.pi
                        ),
                    ),
                )
            if len(corners) >= 3:
                piece = Polygon(vertices=points[corners])
                if piece.area > tolerance**2:
                    pieces.append(piece)
        return pieces

    def shared_edge_length(
        self, *, segment: FoldSegment, tolerance: float = 1e-6
    ) -> float:
        """
        :param segment: A segment.
        :param tolerance: How far from the segment's line a vertex may lie to count as
            lying on it, in metres.
        :return: How long the polygon's edges run along the segment, in metres.
        """
        on_line = numpy.abs(segment.signed_distances(self.vertices)) <= tolerance
        positions = segment.foot_distances(self.vertices)
        following_positions = numpy.roll(positions, -1)
        along = on_line & numpy.roll(on_line, -1)
        lower = numpy.maximum(numpy.minimum(positions, following_positions), 0.0)
        upper = numpy.minimum(
            numpy.maximum(positions, following_positions), segment.length
        )
        return float(numpy.clip(upper - lower, 0.0, None)[along].sum())

    def reflected(self, *, segment: FoldSegment) -> Polygon:
        """
        :param segment: The segment across whose line the polygon is mirrored.
        :return: The mirror image of the polygon.
        """
        return Polygon(vertices=segment.reflect(self.vertices))

    def transformed(self, transform: PlaneTransform) -> Polygon:
        """
        :param transform: Carries the frame the polygon is given in onto another.
        :return: The same polygon in the other frame.
        """
        return Polygon(vertices=transform.apply(self.vertices))

    def free_hanging_vertices(
        self, *, upward: numpy.ndarray, tolerance: float = 1e-9
    ) -> numpy.ndarray:
        """
        :param upward: The direction on the table's plane that points up while the
            polygon hangs.
        :param tolerance: How far a direction may turn out of the polygon and still
            count as pointing into it, in radians.
        :return: The indices of the convex vertices at which ``upward`` points out of
            the polygon, neither into it nor along an edge: hanging that way up, no
            cloth above them holds them.
        """
        to_next = numpy.roll(self.vertices, -1, axis=0) - self.vertices
        to_previous = numpy.roll(self.vertices, 1, axis=0) - self.vertices
        convex = self._cross(to_next, to_previous) > 0
        next_length = numpy.linalg.norm(to_next, axis=1)
        previous_length = numpy.linalg.norm(to_previous, axis=1)
        upward_after_next = self._cross(to_next, upward) >= -tolerance * next_length
        upward_before_previous = (
            self._cross(upward, to_previous) >= -tolerance * previous_length
        )
        return numpy.flatnonzero(convex & ~(upward_after_next & upward_before_previous))

    @staticmethod
    def _cross(first: numpy.ndarray, second: numpy.ndarray) -> numpy.ndarray:
        """
        :param first: Vectors of shape (n, 2) or (2,).
        :param second: Vectors of shape (n, 2) or (2,).
        :return: The z component of the cross product of each pair of vectors.
        """
        return first[..., 0] * second[..., 1] - first[..., 1] * second[..., 0]
