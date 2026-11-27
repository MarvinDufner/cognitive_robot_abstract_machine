"""
Tests for the plane geometry of cloth lying on a table.
"""

from __future__ import annotations

import numpy
import pytest

from semantic_digital_twin.folding.geometry import FoldSegment, PlaneTransform, Polygon

# %% fold segments


def test_a_fold_segment_through_two_points_folds_the_side_of_the_folded_point() -> None:
    """
    The segment is directed so that the point to fold lies on its left.
    """
    segment = FoldSegment.through_points(
        first=numpy.array([0.0, 0.0]),
        second=numpy.array([1.0, 0.0]),
        folded_point=numpy.array([0.5, -1.0]),
    )

    assert segment.signed_distances(numpy.array([0.5, -1.0])) == pytest.approx(1.0)
    assert numpy.allclose(segment.normal, [0.0, -1.0])


def test_a_fold_segment_mirrors_points_across_its_line() -> None:
    """
    A point is mirrored to the same distance on the other side of the line, even beyond
    the segment's ends.
    """
    segment = FoldSegment(start=numpy.array([0.0, 0.0]), end=numpy.array([1.0, 1.0]))

    assert numpy.allclose(
        segment.reflect(numpy.array([[2.0, 0.0], [5.0, 3.0]])),
        [[0.0, 2.0], [3.0, 5.0]],
    )


# %% placements


def test_a_plane_transform_undoes_itself_with_its_inverse() -> None:
    """
    Carrying points onto the table and back leaves them where they were.
    """
    transform = PlaneTransform(
        translation=numpy.array([0.3, -0.2]), rotation_angle=1.1, scale=1.2
    )
    points = numpy.array([[0.1, 0.2], [-0.4, 0.5]])

    assert numpy.allclose(transform.apply_inverse(transform.apply(points)), points)


# %% polygons


@pytest.fixture
def square() -> Polygon:
    """
    :return: The unit square, its vertices given clockwise.
    """
    return Polygon(
        vertices=numpy.array([[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]])
    )


def test_a_polygon_runs_counter_clockwise_once_made(square: Polygon) -> None:
    """
    A polygon given clockwise has its vertices turned around.
    """
    assert square.signed_area == pytest.approx(1.0)


def test_cutting_a_square_along_a_segment_across_it_leaves_two_halves(
    square: Polygon,
) -> None:
    """
    A segment running right across the square cuts it into the parts either side.
    """
    pieces = square.cut(
        segment=FoldSegment(
            start=numpy.array([0.25, -1.0]), end=numpy.array([0.25, 2.0])
        )
    )

    assert sorted(piece.area for piece in pieces) == pytest.approx([0.25, 0.75])


def test_cutting_along_a_segment_that_ends_inside_cuts_nothing_off(
    square: Polygon,
) -> None:
    """
    A cut from the edge to the middle splits no piece off.
    """
    pieces = square.cut(
        segment=FoldSegment(start=numpy.array([0.5, -1.0]), end=numpy.array([0.5, 0.5]))
    )

    assert [piece.area for piece in pieces] == pytest.approx([1.0])


def test_cutting_along_a_segment_keeps_what_lies_beyond_its_ends_whole() -> None:
    """
    An L-shaped polygon cut along the segment where its arm meets its foot loses only
    the arm, even though the segment's line crosses the foot too.
    """
    l_shape = Polygon(
        vertices=numpy.array(
            [[0.0, 0.0], [2.0, 0.0], [2.0, 1.0], [1.0, 1.0], [1.0, 3.0], [0.0, 3.0]]
        )
    )

    pieces = l_shape.cut(
        segment=FoldSegment(start=numpy.array([0.0, 1.5]), end=numpy.array([1.0, 1.5]))
    )

    assert sorted(piece.area for piece in pieces) == pytest.approx([1.5, 2.5])


def test_cutting_along_a_polygon_edge_cuts_nothing_off(square: Polygon) -> None:
    """
    A segment lying along an edge runs through none of the inside.
    """
    pieces = square.cut(
        segment=FoldSegment(start=numpy.array([0.0, 0.0]), end=numpy.array([1.0, 0.0]))
    )

    assert [piece.area for piece in pieces] == pytest.approx([1.0])


def test_the_free_hanging_vertices_of_a_square_are_its_top_corners(
    square: Polygon,
) -> None:
    """
    Hanging with the y-axis up, only the two top corners have no cloth above them; the
    side edges hold up the bottom corners.
    """
    free_hanging = square.vertices[
        square.free_hanging_vertices(upward=numpy.array([0.0, 1.0]))
    ]

    assert sorted(map(tuple, free_hanging)) == [(0.0, 1.0), (1.0, 1.0)]


def test_only_a_corner_with_cloth_above_it_is_held_up_by_it() -> None:
    """
    Hanging a diamond from its top corner, its side corners would swing down; only its
    bottom corner, whose upward direction points between its edges into the diamond, is
    held up by the cloth above.
    """
    diamond = Polygon(
        vertices=numpy.array([[0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    )

    free_hanging = diamond.vertices[
        diamond.free_hanging_vertices(upward=numpy.array([0.0, 1.0]))
    ]

    assert free_hanging.tolist() == [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]
