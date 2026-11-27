"""
A T-shirt on a table and two hands in MuJoCo, without a robot, to fold the T-shirt by
g-folds and see how its folding planner copes with what the simulation makes of them.

.. warning:: The simulated T-shirt is not a model of a real one: its layers pass through
    each other, a hand holds every layer at the point it pinches, and the friction is a
    guess.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import timedelta
from enum import StrEnum
from pathlib import Path

import mujoco
import numpy
import trimesh
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from experiments.t_shirt_folding.exceptions import NothingToPinch, TooFewHands
from krrood.patterns.path_enumeration import PathEnumeration
from krrood_spatial_math.spatial_types import HomogeneousTransformationMatrix
from physics_simulators.base_simulator import SimulatorCallbackResult
from semantic_digital_twin.adapters.multi_sim import (
    MujocoBody,
    MujocoCamera,
    MujocoSim,
)
from semantic_digital_twin.folding.g_fold import FoldGrasp
from semantic_digital_twin.folding.geometry import FoldSegment, PlaneTransform
from semantic_digital_twin.folding.silhouette import ClothSilhouette
from semantic_digital_twin.folding.strategy import FoldPlan
from semantic_digital_twin.folding.t_shirt import TShirtFold, TShirtTemplate
from semantic_digital_twin.semantic_annotations.garments import TShirtLandmark
from semantic_digital_twin.specifications.kinematic_structure_entities import (
    BodySpecification,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.air import Air
from semantic_digital_twin.world_description.cloth import (
    ClothElasticity,
    ClothSheet,
    GarmentFile,
)
from semantic_digital_twin.world_description.geometry import Scale

# %% what the bench holds


class BenchEntity(StrEnum):
    """
    The names of what the bench holds.
    """

    ROOT = "world"
    """
    The root of the world, named as MuJoCo names its own, which a body the simulation
    moves by hand has to be fixed to directly.
    """

    FLOOR = "floor"
    TABLE = "table"
    T_SHIRT = "t_shirt"
    TOP_CAMERA = "top_camera"


class BenchHand(StrEnum):
    """
    The hands of the bench, each a point that pins the T-shirt where it closes.
    """

    LEFT = "left_hand"
    RIGHT = "right_hand"


class BenchResource(PathEnumeration):
    """
    The files the bench reads.
    """

    T_SHIRT_LANDMARK_VERTICES = Path(__file__).parent / "t_shirt_landmark_vertices.json"
    """
    The index of the vertex of :attr:`GarmentFile.T_SHIRT` at each
    :class:`TShirtLandmark`.
    """


# %% what the camera sees


@dataclass(kw_only=True)
class TopView:
    """
    An image of the table taken straight from above with an orthographic camera, so
    that its pixels lie on a square grid on the table's plane.
    """

    image: numpy.ndarray
    """
    The colour of each pixel, of shape (rows, columns, 3), the first row furthest along
    the plane's y-axis.
    """

    centre: numpy.ndarray
    """
    The point of the table's plane in the middle of the image.
    """

    side_length: float
    """
    How long a stretch of the table's plane the image shows from top to bottom, in
    metres.
    """

    @property
    def extent(self) -> list[float]:
        """
        The plane's coordinates at the image's left, right, bottom and top edges.
        """
        half_width = self.side_length * self.image.shape[1] / self.image.shape[0] / 2
        half_height = self.side_length / 2
        return [
            self.centre[0] - half_width,
            self.centre[0] + half_width,
            self.centre[1] - half_height,
            self.centre[1] + half_height,
        ]


# %% the bench


@dataclass(kw_only=True)
class FoldingBench:
    """
    A T-shirt lying spread out on a table, and two hands that pinch it, carry it along
    fold paths and let go.

    The table's frame is the world's, moved up to the table top. What perception would
    report is read off the simulation.
    """

    placement: PlaneTransform = field(
        default_factory=lambda: PlaneTransform(translation=numpy.zeros(2))
    )
    """
    Where the T-shirt is laid out: carries the frame of :attr:`t_shirt_template` onto
    the table's plane.
    """

    turned_over: bool = False
    """
    Whether the T-shirt lies on its other face, mirrored, as turning it over leaves it.
    """

    folds_made: list[TShirtFold] = field(default_factory=list)
    """
    The folds the T-shirt was folded by, in order, before it is laid on the table; the
    simulation starts with it folded.
    """

    crease_radius: float = 0.015
    """
    The radius of the bend each of :attr:`folds_made` leaves the cloth with along its
    segment, in metres.
    """

    table_side_length: float = 1.2
    """
    How long each side of the square table top is, in metres.
    """

    table_height: float = 0.75
    """
    How high the table top is, in metres.
    """

    collar_to_hem_length: float = 0.62
    """
    How long the T-shirt is from its collar to its hem, in metres.
    """

    sleeve_to_sleeve_width: float = 0.61
    """
    How wide the T-shirt is from the end of one sleeve to the end of the other, in
    metres.
    """

    t_shirt_mass: float = 0.13
    """
    The mass of the T-shirt, in kilograms: that of RGBench's T-shirt, whose mesh
    :attr:`GarmentFile.T_SHIRT` is, scaled with its area.
    """

    t_shirt_thickness: float = 1e-3
    """
    The thickness of the T-shirt's fabric, in metres; thinner than the 3 mm RGBench
    states, at which a folded sleeve springs back open in the simulation.
    """

    t_shirt_elasticity: ClothElasticity = field(
        default_factory=lambda: ClothElasticity(
            young_modulus=3 / 4 * (1 - 0.35) * 2.3e5, poisson_ratio=0.35
        )
    )
    """
    How stiffly the T-shirt resists bending, as RGBench states it and
    :class:`~experiments.t_shirt_demo.demo.TShirtSwingDemonstration` converts it.
    """

    sliding_friction: float = 2.0
    """
    The coefficient of friction between the T-shirt and the table, high enough that
    the T-shirt does not slide over the table while it is folded, as g-folds assume.
    """

    laying_height: float = 0.01
    """
    How far above the table top the lowest vertex of the T-shirt is laid out, to settle
    onto it, in metres.
    """

    pinch_radius: float = 0.03
    """
    How far from a hand a vertex of the T-shirt may lie to be held when the hand
    closes, in metres.
    """

    hand_speed: float = 0.2
    """
    How fast the hand with the longest path moves, in metres per second.
    """

    silhouette_cell_size: float = 4e-3
    """
    The length of a side of the cells the T-shirt's silhouette is seen on, in metres.
    """

    step_size: float = 4e-3
    """
    The time step of the simulation, in seconds.
    """

    headless: bool = True
    """
    Whether the simulation runs without a viewer.
    """

    t_shirt_template: TShirtTemplate = field(init=False)
    """
    The T-shirt as it lies spread out, in a frame of its own centred on it, its hem
    along the y-axis.
    """

    simulation: MujocoSim = field(init=False)
    """
    The simulation of the table, the hands and the T-shirt.
    """

    _cloth: ClothSheet = field(init=False)
    """
    The cloth the T-shirt is simulated as.
    """

    _landmark_vertices: list[int] = field(init=False)
    """
    The index of the vertex of the T-shirt at each landmark, in the order of
    :class:`TShirtLandmark`.
    """

    def __post_init__(self) -> None:
        """
        Lay the T-shirt out flat where :attr:`placement` puts it and start the
        simulation.
        """
        vertex_of_landmark = json.loads(
            BenchResource.T_SHIRT_LANDMARK_VERTICES.read_text()
        )
        self._landmark_vertices = [
            vertex_of_landmark[landmark] for landmark in TShirtLandmark
        ]
        flat_surface = self._flat_surface()
        self.t_shirt_template = TShirtTemplate(
            landmark_positions=dict(
                zip(
                    TShirtLandmark,
                    flat_surface.vertices[self._landmark_vertices, :2],
                )
            )
        )
        self._cloth = ClothSheet(
            name=BenchEntity.T_SHIRT,
            surface=self._laid_out_surface(self._folded_surface(flat_surface)),
            mass=self.t_shirt_mass,
            thickness=self.t_shirt_thickness,
            elasticity=self.t_shirt_elasticity,
            sliding_friction=self.sliding_friction,
        )
        self.simulation = MujocoSim(
            world=self._build_world(),
            headless=self.headless,
            step_size=self.step_size,
            integrator=mujoco.mjtIntegrator.mjINT_DISCRETE,
        )
        self.simulation.start_stepped_simulation()

    def _flat_surface(self) -> trimesh.Trimesh:
        """
        :return: The T-shirt's mesh stretched to the T-shirt's length and width, in the
            frame of :attr:`t_shirt_template`, its relief above zero.
        """
        surface = GarmentFile.T_SHIRT.create_surface(HomogeneousTransformationMatrix())
        length, width, _ = surface.extents
        surface.vertices = (surface.vertices - surface.bounds.mean(axis=0)) * [
            self.collar_to_hem_length / length,
            self.sleeve_to_sleeve_width / width,
            1.0,
        ]
        surface.vertices[:, 2] -= surface.vertices[:, 2].min()
        return surface

    def _folded_surface(self, flat_surface: trimesh.Trimesh) -> trimesh.Trimesh:
        """
        :param flat_surface: The T-shirt's mesh in its own frame.
        :return: The mesh folded by each of :attr:`folds_made` in turn: the cloth each
            fold lifts is wrapped half around a cylinder of :attr:`crease_radius` lying
            along the fold's segment, and the rest of it laid on top, mirrored.
        """
        vertices = flat_surface.vertices.copy()
        for fold in self.folds_made:
            segment = fold.fold_segment(template=self.t_shirt_template)
            lifted = self._lifted_vertices(
                surface=flat_surface, vertices=vertices, segment=segment
            )
            distances = segment.signed_distances(vertices[lifted, :2])
            angles = numpy.minimum(distances / self.crease_radius, numpy.pi)
            beyond = numpy.maximum(distances - numpy.pi * self.crease_radius, 0.0)
            folded_distances = self.crease_radius * numpy.sin(angles) - beyond
            vertices[lifted, :2] += numpy.multiply.outer(
                folded_distances - distances, segment.normal
            )
            vertices[lifted, 2] += self.crease_radius * (1 - numpy.cos(angles))
        surface = flat_surface.copy()
        surface.vertices = vertices
        return surface

    @staticmethod
    def _lifted_vertices(
        *, surface: trimesh.Trimesh, vertices: numpy.ndarray, segment: FoldSegment
    ) -> numpy.ndarray:
        """
        :param surface: The T-shirt's mesh, whose edges join its vertices.
        :param vertices: Where the mesh's vertices lie now.
        :param segment: The segment the T-shirt is folded over.
        :return: The indices of the vertices the fold lifts: those left of the
            segment's line that the mesh joins, through vertices left of it, to a vertex
            next to the segment itself.
        """
        distances = segment.signed_distances(vertices[:, :2])
        left = distances > 0
        edges = surface.edges_unique
        edge_length = float(numpy.mean(surface.edges_unique_length))
        joined = edges[left[edges].all(axis=1)]
        _, component = connected_components(
            coo_matrix(
                (numpy.ones(len(joined)), (joined[:, 0], joined[:, 1])),
                shape=(len(vertices), len(vertices)),
            ),
            directed=False,
        )
        feet = segment.foot_distances(vertices[:, :2])
        next_to_segment = (
            left & (distances < edge_length) & (feet > 0.0) & (feet < segment.length)
        )
        return numpy.flatnonzero(
            left & numpy.isin(component, component[next_to_segment])
        )

    def _laid_out_surface(self, flat_surface: trimesh.Trimesh) -> trimesh.Trimesh:
        """
        :param flat_surface: The T-shirt's mesh in its own frame.
        :return: The mesh, turned over if :attr:`turned_over`, placed by
            :attr:`placement` just above the table top.
        """
        vertices = flat_surface.vertices.copy()
        if self.turned_over:
            vertices[:, 1:] *= -1
            vertices[:, 2] -= vertices[:, 2].min()
        surface = flat_surface.copy()
        surface.vertices = numpy.column_stack(
            [
                self.placement.apply(vertices[:, :2]),
                self.table_height + self.laying_height + vertices[:, 2],
            ]
        )
        return surface

    def _build_world(self) -> World:
        """
        :return: A world with a floor, the table, the hands, a camera above the table
            and the T-shirt.
        """
        world = World.create_with_root_body(BenchEntity.ROOT)
        table_thickness = 0.04
        BodySpecification.box(
            BenchEntity.FLOOR,
            Scale(4.0, 4.0, 0.02),
            parent_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(z=-0.01),
        ).spawn(world)
        BodySpecification.box(
            BenchEntity.TABLE,
            Scale(self.table_side_length, self.table_side_length, table_thickness),
            parent_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(
                z=self.table_height - table_thickness / 2
            ),
        ).spawn(world)
        for hand in BenchHand:
            BodySpecification(hand).spawn(world).add_simulator_property(
                MujocoBody(motion_capture=True)
            )
        world.root.add_simulator_property(
            MujocoCamera(
                name=BenchEntity.TOP_CAMERA,
                body=world.root,
                orthographic=True,
                fovy=self.table_side_length,
                position=[0.0, 0.0, self.table_height + 1.0],
                quaternion=[1.0, 0.0, 0.0, 0.0],
            )
        )
        world.add_simulator_property(Air())
        world.add_simulator_property(self._cloth)
        return world

    # %% the T-shirt

    @property
    def t_shirt_vertex_positions(self) -> numpy.ndarray:
        """
        Where each vertex of the T-shirt is, in the world frame, of shape (n, 3).
        """
        return self.simulation.simulator.get_flex_vertex_positions(
            flex_name=BenchEntity.T_SHIRT
        ).result

    @property
    def landmark_positions(self) -> dict[TShirtLandmark, numpy.ndarray]:
        """
        Where the vertex at each landmark of the T-shirt is, in the world frame.
        """
        return dict(
            zip(TShirtLandmark, self.t_shirt_vertex_positions[self._landmark_vertices])
        )

    def let_t_shirt_settle(
        self,
        longest: timedelta = timedelta(seconds=4),
        still_below: float = 2e-3,
    ) -> None:
        """
        Advance the simulation until the T-shirt lies still.

        :param longest: How long to wait for that at most, in simulated time.
        :param still_below: How far any vertex may move within a fifth of a second for
            the T-shirt to count as lying still, in metres.
        """
        interval = timedelta(seconds=0.2)
        before = self.t_shirt_vertex_positions
        for _ in range(round(longest / interval)):
            self.simulation.step_simulation(interval)
            now = self.t_shirt_vertex_positions
            if numpy.abs(now - before).max() < still_below:
                return
            before = now

    def observe_silhouette(self) -> ClothSilhouette:
        """
        :return: The silhouette of the T-shirt as a camera straight above the table
            sees it, on the table's plane.
        """
        return ClothSilhouette.from_surface(
            surface=trimesh.Trimesh(
                self.t_shirt_vertex_positions, self._cloth.surface.faces, process=False
            ),
            cell_size=self.silhouette_cell_size,
        )

    def capture_top_view(self, *, pixel_count: int = 480) -> TopView:
        """
        :param pixel_count: How many pixels the image has along each side.
        :return: An image of the whole table top taken straight from above.
        """
        return TopView(
            image=self.simulation.simulator.capture_rgb(
                camera_name=BenchEntity.TOP_CAMERA,
                height=pixel_count,
                width=pixel_count,
            ).result,
            centre=numpy.zeros(2),
            side_length=self.table_side_length,
        )

    # %% the hands

    def perform(self, *, plan: FoldPlan) -> None:
        """
        Make a fold with the hands as the plan says: each hand takes the T-shirt where
        a grasp does, all of them follow their paths together, turning the cloth they
        hold over the fold's line as they go, let go at the end, and the T-shirt comes
        to rest.

        :param plan: The fold to make.
        """
        if len(plan.grasps) > len(BenchHand):
            raise TooFewHands(grasp_count=len(plan.grasps), hand_count=len(BenchHand))
        hands = list(zip(BenchHand, plan.grasps))
        paths = [self._world_path(grasp=grasp, hand=hand) for hand, grasp in hands]
        longest = max(
            numpy.linalg.norm(numpy.diff(path, axis=0), axis=1).sum() for path in paths
        )
        segment = plan.g_fold.segment
        turning_axis = numpy.array([*segment.direction, 0.0])
        step_count = max(1, round(longest / self.hand_speed / self.step_size))
        for step in range(1, step_count + 1):
            for (hand, _), path in zip(hands, paths):
                position = self._point_along(path, step / step_count)
                start_distance, distance = segment.signed_distances(
                    numpy.array([path[0, :2], position[:2]])
                )
                half_turn = numpy.pi * (start_distance - distance) / 4 / start_distance
                self.simulation.simulator.set_motion_capture_body_pose(
                    body_name=hand,
                    position=position,
                    quaternion=numpy.array(
                        [numpy.cos(half_turn), *(numpy.sin(half_turn) * turning_axis)]
                    ),
                )
            self.simulation.step_simulation(timedelta(seconds=self.step_size))
        for hand, _ in hands:
            self.simulation.simulator.unpin_flex_vertices(
                flex_name=BenchEntity.T_SHIRT, body_name=hand
            )
        self.let_t_shirt_settle()

    def _world_path(self, *, grasp: FoldGrasp, hand: BenchHand) -> numpy.ndarray:
        """
        Put a hand where a grasp takes the T-shirt and close it there.

        :param grasp: The grasp.
        :param hand: The hand that makes it.
        :return: The grasp's path in the world frame, starting halfway between the
            lowest and the highest layer of cloth the hand closed on.
        """
        vertices = self.t_shirt_vertex_positions
        nearby = (
            numpy.linalg.norm(vertices[:, :2] - grasp.grasp_position, axis=1)
            <= self.pinch_radius
        )
        if not nearby.any():
            raise NothingToPinch(
                position=numpy.array([*grasp.grasp_position, self.table_height])
            )
        lowest, highest = vertices[nearby, 2].min(), vertices[nearby, 2].max()
        path = grasp.path + [0.0, 0.0, (lowest + highest) / 2]
        self.simulation.simulator.set_motion_capture_body_position(
            body_name=hand, position=path[0]
        )
        pinch = self.simulation.simulator.pin_flex_vertices(
            flex_name=BenchEntity.T_SHIRT,
            body_name=hand,
            body_P_centre=numpy.zeros(3),
            # closing between the layers, the hand reaches the top and bottom ones
            radius=float(numpy.hypot(self.pinch_radius, (highest - lowest) / 2)),
        )
        if (
            pinch.type
            is not SimulatorCallbackResult.ResultType.SUCCESS_AFTER_EXECUTION_ON_MODEL
        ):
            raise NothingToPinch(position=path[0])
        return path

    @staticmethod
    def _point_along(path: numpy.ndarray, share: float) -> numpy.ndarray:
        """
        :param path: Points of shape (n, 3).
        :param share: How far along the path, as a share of its length.
        :return: The point that far along the path.
        """
        distances = numpy.concatenate(
            [[0.0], numpy.cumsum(numpy.linalg.norm(numpy.diff(path, axis=0), axis=1))]
        )
        return numpy.array(
            [
                numpy.interp(share * distances[-1], distances, coordinate)
                for coordinate in path.T
            ]
        )

    def stop(self) -> None:
        """
        Stop the simulation.
        """
        self.simulation.stop_simulation()
