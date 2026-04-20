"""Open3D-based visualization for RoboKudo pipelines.

This module provides 3D visualization capabilities for RoboKudo pipelines using Open3D.
It handles:

* 3D geometry visualization
* Point cloud rendering
* Camera control
* Coordinate frame display
* Window management
"""

from __future__ import annotations

import logging
import queue
import time
from collections import defaultdict
from dataclasses import dataclass, field
from multiprocessing import Pipe, Process, Queue, shared_memory
from multiprocessing.connection import Connection
from threading import Event, Lock, Thread

import numpy as np
import open3d as o3d  # this import creates a SIGINT during unit test execution....
from typing_extensions import (
    TYPE_CHECKING,
    Any,
    Dict,
    Iterator,
    List,
    Optional,
    Tuple,
    Type,
    Union,
)

from robokudo.annotators.core import BaseAnnotator
from robokudo.defs import PACKAGE_NAME
from robokudo.utils.decorators import record_time, timer_decorator
from robokudo.vis.o3d_visualizer import Viewer3D
from robokudo.vis.visualizer import Visualizer

if TYPE_CHECKING:
    import numpy.typing as npt


class O3DVisualizer(Visualizer, Visualizer.Observer):
    """Open3D-based visualizer for 3D geometry data.

    This class provides visualization of 3D geometry data from pipeline annotators using
    Open3D windows. It supports:

    * 3D geometry visualization
    * Point cloud rendering
    * Camera control
    * Coordinate frame display
    * Shared visualization state

    .. note::
        This Visualizer works with a shared state and needs notifications
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the Open3D visualizer."""
        super().__init__(*args, **kwargs)

        self.viewer3d: Optional[MultiprocessedViewer3D] = None
        """Open3D viewer instance"""

        self.shared_visualizer_state.register_observer(self)

    def notify(
        self,
        observable: Visualizer.Observable,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Handle notification of state changes.

        :param observable: The object that sent the notification
        """
        self.update_output = True

    @timer_decorator
    def tick(self) -> None:
        """Update the visualization display.

        This method:

        * Initializes viewer if needed
        * Gets current annotator outputs
        * Updates display if needed
        * Handles viewer lifecycle

        :returns: False if visualization should terminate, True otherwise
        """
        if self.viewer3d is None:
            self.viewer3d = MultiprocessedViewer3D(self.window_title() + "_3D")

        annotator_outputs = self.get_visualized_annotator_outputs_for_pipeline()

        active_annotator_instance: BaseAnnotator = (
            self.shared_visualizer_state.active_annotator
        )

        self.update_output_flag_for_new_data()

        if self.update_output:
            self.update_output = False

            geometries = None
            # We might not yet have visual output set up for this annotator
            # This might happen in dynamic perception pipelines, where annotators have not been set up
            # during construction of the tree AND don't generate cloud outputs.
            # => Fetch geometry if present
            if active_annotator_instance.name in annotator_outputs.outputs:
                geometries = annotator_outputs.outputs[
                    active_annotator_instance.name
                ].geometries

            self.viewer3d.update_cloud(geometries)

        tick_result = (
            self.viewer3d.tick()
        )  # right now, this is the last update call. if that's true, the GUI is happy.

        if not tick_result:
            self.indicate_termination_var = True

    def window_title(self) -> str:
        """Get the window title for this visualizer."""
        return self.identifier()


@dataclass(slots=True)
class MemoryMap(object):
    """A base memory map for shared memory."""

    size: int
    """Size of the underlying data in bytes."""


@dataclass(slots=True)
class ArrayMemoryMap(MemoryMap):
    """A memory map for a numpy array in shared memory."""

    shape: Tuple
    """Shape of the underlying array."""

    dtype: str
    """Datatype of the underlying data as a string."""

    @classmethod
    def from_numpy_array(cls, array: npt.NDArray) -> "ArrayMemoryMap":
        """Create a new memory map for the given numpy array."""
        return cls(
            shape=array.shape,
            dtype=str(array.dtype),
            size=array.size * array.dtype.itemsize,
        )


@dataclass(slots=True, kw_only=True)
class GeometryMemoryMap(MemoryMap):
    """A memory map for a geometry in shared memory."""

    name: str
    """Name of the underlying geometry."""

    type: Type

    # TODO: Handle this
    material: Optional[o3d.visualization.rendering.MaterialRecord] = None

    group: Optional[str] = None

    time: Optional[float] = None

    is_visible: Optional[bool] = None

    mapped_attributes = []
    """A list of (attribute name, attribute type) for the open3d attributes mapped by the memory map."""

    @classmethod
    def from_geometry(
        cls,
        name: str,
        geometry: o3d.geometry.Geometry3D,
        material: Optional[o3d.visualization.rendering.MaterialRecord] = None,
        group: Optional[str] = None,
        time: Optional[float] = None,
        is_visible: Optional[bool] = None,
    ) -> "GeometryMemoryMap":
        """Create a new memory memory map for the given geometry."""
        size = 0
        attribute_dict: Dict[str, ArrayMemoryMap] = {}

        for attribute, _ in cls.mapped_attributes:
            attribute_dict[attribute] = ArrayMemoryMap.from_numpy_array(
                np.asarray(getattr(geometry, attribute))
            )
            size += attribute_dict[attribute].size

        return cls(
            name=name,
            type=type(geometry),
            material=material,
            group=group,
            time=time,
            is_visible=is_visible,
            size=size,
            **attribute_dict,
        )

    @classmethod
    def from_geometry_dict(cls, geometry: Dict) -> "GeometryMemoryMap":
        """Create a new memory map from a geometry dictionary."""
        instance = cls.from_geometry(**geometry)
        return instance

    def as_geometry_dict(
        self, shm: shared_memory.SharedMemory, read_idx: int
    ) -> Tuple[Dict, int]:
        """Create an open3d geometry dict from the memory map."""
        geometry, read_idx = self.to_geometry(shm, read_idx)

        geometry_dict: Dict[str, Any] = {
            "name": self.name,
            "geometry": geometry,
        }

        if self.material is not None:
            geometry_dict["material"] = self.material
        if self.group is not None:
            geometry_dict["group"] = self.group
        if self.time is not None:
            geometry_dict["time"] = self.time
        if self.is_visible is not None:
            geometry_dict["is_visible"] = self.is_visible

        return geometry_dict, read_idx

    def write_geometry(
        self,
        shm: shared_memory.SharedMemory,
        write_idx: int,
        geometry: o3d.geometry.Geometry3D,
    ) -> int:
        """Write the given geometry to the shared memory using the memory map."""
        write_buf = shm.buf

        for attribute, _ in self.mapped_attributes:
            attribute_map = getattr(self, attribute)
            if attribute_map.size == 0:
                continue
            buf = np.ndarray(
                attribute_map.shape,
                dtype=attribute_map.dtype,
                buffer=write_buf[write_idx : write_idx + attribute_map.size],
            )
            buf[:] = np.asarray(getattr(geometry, attribute))[:]
            write_idx += attribute_map.size
        return write_idx

    def to_geometry(
        self, shm: shared_memory.SharedMemory, read_idx: int
    ) -> Tuple[o3d.geometry.PointCloud, int]:
        """Read the geometry from the shared memory using the memory map."""
        read_buf = shm.buf

        geometry = self.type()
        for attribute, attribute_type in self.mapped_attributes:
            attribute_map = getattr(self, attribute)
            if attribute_map.size == 0:
                continue
            buf = np.ndarray(
                attribute_map.shape,
                dtype=attribute_map.dtype,
                buffer=read_buf[read_idx : read_idx + attribute_map.size],
            )
            if attribute_type == np.ndarray:
                setattr(geometry, attribute, buf)
            else:
                setattr(geometry, attribute, attribute_type(buf))
            read_idx += attribute_map.size
        return geometry, read_idx


@dataclass(slots=True)
class PointCloudMemoryMap(GeometryMemoryMap):
    points: ArrayMemoryMap
    """Memory map of the point clouds points."""

    normals: ArrayMemoryMap
    """Memory map of the point clouds point normals."""

    colors: ArrayMemoryMap
    """Memory map of the point clouds point colors."""

    covariances: ArrayMemoryMap
    """Memory map of the point clouds point covariances."""

    mapped_attributes = [
        ("points", o3d.utility.Vector3dVector),
        ("colors", o3d.utility.Vector3dVector),
        ("normals", o3d.utility.Vector3dVector),
        ("covariances", o3d.utility.Matrix3dVector),
    ]


@dataclass(slots=True)
class TriangleMeshMemoryMap(GeometryMemoryMap):
    vertices: ArrayMemoryMap
    """Memory map of the mesh vertices."""

    vertex_normals: ArrayMemoryMap
    """Memory map of the vertex normals."""

    vertex_colors: ArrayMemoryMap
    """Memory map of the vertex colors."""

    triangles: ArrayMemoryMap
    """Memory map of the mesh triangles."""

    triangle_normals: ArrayMemoryMap
    """Memory map of the mesh triangle normals."""

    triangle_uvs: ArrayMemoryMap
    """Memory map of the mesh triangle uvs."""

    adjacency_list: ArrayMemoryMap
    """Memory map of the mesh adjacency list."""

    mapped_attributes = [
        ("vertices", o3d.utility.Vector3dVector),
        ("vertex_normals", o3d.utility.Vector3dVector),
        ("vertex_colors", o3d.utility.Vector3dVector),
        ("triangles", o3d.utility.Vector3iVector),
        ("triangle_normals", o3d.utility.Vector3dVector),
        ("triangle_uvs", o3d.utility.Vector3dVector),
        ("adjacency_list", o3d.utility.Vector3dVector),
        # TODO:
        # ("textures", o3d.utility.Vector3dVector)
        # ("triangle_material_ids", o3d.utility.Vector3dVector),
    ]


@dataclass(slots=True)
class OrientedBoundingBoxMap(GeometryMemoryMap):
    center: ArrayMemoryMap
    """Memory map of the oriented bounding box center."""

    color: ArrayMemoryMap
    """Memory map of the oriented bounding box color."""

    extent: ArrayMemoryMap
    """Memory map of the oriented bounding box extent."""

    mapped_attributes = [
        ("center", np.ndarray),
        ("color", np.ndarray),
        ("extent", np.ndarray),
    ]


class GeometryMemoryMapFactory:
    """A factory class for creating geometry memory maps from open3d geometry objects."""

    geometry_proxies: Dict[Type, Type[GeometryMemoryMap]] = {
        o3d.geometry.PointCloud: PointCloudMemoryMap,
        o3d.geometry.TriangleMesh: TriangleMeshMemoryMap,
        o3d.geometry.OrientedBoundingBox: OrientedBoundingBoxMap,
    }
    """Map of open3d geometry types to their corresponding memory map types."""

    @classmethod
    def from_geometry(
        cls, name: str, geometry: o3d.geometry.Geometry3D
    ) -> GeometryMemoryMap:
        """Create a geometry proxy from a geometry3d object."""
        return cls.geometry_proxies[type(geometry)].from_geometry(name, geometry)

    @classmethod
    def from_geometry_dict(cls, geometry: Dict) -> GeometryMemoryMap:
        """Create a geometry proxy from a geometry3d object."""
        return cls.geometry_proxies[type(geometry["geometry"])].from_geometry(
            **geometry
        )


@dataclass(slots=True, frozen=True)
class MemoryMapTransport(object):
    """A message containing geometry data for visualization."""

    shm_name: str
    """The shared memory to read from."""

    memory_maps: List[GeometryMemoryMap] = field(default_factory=list)
    """The memory mappings for the geometries."""


@dataclass(slots=True)
class SharedMemoryManager(object):
    """A manager for geometries in shared memory."""

    memory_maps: List[GeometryMemoryMap] = field(default_factory=list)
    """A list of all memory maps managed by this object."""

    write_cursor: int = 0
    """The current end byte index of the shared memory (sum of all memory map sizes)."""

    read_cursor: int = 0
    """The current start byte index of the shared memory (sum of all memory map sizes)."""

    def append(self, memory_map: GeometryMemoryMap) -> int:
        """Add a memory map to the shared memory manager.

        :param memory_map: MemoryMap to add to the shared memory manager.
        :return: The byte index to start writing to for the appended memory map
        """
        self.memory_maps.append(memory_map)

        write_cursor = self.write_cursor
        self.write_cursor += memory_map.size
        return write_cursor

    def extend(self, memory_maps: List[GeometryMemoryMap]) -> List[int]:
        """Add a list of memory maps to the shared memory manager.

        :return: The byte indices to start writing to for each of the appended memory maps
        """
        self.memory_maps.extend(memory_maps)

        write_cursors = []
        for memory_map in memory_maps:
            write_cursors.append(self.write_cursor)
            self.write_cursor += memory_map.size
        return write_cursors

    def read(self) -> Iterator[Tuple[int, GeometryMemoryMap]]:
        """Read all memory maps from the shared memory manager."""
        for memory_map in self.memory_maps:
            yield self.read_cursor, memory_map
            self.read_cursor += memory_map.size

    def reset(self) -> None:
        """Reset the shared memory manager to its initial state."""
        self.write_cursor = 0
        self.read_cursor = 0
        self.memory_maps.clear()


class MultiprocessedViewer3DClient(object):
    def __init__(self, title: str, cmd_conn: Connection) -> None:
        self.rk_logger: logging.Logger = logging.getLogger(PACKAGE_NAME)
        """Logger instance"""

        self.viewer3d = Viewer3D(title)
        """Viewer3D instance for visualization."""

        self.cmd_conn = cmd_conn
        """Communication connection for sending and receiving commands from the main process."""

        self.name_to_shm = {}
        """Mapping of shared memory names to shared memory instances."""

        self.name_to_shm_manager = defaultdict(SharedMemoryManager)
        """Mapping of shared memory names to shared memory instances."""

        self.visualized_geometries: List[str] = []
        """List of the names of the currently visualized geometries"""

        self.geometries_lock = Lock()
        self.geometries = []

        self.receiver_thread = Thread(target=self.listen, daemon=True)
        """A thread for listening to commands from the main process."""

    def get_shm(self, shm_name: str) -> shared_memory.SharedMemory:
        """Get the shared memory instance for the given shared memory name."""
        if shm_name not in self.name_to_shm:
            self.name_to_shm[shm_name] = shared_memory.SharedMemory(name=shm_name)
        return self.name_to_shm[shm_name]

    def get_shm_manager(self, shm_name: str) -> SharedMemoryManager:
        return self.name_to_shm_manager[shm_name]

    def run(self) -> None:
        self.receiver_thread.start()

        def tick():
            self.viewer3d.main_vis.post_redraw()
            return True  # keep running

        o3d.visualization.gui.Application.instance.run()

    def update_geometry(self) -> None:
        with self.geometries_lock:
            self.viewer3d.update_cloud(self.geometries)

    def listen(self):
        """Listen for commands from the main process and handle them accordingly."""
        interval = 1.0 / 60.0
        frame_count = 0
        start_time = time.monotonic()

        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        self.viewer3d.main_vis.add_geometry("dummy", coordinate_frame)

        last_tick = time.monotonic() - interval
        while True:
            now = time.monotonic()
            delta = now - last_tick
            sleep_time = max(0.0, interval - delta)
            last_tick = now
            if self.cmd_conn.poll(timeout=sleep_time):
                cmd = self.cmd_conn.recv()
                if isinstance(cmd, MemoryMapTransport):
                    self.rk_logger.debug(f"Received memory map: {cmd}")
                    start_t = time.perf_counter()

                    shm = self.get_shm(cmd.shm_name)
                    shm_manager = self.get_shm_manager(cmd.shm_name)
                    shm_manager.reset()

                    # Load into memory manager
                    _ = shm_manager.extend(cmd.memory_maps)

                    # Reconstruct geometries
                    with self.geometries_lock:
                        self.geometries = []
                        for read_idx, memory_map in shm_manager.read():
                            geometry, read_idx = memory_map.as_geometry_dict(
                                shm, read_idx
                            )
                            self.geometries.append(geometry)

                    # self.viewer3d.update_cloud(geometries)
                    o3d.visualization.gui.Application.instance.post_to_main_thread(
                        self.viewer3d.main_vis, self.update_geometry
                    )

                    self.cmd_conn.send(True)
                    self.rk_logger.debug(
                        f"Processed memory map in {time.perf_counter() - start_t:.4f}s"
                    )

            # self.viewer3d.main_vis.remove_geometry("dummy")
            # self.viewer3d.main_vis.add_geometry("dummy", coordinate_frame)
            # self.viewer3d.tick()
            # frame_count += 1

            # elapsed = now - start_time
            # if elapsed > 0:
            #     avg_fps = frame_count / elapsed
            #     self.rk_logger.info(f"Average FPS: {avg_fps:.2f}")


class MultiprocessedViewer3D(object):
    """A wrapper class for the Viewer3D class to run it in a separate process."""

    def __init__(self, title: str) -> None:
        """Initialize the 3D viewer.

        :param title: Window title for the viewer
        """

        self.rk_logger: logging.Logger = logging.getLogger(PACKAGE_NAME)
        """Logger instance"""

        self.draw_queue: Queue = Queue()
        """Multiprocessing queue for triggering drawing events."""

        self.buffer_count = 2
        """Number of buffers to use for communication."""

        self.buffer_write_cursor = 0
        """Index of the shm to write to."""

        self.buffer_read_cursor = 1
        """Index of the shm to read to."""

        self.shms = [
            shared_memory.SharedMemory(create=True, size=5_000_000_000)
            for _ in range(self.buffer_count)
        ]
        """Shared memory instances for communicating with the viewer process."""

        self.shm_names = [shm.name for shm in self.shms]
        """Names of the shared memory instances."""

        self.memory_manager = [SharedMemoryManager() for _ in self.shms]
        """A manager for underlying data in shared memory."""

        parent_cmd_conn, child_cmd_conn = Pipe()

        self.parent_cmd_conn: Connection = parent_cmd_conn
        """Pipe connection for sending and receiving commands from the main process."""

        self.child_cmd_conn: Connection = child_cmd_conn
        """Pipe connection for sending and receiving commands on the visualizer process."""

        self.visualizer_process: Process = Process(
            target=self.run_visualizer, args=(title, self.child_cmd_conn)
        )
        """A process running a viewer3d instance."""

        self.visualizer_process.start()

    @staticmethod
    def run_visualizer(title: str, cmd_conn: Connection) -> None:
        """Run the viewer3d instance in a separate process.

        :param title: Window title for the viewer.
        :param cmd_conn: Connection for sending and receiving commands from the main process.
        """
        client = MultiprocessedViewer3DClient(title, cmd_conn)
        client.run()
        # client.listen()

    @property
    def _read_shm(self) -> shared_memory.SharedMemory:
        return self.shms[self.buffer_read_cursor]

    @property
    def _write_shm(self) -> shared_memory.SharedMemory:
        return self.shms[self.buffer_write_cursor]

    @property
    def _read_manager(self) -> SharedMemoryManager:
        return self.memory_manager[self.buffer_read_cursor]

    @property
    def _write_manager(self) -> SharedMemoryManager:
        return self.memory_manager[self.buffer_write_cursor]

    def _swap(self) -> int:
        self.buffer_read_cursor = (self.buffer_read_cursor + 1) % self.buffer_count
        self.buffer_write_cursor = (self.buffer_write_cursor + 1) % self.buffer_count
        return self.buffer_read_cursor

    def tick(self) -> Any:
        """Update the viewer display.

        :returns: False if visualization should terminate, True otherwise
        """

        # self.parent_cmd_conn.send("tick")
        # tick_return = self.parent_cmd_conn.recv()
        # return tick_return
        return True

    def update_cloud(
        self, geometries: Optional[Union[o3d.geometry.Geometry, Dict, List]]
    ) -> None:
        """Update the displayed geometries.

        This method updates the Open3D visualizer based on the outputs of the annotators.
        For the first update, it also sets up the camera and coordinate frame.

        :param geometries: Geometries to display. Can be:

        .. note::
            The dict format follows Open3D's draw() convention. See:
            https://github.com/isl-org/Open3D/blob/master/examples/python/visualization/draw.py
        """
        if geometries is None:
            return
        if isinstance(geometries, list) and len(geometries) == 0:
            return
        if isinstance(geometries, dict) and len(geometries) == 0:
            return

        # local method to add a single geometry. either based on the geometry being fully
        # defined with a dict or being a plain geometry object
        def add(g: Union[o3d.geometry.PointCloud, Dict], n: int) -> None:
            # Skip empty point clouds as they generate errors during the update
            if isinstance(g, o3d.geometry.PointCloud) and len(g.points) == 0:
                return

            if isinstance(g, dict):
                geometry = g["geometry"]
                memory_map = GeometryMemoryMapFactory.from_geometry_dict(g)
            else:
                try:
                    geometry = g
                    name = "Object " + str(n)
                    memory_map = GeometryMemoryMapFactory.from_geometry(name, g)
                except KeyError as e:
                    self.rk_logger.warning(
                        f"Could not create a memory map for {g}: {e}"
                    )
                    return

            write_idx = self._write_manager.append(memory_map)
            memory_map.write_geometry(self._write_shm, write_idx, geometry)

        self._write_manager.reset()

        n = 1
        if isinstance(geometries, list):
            for g in geometries:
                add(g, n)
                n += 1
        elif geometries is not None:
            add(geometries, n)

        transport = MemoryMapTransport(
            shm_name=self._write_shm.name,
            memory_maps=self._write_manager.memory_maps,
        )

        self._swap()  # Swap buffers

        self.parent_cmd_conn.send(transport)
        _ = self.parent_cmd_conn.recv()
