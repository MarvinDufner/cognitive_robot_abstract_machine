"""
Pictures of the garment on the folding bench seen from above, with the next fold its
planner decided on drawn over it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy
from matplotlib.figure import Figure
from matplotlib.patches import Circle

from experiments.garment_folding.bench import TopView
from semantic_digital_twin.folding.strategy import GarmentFoldPlan

# %% pictures


@dataclass(kw_only=True)
class NextFoldPicture:
    """
    A view of the table from above with the next fold of a garment drawn on it: the
    outline and landmarks of the state the planner recognised, the fold segment, and for
    each grasp the vertices it holds, the gripper's reach, its path and where it lays
    the cloth down.
    """

    view: TopView
    """
    The table seen from above.
    """

    plan: GarmentFoldPlan
    """
    The next fold.
    """

    gripper_radius: float
    """
    How far from its centre a gripper holds cloth, in metres.
    """

    width: float = 8.0
    """
    The width of the picture, in inches.
    """

    height: float = 8.6
    """
    The height of the picture, in inches.
    """

    dots_per_inch: int = 120
    """
    How many pixels the picture has per inch.
    """

    margin: float = 0.06
    """
    How much of the table around the garment and the grippers' paths the picture shows,
    in metres.
    """

    def save(self, *, path: Path) -> None:
        """
        :param path: The image file to write.
        """
        state = self.plan.state
        figure = Figure(figsize=(self.width, self.height))
        axes = figure.subplots()
        axes.imshow(self.view.image, extent=self.view.extent)
        for index, layer in enumerate(state.cloth.layers):
            outline = numpy.vstack([layer.vertices, layer.vertices[:1]])
            axes.plot(
                *outline.T,
                color="white",
                linewidth=0.8,
                label="recognised state" if index == 0 else None,
            )
        landmarks = numpy.array(list(state.landmark_positions.values()))
        axes.scatter(*landmarks.T, s=10, color="cyan", label="landmarks", zorder=3)
        segment = self.plan.g_fold.segment
        axes.plot(
            *numpy.array([segment.start, segment.end]).T,
            linestyle="--",
            linewidth=2,
            color="orange",
            label="fold segment",
        )
        for number, grasp in enumerate(self.plan.grasps, start=1):
            axes.plot(*grasp.path[:, :2].T, color="magenta", linewidth=1.5)
            axes.add_patch(
                Circle(
                    grasp.grasp_position,
                    self.gripper_radius,
                    fill=False,
                    color="red",
                    linewidth=1,
                )
            )
            axes.scatter(*grasp.held_vertices.T, marker="x", s=40, color="red")
            axes.scatter(*grasp.grasp_position, s=70, color="red", zorder=4)
            axes.scatter(*grasp.placement_position, s=70, color="dodgerblue", zorder=4)
            axes.annotate(
                f"grasp {number}",
                grasp.grasp_position,
                xytext=(6, 6),
                textcoords="offset points",
                color="red",
                fontweight="bold",
            )
            axes.annotate(
                f"place {number}",
                grasp.placement_position,
                xytext=(6, -12),
                textcoords="offset points",
                color="dodgerblue",
                fontweight="bold",
            )
        axes.plot([], [], color="magenta", label="gripper path, seen from above")
        axes.scatter([], [], s=70, color="red", label="grasp (reach circled)")
        axes.scatter([], [], marker="x", color="red", label="vertex the grasp holds")
        axes.scatter([], [], s=70, color="dodgerblue", label="placement")
        shown = numpy.concatenate(
            [layer.vertices for layer in state.cloth.layers]
            + [grasp.path[:, :2] for grasp in self.plan.grasps]
        )
        lower, upper = shown.min(axis=0) - self.margin, shown.max(axis=0) + self.margin
        axes.set_xlim(lower[0], upper[0])
        axes.set_ylim(lower[1], upper[1])
        axes.set_xlabel("x (m)")
        axes.set_ylabel("y (m)")
        axes.set_title(
            f"{len(state.garment.folds)} of {len(state.sequence.folds)} folds made, "
            f"fit IoU {state.overlap:.2f}\n"
            f"next: {self.plan.fold.description}, "
            f"{len(self.plan.grasps)} gripper(s)"
        )
        figure.legend(loc="lower center", ncol=4, fontsize=8, facecolor="darkgray")
        figure.tight_layout(rect=(0.0, 0.07, 1.0, 1.0))
        figure.savefig(path, dpi=self.dots_per_inch)
