"""
The air a physical simulation moves everything through.
"""

from __future__ import annotations

from dataclasses import dataclass

from semantic_digital_twin.mixin import UniqueSimulatorProperty

# %% air


@dataclass
class Air(UniqueSimulatorProperty):
    """
    The air around everything a physical simulation of the world moves, which drags on
    whatever moves through it; a world without it is simulated in a vacuum.
    """

    density: float = 1.204
    """
    Mass per volume of the air, in kilograms per cubic metre; that of dry air at 20 °C
    and sea-level pressure by default.
    """

    viscosity: float = 1.81e-5
    """
    Dynamic viscosity of the air, in pascal seconds; that of air at 20 °C by default.
    """
