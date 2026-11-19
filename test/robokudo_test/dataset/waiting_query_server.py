"""
A stand-in for RoboKudo's query action server that only holds the query waiting to be
processed.
"""

from __future__ import annotations

from dataclasses import dataclass

from robokudo_msgs.action import Query

# %% the server


@dataclass(kw_only=True)
class WaitingQueryServer:
    """
    Holds the query a client sent that no pipeline has started processing yet.
    """

    new_query: Query.Goal | None = None
    """
    The query waiting to be processed, or ``None`` if none is waiting.
    """

    def destroy(self) -> None:
        """
        Release nothing, as the stand-in holds no ROS resources.
        """
