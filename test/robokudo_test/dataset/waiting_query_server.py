"""
A stand-in for RoboKudo's query action server that only holds the current query and the
query waiting to be processed.
"""

from __future__ import annotations

from dataclasses import dataclass

from robokudo_msgs.action import Query

# %% the server


@dataclass(kw_only=True)
class WaitingQueryServer:
    """
    Holds the current query and the query a client sent that no pipeline has started
    processing yet.
    """

    new_query: Query.Goal | None = None
    """
    The query waiting to be processed, or ``None`` if none is waiting.
    """

    query: Query.Goal | None = None
    """
    The current query, taken up or still waiting, or ``None`` once it ended.
    """

    def is_active(self) -> bool:
        """
        :return: Whether a pipeline took up the current query and has not finished it.
        """
        return self.query is not None and self.new_query is None

    def destroy(self) -> None:
        """
        Release nothing, as the stand-in holds no ROS resources.
        """
