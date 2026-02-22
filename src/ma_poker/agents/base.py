from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Protocol


@dataclass(frozen=True)
class ActOutput:
    action: int
    info: Dict[str, Any]


class Agent(Protocol):
    """A minimal agent interface for turn-based multi-agent poker."""

    def reset(self) -> None: ...

    def act(self, obs: Any, legal_actions: List[int]) -> ActOutput: ...

