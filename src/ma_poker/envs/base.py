from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, Tuple

import numpy as np


@dataclass(frozen=True)
class StepOutput:
    obs: Dict[int, Any]
    rewards: Dict[int, float]
    terminated: bool
    info: Dict[str, Any]


class MultiAgentEnv(Protocol):
    """A minimal multi-agent env interface (player-id keyed)."""

    @property
    def num_players(self) -> int: ...

    @property
    def action_space_sizes(self) -> Dict[int, int]: ...

    def reset(self, seed: Optional[int] = None) -> Dict[int, Any]: ...

    def step(self, actions: Dict[int, int]) -> StepOutput: ...

    def current_player(self) -> int: ...

    def legal_actions(self, player_id: int) -> List[int]: ...


def one_hot(index: int, size: int) -> np.ndarray:
    x = np.zeros((size,), dtype=np.float32)
    x[index] = 1.0
    return x

