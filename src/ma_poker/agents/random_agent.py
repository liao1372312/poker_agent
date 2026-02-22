from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, List

from ma_poker.agents.base import ActOutput, Agent


@dataclass
class RandomAgent(Agent):
    seed: int | None = None

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    def reset(self) -> None:
        return

    def act(self, obs: Any, legal_actions: List[int]) -> ActOutput:
        if not legal_actions:
            raise ValueError("No legal actions provided to agent.")
        a = self._rng.choice(list(legal_actions))
        return ActOutput(action=int(a), info={})

