from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from ma_poker.envs.base import MultiAgentEnv, StepOutput


@dataclass
class RLCardHoldemConfig:
    """RLCard env name and options.

    env_id:
      - "limit-holdem" (Texas Hold'em Limit)
      - "no-limit-holdem" (Texas Hold'em No-Limit)  # depending on RLCard version
    
    Note: RLCard's limit-holdem uses very small initial chips (1-2 units).
    This is by design for the limit hold'em variant. Each player typically starts
    with minimal chips relative to blinds, making it a short-stack game.
    """

    env_id: str = "limit-holdem"
    num_players: int = 2


class RLCardHoldemEnv(MultiAgentEnv):
    """A thin adapter around RLCard's env to a simple multi-agent interface.

    RLCard is turn-based: only one player acts each step.
    We keep `step(actions)` signature for consistency; only action of `current_player()` is used.
    """

    def __init__(self, cfg: RLCardHoldemConfig):
        import rlcard  # local import to keep optionality

        self._cfg = cfg
        self._env = rlcard.make(cfg.env_id, config={"game_num_players": int(cfg.num_players)})
        self._num_players = int(self._env.num_players)
        self._last_obs: Optional[Dict[str, Any]] = None

        # action space is shared across players in rlcard
        self._action_size = int(self._env.num_actions)

    @property
    def num_players(self) -> int:
        return self._num_players

    @property
    def action_space_sizes(self) -> Dict[int, int]:
        return {pid: self._action_size for pid in range(self._num_players)}

    def reset(self, seed: Optional[int] = None) -> Dict[int, Any]:
        if seed is not None:
            # best-effort: RLCard does not expose a stable unified seed API across versions
            np.random.seed(seed)
        obs, player_id = self._env.reset()
        self._last_obs = obs
        return {int(player_id): obs}

    def current_player(self) -> int:
        if self._last_obs is None:
            raise RuntimeError("Call reset() first.")
        return int(self._env.get_player_id())

    def legal_actions(self, player_id: int) -> List[int]:
        if self._last_obs is None:
            raise RuntimeError("Call reset() first.")
        if int(player_id) != int(self._env.get_player_id()):
            # Only current player has legal moves in turn-based rlcard step
            return []
        legal = self._last_obs.get("legal_actions", {})
        # RLCard returns OrderedDict with action_id as keys
        if isinstance(legal, dict):
            return sorted([int(a) for a in legal.keys()])
        return []

    def step(self, actions: Dict[int, int]) -> StepOutput:
        if self._last_obs is None:
            raise RuntimeError("Call reset() first.")

        cur = int(self._env.get_player_id())
        if cur not in actions:
            raise ValueError(f"Missing action for current player {cur}. Got keys={list(actions.keys())}")

        action = int(actions[cur])
        next_obs, next_player = self._env.step(action)
        self._last_obs = next_obs

        done = self._env.is_over()
        
        # RLCard only provides payoffs when game is over
        if done:
            payoffs = self._env.get_payoffs()
            reward_dict = {pid: float(payoffs[pid]) for pid in range(self._num_players)}
        else:
            # During game, rewards are 0 (only terminal rewards matter)
            reward_dict = {pid: 0.0 for pid in range(self._num_players)}

        obs_dict: Dict[int, Any] = {}
        if not done:
            obs_dict[int(next_player)] = next_obs

        info = {
            "rlcard": {
                "next_player": int(next_player),
                "raw_obs_keys": list(next_obs.keys()) if isinstance(next_obs, dict) else None,
            }
        }
        return StepOutput(obs=obs_dict, rewards=reward_dict, terminated=bool(done), info=info)

