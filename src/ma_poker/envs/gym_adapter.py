"""Gym adapter for RLCard environment to work with Stable-Baselines3."""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np

from ma_poker.envs.rlcard_holdem import RLCardHoldemEnv, RLCardHoldemConfig


class RLCardGymAdapter(gym.Env):
    """Adapter to convert RLCardHoldemEnv to Gymnasium interface for SB3 training.
    
    This adapter wraps a single-agent view of the multi-agent environment,
    allowing SB3 algorithms to train on it.
    """

    metadata = {"render_modes": []}

    def __init__(self, rlcard_env: RLCardHoldemEnv, player_id: int = 0):
        """Initialize adapter.
        
        Args:
            rlcard_env: The underlying RLCardHoldemEnv
            player_id: Which player this adapter represents (for multi-agent training)
        """
        super().__init__()
        self._env = rlcard_env
        self._player_id = player_id
        self._current_obs: Optional[Dict[int, Any]] = None
        self._current_player: Optional[int] = None
        
        # Get observation and action space from environment
        # RLCard obs is a dict with 'obs' key containing the vector
        obs_sample = self._env.reset()
        if obs_sample:
            sample_obs = list(obs_sample.values())[0]
            if isinstance(sample_obs, dict) and "obs" in sample_obs:
                obs_vec = sample_obs["obs"]
                if isinstance(obs_vec, np.ndarray):
                    obs_shape = obs_vec.shape
                else:
                    obs_shape = (len(obs_vec),)
            else:
                # Fallback
                obs_shape = (100,)  # Default size
        else:
            obs_shape = (100,)
        
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=obs_shape, dtype=np.float32
        )
        
        # Action space: number of possible actions
        action_size = self._env.action_space_sizes.get(player_id, 4)
        self.action_space = gym.spaces.Discrete(action_size)

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset environment and return initial observation."""
        self._current_obs = self._env.reset(seed=seed)
        self._current_player = self._env.current_player()
        
        # Get observation for current player
        obs = self._get_obs()
        info = {"player_id": self._current_player}
        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Step environment with action."""
        if self._current_obs is None:
            raise RuntimeError("Call reset() first")
        
        # Only current player can act
        if self._current_player != self._player_id:
            # Not this player's turn, return zero reward and continue
            obs = self._get_obs()
            return obs, 0.0, False, False, {"player_id": self._current_player, "not_my_turn": True}
        
        # Execute action
        step_out = self._env.step({self._player_id: action})
        self._current_obs = step_out.obs
        
        # Get reward for this player
        reward = float(step_out.rewards.get(self._player_id, 0.0))
        
        # Check if done
        terminated = step_out.terminated
        truncated = False  # RLCard doesn't have truncation
        
        # Get next observation
        if terminated:
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        else:
            obs = self._get_obs()
            self._current_player = self._env.current_player()
        
        info = {
            "player_id": self._current_player,
            "terminated": terminated,
        }
        
        return obs, reward, terminated, truncated, info

    def _get_obs(self) -> np.ndarray:
        """Extract observation vector for current player."""
        if self._current_obs is None:
            return np.zeros(self.observation_space.shape, dtype=np.float32)
        
        # Get observation for current player
        player_obs = self._current_obs.get(self._current_player)
        if player_obs is None:
            return np.zeros(self.observation_space.shape, dtype=np.float32)
        
        # Extract observation vector
        if isinstance(player_obs, dict) and "obs" in player_obs:
            obs_vec = player_obs["obs"]
        else:
            obs_vec = player_obs
        
        # Convert to numpy array
        if not isinstance(obs_vec, np.ndarray):
            obs_vec = np.array(obs_vec, dtype=np.float32)
        
        # Ensure correct shape
        if obs_vec.shape != self.observation_space.shape:
            # Pad or truncate to match
            target_size = self.observation_space.shape[0]
            if len(obs_vec) < target_size:
                obs_vec = np.pad(obs_vec, (0, target_size - len(obs_vec)), mode='constant')
            else:
                obs_vec = obs_vec[:target_size]
        
        return obs_vec.astype(np.float32)

