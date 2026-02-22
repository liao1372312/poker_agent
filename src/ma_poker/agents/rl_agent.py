from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional

import numpy as np

from ma_poker.agents.base import ActOutput, Agent


@dataclass
class RLAgent(Agent):
    """RL policy agent (PPO/DQN) using Stable-Baselines3.

    This agent can:
    1. Load a pre-trained model from a checkpoint file
    2. Train a new model from scratch
    3. Continue training from a checkpoint
    4. Save model checkpoints during/after training
    """

    model_path: Optional[str] = None
    algorithm: str = "ppo"  # "ppo" or "dqn"
    seed: Optional[int] = None
    trainable: bool = True  # Whether this agent can be trained

    def __post_init__(self) -> None:
        try:
            from stable_baselines3 import PPO, DQN
            from stable_baselines3.common.vec_env import DummyVecEnv
        except ImportError:
            raise ImportError(
                "stable-baselines3 is required for RL agents. Install with: pip install stable-baselines3"
            )

        self._model = None
        if self.model_path:
            path = Path(self.model_path)
            if not path.exists():
                raise FileNotFoundError(f"Model checkpoint not found: {self.model_path}")
            if self.algorithm.lower() == "ppo":
                self._model = PPO.load(str(path))
            elif self.algorithm.lower() == "dqn":
                self._model = DQN.load(str(path))
            else:
                raise ValueError(f"Unknown algorithm: {self.algorithm}")

        # If no model provided, use random fallback until model is created
        if self._model is None:
            import random
            self._rng = random.Random(self.seed)
            self._use_random = True
        else:
            self._use_random = False

    def reset(self) -> None:
        return

    def act(self, obs: Any, legal_actions: List[int]) -> ActOutput:
        if not legal_actions:
            raise ValueError("No legal actions provided.")

        # If no model and trainable, use random fallback until model is trained
        if self._model is None:
            import random
            if not hasattr(self, '_rng'):
                self._rng = random.Random(self.seed)
            action = self._rng.choice(legal_actions)
            return ActOutput(action=int(action), info={"policy": f"{self.algorithm}_random_fallback"})

        if self._use_random:
            # Fallback: random action
            action = self._rng.choice(legal_actions)
            return ActOutput(action=int(action), info={"policy": f"{self.algorithm}_random_fallback"})

        # Extract observation vector from RLCard obs dict
        if isinstance(obs, dict):
            obs_vec = obs.get("obs")
            if obs_vec is None:
                # Fallback if obs format unexpected
                action = self._rng.choice(legal_actions)
                return ActOutput(action=int(action), info={"policy": f"{self.algorithm}_fallback"})
        else:
            obs_vec = obs

        if not isinstance(obs_vec, np.ndarray):
            obs_vec = np.array(obs_vec, dtype=np.float32)

        # Reshape for SB3 (needs batch dimension)
        obs_vec = obs_vec.reshape(1, -1)

        # Predict action
        action_raw, _ = self._model.predict(obs_vec, deterministic=True)

        action_id = int(action_raw[0])

        # Mask to legal actions only
        if action_id not in legal_actions:
            # If predicted action is illegal, pick closest legal action or first legal
            # Simple strategy: pick first legal action as fallback
            action_id = legal_actions[0]

        return ActOutput(action=action_id, info={"policy": self.algorithm, "raw_pred": int(action_raw[0])})

    def get_model(self):
        """Get the underlying SB3 model (for training)."""
        return self._model

    def set_model(self, model) -> None:
        """Set the underlying SB3 model (after training)."""
        self._model = model
        self._use_random = False

    def save_model(self, path: str) -> None:
        """Save model to checkpoint file."""
        if self._model is None:
            raise ValueError("No model to save. Train or load a model first.")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._model.save(str(path))

    def train_step(self, env, total_timesteps: int = 10000, save_path: Optional[str] = None) -> None:
        """Train the model for one step (for online learning).
        
        Args:
            env: Gym environment (should be RLCardGymAdapter)
            total_timesteps: Number of timesteps to train
            save_path: Optional path to save checkpoint after training
        """
        if not self.trainable:
            return
        
        if self._model is None:
            # Create new model
            from stable_baselines3 import PPO, DQN
            from stable_baselines3.common.callbacks import CheckpointCallback
            
            if self.algorithm.lower() == "ppo":
                self._model = PPO(
                    "MlpPolicy",
                    env,
                    verbose=0,
                    seed=self.seed,
                )
            elif self.algorithm.lower() == "dqn":
                self._model = DQN(
                    "MlpPolicy",
                    env,
                    verbose=0,
                    seed=self.seed,
                )
            else:
                raise ValueError(f"Unknown algorithm: {self.algorithm}")
        
        # Train
        self._model.learn(total_timesteps=total_timesteps, reset_num_timesteps=False)
        
        # Save if path provided
        if save_path:
            self.save_model(save_path)
