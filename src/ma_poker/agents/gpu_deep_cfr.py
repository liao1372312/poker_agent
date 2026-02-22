"""GPU-accelerated DeepCFR implementation using PyTorch.

This is a simplified DeepCFR implementation that uses neural networks
to approximate CFR strategies and values, enabling GPU acceleration.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
import collections

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from ma_poker.agents.base import ActOutput, Agent


class StrategyNetwork(nn.Module):
    """Neural network to approximate CFR strategy (action probabilities)."""
    
    def __init__(self, obs_dim: int, num_actions: int, hidden_dim: int = 128):
        super().__init__()
        self.num_actions = num_actions
        
        self.network = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions),
        )
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Output logits for action probabilities."""
        return self.network(obs)
    
    def get_strategy(self, obs: torch.Tensor, legal_actions: List[int]) -> torch.Tensor:
        """Get strategy (action probabilities) for legal actions only."""
        logits = self.forward(obs)
        # Mask illegal actions
        mask = torch.zeros(self.num_actions, device=obs.device)
        for action in legal_actions:
            if 0 <= action < self.num_actions:
                mask[action] = 1.0
        
        # Expand mask to match batch dimension
        if len(logits.shape) == 2:
            mask = mask.unsqueeze(0).expand_as(logits)
        
        masked_logits = logits * mask + (1 - mask) * (-1e9)
        probs = torch.softmax(masked_logits, dim=-1)
        return probs


class ValueNetwork(nn.Module):
    """Neural network to approximate counterfactual values."""
    
    def __init__(self, obs_dim: int, hidden_dim: int = 128):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),  # Single value output
        )
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Output counterfactual value."""
        return self.network(obs).squeeze(-1)


@dataclass
class GPUDeepCFRAgent(Agent):
    """GPU-accelerated DeepCFR agent using PyTorch neural networks.
    
    This implementation uses neural networks to approximate:
    - Strategy network: approximates action probabilities
    - Value network: approximates counterfactual values
    
    This enables GPU acceleration for CFR training on large games.
    """
    
    game_name: str = "limit-holdem"
    iterations: int = 1000
    model_path: Optional[str] = None
    seed: Optional[int] = None
    device: Optional[str] = None  # 'cuda', 'cpu', or None (auto-detect)
    learning_rate: float = 0.0005  # Reduced for stability
    batch_size: int = 64  # Increased for better stability
    hidden_dim: int = 128
    replay_buffer_size: int = 10000  # Replay buffer for better training
    update_frequency: int = 10  # Update networks every N iterations
    
    def __post_init__(self) -> None:
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for GPUDeepCFRAgent. Install with: pip install torch")
        
        try:
            import rlcard
        except ImportError:
            raise ImportError("rlcard is required. Install with: pip install rlcard")
        
        # Determine device
        if self.device:
            self._device = torch.device(self.device)
            device_source = "user-specified"
        else:
            if torch.cuda.is_available():
                self._device = torch.device('cuda')
                device_source = "auto-detected (GPU available)"
            else:
                self._device = torch.device('cpu')
                device_source = "auto-detected (GPU not available)"
        
        # Print device information
        if self._device.type == 'cuda':
            gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Unknown"
            gpu_count = torch.cuda.device_count()
            print(f"[INFO] GPUDeepCFR: Using GPU for training")
            print(f"    Device: {self._device}")
            print(f"    GPU Name: {gpu_name}")
            print(f"    GPU Count: {gpu_count}")
            print(f"    Source: {device_source}")
        else:
            print(f"[INFO] GPUDeepCFR: Using CPU for training")
            print(f"    Device: {self._device}")
            print(f"    Source: {device_source}")
            if torch.cuda.is_available():
                print(f"    [NOTE] GPU is available but not being used. Set device='cuda' to use GPU.")
        
        # Create rlcard environment
        config = {'seed': self.seed if self.seed is not None else 0, 'allow_step_back': True}
        self._rlcard_env = rlcard.make(self.game_name, config=config)
        
        # Get observation and action dimensions
        # state_shape can be a list or tuple, we need the first element as integer
        state_shape_raw = self._rlcard_env.state_shape[0]
        if isinstance(state_shape_raw, (list, tuple)):
            # If it's a list/tuple, get the first element or product of all elements
            if len(state_shape_raw) == 1:
                obs_shape = int(state_shape_raw[0])
            else:
                # If multi-dimensional, use the product (flattened size)
                obs_shape = int(np.prod(state_shape_raw))
        else:
            # If it's already an integer
            obs_shape = int(state_shape_raw)
        
        num_actions = int(self._rlcard_env.num_actions)
        
        # Debug: print dimensions
        print(f"[DEBUG] GPUDeepCFR: obs_shape={obs_shape}, num_actions={num_actions}")
        print(f"[DEBUG] GPUDeepCFR: state_shape={self._rlcard_env.state_shape}, state_shape[0]={state_shape_raw}, type={type(state_shape_raw)}")
        
        # Initialize networks
        self.strategy_network = StrategyNetwork(obs_shape, num_actions, self.hidden_dim).to(self._device)
        self.value_network = ValueNetwork(obs_shape, self.hidden_dim).to(self._device)
        
        # Optimizers
        self.strategy_optimizer = optim.Adam(self.strategy_network.parameters(), lr=self.learning_rate)
        self.value_optimizer = optim.Adam(self.value_network.parameters(), lr=self.learning_rate)
        
        # Replay buffers for better training stability
        self.strategy_buffer: List[tuple] = []  # (obs, legal_actions, strategy_target)
        self.value_buffer: List[tuple] = []  # (obs, value_target)
        
        # Average strategy (for evaluation)
        self.average_strategy_buffer: List[tuple] = []
        
        # Training state
        self._trained = False
        self._training_iterations = 0
        
        # Try to load model if exists
        if self.model_path:
            try:
                self.load_model(self.model_path)
                self._trained = True
            except Exception as e:
                print(f"[WARN] GPUDeepCFR: Failed to load model from {self.model_path}: {e}")
    
    def reset(self) -> None:
        """Reset agent state."""
        pass
    
    def _sample_trajectory(self) -> List[Dict[str, Any]]:
        """Sample a trajectory by playing a game."""
        trajectory = []
        obs, player_id = self._rlcard_env.reset()
        done = False
        
        while not done:
            state = self._rlcard_env.get_state(player_id)
            legal_actions = list(state['legal_actions'].keys())
            
            # Get strategy from network
            obs_tensor = torch.FloatTensor(state['obs']).unsqueeze(0).to(self._device)
            with torch.no_grad():
                strategy = self.strategy_network.get_strategy(obs_tensor, legal_actions)
            
            # Sample action - extract probabilities only for legal actions
            action_probs_full = strategy.cpu().numpy()[0]  # Full probability vector (all actions)
            action_probs_legal = np.array([action_probs_full[a] for a in legal_actions])  # Only legal actions
            # Normalize to ensure probabilities sum to 1
            action_probs_legal = action_probs_legal / (action_probs_legal.sum() + 1e-8)
            action = np.random.choice(legal_actions, p=action_probs_legal)
            
            trajectory.append({
                'obs': state['obs'],
                'legal_actions': legal_actions,
                'action': action,
                'player_id': player_id,
            })
            
            obs, next_player = self._rlcard_env.step(action)
            player_id = next_player
            done = self._rlcard_env.is_over()
        
        # Calculate payoffs
        payoffs = self._rlcard_env.get_payoffs()
        for i, step in enumerate(trajectory):
            step['payoff'] = payoffs[step['player_id']]
        
        return trajectory
    
    def _compute_counterfactual_values(self, trajectory: List[Dict[str, Any]]) -> Dict[int, float]:
        """Compute counterfactual values for each information set in trajectory.
        
        Improved version: uses value network for non-terminal states and actual payoffs for terminal.
        """
        values = {}
        
        # Get final payoffs for all players
        if trajectory:
            final_payoffs = trajectory[-1].get('payoff', 0.0)
            # If payoff is a dict (per player), extract for each step's player
            if isinstance(final_payoffs, dict):
                # Use the last step's payoff dict
                final_payoffs_dict = final_payoffs
            else:
                # Single value, use for all
                final_payoffs_dict = {step['player_id']: final_payoffs for step in trajectory}
        else:
            final_payoffs_dict = {}
        
        # Backward pass: compute values from terminal to root
        for i, step in enumerate(trajectory):
            info_set_key = tuple(step['obs'])
            player_id = step['player_id']
            
            # Get payoff for this player
            if isinstance(final_payoffs_dict, dict):
                payoff = final_payoffs_dict.get(player_id, step.get('payoff', 0.0))
            else:
                payoff = step.get('payoff', 0.0)
            
            # Use value network for non-terminal states if available
            if i < len(trajectory) - 1 and self._training_iterations > 10:
                # Non-terminal: blend network value with actual payoff
                obs_tensor = torch.FloatTensor(step['obs']).unsqueeze(0).to(self._device)
                with torch.no_grad():
                    network_value = self.value_network(obs_tensor).cpu().item()
                    # Weight increases with training iterations
                    weight = min(0.7, self._training_iterations / 200.0)
                    values[info_set_key] = weight * network_value + (1 - weight) * payoff
            else:
                # Terminal or early training: use actual payoff
                values[info_set_key] = payoff
        
        return values
    
    def train(self, num_iterations: Optional[int] = None) -> None:
        """Train DeepCFR using GPU-accelerated neural networks."""
        if self._trained and num_iterations is None:
            return
        
        iterations = num_iterations or self.iterations
        
        print(f"[INFO] GPUDeepCFR: Training {self.game_name} agent for {iterations} iterations...")
        if self._device.type == 'cuda':
            gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Unknown"
            print(f"    Training Device: GPU ({gpu_name})")
        else:
            print(f"    Training Device: CPU")
        
        import time
        start_time = time.time()
        
        for iteration in range(iterations):
            # Sample trajectory
            trajectory = self._sample_trajectory()
            
            # Compute counterfactual values
            cf_values = self._compute_counterfactual_values(trajectory)
            
            # Store in replay buffer
            self._store_in_replay_buffer(trajectory, cf_values)
            
            # Update networks periodically (not every iteration for stability)
            if (iteration + 1) % self.update_frequency == 0 or (iteration + 1) == iterations:
                self._update_networks_from_buffer()
            
            self._training_iterations += 1
            
            # Progress reporting
            if (iteration + 1) % max(1, iterations // 10) == 0 or (iteration + 1) == iterations:
                elapsed = time.time() - start_time
                rate = (iteration + 1) / elapsed if elapsed > 0 else 0
                remaining = (iterations - (iteration + 1)) / rate if rate > 0 else 0
                progress_pct = 100 * (iteration + 1) / iterations
                print(f"  GPUDeepCFR training: {iteration + 1}/{iterations} ({progress_pct:.1f}%) | "
                      f"Rate: {rate:.2f} it/s | "
                      f"Elapsed: {elapsed:.1f}s | "
                      f"Remaining: {remaining:.1f}s")
        
        self._trained = True
        
        # Save model
        if self.model_path:
            self.save_model(self.model_path)
            print(f"[OK] GPUDeepCFR: Saved model to {self.model_path}")
    
    def _store_in_replay_buffer(self, trajectory: List[Dict[str, Any]], cf_values: Dict[int, float]) -> None:
        """Store training data in replay buffer."""
        for step in trajectory:
            obs = step['obs']
            legal_actions = step['legal_actions']
            action = step['action']
            player_id = step['player_id']
            
            info_set_key = tuple(obs)
            cf_value = cf_values.get(info_set_key, 0.0)
            
            # Store value data
            self.value_buffer.append((obs, cf_value))
            if len(self.value_buffer) > self.replay_buffer_size:
                self.value_buffer.pop(0)  # Remove oldest
            
            # Store strategy data
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self._device)
            with torch.no_grad():
                current_strategy = self.strategy_network.get_strategy(obs_tensor, legal_actions)
            
            # Compute regret (improved: consider all actions, not just taken one)
            regrets = {}
            for a in legal_actions:
                if a == action:
                    regrets[a] = cf_value
                else:
                    # Use value network to estimate value of other actions
                    regrets[a] = 0.0  # Simplified: could use value network here
            
            # Regret matching
            positive_regrets = {a: max(r, 0) for a, r in regrets.items()}
            total_regret = sum(positive_regrets.values())
            if total_regret > 0:
                strategy_target = {a: r / total_regret for a, r in positive_regrets.items()}
            else:
                strategy_target = {a: 1.0 / len(legal_actions) for a in legal_actions}
            
            self.strategy_buffer.append((obs, legal_actions, strategy_target))
            if len(self.strategy_buffer) > self.replay_buffer_size:
                self.strategy_buffer.pop(0)  # Remove oldest
    
    def _update_networks_from_buffer(self) -> None:
        """Update networks from replay buffer."""
        if len(self.value_buffer) > 0:
            self._update_value_network(self.value_buffer)
        if len(self.strategy_buffer) > 0:
            self._update_strategy_network(self.strategy_buffer)
    
    def _update_networks(self, trajectory: List[Dict[str, Any]], cf_values: Dict[int, float]) -> None:
        """Update strategy and value networks."""
        # Collect training data
        strategy_data = []
        value_data = []
        
        for step in trajectory:
            obs = step['obs']
            legal_actions = step['legal_actions']
            action = step['action']
            player_id = step['player_id']
            
            info_set_key = tuple(obs)
            cf_value = cf_values.get(info_set_key, 0.0)
            
            # Value network target
            value_data.append((obs, cf_value))
            
            # Strategy network target (simplified: use regret matching)
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self._device)
            with torch.no_grad():
                current_strategy = self.strategy_network.get_strategy(obs_tensor, legal_actions)
            
            # Compute regret (simplified)
            regrets = {}
            for a in legal_actions:
                if a == action:
                    regrets[a] = cf_value
                else:
                    regrets[a] = 0.0
            
            # Regret matching
            positive_regrets = {a: max(r, 0) for a, r in regrets.items()}
            total_regret = sum(positive_regrets.values())
            if total_regret > 0:
                strategy_target = {a: r / total_regret for a, r in positive_regrets.items()}
            else:
                strategy_target = {a: 1.0 / len(legal_actions) for a in legal_actions}
            
            strategy_data.append((obs, legal_actions, strategy_target))
        
        # Update value network
        if value_data:
            self._update_value_network(value_data)
        
        # Update strategy network
        if strategy_data:
            self._update_strategy_network(strategy_data)
    
    def _update_value_network(self, value_data: List[tuple]) -> None:
        """Update value network using batch training."""
        if len(value_data) < self.batch_size:
            return
        
        # Sample batch
        if len(value_data) <= self.batch_size:
            batch_indices = list(range(len(value_data)))
        else:
            batch_indices = np.random.choice(len(value_data), self.batch_size, replace=False).tolist()
        
        obs_batch = torch.FloatTensor([value_data[i][0] for i in batch_indices]).to(self._device)
        value_targets = torch.FloatTensor([value_data[i][1] for i in batch_indices]).to(self._device)
        
        # Forward pass
        self.value_optimizer.zero_grad()
        values = self.value_network(obs_batch)
        loss = nn.MSELoss()(values, value_targets)
        
        # Backward pass
        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.value_network.parameters(), max_norm=1.0)
        self.value_optimizer.step()
    
    def _update_strategy_network(self, strategy_data: List[tuple]) -> None:
        """Update strategy network using batch training."""
        if len(strategy_data) < self.batch_size:
            return
        
        # Sample batch
        batch_indices = np.random.choice(len(strategy_data), min(self.batch_size, len(strategy_data)), replace=False)
        
        obs_batch = []
        strategy_targets_batch = []
        legal_actions_batch = []
        
        for idx in batch_indices:
            obs, legal_actions, strategy_target = strategy_data[idx]
            obs_batch.append(obs)
            legal_actions_batch.append(legal_actions)
            
            # Convert strategy target to tensor
            target = np.zeros(self._rlcard_env.num_actions)
            for action, prob in strategy_target.items():
                if 0 <= action < len(target):
                    target[action] = prob
            strategy_targets_batch.append(target)
        
        obs_tensor = torch.FloatTensor(obs_batch).to(self._device)
        strategy_targets = torch.FloatTensor(strategy_targets_batch).to(self._device)
        
        # Forward pass
        self.strategy_optimizer.zero_grad()
        strategies = []
        for i, legal_actions in enumerate(legal_actions_batch):
            strategy = self.strategy_network.get_strategy(obs_tensor[i:i+1], legal_actions)
            strategies.append(strategy)
        
        strategies_tensor = torch.stack(strategies).squeeze(1)
        
        # KL divergence loss
        loss = nn.KLDivLoss(reduction='batchmean')(
            torch.log(strategies_tensor + 1e-8),
            strategy_targets
        )
        
        # Backward pass
        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.strategy_network.parameters(), max_norm=1.0)
        self.strategy_optimizer.step()
    
    def eval_step(self, state: Dict[str, Any]) -> tuple:
        """Get action and action probabilities for given state.
        
        This model is for limit-holdem (num_actions=4, IDs 0..3). If state contains
        action IDs >= num_actions (e.g. from no-limit env), we only use in-range IDs
        for strategy and fallback to first legal action when none are in range.
        """
        obs = state['obs']
        legal_actions = list(state['legal_actions'].keys())
        num_actions = self.strategy_network.num_actions
        legal_in_range = [a for a in legal_actions if 0 <= a < num_actions]
        if legal_actions and not legal_in_range:
            print(f"[WARN] GPUDeepCFR: All legal action IDs {legal_actions} are out of range [0,{num_actions-1}] "
                  "(limit-holdem). Use limit-holdem env for DeepCFR. Falling back to first legal action.")
            return (legal_actions[0], {a: 0.0 for a in legal_actions})
        use_actions = legal_in_range if legal_in_range else legal_actions
        
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self._device)
        with torch.no_grad():
            strategy = self.strategy_network.get_strategy(obs_tensor, use_actions)
        
        action_probs_full = strategy.cpu().numpy()[0]
        action_probs_legal = np.array([action_probs_full[a] for a in use_actions], dtype=np.float64)
        action_probs_legal = action_probs_legal / (action_probs_legal.sum() + 1e-8)
        action = int(np.random.choice(use_actions, p=action_probs_legal))
        action_probs_dict = {a: float(action_probs_full[a]) for a in legal_actions if 0 <= a < num_actions}
        return action, action_probs_dict
    
    def save_model(self, path: str) -> None:
        """Save model to disk."""
        model_dir = Path(path)
        model_dir.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            'strategy_network': self.strategy_network.state_dict(),
            'value_network': self.value_network.state_dict(),
            'strategy_optimizer': self.strategy_optimizer.state_dict(),
            'value_optimizer': self.value_optimizer.state_dict(),
            'training_iterations': self._training_iterations,
        }, path)
    
    def load_model(self, path: str) -> None:
        """Load model from disk."""
        checkpoint = torch.load(path, map_location=self._device)
        self.strategy_network.load_state_dict(checkpoint['strategy_network'])
        self.value_network.load_state_dict(checkpoint['value_network'])
        self.strategy_optimizer.load_state_dict(checkpoint['strategy_optimizer'])
        self.value_optimizer.load_state_dict(checkpoint['value_optimizer'])
        self._training_iterations = checkpoint.get('training_iterations', 0)
        print(f"[OK] GPUDeepCFR: Loaded model from {path}")
    
    def act(self, obs: Any, legal_actions: List[int]) -> ActOutput:
        """Select action using DeepCFR strategy."""
        if not legal_actions:
            raise ValueError("No legal actions provided.")
        
        # Train if not trained yet
        if not self._trained:
            self.train()
        
        # Convert observation to state format (model expects limit-holdem 72-dim)
        try:
            sh0 = self._rlcard_env.state_shape[0]
            expected_len = int(np.prod(sh0)) if isinstance(sh0, (list, tuple)) else int(sh0)
            obs_vec = obs.get("obs") if isinstance(obs, dict) else obs
            if obs_vec is None:
                if not getattr(self, "_warned_obs_missing", False):
                    print("[WARN] GPUDeepCFR: obs['obs'] missing — using zeros. Use env_id: limit-holdem when evaluating DeepCFR.")
                    self._warned_obs_missing = True
                obs_vec = np.zeros(expected_len, dtype=np.float32)
            else:
                if not isinstance(obs_vec, np.ndarray):
                    obs_vec = np.array(obs_vec, dtype=np.float32)
                n = len(obs_vec)
                if n != expected_len:
                    if not getattr(self, "_warned_obs_len", False):
                        print(f"[WARN] GPUDeepCFR: obs length {n} != model expected {expected_len} (limit-holdem). Resizing. Use env_id: limit-holdem for DeepCFR.")
                        self._warned_obs_len = True
                    obs_vec = np.concatenate([obs_vec, np.zeros(expected_len - n, dtype=np.float32)]) if n < expected_len else np.array(obs_vec[:expected_len], dtype=np.float32)
            
            # Convert legal actions to dict format
            legal_actions_dict = {int(a): None for a in legal_actions}
            
            state = {
                'obs': obs_vec,
                'legal_actions': legal_actions_dict,
            }
            
            # Get action from network
            action_id, action_probs = self.eval_step(state)
            
            # Ensure action_id is in legal_actions
            if action_id not in legal_actions:
                action_id = legal_actions[0] if legal_actions else 0
            
            return ActOutput(
                action=int(action_id),
                info={
                    "policy": "gpu_deep_cfr",
                    "action_probs": {int(a): float(action_probs.get(a, 0.0)) for a in legal_actions},
                    "game": self.game_name,
                }
            )
        except Exception as e:
            # Fallback to random
            import random
            rng = random.Random(self.seed)
            action = rng.choice(legal_actions)
            return ActOutput(
                action=int(action),
                info={"policy": "gpu_deep_cfr_fallback_random", "error": str(e)}
            )
