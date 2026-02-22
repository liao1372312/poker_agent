"""Opponent Analysis Agent: Analyzes opponents and predicts hand probability distributions.

This agent is responsible for:
1. Analyzing opponent behavior patterns (VPIP, aggression, etc.)
2. Learning opponent portraits (e_u vectors)
3. Predicting hand probability distributions (169-dim belief) based on action sequences
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ma_poker.agents.hand_utils import (
    cards_to_hand_type,
    get_top_hands,
    format_top_hands_for_prompt,
    INDEX_TO_HAND_TYPE,
)


def _parse_rlcard_raw_obs(obs: Any) -> Dict[str, Any]:
    """Extract raw observation dict from RLCard obs."""
    if not isinstance(obs, dict):
        return {}
    raw = obs.get("raw_obs")
    return raw if isinstance(raw, dict) else {}


@dataclass
class OpponentStats:
    """Statistics about opponent behavior."""
    vpip: float = 0.0  # Voluntarily Put money In Pot (0-1)
    pfr: float = 0.0  # Pre-Flop Raise (0-1)
    aggression: float = 0.0  # Aggression factor (0-1)
    fold_rate: float = 0.0  # Fold rate (0-1)
    call_rate: float = 0.0  # Call rate (0-1)
    raise_rate: float = 0.0  # Raise rate (0-1)
    total_actions: int = 0


@dataclass
class OpponentBelief:
    """Belief about opponent's hand distribution."""
    # 169-dimensional probability distribution over hand types
    belief: np.ndarray = field(default_factory=lambda: np.ones(169) / 169)  # Uniform prior
    # User portrait vector (embedding)
    portrait_vector: np.ndarray = field(default_factory=lambda: np.zeros(32))  # 32-dim embedding
    # Total reward accumulated from interactions with this opponent
    total_reward: float = 0.0
    # Number of interactions
    interaction_count: int = 0


@dataclass
class OpponentAnalysisAgent:
    """Agent responsible for analyzing opponents and predicting hand distributions.
    
    This agent:
    1. Tracks opponent statistics (VPIP, aggression, etc.)
    2. Maintains opponent portraits (e_u vectors)
    3. Predicts hand probability distributions based on action sequences
    4. Uses a learnable neural network (if available) or heuristic methods
    """
    
    # Belief update settings
    belief_update_rate: float = 0.1  # Learning rate for belief updates
    portrait_update_rate: float = 0.01  # Learning rate for portrait vector updates
    seed: Optional[int] = None
    
    def __post_init__(self) -> None:
        # Opponent statistics tracking
        self._opponent_stats: Dict[int, OpponentStats] = defaultdict(OpponentStats)
        self._opponent_action_history: Dict[int, List[str]] = defaultdict(list)
        
        # Store detailed action sequence for each opponent (for belief network)
        # Format: List of (action, betting_round, bet_size_ratio, context_features)
        self._opponent_action_sequence: Dict[int, List[Tuple[str, int, float, np.ndarray]]] = defaultdict(list)
        
        # Opponent belief system: 169-dim hand probability distribution per opponent
        self._opponent_beliefs: Dict[int, OpponentBelief] = defaultdict(
            lambda: OpponentBelief(
                belief=self._initialize_belief(),
                portrait_vector=self._initialize_portrait_vector(),
            )
        )
        
        # Belief network: learns P(hand_type | action_sequence, context)
        self._belief_network = None  # Will be initialized lazily if PyTorch is available
        self._use_learned_belief_network = False  # Flag to enable/disable learned network
        self._belief_training_data = []  # Store training examples
        
        # Try to initialize belief network (requires PyTorch)
        try:
            import torch
            import torch.nn as nn
            self._belief_network = self._create_belief_network()
            self._use_learned_belief_network = True
            self._belief_optimizer = None  # Will be created when training
        except ImportError:
            # PyTorch not available, use heuristic method
            self._use_learned_belief_network = False
        
        # Random number generator
        import random
        self._rng = random.Random(self.seed)
    
    def _initialize_belief(self) -> np.ndarray:
        """Initialize belief with a generic prior distribution."""
        # Start with uniform distribution (can be replaced with learned prior)
        return np.ones(169) / 169.0
    
    def _initialize_portrait_vector(self) -> np.ndarray:
        """Initialize user portrait vector."""
        # Initialize with small random values
        return np.random.normal(0, 0.1, size=32).astype(np.float32)
    
    def _create_belief_network(self):
        """Create a neural network to learn P(hand_type | action_sequence, context)."""
        try:
            import torch
            import torch.nn as nn
            
            class BeliefNetwork(nn.Module):
                """Neural network to predict hand type probability distribution given action sequence."""
                def __init__(self, max_sequence_length=10):
                    super().__init__()
                    self.max_sequence_length = max_sequence_length
                    
                    # Each action feature: action_type (3) + betting_round (4) + bet_size_ratio (1) = 8
                    action_feature_dim = 3 + 4 + 1
                    
                    # LSTM to process action sequence
                    lstm_hidden_dim = 64
                    self.lstm = nn.LSTM(
                        input_size=action_feature_dim,
                        hidden_size=lstm_hidden_dim,
                        num_layers=2,
                        batch_first=True,
                        dropout=0.2
                    )
                    
                    # Context features: current context (10) + portrait vector (32) = 42
                    context_dim = 10 + 32
                    combined_dim = lstm_hidden_dim + context_dim
                    hidden_dim = 128
                    output_dim = 169  # 169 hand types
                    
                    self.fc = nn.Sequential(
                        nn.Linear(combined_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Dropout(0.2),
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Linear(hidden_dim, output_dim),
                        nn.Softmax(dim=-1)
                    )
                
                def forward(self, action_sequence, context_features, portrait_vector):
                    """Forward pass."""
                    lstm_out, (h_n, c_n) = self.lstm(action_sequence)
                    sequence_encoding = h_n[-1]  # [batch_size, lstm_hidden_dim]
                    
                    context_combined = torch.cat([context_features, portrait_vector], dim=-1)
                    combined = torch.cat([sequence_encoding, context_combined], dim=-1)
                    
                    hand_type_probs = self.fc(combined)
                    return hand_type_probs
            
            return BeliefNetwork()
        except ImportError:
            return None
    
    def _encode_action_sequence(self, action_sequence: List[Tuple[str, int, float, np.ndarray]], max_length: int = 10) -> np.ndarray:
        """Encode action sequence into tensor format for network."""
        encoded = np.zeros((max_length, 8))
        
        seq_len = min(len(action_sequence), max_length)
        for i in range(seq_len):
            action, betting_round, bet_size_ratio, _ = action_sequence[-(seq_len-i)]
            
            # Action type one-hot: fold=0, call=1, raise=2
            action_map = {"fold": 0, "call": 1, "raise": 2, "all_in": 2, "check": 1}
            action_idx = action_map.get(action.lower(), 1)
            encoded[i, action_idx] = 1.0
            
            # Betting round one-hot: pre-flop=0, flop=1, turn=2, river=3
            betting_round = max(0, min(3, betting_round))
            encoded[i, 3 + betting_round] = 1.0
            
            # Bet size ratio (normalized)
            encoded[i, 7] = min(1.0, max(0.0, bet_size_ratio))
        
        return encoded
    
    def _extract_context_features(self, context: Dict[str, Any]) -> np.ndarray:
        """Extract context features for belief network."""
        features = np.zeros(10)
        
        # Feature 0: Public cards count (0-5)
        public_cards = context.get("public_cards", [])
        features[0] = len(public_cards) / 5.0
        
        # Feature 1: Pot size (normalized)
        pot_size = context.get("pot_size", 0)
        features[1] = min(pot_size / 1000.0, 1.0)
        
        # Feature 2: Position (0-1, normalized)
        position = context.get("position", 0)
        features[2] = position / 9.0
        
        # Feature 3: Betting round (0-3: pre-flop, flop, turn, river)
        betting_round = context.get("betting_round", 0)
        features[3] = betting_round / 3.0
        
        # Features 4-9: Reserved for future expansion
        
        return features
    
    def _compute_action_likelihoods(self, action: str, context: Dict[str, Any], portrait_vector: np.ndarray, opponent_id: Optional[int] = None) -> np.ndarray:
        """Compute likelihood of action given each hand type."""
        # If learned network is available and trained, use it
        if self._use_learned_belief_network and self._belief_network is not None and opponent_id is not None:
            try:
                import torch
                
                action_sequence = self._opponent_action_sequence.get(opponent_id, [])
                
                if len(action_sequence) > 0:
                    context_features = self._extract_context_features(context)
                    action_seq_encoded = self._encode_action_sequence(action_sequence, max_length=10)
                    
                    action_seq_tensor = torch.FloatTensor(action_seq_encoded).unsqueeze(0)
                    context_tensor = torch.FloatTensor(context_features).unsqueeze(0)
                    portrait_tensor = torch.FloatTensor(portrait_vector).unsqueeze(0)
                    
                    with torch.no_grad():
                        hand_type_probs = self._belief_network(action_seq_tensor, context_tensor, portrait_tensor)
                    
                    likelihoods = hand_type_probs[0].cpu().numpy()
                    
                    if likelihoods.sum() > 0:
                        likelihoods = likelihoods / likelihoods.sum()
                    else:
                        likelihoods = np.ones(169) / 169.0
                    
                    return likelihoods
            except Exception as e:
                print(f"Warning: Belief network failed, using heuristic: {e}")
        
        # Fallback to heuristic method
        return self._compute_action_likelihoods_heuristic(action, context, portrait_vector)
    
    def _compute_action_likelihoods_heuristic(self, action: str, context: Dict[str, Any], portrait_vector: np.ndarray) -> np.ndarray:
        """Heuristic method for computing action likelihoods (fallback)."""
        likelihoods = np.ones(169)
        
        if action == "fold":
            for i in range(169):
                hand_str = INDEX_TO_HAND_TYPE.get(i, "")
                if hand_str and len(hand_str) >= 2:
                    if hand_str[0] == hand_str[1]:
                        likelihoods[i] *= 0.3
                    elif hand_str[0] in "AKQ":
                        likelihoods[i] *= 0.5
                    else:
                        likelihoods[i] *= 1.5
        
        elif action == "raise":
            for i in range(169):
                hand_str = INDEX_TO_HAND_TYPE.get(i, "")
                if hand_str and len(hand_str) >= 2:
                    if hand_str[0] == hand_str[1]:
                        likelihoods[i] *= 2.0
                    elif hand_str[0] in "AKQ":
                        likelihoods[i] *= 1.5
                    else:
                        likelihoods[i] *= 0.5
        
        elif action == "call":
            for i in range(169):
                hand_str = INDEX_TO_HAND_TYPE.get(i, "")
                if hand_str and len(hand_str) >= 2:
                    if hand_str[0] == hand_str[1]:
                        likelihoods[i] *= 0.8
                    elif hand_str[0] in "AKQ":
                        likelihoods[i] *= 1.2
                    else:
                        likelihoods[i] *= 1.0
        
        portrait_factor = 1.0 + 0.1 * portrait_vector[0]
        likelihoods *= portrait_factor
        
        if likelihoods.sum() > 0:
            likelihoods = likelihoods / likelihoods.sum()
        
        return likelihoods
    
    def update_opponent_action(self, opponent_id: int, action: str, context: Dict[str, Any]) -> None:
        """Update opponent statistics and belief based on their action.
        
        Args:
            opponent_id: Opponent player ID
            action: Action taken by opponent
            context: Context information (public cards, pot size, etc.)
        """
        # Update statistics
        if opponent_id not in self._opponent_stats:
            self._opponent_stats[opponent_id] = OpponentStats()
        
        stats = self._opponent_stats[opponent_id]
        stats.total_actions += 1
        self._opponent_action_history[opponent_id].append(action)
        
        # Keep only recent history (last 50 actions)
        if len(self._opponent_action_history[opponent_id]) > 50:
            self._opponent_action_history[opponent_id] = self._opponent_action_history[opponent_id][-50:]
        
        # Update statistics based on recent actions
        recent_actions = self._opponent_action_history[opponent_id]
        if recent_actions:
            stats.fold_rate = recent_actions.count("fold") / len(recent_actions)
            stats.call_rate = recent_actions.count("call") / len(recent_actions)
            stats.raise_rate = recent_actions.count("raise") / len(recent_actions)
            stats.vpip = 1.0 - stats.fold_rate
            stats.pfr = stats.raise_rate
            if stats.call_rate > 0:
                stats.aggression = stats.raise_rate / (stats.raise_rate + stats.call_rate)
            else:
                stats.aggression = stats.raise_rate
        
        # Update belief
        self._update_belief(opponent_id, action, context)
    
    def _update_belief(self, opponent_id: int, action: str, context: Dict[str, Any]) -> None:
        """Update opponent's hand belief using Bayesian update."""
        if opponent_id not in self._opponent_beliefs:
            self._opponent_beliefs[opponent_id] = OpponentBelief(
                belief=self._initialize_belief(),
                portrait_vector=self._initialize_portrait_vector(),
            )
        
        belief_obj = self._opponent_beliefs[opponent_id]
        
        # Determine betting round from context
        public_cards = context.get("public_cards", [])
        betting_round = 0  # pre-flop
        if len(public_cards) >= 3:
            betting_round = 1  # flop
        if len(public_cards) >= 4:
            betting_round = 2  # turn
        if len(public_cards) >= 5:
            betting_round = 3  # river
        
        # Estimate bet size ratio
        pot_size = context.get("pot_size", 1.0)
        bet_size = context.get("bet_size", 0.0)
        bet_size_ratio = bet_size / max(pot_size, 1.0)
        
        # Extract context features
        context_features = self._extract_context_features(context)
        
        # Add current action to sequence
        self._opponent_action_sequence[opponent_id].append((action, betting_round, bet_size_ratio, context_features))
        
        # Keep only recent actions (last 20)
        if len(self._opponent_action_sequence[opponent_id]) > 20:
            self._opponent_action_sequence[opponent_id] = self._opponent_action_sequence[opponent_id][-20:]
        
        # Compute likelihood
        likelihoods = self._compute_action_likelihoods(action, context, belief_obj.portrait_vector, opponent_id)
        
        # Bayesian update
        new_belief = belief_obj.belief * likelihoods
        
        # Normalize
        if new_belief.sum() > 0:
            new_belief = new_belief / new_belief.sum()
        else:
            new_belief = self._initialize_belief()
        
        # Smooth update (exponential moving average)
        belief_obj.belief = (1 - self.belief_update_rate) * belief_obj.belief + self.belief_update_rate * new_belief
        belief_obj.belief = belief_obj.belief / belief_obj.belief.sum()  # Renormalize
    
    def update_portrait_vector(self, opponent_id: int, reward: float) -> None:
        """Update user portrait vector based on reward."""
        if opponent_id not in self._opponent_beliefs:
            return
        
        belief_obj = self._opponent_beliefs[opponent_id]
        
        # Simple gradient-based update
        update = reward * self.portrait_update_rate
        belief_obj.portrait_vector += update * np.random.normal(0, 0.1, size=32)
        belief_obj.portrait_vector = np.clip(belief_obj.portrait_vector, -1.0, 1.0)
        
        # Update statistics
        belief_obj.total_reward += reward
        belief_obj.interaction_count += 1
    
    def get_opponent_stats(self, opponent_id: int) -> OpponentStats:
        """Get opponent statistics."""
        if opponent_id not in self._opponent_stats:
            self._opponent_stats[opponent_id] = OpponentStats()
        return self._opponent_stats[opponent_id]
    
    def get_opponent_belief(self, opponent_id: int) -> OpponentBelief:
        """Get opponent belief (hand probability distribution)."""
        if opponent_id not in self._opponent_beliefs:
            self._opponent_beliefs[opponent_id] = OpponentBelief(
                belief=self._initialize_belief(),
                portrait_vector=self._initialize_portrait_vector(),
            )
        return self._opponent_beliefs[opponent_id]
    
    def get_top_hands_for_opponent(self, opponent_id: int, top_k: int = 10) -> List[Tuple[int, float, str]]:
        """Get top K most likely hands for an opponent."""
        belief_obj = self.get_opponent_belief(opponent_id)
        return get_top_hands(belief_obj.belief.tolist(), top_k=top_k)
    
    def add_belief_training_example(self, hand_type_idx: int, action_sequence: List[Tuple[str, int, float, np.ndarray]], context_features: np.ndarray, opponent_id: int) -> None:
        """Add a training example for belief network."""
        self._belief_training_data.append((hand_type_idx, action_sequence, context_features, opponent_id))
    
    def train_belief_network(self) -> None:
        """Train belief network using collected training data."""
        if not self._use_learned_belief_network or self._belief_network is None:
            return
        
        if len(self._belief_training_data) == 0:
            return
        
        try:
            import torch
            import torch.nn as nn
            
            if self._belief_optimizer is None:
                self._belief_optimizer = torch.optim.Adam(self._belief_network.parameters(), lr=0.001)
            
            action_sequences = []
            context_features_list = []
            portrait_vectors_list = []
            hand_type_targets = []
            
            for hand_type_idx, action_sequence, context_features, opponent_id in self._belief_training_data:
                if opponent_id in self._opponent_beliefs:
                    portrait_vector = self._opponent_beliefs[opponent_id].portrait_vector
                else:
                    portrait_vector = self._initialize_portrait_vector()
                
                action_seq_encoded = self._encode_action_sequence(action_sequence, max_length=10)
                
                action_sequences.append(torch.FloatTensor(action_seq_encoded))
                context_features_list.append(torch.FloatTensor(context_features))
                portrait_vectors_list.append(torch.FloatTensor(portrait_vector))
                hand_type_targets.append(hand_type_idx)
            
            batch_size = min(32, len(self._belief_training_data))
            num_batches = (len(self._belief_training_data) + batch_size - 1) // batch_size
            
            self._belief_network.train()
            criterion = nn.CrossEntropyLoss()
            
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(self._belief_training_data))
                
                batch_action_seqs = torch.stack(action_sequences[start_idx:end_idx])
                batch_context = torch.stack(context_features_list[start_idx:end_idx])
                batch_portrait = torch.stack(portrait_vectors_list[start_idx:end_idx])
                batch_targets = torch.LongTensor(hand_type_targets[start_idx:end_idx])
                
                self._belief_optimizer.zero_grad()
                hand_type_probs = self._belief_network(batch_action_seqs, batch_context, batch_portrait)
                
                log_probs = torch.log(hand_type_probs + 1e-8)
                loss = criterion(log_probs, batch_targets)
                
                loss.backward()
                self._belief_optimizer.step()
            
            self._belief_network.eval()
            self._belief_training_data = []  # Clear after training
            
        except Exception as e:
            print(f"Warning: Failed to train belief network: {e}")
    
    def save_memory(self, path: str) -> None:
        """Save opponent analysis data to file."""
        data = {
            "opponent_stats": {
                str(pid): {
                    "vpip": stats.vpip,
                    "pfr": stats.pfr,
                    "aggression": stats.aggression,
                    "fold_rate": stats.fold_rate,
                    "call_rate": stats.call_rate,
                    "raise_rate": stats.raise_rate,
                    "total_actions": stats.total_actions,
                }
                for pid, stats in self._opponent_stats.items()
            },
            "opponent_beliefs": {
                str(pid): {
                    "belief": belief_obj.belief.tolist(),
                    "portrait_vector": belief_obj.portrait_vector.tolist(),
                    "total_reward": belief_obj.total_reward,
                    "interaction_count": belief_obj.interaction_count,
                }
                for pid, belief_obj in self._opponent_beliefs.items()
            },
            "use_learned_belief_network": self._use_learned_belief_network,
        }
        
        # Save belief network weights if available
        if self._use_learned_belief_network and self._belief_network is not None:
            try:
                import torch
                network_path = path.replace(".json", "_belief_network.pt")
                torch.save(self._belief_network.state_dict(), network_path)
                data["belief_network_path"] = network_path
            except Exception as e:
                print(f"Warning: Failed to save belief network: {e}")
        
        Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")
    
    def load_memory(self, path: str) -> None:
        """Load opponent analysis data from file."""
        if not Path(path).exists():
            return
        
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        
        # Load opponent stats
        opponent_stats_data = data.get("opponent_stats", {})
        self._opponent_stats.clear()
        for pid_str, stats_data in opponent_stats_data.items():
            pid = int(pid_str)
            stats = OpponentStats(
                vpip=stats_data.get("vpip", 0.0),
                pfr=stats_data.get("pfr", 0.0),
                aggression=stats_data.get("aggression", 0.0),
                fold_rate=stats_data.get("fold_rate", 0.0),
                call_rate=stats_data.get("call_rate", 0.0),
                raise_rate=stats_data.get("raise_rate", 0.0),
                total_actions=stats_data.get("total_actions", 0),
            )
            self._opponent_stats[pid] = stats
        
        # Load opponent beliefs
        opponent_beliefs_data = data.get("opponent_beliefs", {})
        self._opponent_beliefs.clear()
        for pid_str, belief_data in opponent_beliefs_data.items():
            pid = int(pid_str)
            belief_obj = OpponentBelief(
                belief=np.array(belief_data.get("belief", [1.0/169] * 169)),
                portrait_vector=np.array(belief_data.get("portrait_vector", [0.0] * 32)),
                total_reward=belief_data.get("total_reward", 0.0),
                interaction_count=belief_data.get("interaction_count", 0),
            )
            self._opponent_beliefs[pid] = belief_obj
        
        # Load belief network weights if available
        if data.get("use_learned_belief_network", False) and self._use_learned_belief_network:
            network_path = data.get("belief_network_path")
            if network_path and Path(network_path).exists():
                try:
                    import torch
                    if self._belief_network is None:
                        self._belief_network = self._create_belief_network()
                    self._belief_network.load_state_dict(torch.load(network_path))
                    self._belief_network.eval()
                except Exception as e:
                    print(f"Warning: Failed to load belief network: {e}")

