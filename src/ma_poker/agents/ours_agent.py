"""Ours Agent: Multi-agent architecture coordinating opponent analysis and LLM decision-making.

This agent coordinates:
1. OpponentAnalysisAgent: Analyzes opponents and predicts hand probability distributions
2. LLMDecisionAgent: Makes decisions using LLM based on opponent analysis results
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from ma_poker.agents.base import ActOutput, Agent
from ma_poker.agents.opponent_analysis_agent import OpponentAnalysisAgent, OpponentStats
from ma_poker.agents.llm_decision_agent import LLMDecisionAgent


def _parse_rlcard_raw_obs(obs: Any) -> Dict[str, Any]:
    """Extract raw observation dict from RLCard obs."""
    if not isinstance(obs, dict):
        return {}
    raw = obs.get("raw_obs")
    return raw if isinstance(raw, dict) else {}


@dataclass
class OursAgent(Agent):
    """Multi-agent architecture coordinating opponent analysis and LLM decision-making.
    
    Architecture:
    1. OpponentAnalysisAgent: Analyzes opponents, learns portraits, predicts hand distributions
    2. LLMDecisionAgent: Makes decisions using LLM based on opponent analysis
    
    Workflow:
    1. Observe opponent actions -> OpponentAnalysisAgent updates beliefs
    2. When making decision:
       - OpponentAnalysisAgent provides top hands and stats
       - LLMDecisionAgent uses this info to make decision via LLM
    3. After episode: Both agents update based on outcomes
    """
    
    # LLM settings
    api_url: str = "https://api.vveai.com/v1"
    api_key: str = "sk-nY2KbJ4rtdbjwHyK9eCe56787c3e451a9fE4Ef81274e8a9d"
    model: str = "gpt-4.1-mini"
    temperature: float = 0.7
    max_tokens: int = 200
    seed: Optional[int] = None
    
    # RL settings for prompt selection
    learning_rate: float = 0.1
    discount_factor: float = 0.9
    epsilon: float = 0.1
    epsilon_decay: float = 0.995
    epsilon_min: float = 0.05
    
    # Belief update settings
    belief_update_rate: float = 0.1
    portrait_update_rate: float = 0.01
    
    def __post_init__(self) -> None:
        # Initialize sub-agents
        self._opponent_analysis_agent = OpponentAnalysisAgent(
            belief_update_rate=self.belief_update_rate,
            portrait_update_rate=self.portrait_update_rate,
            seed=self.seed,
        )
        
        self._llm_decision_agent = LLMDecisionAgent(
            api_url=self.api_url,
            api_key=self.api_key,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            seed=self.seed,
            learning_rate=self.learning_rate,
            discount_factor=self.discount_factor,
            epsilon=self.epsilon,
            epsilon_decay=self.epsilon_decay,
            epsilon_min=self.epsilon_min,
        )
        
        # Episode tracking
        self._current_episode_id: int = 0
        self._episode_count: int = 0
    
    def reset(self) -> None:
        """Reset agent state for new episode."""
        self._current_episode_id += 1
        self._llm_decision_agent.reset()
    
    def act(self, obs: Any, legal_actions: List[int]) -> ActOutput:
        """Make decision by coordinating opponent analysis and LLM decision-making.
        
        Workflow:
        1. Extract current game state
        2. Get opponent analysis (stats and top hands) from OpponentAnalysisAgent
        3. Use LLMDecisionAgent to make decision based on analysis
        """
        if not legal_actions:
            raise ValueError("No legal actions provided.")
        
        raw = _parse_rlcard_raw_obs(obs)
        all_chips = raw.get("all_chips", [])
        num_players = len(all_chips) if all_chips else 9
        
        # Get opponent statistics (use average or most relevant opponent)
        # For simplicity, use the first opponent's stats
        # In practice, you might want to select based on context (e.g., current opponent)
        opponent_id = 0  # Can be improved to select based on context
        if opponent_id >= num_players:
            opponent_id = 0
        
        opponent_stats = self._opponent_analysis_agent.get_opponent_stats(opponent_id)
        
        # Get top hands for opponent
        top_hands = self._opponent_analysis_agent.get_top_hands_for_opponent(opponent_id, top_k=10)
        
        # Format top hands information
        from ma_poker.agents.hand_utils import format_top_hands_for_prompt
        top_hands_info = format_top_hands_for_prompt(top_hands) if top_hands else ""
        
        # Use LLMDecisionAgent to make decision
        return self._llm_decision_agent.make_decision(
            obs=obs,
            legal_actions=legal_actions,
            opponent_stats=opponent_stats,
            top_hands_info=top_hands_info,
        )
    
    def update_opponent_action(self, opponent_id: int, action: str, context: Optional[Dict[str, Any]] = None) -> None:
        """Update opponent analysis when opponent takes an action.
        
        This should be called by the environment/evaluator when an opponent acts.
        
        Args:
            opponent_id: Opponent player ID
            action: Action taken by opponent
            context: Context information (public cards, pot size, etc.)
        """
        if context is None:
            context = {}
        
        self._opponent_analysis_agent.update_opponent_action(opponent_id, action, context)
    
    def update_episode_outcome(self, final_reward: float) -> None:
        """Update both sub-agents with episode outcome."""
        # Update LLMDecisionAgent (Q-table for prompt selection)
        self._llm_decision_agent.update_episode_outcome(final_reward)
        
        # Update OpponentAnalysisAgent (portrait vectors)
        # Distribute reward among opponents (simplified - equal distribution)
        num_opponents = len(self._opponent_analysis_agent._opponent_beliefs)
        if num_opponents > 0:
            reward_per_opponent = final_reward / num_opponents
            for opponent_id in self._opponent_analysis_agent._opponent_beliefs.keys():
                self._opponent_analysis_agent.update_portrait_vector(opponent_id, reward_per_opponent)
        
        # Train belief network if training data is available
        self._opponent_analysis_agent.train_belief_network()
        
        self._episode_count += 1
    
    def add_belief_training_example(
        self,
        hand_type_idx: int,
        action_sequence: List[tuple],
        context_features: Any,
        opponent_id: int,
    ) -> None:
        """Add a training example for belief network.
        
        This should be called at the end of an episode when actual hands are revealed.
        """
        self._opponent_analysis_agent.add_belief_training_example(
            hand_type_idx, action_sequence, context_features, opponent_id
        )
    
    def save_memory(self, path: str) -> None:
        """Save memory for both sub-agents."""
        # Save opponent analysis agent memory
        analysis_path = path.replace(".json", "_analysis.json")
        self._opponent_analysis_agent.save_memory(analysis_path)
        
        # Save LLM decision agent memory
        decision_path = path.replace(".json", "_decision.json")
        self._llm_decision_agent.save_memory(decision_path)
        
        # Save main agent metadata
        from ma_poker.agents.opponent_analysis_agent import OpponentAnalysisAgent
        import json
        
        metadata = {
            "episode_count": self._episode_count,
            "current_episode_id": self._current_episode_id,
            "analysis_memory_path": analysis_path,
            "decision_memory_path": decision_path,
        }
        
        Path(path).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    
    def load_memory(self, path: str) -> None:
        """Load memory for both sub-agents."""
        if not Path(path).exists():
            return
        
        import json
        metadata = json.loads(Path(path).read_text(encoding="utf-8"))
        
        self._episode_count = metadata.get("episode_count", 0)
        self._current_episode_id = metadata.get("current_episode_id", 0)
        
        # Load sub-agent memories
        analysis_path = metadata.get("analysis_memory_path", path.replace(".json", "_analysis.json"))
        if Path(analysis_path).exists():
            self._opponent_analysis_agent.load_memory(analysis_path)
        
        decision_path = metadata.get("decision_memory_path", path.replace(".json", "_decision.json"))
        if Path(decision_path).exists():
            self._llm_decision_agent.load_memory(decision_path)
