"""LLM Decision Agent: Makes decisions using LLM based on opponent analysis results.

This agent:
1. Receives opponent analysis results (top hands, stats, etc.)
2. Uses LLM with prompt templates to make decisions
3. Uses RL to select the best prompt template based on opponent behavior
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ma_poker.agents.base import ActOutput
from ma_poker.agents.hand_utils import format_top_hands_for_prompt


def _parse_rlcard_raw_obs(obs: Any) -> Dict[str, Any]:
    """Extract raw observation dict from RLCard obs."""
    if not isinstance(obs, dict):
        return {}
    raw = obs.get("raw_obs")
    return raw if isinstance(raw, dict) else {}


def _format_cards(cards: List[str]) -> str:
    """Format card list to readable string."""
    if not cards:
        return "None"
    return ", ".join(cards)


@dataclass
class OpponentStats:
    """Statistics about opponent behavior."""
    vpip: float = 0.0
    pfr: float = 0.0
    aggression: float = 0.0
    fold_rate: float = 0.0
    call_rate: float = 0.0
    raise_rate: float = 0.0
    total_actions: int = 0


@dataclass
class LLMDecisionAgent:
    """Agent responsible for making decisions using LLM.
    
    This agent:
    1. Receives opponent analysis results from OpponentAnalysisAgent
    2. Selects prompt template using RL based on opponent behavior
    3. Calls LLM to make decisions
    """
    
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
    
    def __post_init__(self) -> None:
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai library is required. Install with: pip install openai")
        
        self._client = OpenAI(api_key=self.api_key, base_url=self.api_url)
        
        # 5 prompt types
        self._prompt_types = [
            "balanced_gto",
            "loose_passive",
            "loose_aggressive",
            "tight_passive",
            "tight_aggressive",
        ]
        
        # Q-table: state (opponent stats discretized) -> action (prompt type) -> Q-value
        self._q_table: Dict[Tuple[int, int, int], Dict[int, float]] = defaultdict(
            lambda: defaultdict(float)
        )
        
        # Episode tracking
        self._episode_prompt_selections: List[Tuple[int, Tuple[int, int, int]]] = []  # (prompt_type_idx, state)
        self._episode_count: int = 0
        
        # Random number generator
        import random
        self._rng = random.Random(self.seed)
    
    def _get_prompt_template(self, prompt_type: str) -> str:
        """Get prompt template for a specific type from file."""
        current_file = Path(__file__)
        prompts_dir = current_file.parent / "prompts"
        prompt_file = prompts_dir / f"{prompt_type}.md"
        
        if prompt_file.exists():
            try:
                return prompt_file.read_text(encoding="utf-8").strip()
            except Exception as e:
                print(f"Warning: Failed to read prompt file {prompt_file}: {e}")
        
        # Fallback to default template
        default_template = """You are an expert Texas Hold'em poker player.

Current situation:
- Your hand: {hand}
- Public cards: {public_cards}
- Your chips: {my_chips}
- All chips: {all_chips}
- Legal actions: {legal_actions}

{opponent_analysis}

Make a strategic decision. Respond with the action name (e.g., 'call', 'raise', 'fold', 'check')."""
        
        return default_template
    
    def _discretize_state(self, opponent_stats: OpponentStats) -> Tuple[int, int, int]:
        """Discretize opponent stats into state representation for Q-table."""
        # Discretize VPIP: 0-0.3 (tight), 0.3-0.6 (medium), 0.6-1.0 (loose)
        vpip_bin = 0 if opponent_stats.vpip < 0.3 else (1 if opponent_stats.vpip < 0.6 else 2)
        
        # Discretize aggression: 0-0.3 (passive), 0.3-0.6 (medium), 0.6-1.0 (aggressive)
        aggression_bin = 0 if opponent_stats.aggression < 0.3 else (1 if opponent_stats.aggression < 0.6 else 2)
        
        # Tightness = 1 - VPIP
        tightness = 1.0 - opponent_stats.vpip
        tightness_bin = 0 if tightness < 0.3 else (1 if tightness < 0.6 else 2)
        
        return (vpip_bin, aggression_bin, tightness_bin)
    
    def _select_prompt_with_rl(self, state: Tuple[int, int, int]) -> int:
        """Select prompt type using epsilon-greedy Q-learning."""
        if self._rng.random() < self.epsilon:
            return self._rng.randint(0, len(self._prompt_types) - 1)
        else:
            q_values = self._q_table[state]
            if not q_values:
                return self._rng.randint(0, len(self._prompt_types) - 1)
            
            best_action = max(q_values.items(), key=lambda x: x[1])[0]
            return best_action
    
    def _update_q_table(self, state: Tuple[int, int, int], action: int, reward: float, next_state: Optional[Tuple[int, int, int]] = None) -> None:
        """Update Q-table using Q-learning."""
        current_q = self._q_table[state][action]
        
        if next_state is not None:
            next_max_q = max(self._q_table[next_state].values()) if self._q_table[next_state] else 0.0
            target_q = reward + self.discount_factor * next_max_q
        else:
            target_q = reward
        
        new_q = current_q + self.learning_rate * (target_q - current_q)
        self._q_table[state][action] = new_q
    
    def _call_llm(self, prompt: str) -> str:
        """Call LLM API and return response text."""
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"ERROR: {str(e)}"
    
    def _parse_action_from_response(self, response: str, legal_actions: List[int], raw_legal_actions: List[str]) -> int:
        """Parse action from LLM response text."""
        response_lower = response.lower()
        
        # Convert Action enum objects to strings
        raw_legal_str = []
        for la in raw_legal_actions:
            if isinstance(la, str):
                raw_legal_str.append(la)
            elif hasattr(la, "name"):
                enum_name = la.name.lower()
                if "fold" in enum_name:
                    raw_legal_str.append("fold")
                elif "check" in enum_name and "call" in enum_name:
                    raw_legal_str.append("check_call")
                elif "check" in enum_name:
                    raw_legal_str.append("check")
                elif "call" in enum_name:
                    raw_legal_str.append("call")
                elif "raise" in enum_name or "pot" in enum_name:
                    raw_legal_str.append("raise")
                elif "all" in enum_name and "in" in enum_name:
                    raw_legal_str.append("all_in")
                else:
                    raw_legal_str.append(enum_name)
            else:
                raw_legal_str.append(str(la))
        
        # Map action names to IDs
        action_map = {}
        if isinstance(raw_legal_str, list):
            for i, action_id in enumerate(sorted(legal_actions)):
                if i < len(raw_legal_str):
                    action_name = raw_legal_str[i].lower()
                    action_map[action_name] = action_id
                    if "check" in action_name and "call" in action_name:
                        action_map["check"] = action_id
                        action_map["call"] = action_id
        
        # Check for explicit action mentions
        for action_name, action_id in action_map.items():
            if action_name in response_lower:
                return action_id
        
        # Try JSON parsing
        try:
            json_match = re.search(r"\{[^}]+\}", response)
            if json_match:
                data = json.loads(json_match.group())
                if "action" in data:
                    action_name = str(data["action"]).lower()
                    if action_name in action_map:
                        return action_map[action_name]
        except:
            pass
        
        # Fallback: pick first legal action
        return legal_actions[0] if legal_actions else 0
    
    def make_decision(
        self,
        obs: Any,
        legal_actions: List[int],
        opponent_stats: OpponentStats,
        top_hands_info: str = "",
    ) -> ActOutput:
        """Make decision using LLM based on opponent analysis.
        
        Args:
            obs: Current observation
            legal_actions: List of legal action IDs
            opponent_stats: Opponent statistics from OpponentAnalysisAgent
            top_hands_info: Formatted top hands information from OpponentAnalysisAgent
        
        Returns:
            ActOutput with selected action
        """
        if not legal_actions:
            raise ValueError("No legal actions provided.")
        
        raw = _parse_rlcard_raw_obs(obs)
        hand = raw.get("hand", [])
        public_cards = raw.get("public_cards", [])
        legal_action_names = obs.get("raw_legal_actions", []) if isinstance(obs, dict) else []
        my_chips = raw.get("my_chips", 0)
        all_chips = raw.get("all_chips", [])
        
        # Discretize state for RL
        state = self._discretize_state(opponent_stats)
        
        # Select prompt type using RL
        prompt_type_idx = self._select_prompt_with_rl(state)
        prompt_type = self._prompt_types[prompt_type_idx]
        
        # Build prompt
        prompt_template = self._get_prompt_template(prompt_type)
        
        # Convert legal actions to strings
        legal_actions_str = []
        for la in legal_action_names:
            if isinstance(la, str):
                legal_actions_str.append(la)
            elif hasattr(la, "name"):
                enum_name = la.name.lower()
                if "fold" in enum_name:
                    legal_actions_str.append("fold")
                elif "check" in enum_name and "call" in enum_name:
                    legal_actions_str.append("check/call")
                elif "check" in enum_name:
                    legal_actions_str.append("check")
                elif "call" in enum_name:
                    legal_actions_str.append("call")
                elif "raise" in enum_name or "pot" in enum_name:
                    legal_actions_str.append("raise")
                elif "all" in enum_name and "in" in enum_name:
                    legal_actions_str.append("all-in")
                else:
                    legal_actions_str.append(enum_name)
            else:
                legal_actions_str.append(str(la))
        
        # Format opponent analysis information
        opponent_analysis = ""
        if top_hands_info:
            opponent_analysis += f"{top_hands_info}\n\n"
        opponent_analysis += f"Opponent Statistics:\n"
        opponent_analysis += f"- VPIP: {opponent_stats.vpip:.2%}\n"
        opponent_analysis += f"- Aggression: {opponent_stats.aggression:.2%}\n"
        opponent_analysis += f"- Fold Rate: {opponent_stats.fold_rate:.2%}\n"
        opponent_analysis += f"- Raise Rate: {opponent_stats.raise_rate:.2%}\n"
        
        # Build prompt
        prompt = prompt_template.format(
            hand=_format_cards(hand),
            public_cards=_format_cards(public_cards),
            my_chips=my_chips,
            all_chips=", ".join(map(str, all_chips)) if all_chips else "N/A",
            legal_actions=", ".join(legal_actions_str) if legal_actions_str else "unknown",
            opponent_analysis=opponent_analysis,
        )
        
        # Call LLM
        response = self._call_llm(prompt)
        
        # Parse action
        action_id = self._parse_action_from_response(response, legal_actions, legal_action_names)
        
        # Store prompt selection for later Q-learning update
        self._episode_prompt_selections.append((prompt_type_idx, state))
        
        return ActOutput(
            action=action_id,
            info={
                "policy": f"llm_decision_{prompt_type}",
                "prompt_type": prompt_type,
                "prompt_type_idx": prompt_type_idx,
                "state": state,
                "llm_response": response[:100],
            },
        )
    
    def update_episode_outcome(self, final_reward: float) -> None:
        """Update Q-table with episode outcome."""
        # Update Q-table for each prompt selection in this episode
        for i, (prompt_type_idx, state) in enumerate(self._episode_prompt_selections):
            next_state = None  # Terminal state
            self._update_q_table(state, prompt_type_idx, final_reward, next_state)
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        self._episode_count += 1
        self._episode_prompt_selections = []
    
    def reset(self) -> None:
        """Reset agent state for new episode."""
        self._episode_prompt_selections = []
    
    def save_memory(self, path: str) -> None:
        """Save Q-table and agent state to file."""
        q_table_serializable = {}
        for state, actions in self._q_table.items():
            q_table_serializable[str(state)] = dict(actions)
        
        data = {
            "episode_count": self._episode_count,
            "epsilon": self.epsilon,
            "q_table": q_table_serializable,
        }
        
        Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")
    
    def load_memory(self, path: str) -> None:
        """Load Q-table and agent state from file."""
        if not Path(path).exists():
            return
        
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        self._episode_count = data.get("episode_count", 0)
        self.epsilon = data.get("epsilon", self.epsilon)
        
        # Load Q-table
        q_table_data = data.get("q_table", {})
        self._q_table.clear()
        for state_str, actions in q_table_data.items():
            state = eval(state_str)  # Convert string back to tuple
            self._q_table[state] = defaultdict(float, actions)

