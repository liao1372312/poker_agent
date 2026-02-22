from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping

from ma_poker.agents.base import Agent
from ma_poker.agents.random_agent import RandomAgent
from ma_poker.agents.rule_based_agent import RuleBasedAgent
from ma_poker.agents.rl_agent import RLAgent
from ma_poker.agents.cfr_agent import CFRAgent, DeepCFRAgent
from ma_poker.agents.llm_agent import (
    LLMOnlyAgent,
    LLMHeuristicRangeAgent,
    LLMStaticBeliefAgent,
    LLMFixedPromptAgent,
    LLMFixedStyleOpponentAgent,
)


class NotImplementedBaselineError(NotImplementedError):
    pass


@dataclass(frozen=True)
class SeatSpec:
    """Spec for one seat/player."""

    type: str
    params: Dict[str, Any]


def build_agent(spec: SeatSpec) -> Agent:
    t = spec.type.lower().strip()
    p = spec.params or {}

    if t == "random":
        return RandomAgent(seed=p.get("seed"))
    if t in {"rule", "rule_based", "rule-based", "heuristic"}:
        return RuleBasedAgent(
            tightness=float(p.get("tightness", 0.65)),
            aggression=float(p.get("aggression", 0.35)),
            seed=p.get("seed"),
        )

    # RL policy agents (PPO/DQN)
    if t in {"ppo", "dqn", "rl_policy", "rl"}:
        algorithm = p.get("algorithm", "ppo" if t == "ppo" else "dqn" if t == "dqn" else "ppo")
        return RLAgent(
            model_path=p.get("model_path"),
            algorithm=str(algorithm),
            seed=p.get("seed"),
            trainable=bool(p.get("trainable", True)),  # Default to trainable
        )

    # CFR agents
    if t == "cfr":
        return CFRAgent(
            game_name=str(p.get("game_name", "limit_holdem")),
            game_params=p.get("game_params"),  # Pass game_params to configure universal_poker
            iterations=int(p.get("iterations", 1000)),
            model_path=p.get("model_path"),
            seed=p.get("seed"),
            device=p.get("device"),  # Device for DeepCFR: 'cuda', 'cpu', or None (auto-detect)
        )
    if t in {"deep_cfr", "deepcfr"}:
        return DeepCFRAgent(
            game_name=str(p.get("game_name", "limit_holdem")),
            game_params=p.get("game_params"),  # Pass game_params to configure universal_poker
            model_path=p.get("model_path"),
            seed=p.get("seed"),
            device=p.get("device"),  # Device for DeepCFR: 'cuda', 'cpu', or None (auto-detect)
        )

    # LLM agents
    if t in {"llm_only", "llm-only", "llm"}:
        return LLMOnlyAgent(
            api_url=str(p.get("api_url", "https://api.vveai.com/v1")),
            api_key=str(p.get("api_key", "sk-nY2KbJ4rtdbjwHyK9eCe56787c3e451a9fE4Ef81274e8a9d")),
            model=str(p.get("model", "gpt-5-mini")),
            temperature=float(p.get("temperature", 0.7)),
            max_tokens=int(p.get("max_tokens", 6000)),  # High limit for reasoning tokens + response
            seed=p.get("seed"),
        )
    if t in {"llm_heuristic_range", "llm+heuristic", "llm_heuristic"}:
        return LLMHeuristicRangeAgent(
            api_url=str(p.get("api_url", "https://api.vveai.com/v1")),
            api_key=str(p.get("api_key", "sk-nY2KbJ4rtdbjwHyK9eCe56787c3e451a9fE4Ef81274e8a9d")),
            model=str(p.get("model", "gpt-5-mini")),
            temperature=float(p.get("temperature", 0.7)),
            max_tokens=int(p.get("max_tokens", 6000)),  # High limit for reasoning tokens + response
            seed=p.get("seed"),
        )
    if t in {"llm_static_belief", "llm+static_belief", "llm_belief"}:
        return LLMStaticBeliefAgent(
            api_url=str(p.get("api_url", "https://api.vveai.com/v1")),
            api_key=str(p.get("api_key", "sk-nY2KbJ4rtdbjwHyK9eCe56787c3e451a9fE4Ef81274e8a9d")),
            model=str(p.get("model", "gpt-5-mini")),
            temperature=float(p.get("temperature", 0.7)),
            max_tokens=int(p.get("max_tokens", 6000)),  # High limit for reasoning tokens + response
            seed=p.get("seed"),
        )
    if t in {"llm_fixed_prompt", "llm+fixed_prompt"}:
        return LLMFixedPromptAgent(
            api_url=str(p.get("api_url", "https://api.vveai.com/v1")),
            api_key=str(p.get("api_key", "sk-nY2KbJ4rtdbjwHyK9eCe56787c3e451a9fE4Ef81274e8a9d")),
            model=str(p.get("model", "gpt-5-mini")),
            temperature=float(p.get("temperature", 0.7)),
            max_tokens=int(p.get("max_tokens", 6000)),  # High limit for reasoning tokens + response
            seed=p.get("seed"),
        )
    if t in {"llm_fixed_style_opponent", "llm_opponent", "llm_style_opponent", "opponent_llm"}:
        return LLMFixedStyleOpponentAgent(
            api_url=str(p.get("api_url", "https://api.vveai.com/v1")),
            api_key=str(p.get("api_key", "sk-nY2KbJ4rtdbjwHyK9eCe56787c3e451a9fE4Ef81274e8a9d")),
            model=str(p.get("model", "gpt-5-mini")),  # Use GPT-5-mini as specified
            temperature=float(p.get("temperature", 0.7)),
            max_tokens=int(p.get("max_tokens", 6000)),  # High limit for reasoning tokens + response
            player_style=str(p.get("player_style", "tag")),  # nit_rock, calling_station, tag, lag
            seed=p.get("seed"),
        )

    # Ours baseline: LLM with RL-based prompt selection
    if t == "ours":
        from ma_poker.agents.ours_agent import OursAgent

        return OursAgent(
            api_url=str(p.get("api_url", "https://api.vveai.com/v1")),
            api_key=str(p.get("api_key", "sk-nY2KbJ4rtdbjwHyK9eCe56787c3e451a9fE4Ef81274e8a9d")),
            model=str(p.get("model", "gpt-5-mini")),
            temperature=float(p.get("temperature", 0.7)),
            max_tokens=int(p.get("max_tokens", 6000)),
            learning_rate=float(p.get("learning_rate", 0.1)),
            discount_factor=float(p.get("discount_factor", 0.9)),
            epsilon=float(p.get("epsilon", 0.1)),
            epsilon_decay=float(p.get("epsilon_decay", 0.995)),
            epsilon_min=float(p.get("epsilon_min", 0.05)),
            seed=p.get("seed"),
        )

    raise ValueError(f"Unknown agent type: {spec.type}")


def build_seated_agents(seats: List[SeatSpec]) -> Dict[int, Agent]:
    return {i: build_agent(seat) for i, seat in enumerate(seats)}

