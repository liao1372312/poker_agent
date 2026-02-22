from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ma_poker.agents.base import ActOutput, Agent


def _parse_rlcard_raw_obs(obs: Any) -> Dict[str, Any]:
    if not isinstance(obs, dict):
        return {}
    raw = obs.get("raw_obs")
    return raw if isinstance(raw, dict) else {}


@dataclass
class RuleBasedAgent(Agent):
    """A lightweight heuristic baseline for RLCard Hold'em.

    This is intentionally simple and fully self-contained (no hand evaluator dependency).
    Policy sketch:
    - Prefer check/call when available.
    - Fold only when forced (no check/call) and hand is likely weak (preflop heuristic).
    - Raise more often with stronger-looking preflop holdings (pairs, high cards, suited).
    """

    tightness: float = 0.65  # higher => tighter (more folds)
    aggression: float = 0.35  # higher => more raises
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        import random

        self._rng = random.Random(self.seed)

    def reset(self) -> None:
        return

    def act(self, obs: Any, legal_actions: List[int]) -> ActOutput:
        if not legal_actions:
            raise ValueError("No legal actions provided.")

        # RLCard raw_legal_actions contains strings like: ['raise','fold','check'].
        raw = _parse_rlcard_raw_obs(obs)
        raw_legal = raw.get("legal_actions", [])
        if not isinstance(raw_legal, list):
            raw_legal = []

        # Map action_id -> name using obs['raw_legal_actions'] alignment if present.
        # In RLCard, obs has both 'legal_actions' (id dict) and 'raw_legal_actions' (name list).
        # Note: limit-holdem uses strings, no-limit-holdem may use Action enum objects
        raw_legal_actions = obs.get("raw_legal_actions", []) if isinstance(obs, dict) else []
        id_legal = sorted(list(legal_actions))

        id_to_name: Dict[int, str] = {}
        if isinstance(raw_legal_actions, list) and len(raw_legal_actions) == len(id_legal):
            for a_id, a_name in zip(id_legal, raw_legal_actions):
                # Handle both string names and Action enum objects
                if isinstance(a_name, str):
                    id_to_name[int(a_id)] = a_name
                elif hasattr(a_name, "name"):  # Action enum
                    # Convert enum to string (e.g., Action.FOLD -> "fold")
                    enum_name = a_name.name.lower()
                    # Map common enum names to action strings
                    if "fold" in enum_name:
                        id_to_name[int(a_id)] = "fold"
                    elif "check" in enum_name or "call" in enum_name:
                        if "check" in enum_name:
                            id_to_name[int(a_id)] = "check"
                        else:
                            id_to_name[int(a_id)] = "call"
                    elif "raise" in enum_name or "pot" in enum_name:
                        id_to_name[int(a_id)] = "raise"
                    elif "all" in enum_name and "in" in enum_name:
                        id_to_name[int(a_id)] = "raise"  # Treat all-in as raise
                    else:
                        id_to_name[int(a_id)] = enum_name

        # Convenience pickers
        def pick_by_name(preferred: List[str]) -> Optional[int]:
            for name in preferred:
                for a_id in id_legal:
                    action_name = id_to_name.get(int(a_id), "").lower()
                    # Flexible matching: "check_call" matches both "check" and "call"
                    if action_name == name.lower() or name.lower() in action_name:
                        return int(a_id)
            return None

        # Estimate preflop strength cheaply from raw hand strings like 'SA', 'D9'
        hand = raw.get("hand", [])
        hand_strength = 0.0
        if isinstance(hand, list) and len(hand) == 2 and all(isinstance(c, str) and len(c) >= 2 for c in hand):
            ranks = [c[1] for c in hand]
            suits = [c[0] for c in hand]
            rank_order = "23456789TJQKA"
            vals = [rank_order.index(r) + 2 if r in rank_order else 2 for r in ranks]
            hi = max(vals)
            lo = min(vals)
            pair = 1.0 if ranks[0] == ranks[1] else 0.0
            suited = 0.6 if suits[0] == suits[1] else 0.0
            connected = 0.3 if abs(vals[0] - vals[1]) == 1 else 0.0
            broadway = 0.4 if (hi >= 11 and lo >= 10) else 0.0
            hand_strength = 0.15 * (hi / 14.0) + 0.10 * (lo / 14.0) + pair + suited + connected + broadway

        # Default: check/call if possible
        a_check = pick_by_name(["check"])
        a_call = pick_by_name(["call"])
        a_fold = pick_by_name(["fold"])
        a_raise = pick_by_name(["raise"])

        # If can check, usually check; sometimes raise with strong hand.
        if a_check is not None:
            if a_raise is not None:
                p_raise = min(0.95, max(0.0, self.aggression + 0.35 * hand_strength))
                if self._rng.random() < p_raise:
                    return ActOutput(action=a_raise, info={"policy": "rule_based", "choice": "raise"})
            return ActOutput(action=a_check, info={"policy": "rule_based", "choice": "check"})

        # If must respond to bet: call more with stronger hand; fold otherwise (if allowed)
        if a_call is not None:
            p_fold = min(0.95, max(0.0, self.tightness - 0.40 * hand_strength))
            if a_fold is not None and self._rng.random() < p_fold:
                return ActOutput(action=a_fold, info={"policy": "rule_based", "choice": "fold"})
            # occasionally raise
            if a_raise is not None:
                p_raise = min(0.90, max(0.0, self.aggression + 0.25 * hand_strength))
                if self._rng.random() < p_raise:
                    return ActOutput(action=a_raise, info={"policy": "rule_based", "choice": "raise"})
            return ActOutput(action=a_call, info={"policy": "rule_based", "choice": "call"})

        # Fallback: pick any legal action deterministically
        return ActOutput(action=int(id_legal[0]), info={"policy": "rule_based", "choice": "fallback"})

