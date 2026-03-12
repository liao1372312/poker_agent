"""Ours Agent: Multi-agent architecture coordinating opponent analysis and LLM decision-making.

This agent coordinates:
1. OpponentAnalysisAgent: Analyzes opponents and predicts hand probability distributions
2. LLMDecisionAgent: Makes decisions using LLM based on opponent analysis results
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from ma_poker.agents.base import ActOutput, Agent
from ma_poker.agents.opponent_analysis_agent import OpponentAnalysisAgent, OpponentStats
from ma_poker.agents.llm_decision_agent import LLMDecisionAgent
from ma_poker.agents.hand_utils import format_top_hands_for_prompt


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
    api_key: str = ""
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

    # Hand-memory & gate settings (paper Sec. 3.8-3.10)
    hand_memory_max_size: int = 2000
    gate_learning_rate: float = 0.02
    gate_ev_threshold: float = 0.05
    cf_rollout_hands: int = 128
    uncertainty_penalty_coef: float = 0.05
    deviation_penalty_lambda: float = 0.02

    # PPO gate settings (paper Sec. 3.8, Eq. 13-16)
    ppo_clip_eps: float = 0.2
    ppo_epochs: int = 4
    ppo_gamma: float = 0.99
    ppo_gae_lambda: float = 0.95
    ppo_entropy_coef: float = 0.01
    ppo_value_coef: float = 0.5
    ppo_value_lr: float = 0.02
    
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

        # Runtime trackers
        import random

        self._rng = random.Random(self.seed)
        self._last_observed_opponent_id: Optional[int] = None
        self._self_player_id: Optional[int] = None

        # Simple RL gate: sigmoid(w·x + b) -> choose exploit vs anchor
        self._gate_w = np.zeros(6, dtype=np.float32)
        self._gate_b = 0.0
        self._value_w = np.zeros(6, dtype=np.float32)
        self._value_b = 0.0

        # Hand memory buffer + current hand cache
        self._hand_memory: List[Dict[str, Any]] = []
        self._hand_in_progress: bool = False
        self._current_hand_trace: List[Dict[str, Any]] = []
        self._current_hand_decisions: List[Dict[str, Any]] = []
        self._current_profile_snapshot: Dict[str, Any] = {}
        self._current_belief_summaries: List[Dict[str, Any]] = []

    def _start_new_hand_if_needed(self) -> None:
        if self._hand_in_progress:
            return
        self._hand_in_progress = True
        self._current_hand_trace = []
        self._current_hand_decisions = []
        self._current_profile_snapshot = {}
        self._current_belief_summaries = []

    def _select_target_opponent_id(self, num_players: int) -> int:
        if self._last_observed_opponent_id is not None:
            return int(self._last_observed_opponent_id)

        stats_items = list(self._opponent_analysis_agent._opponent_stats.items())
        if stats_items:
            stats_items.sort(key=lambda kv: kv[1].total_actions, reverse=True)
            return int(stats_items[0][0])

        return 0 if num_players <= 1 else 1

    def _belief_entropy(self, belief: np.ndarray) -> float:
        x = np.clip(belief, 1e-12, 1.0)
        h = float(-np.sum(x * np.log(x)))
        return h / math.log(len(x)) if len(x) > 1 else 0.0

    def _profile_snapshot(self, opponent_id: int) -> Dict[str, Any]:
        stats = self._opponent_analysis_agent.get_opponent_stats(opponent_id)
        style = self._classify_exploit_prompt(stats)
        return {
            "opponent_id": opponent_id,
            "vpip": float(stats.vpip),
            "pfr": float(stats.pfr),
            "aggression": float(stats.aggression),
            "fold_rate": float(stats.fold_rate),
            "raise_rate": float(stats.raise_rate),
            "style_hint": style,
        }

    def _belief_summary(self, opponent_id: int) -> Dict[str, Any]:
        belief_obj = self._opponent_analysis_agent.get_opponent_belief(opponent_id)
        top_hands = self._opponent_analysis_agent.get_top_hands_for_opponent(opponent_id, top_k=10)
        entropy = self._belief_entropy(belief_obj.belief)
        top_mass = float(sum(float(x[1]) for x in top_hands)) if top_hands else 0.0
        return {
            "opponent_id": opponent_id,
            "entropy": entropy,
            "top10_mass": top_mass,
            "top_hands": [(int(i), float(p), str(name)) for i, p, name in top_hands],
        }

    def _memory_summary(self, opponent_id: int) -> Dict[str, Any]:
        items = [m for m in self._hand_memory if int(m.get("opponent_id", -1)) == int(opponent_id)]
        if not items:
            return {
                "hands_seen": 0,
                "avg_relative_reward": 0.0,
                "win_rate": 0.0,
                "best_exploit_prompt": "tight_aggressive",
            }

        recent = items[-200:]
        rel = [float(m.get("relative_reward", 0.0)) for m in recent]
        wr = [1.0 if float(m.get("final_reward", 0.0)) > 0 else 0.0 for m in recent]

        prompt_score: Dict[str, List[float]] = {}
        for m in recent:
            for d in m.get("selector_decisions", []):
                p = str(d.get("exploit_prompt", "tight_aggressive"))
                prompt_score.setdefault(p, []).append(float(m.get("relative_reward", 0.0)))

        best_prompt = "tight_aggressive"
        if prompt_score:
            best_prompt = max(prompt_score.items(), key=lambda kv: (sum(kv[1]) / max(len(kv[1]), 1)))[0]

        return {
            "hands_seen": len(recent),
            "avg_relative_reward": float(sum(rel) / max(len(rel), 1)),
            "win_rate": float(sum(wr) / max(len(wr), 1)),
            "best_exploit_prompt": best_prompt,
        }

    def _classify_exploit_prompt(self, s: OpponentStats) -> str:
        # Map opponent tendencies -> exploit prompt family
        if s.vpip >= 0.55 and s.aggression < 0.45:
            return "loose_passive"
        if s.vpip >= 0.55 and s.aggression >= 0.45:
            return "tight_aggressive"
        if s.vpip < 0.35 and s.aggression >= 0.5:
            return "tight_passive"
        if s.vpip < 0.35 and s.aggression < 0.5:
            return "loose_aggressive"
        return "tight_aggressive"

    def _estimate_counterfactual_ev(
        self,
        opponent_id: int,
        street_bucket: int,
        anchor_prompt: str,
        exploit_prompt: str,
        opponent_stats: OpponentStats,
        belief_entropy: float,
        memory_summary: Dict[str, Any],
    ) -> Dict[str, float]:
        # Belief-guided counterfactual estimator aligned with paper Sec.3.7:
        # use hand memory as rollout source and profile/belief as conditioning weights.
        # Returns dEV(anchor), dEV(exploit), uncertainty u, and deltaEV.

        def _score_sample(rec: Dict[str, Any], prompt: str) -> float:
            # Base terminal proxy from memory.
            r = float(rec.get("final_reward", 0.0))

            # Condition by street and prompt match as proxy for policy consistency.
            decisions = rec.get("selector_decisions", [])
            prompt_bonus = 0.0
            if decisions:
                used = str(decisions[-1].get("selected_prompt", ""))
                if used == prompt:
                    prompt_bonus += 0.15

            # Belief-guided weighting by profile similarity.
            ps = rec.get("profile_snapshot", {})
            dvpip = abs(float(ps.get("vpip", 0.5)) - float(opponent_stats.vpip))
            dagg = abs(float(ps.get("aggression", 0.5)) - float(opponent_stats.aggression))
            sim = max(0.0, 1.0 - 0.8 * dvpip - 0.8 * dagg)

            # Street consistency bonus.
            bs = rec.get("belief_summaries", [])
            sb = len(bs[-1].get("summary", {}).get("top_hands", [])) if bs else 0
            street_like = 1.0 if sb > 0 and street_bucket >= 0 else 0.8

            return (r + prompt_bonus) * (0.6 + 0.4 * sim) * street_like

        # Build candidate memory pool for this opponent first, then global fallback.
        pool = [m for m in self._hand_memory if int(m.get("opponent_id", -1)) == int(opponent_id)]
        if not pool:
            pool = list(self._hand_memory)

        if not pool:
            # Cold start fallback before memory exists.
            exploitability = max(0.0, opponent_stats.vpip - 0.45) + max(0.0, 0.45 - opponent_stats.fold_rate)
            risk = max(0.0, opponent_stats.aggression - 0.55) + 0.5 * belief_entropy
            d_ev_anchor = 0.02 - 0.05 * risk
            d_ev_exploit = 0.03 + 0.09 * exploitability - 0.06 * risk + 0.03 * float(memory_summary.get("avg_relative_reward", 0.0))
            uncertainty = float(np.clip(0.7 + 0.3 * belief_entropy, 0.0, 1.0))
            return {
                "d_ev_anchor": float(d_ev_anchor),
                "d_ev_exploit": float(d_ev_exploit),
                "uncertainty": uncertainty,
                "delta_ev": float(d_ev_exploit - d_ev_anchor),
            }

        # Monte-Carlo style resampling from hand memory.
        n = min(max(16, int(self.cf_rollout_hands)), len(pool))
        samples = self._rng.sample(pool, n) if len(pool) > n else pool

        anchor_vals = [_score_sample(rec, anchor_prompt) for rec in samples]
        exploit_vals = [_score_sample(rec, exploit_prompt) for rec in samples]

        d_ev_anchor = float(np.mean(anchor_vals)) if anchor_vals else 0.0
        d_ev_exploit = float(np.mean(exploit_vals)) if exploit_vals else 0.0

        # Uncertainty from dispersion + belief entropy.
        var_anchor = float(np.var(anchor_vals)) if len(anchor_vals) > 1 else 0.0
        var_exploit = float(np.var(exploit_vals)) if len(exploit_vals) > 1 else 0.0
        est_std = math.sqrt(max(0.0, 0.5 * (var_anchor + var_exploit)))
        uncertainty = float(np.clip(0.5 * belief_entropy + 0.2 * est_std, 0.0, 1.0))

        # Penalize exploit candidate by uncertainty (safe selection tendency).
        d_ev_exploit -= float(self.uncertainty_penalty_coef) * uncertainty
        return {
            "d_ev_anchor": float(d_ev_anchor),
            "d_ev_exploit": float(d_ev_exploit),
            "uncertainty": uncertainty,
            "delta_ev": float(d_ev_exploit - d_ev_anchor),
        }

    def _prompt_distance(self, p1: str, p2: str) -> float:
        if p1 == p2:
            return 0.0
        # Coarse policy divergence proxy over prompt families (Eq.11 style penalty proxy).
        groups = {
            "balanced_gto": 0,
            "tight_aggressive": 1,
            "tight_passive": 2,
            "loose_aggressive": 3,
            "loose_passive": 4,
        }
        a = groups.get(p1, 0)
        b = groups.get(p2, 0)
        return float(abs(a - b) / 4.0)

    def _sigmoid(self, x: float) -> float:
        if x >= 0:
            z = math.exp(-x)
            return 1.0 / (1.0 + z)
        z = math.exp(x)
        return z / (1.0 + z)

    def _gate_select(self, features: np.ndarray, delta_ev: float) -> Dict[str, Any]:
        anchor = "balanced_gto"
        if abs(delta_ev) < self.gate_ev_threshold:
            p_exploit = 0.0
            value_est = float(np.dot(self._value_w, features) + self._value_b)
            action = 0
            logp = 0.0
            return {
                "selected": action,
                "prob_exploit": p_exploit,
                "anchor_prompt": anchor,
                "logp_old": logp,
                "value_est": value_est,
            }

        z = float(np.dot(self._gate_w, features) + self._gate_b)
        p_exploit = self._sigmoid(z)
        selected = 1 if self._rng.random() < p_exploit else 0
        p_sel = p_exploit if selected == 1 else (1.0 - p_exploit)
        logp = float(math.log(max(1e-8, p_sel)))
        value_est = float(np.dot(self._value_w, features) + self._value_b)
        return {
            "selected": selected,
            "prob_exploit": p_exploit,
            "anchor_prompt": anchor,
            "logp_old": logp,
            "value_est": value_est,
        }

    def _ppo_update_gate(self, relative_reward: float) -> None:
        """PPO Actor-Critic update on current hand decisions.

        Rewards are sparse terminal: r_t=0 for non-terminal, r_T=relative_reward.
        """
        if not self._current_hand_decisions:
            return

        traj = self._current_hand_decisions
        n = len(traj)
        rewards = np.zeros(n, dtype=np.float32)
        rewards[-1] = float(relative_reward)
        values = np.array([float(d.get("value_est", 0.0)) for d in traj], dtype=np.float32)

        # GAE advantages
        adv = np.zeros(n, dtype=np.float32)
        gae = 0.0
        next_value = 0.0
        for t in reversed(range(n)):
            delta = rewards[t] + self.ppo_gamma * next_value - values[t]
            gae = delta + self.ppo_gamma * self.ppo_gae_lambda * gae
            adv[t] = gae
            next_value = values[t]

        returns = adv + values
        adv_mean = float(np.mean(adv))
        adv_std = float(np.std(adv) + 1e-8)
        adv = (adv - adv_mean) / adv_std

        # PPO epochs over the same on-policy batch
        for _ in range(max(1, int(self.ppo_epochs))):
            for i, d in enumerate(traj):
                s = np.array(d.get("selector_state", [0.0] * 6), dtype=np.float32)
                a = int(d.get("gate_action", 0))
                logp_old = float(d.get("gate_logp_old", d.get("logp_old", 0.0)))
                A = float(adv[i])
                ret = float(returns[i])

                # Actor current policy
                z = float(np.dot(self._gate_w, s) + self._gate_b)
                p = float(np.clip(self._sigmoid(z), 1e-6, 1.0 - 1e-6))
                logp = float(math.log(p) if a == 1 else math.log(1.0 - p))
                ratio = float(math.exp(logp - logp_old))

                # Clipped surrogate gradient condition
                use_clip = (A >= 0.0 and ratio > 1.0 + self.ppo_clip_eps) or (
                    A < 0.0 and ratio < 1.0 - self.ppo_clip_eps
                )

                # d log pi(a|s) / d z
                dlogp_dz = (1.0 - p) if a == 1 else (-p)

                if not use_clip:
                    # grad of ratio*A wrt z => ratio*A*dlogp_dz
                    grad_actor = ratio * A * dlogp_dz
                    self._gate_w += self.gate_learning_rate * grad_actor * s
                    self._gate_b += self.gate_learning_rate * grad_actor

                # Entropy bonus gradient (for exploration)
                dH_dz = p * (1.0 - p) * math.log(max((1.0 - p) / p, 1e-8))
                self._gate_w += self.gate_learning_rate * self.ppo_entropy_coef * dH_dz * s
                self._gate_b += self.gate_learning_rate * self.ppo_entropy_coef * dH_dz

                # Critic update (MSE)
                v = float(np.dot(self._value_w, s) + self._value_b)
                dv = (v - ret)  # d(0.5*(v-ret)^2)/dv
                self._value_w -= self.ppo_value_lr * self.ppo_value_coef * dv * s
                self._value_b -= self.ppo_value_lr * self.ppo_value_coef * dv
    
    def reset(self) -> None:
        """Reset long-lived agent state at run start."""
        self._current_episode_id = 0
        self._llm_decision_agent.reset()
        self._hand_in_progress = False
        self._current_hand_trace = []
        self._current_hand_decisions = []
        self._current_profile_snapshot = {}
        self._current_belief_summaries = []
    
    def act(self, obs: Any, legal_actions: List[int]) -> ActOutput:
        """Make decision by coordinating opponent analysis and LLM decision-making.
        
        Workflow:
        1. Extract current game state
        2. Get opponent analysis (stats and top hands) from OpponentAnalysisAgent
        3. Use LLMDecisionAgent to make decision based on analysis
        """
        if not legal_actions:
            raise ValueError("No legal actions provided.")

        self._start_new_hand_if_needed()
        
        raw = _parse_rlcard_raw_obs(obs)
        if self._self_player_id is None:
            maybe_self = raw.get("current_player")
            if isinstance(maybe_self, int):
                self._self_player_id = int(maybe_self)

        all_chips = raw.get("all_chips", [])
        num_players = len(all_chips) if all_chips else 9

        # Select most relevant opponent id from observed memory/actions.
        opponent_id = self._select_target_opponent_id(num_players)
        if opponent_id >= num_players:
            opponent_id = 0
        
        opponent_stats = self._opponent_analysis_agent.get_opponent_stats(opponent_id)

        # Build profile / belief / memory summaries (paper Sec. 3.1/3.2/3.9)
        profile_snapshot = self._profile_snapshot(opponent_id)
        belief_summary = self._belief_summary(opponent_id)
        memory_summary = self._memory_summary(opponent_id)

        self._current_profile_snapshot = profile_snapshot
        self._current_belief_summaries.append(
            {
                "street_public_cards": len(raw.get("public_cards", [])),
                "summary": belief_summary,
            }
        )

        # Get top hands for opponent
        top_hands = self._opponent_analysis_agent.get_top_hands_for_opponent(opponent_id, top_k=10)
        top_hands_info = format_top_hands_for_prompt(top_hands) if top_hands else ""

        street_bucket = 0
        public_cards = raw.get("public_cards", [])
        if len(public_cards) >= 3:
            street_bucket = 1
        if len(public_cards) >= 4:
            street_bucket = 2
        if len(public_cards) >= 5:
            street_bucket = 3

        anchor_prompt = "balanced_gto"
        exploit_prompt = str(memory_summary.get("best_exploit_prompt") or self._classify_exploit_prompt(opponent_stats))

        # Counterfactual EV proxy and gate state (paper Sec. 3.7/3.8)
        ev = self._estimate_counterfactual_ev(
            opponent_id=opponent_id,
            street_bucket=street_bucket,
            anchor_prompt=anchor_prompt,
            exploit_prompt=exploit_prompt,
            opponent_stats=opponent_stats,
            belief_entropy=float(belief_summary["entropy"]),
            memory_summary=memory_summary,
        )

        selector_state = np.array(
            [
                float(opponent_stats.vpip),
                float(opponent_stats.aggression),
                float(belief_summary["entropy"]),
                float(ev["delta_ev"]),
                float(memory_summary["win_rate"]),
                float(memory_summary["avg_relative_reward"]),
            ],
            dtype=np.float32,
        )

        gate = self._gate_select(selector_state, float(ev["delta_ev"]))
        selected_prompt = anchor_prompt if gate["selected"] == 0 else exploit_prompt

        extra_context = (
            "Opponent Profile Snapshot:\n"
            f"- style_hint: {profile_snapshot['style_hint']}\n"
            f"- vpip: {profile_snapshot['vpip']:.3f}, aggression: {profile_snapshot['aggression']:.3f}\n"
            "Belief Summary:\n"
            f"- entropy: {belief_summary['entropy']:.3f}, top10_mass: {belief_summary['top10_mass']:.3f}\n"
            "Hand Memory Summary:\n"
            f"- hands_seen: {memory_summary['hands_seen']}\n"
            f"- memory_win_rate: {memory_summary['win_rate']:.3f}\n"
            f"- avg_relative_reward: {memory_summary['avg_relative_reward']:.3f}\n"
            "EV-Gate Signals:\n"
            f"- dEV(anchor): {ev['d_ev_anchor']:.4f}\n"
            f"- dEV(exploit): {ev['d_ev_exploit']:.4f}\n"
            f"- deltaEV: {ev['delta_ev']:.4f}, uncertainty: {ev['uncertainty']:.4f}\n"
            f"- selected_policy: {'anchor' if gate['selected']==0 else 'exploit'}"
        )
        
        # Use LLMDecisionAgent to make decision
        act_out = self._llm_decision_agent.make_decision(
            obs=obs,
            legal_actions=legal_actions,
            opponent_stats=opponent_stats,
            top_hands_info=top_hands_info,
            forced_prompt_type=selected_prompt,
            extra_context=extra_context,
        )

        self._current_hand_trace.append(
            {
                "actor": "self",
                "opponent_id_focus": int(opponent_id),
                "public_cards": list(raw.get("public_cards", [])),
                "action": int(act_out.action),
                "legal_actions": [int(a) for a in legal_actions],
            }
        )

        self._current_hand_decisions.append(
            {
                "selector_state": [float(x) for x in selector_state.tolist()],
                "gate_action": int(gate["selected"]),
                "gate_prob_exploit": float(gate["prob_exploit"]),
                "gate_logp_old": float(gate.get("logp_old", 0.0)),
                "value_est": float(gate.get("value_est", 0.0)),
                "anchor_prompt": anchor_prompt,
                "exploit_prompt": exploit_prompt,
                "selected_prompt": selected_prompt,
                "prompt_distance": self._prompt_distance(anchor_prompt, exploit_prompt),
                "d_ev_anchor": float(ev["d_ev_anchor"]),
                "d_ev_exploit": float(ev["d_ev_exploit"]),
                "delta_ev": float(ev["delta_ev"]),
                "uncertainty": float(ev["uncertainty"]),
                "llm_policy": act_out.info.get("policy", ""),
            }
        )

        return act_out
    
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

        self._start_new_hand_if_needed()
        self._last_observed_opponent_id = int(opponent_id)

        self._opponent_analysis_agent.update_opponent_action(opponent_id, action, context)

        self._current_hand_trace.append(
            {
                "actor": f"opponent_{int(opponent_id)}",
                "action": str(action),
                "context": {
                    "pot_size": float(context.get("pot_size", 0.0)),
                    "bet_size": float(context.get("bet_size", 0.0)),
                    "betting_round": int(context.get("betting_round", 0)),
                    "position": int(context.get("position", 0)),
                    "public_cards": list(context.get("public_cards", [])),
                },
            }
        )
    
    def update_episode_outcome(self, final_reward: float) -> None:
        """Update both sub-agents with episode outcome (memory-driven self-evolution)."""
        self._current_episode_id += 1

        # Baseline-relative reward (paper Eq.10 style): final - anchor proxy
        if self._current_hand_decisions:
            baseline_proxy = float(np.mean([d.get("d_ev_anchor", 0.0) for d in self._current_hand_decisions]))
        else:
            baseline_proxy = 0.0

        # Optional deviation penalty (paper Eq.11 proxy)
        deviation_penalty = 0.0
        for d in self._current_hand_decisions:
            if int(d.get("gate_action", 0)) == 1:
                deviation_penalty += self.deviation_penalty_lambda * float(d.get("prompt_distance", 0.0))

        relative_reward = float(final_reward - baseline_proxy - deviation_penalty)

        # Update decision agent with relative reward (self-evolution signal)
        self._llm_decision_agent.update_episode_outcome(relative_reward)

        # PPO Actor-Critic gate update (paper-aligned)
        self._ppo_update_gate(relative_reward)
        
        # Update OpponentAnalysisAgent (portrait vectors)
        # Distribute relative reward among opponents (simplified - equal distribution)
        num_opponents = len(self._opponent_analysis_agent._opponent_beliefs)
        if num_opponents > 0:
            reward_per_opponent = relative_reward / num_opponents
            for opponent_id in self._opponent_analysis_agent._opponent_beliefs.keys():
                self._opponent_analysis_agent.update_portrait_vector(opponent_id, reward_per_opponent)
        
        # Train belief network if training data is available
        self._opponent_analysis_agent.train_belief_network()

        # Persist one hand-level memory record (paper Sec. 3.9)
        focus_opponent_id = self._last_observed_opponent_id if self._last_observed_opponent_id is not None else 0
        hand_record = {
            "episode_id": int(self._current_episode_id),
            "opponent_id": int(focus_opponent_id),
            "profile_snapshot": dict(self._current_profile_snapshot),
            "belief_summaries": list(self._current_belief_summaries),
            "action_trace": list(self._current_hand_trace),
            "selector_decisions": list(self._current_hand_decisions),
            "final_reward": float(final_reward),
            "relative_reward": float(relative_reward),
        }
        self._hand_memory.append(hand_record)
        if len(self._hand_memory) > int(self.hand_memory_max_size):
            self._hand_memory = self._hand_memory[-int(self.hand_memory_max_size):]

        # Reset current hand cache
        self._hand_in_progress = False
        self._current_hand_trace = []
        self._current_hand_decisions = []
        self._current_profile_snapshot = {}
        self._current_belief_summaries = []
        
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
            "hand_memory_max_size": int(self.hand_memory_max_size),
            "gate_learning_rate": float(self.gate_learning_rate),
            "gate_ev_threshold": float(self.gate_ev_threshold),
            "cf_rollout_hands": int(self.cf_rollout_hands),
            "uncertainty_penalty_coef": float(self.uncertainty_penalty_coef),
            "deviation_penalty_lambda": float(self.deviation_penalty_lambda),
            "gate_w": [float(x) for x in self._gate_w.tolist()],
            "gate_b": float(self._gate_b),
            "value_w": [float(x) for x in self._value_w.tolist()],
            "value_b": float(self._value_b),
            "ppo_clip_eps": float(self.ppo_clip_eps),
            "ppo_epochs": int(self.ppo_epochs),
            "ppo_gamma": float(self.ppo_gamma),
            "ppo_gae_lambda": float(self.ppo_gae_lambda),
            "ppo_entropy_coef": float(self.ppo_entropy_coef),
            "ppo_value_coef": float(self.ppo_value_coef),
            "ppo_value_lr": float(self.ppo_value_lr),
            "hand_memory": self._hand_memory,
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
        self.hand_memory_max_size = int(metadata.get("hand_memory_max_size", self.hand_memory_max_size))
        self.gate_learning_rate = float(metadata.get("gate_learning_rate", self.gate_learning_rate))
        self.gate_ev_threshold = float(metadata.get("gate_ev_threshold", self.gate_ev_threshold))
        self.cf_rollout_hands = int(metadata.get("cf_rollout_hands", self.cf_rollout_hands))
        self.uncertainty_penalty_coef = float(metadata.get("uncertainty_penalty_coef", self.uncertainty_penalty_coef))
        self.deviation_penalty_lambda = float(metadata.get("deviation_penalty_lambda", self.deviation_penalty_lambda))
        self._gate_w = np.array(metadata.get("gate_w", [0.0] * 6), dtype=np.float32)
        self._gate_b = float(metadata.get("gate_b", 0.0))
        self._value_w = np.array(metadata.get("value_w", [0.0] * 6), dtype=np.float32)
        self._value_b = float(metadata.get("value_b", 0.0))
        self.ppo_clip_eps = float(metadata.get("ppo_clip_eps", self.ppo_clip_eps))
        self.ppo_epochs = int(metadata.get("ppo_epochs", self.ppo_epochs))
        self.ppo_gamma = float(metadata.get("ppo_gamma", self.ppo_gamma))
        self.ppo_gae_lambda = float(metadata.get("ppo_gae_lambda", self.ppo_gae_lambda))
        self.ppo_entropy_coef = float(metadata.get("ppo_entropy_coef", self.ppo_entropy_coef))
        self.ppo_value_coef = float(metadata.get("ppo_value_coef", self.ppo_value_coef))
        self.ppo_value_lr = float(metadata.get("ppo_value_lr", self.ppo_value_lr))
        self._hand_memory = list(metadata.get("hand_memory", []))
        
        # Load sub-agent memories
        analysis_path = metadata.get("analysis_memory_path", path.replace(".json", "_analysis.json"))
        if Path(analysis_path).exists():
            self._opponent_analysis_agent.load_memory(analysis_path)
        
        decision_path = metadata.get("decision_memory_path", path.replace(".json", "_decision.json"))
        if Path(decision_path).exists():
            self._llm_decision_agent.load_memory(decision_path)
