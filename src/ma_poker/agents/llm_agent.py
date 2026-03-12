from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from ma_poker.agents.base import ActOutput, Agent


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


def _action_id_to_name(action_id: int, raw_legal_actions: List[str], legal_ids: List[int]) -> str:
    """Map action ID to action name."""
    if isinstance(raw_legal_actions, list) and len(raw_legal_actions) == len(legal_ids):
        idx = legal_ids.index(action_id) if action_id in legal_ids else -1
        if 0 <= idx < len(raw_legal_actions):
            return raw_legal_actions[idx]
    return f"action_{action_id}"


def _normalize_raw_legal_action_names(raw_legal_actions: List[Any]) -> List[str]:
    """Convert raw_legal_actions (Action enums or strings) to normalized name list for parsing/schema."""
    out: List[str] = []
    for la in raw_legal_actions:
        if isinstance(la, str):
            out.append(la.lower())
        elif hasattr(la, "name"):
            enum_name = la.name.lower()
            if "fold" in enum_name:
                out.append("fold")
            elif "check" in enum_name and "call" in enum_name:
                out.append("check_call")
            elif "check" in enum_name:
                out.append("check")
            elif "call" in enum_name:
                out.append("call")
            elif "raise" in enum_name or "pot" in enum_name:
                out.append("raise")
            elif "all" in enum_name and "in" in enum_name:
                out.append("all_in")
            else:
                out.append(enum_name)
        else:
            out.append(str(la).lower())
    return out


@dataclass
class LLMAgent(Agent):
    """Base LLM agent for poker decision-making.

    This is a base class. Use specific variants:
    - LLMOnlyAgent: Pure LLM without additional information
    - LLMHeuristicRangeAgent: LLM + heuristic hand range
    - LLMStaticBeliefAgent: LLM + static opponent belief
    - LLMFixedPromptAgent: LLM + fixed prompt template
    """

    api_url: str = "https://api.vveai.com/v1"
    api_key: str = ""
    model: str = "gpt-4.1-mini"
    temperature: float = 0.7
    max_tokens: int = 6000  # High limit to allow for reasoning tokens + actual response
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai library is required for LLM agents. Install with: pip install openai")

        resolved_api_key = self.api_key or os.getenv("OPENAI_API_KEY", "")
        if not resolved_api_key:
            raise ValueError("Missing API key. Set OPENAI_API_KEY or pass api_key in config.")

        self._client = OpenAI(api_key=resolved_api_key, base_url=self.api_url)

    def reset(self) -> None:
        return

    def _call_llm(self, prompt: str, response_format: Optional[Dict[str, Any]] = None) -> str:
        """Call LLM API and return response text. If response_format is set, API returns JSON per schema."""
        try:
            kwargs: Dict[str, Any] = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            }
            if response_format is not None:
                kwargs["response_format"] = response_format
            response = self._client.chat.completions.create(**kwargs)
            # DEBUG: Check response structure
            if not response or not response.choices:
                print(f"  [DEBUG LLM Call] WARNING: Empty response or no choices")
                return ""
            
            choice = response.choices[0]
            finish_reason = getattr(choice, 'finish_reason', None)
            
            # Check if response was truncated due to length
            if finish_reason == 'length':
                print(f"  [DEBUG LLM Call] WARNING: Response truncated due to max_tokens limit (finish_reason='length')")
                # Try to get usage info
                if hasattr(response, 'usage'):
                    usage = response.usage
                    print(f"  [DEBUG LLM Call] Usage: completion_tokens={getattr(usage, 'completion_tokens', 'N/A')}, "
                          f"reasoning_tokens={getattr(getattr(usage, 'completion_tokens_details', None), 'reasoning_tokens', 'N/A') if hasattr(usage, 'completion_tokens_details') else 'N/A'}")
            
            content = choice.message.content
            if content is None:
                print(f"  [DEBUG LLM Call] WARNING: Response content is None")
                print(f"  [DEBUG LLM Call] Finish reason: {finish_reason}")
                # If content is None but we have reasoning tokens, the model might be thinking
                # In this case, we should retry with higher max_tokens or different approach
                return ""
            
            result = content.strip()
            if not result:
                print(f"  [DEBUG LLM Call] WARNING: Response content is empty after strip")
                print(f"  [DEBUG LLM Call] Raw content: {repr(content)}")
                print(f"  [DEBUG LLM Call] Finish reason: {finish_reason}")
                # If finish_reason is 'length' and content is empty, all tokens were used for reasoning
                # We need to increase max_tokens or disable reasoning
                if finish_reason == 'length':
                    print(f"  [DEBUG LLM Call] SUGGESTION: Increase max_tokens or check if model supports disabling reasoning")
            
            return result
        except Exception as e:
            # Fallback on API error
            print(f"  [DEBUG LLM Call] ERROR: Exception occurred: {type(e).__name__}: {str(e)}")
            import traceback
            traceback.print_exc()
            import random
            rng = random.Random(self.seed)
            return f"ERROR: {str(e)}"

    def _get_response_format(self, raw_legal_actions: List[Any]) -> Optional[Dict[str, Any]]:
        """Return JSON-schema response_format when this agent uses structured output; else None."""
        return None

    def _parse_action_from_response(self, response: str, legal_actions: List[int], raw_legal_actions: List[Any]) -> int:
        """Parse action from LLM response text. Prefer JSON {\"action\": \"...\"} when present."""
        response_lower = response.lower()
        
        # DEBUG: Print raw response
        print(f"  [DEBUG Action Parse] LLM raw response: {response}")

        # Normalize raw_legal_actions to string names (same as schema enum)
        raw_legal_str = _normalize_raw_legal_action_names(
            list(raw_legal_actions) if isinstance(raw_legal_actions, list) else []
        )

        # DEBUG: Print legal actions mapping
        print(f"  [DEBUG Action Parse] Legal actions (IDs): {legal_actions}")
        print(f"  [DEBUG Action Parse] Legal actions (names): {raw_legal_str}")
        print(f"  [DEBUG Action Parse] Raw legal actions (original): {raw_legal_actions}")

        # Build action map: map action names to action IDs
        # IMPORTANT: RLCard's raw_legal_actions may not be in the same order as legal_actions
        # We need to match them correctly. RLCard typically provides raw_legal_actions as a list
        # that corresponds to legal_actions by index, but we should verify this.
        action_map = {}
        if isinstance(raw_legal_actions, list) and len(raw_legal_actions) > 0:
            # RLCard's legal_actions is typically sorted, but raw_legal_actions may be in a different order
            # We need to match them by finding the action name in raw_legal_actions that corresponds to each action_id
            # However, we don't have a direct mapping, so we'll use index-based matching as a best guess
            # and also build a comprehensive map of all possible action name variations
            
            # First, try index-based matching (assuming they're in the same order)
            for idx, action_id in enumerate(legal_actions):
                if idx < len(raw_legal_str):
                    action_name = raw_legal_str[idx].lower()
                    action_map[action_name] = action_id
            
            # Also try to match by checking if raw_legal_actions contains action names that match action_ids
            # This is a fallback in case the order is different
            # We'll create mappings for common action names
            for action_name_lower in raw_legal_str:
                action_name_lower = action_name_lower.lower()
                # Find the corresponding action_id by index (best guess)
                for idx, action_id in enumerate(legal_actions):
                    if idx < len(raw_legal_str) and raw_legal_str[idx].lower() == action_name_lower:
                        action_map[action_name_lower] = action_id
                        break
            
            # Add common variations and aliases
            for action_name, action_id in list(action_map.items()):
                # Map variations (e.g., "check_call" matches "check" or "call")
                if "check" in action_name and "call" in action_name:
                    action_map["check"] = action_id
                    action_map["call"] = action_id
                    action_map["check/call"] = action_id
                elif action_name == "fold":
                    action_map["fold"] = action_id
                elif action_name == "call":
                    action_map["call"] = action_id
                elif action_name == "raise":
                    action_map["raise"] = action_id
                    action_map["bet"] = action_id  # Some LLMs might say "bet" instead of "raise"
                elif action_name == "check":
                    action_map["check"] = action_id
                elif "all" in action_name and "in" in action_name:
                    action_map["all_in"] = action_id
                    action_map["allin"] = action_id
                    action_map["all-in"] = action_id
        
        # DEBUG: Print action map
        print(f"  [DEBUG Action Parse] Action map: {action_map}")

        # Fast path: when model returns structured JSON {"action": "..."}, use it first
        s = response.strip()
        if s.startswith("{") and s.endswith("}"):
            try:
                data = json.loads(s)
                if isinstance(data, dict) and "action" in data:
                    action_name = str(data["action"]).lower().strip()
                    if action_name in action_map:
                        print(f"  [DEBUG Action Parse] Matched '{action_name}' (JSON) -> action_id {action_map[action_name]}")
                        return action_map[action_name]
            except json.JSONDecodeError:
                pass

        # Check for explicit action mentions in response (use word boundaries for better matching)
        # Try exact matches first (whole words)
        for action_name, action_id in action_map.items():
            # Use word boundaries to avoid partial matches (e.g., "fold" in "folded")
            import re
            pattern = r'\b' + re.escape(action_name) + r'\b'
            if re.search(pattern, response_lower):
                print(f"  [DEBUG Action Parse] Matched '{action_name}' -> action_id {action_id}")
                return action_id
        
        # If no exact match, try substring match (fallback)
        for action_name, action_id in action_map.items():
            if action_name in response_lower:
                print(f"  [DEBUG Action Parse] Matched '{action_name}' (substring) -> action_id {action_id}")
                return action_id

        # Try JSON parsing
        try:
            json_match = re.search(r"\{[^}]+\}", response)
            if json_match:
                data = json.loads(json_match.group())
                if "action" in data:
                    action_name = str(data["action"]).lower()
                    if action_name in action_map:
                        print(f"  [DEBUG Action Parse] Matched '{action_name}' (JSON) -> action_id {action_map[action_name]}")
                        return action_map[action_name]
        except Exception as e:
            print(f"  [DEBUG Action Parse] JSON parsing failed: {e}")

        # Fallback: pick first legal action
        fallback_action = legal_actions[0] if legal_actions else 0
        print(f"  [DEBUG Action Parse] No match found, using fallback: action_id {fallback_action}")
        return fallback_action

    def _build_prompt(self, obs: Any, legal_actions: List[int], variant: str = "only") -> str:
        """Build prompt for LLM based on variant."""
        raw = _parse_rlcard_raw_obs(obs)
        hand = raw.get("hand", [])
        public_cards = raw.get("public_cards", [])
        legal_action_names = obs.get("raw_legal_actions", []) if isinstance(obs, dict) else []
        my_chips = raw.get("my_chips", 0)
        all_chips = raw.get("all_chips", [])

        # Convert Action enum objects to strings
        legal_actions_str = []
        for la in legal_action_names:
            if isinstance(la, str):
                legal_actions_str.append(la)
            elif hasattr(la, "name"):  # Action enum
                enum_name = la.name.lower()
                # Map common enum names
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

        prompt_parts = [
            "You are playing Texas Hold'em poker. Make a decision.",
            "",
            f"Your hand: {_format_cards(hand)}",
            f"Public cards: {_format_cards(public_cards)}",
            f"Your chips: {my_chips}",
            f"Legal actions: {', '.join(legal_actions_str) if legal_actions_str else 'unknown'}",
        ]

        if variant == "heuristic_range":
            # Add heuristic hand range information
            hand_strength = self._estimate_hand_strength(hand, public_cards)
            prompt_parts.append(f"\nHeuristic hand strength estimate: {hand_strength:.2f} (0=weak, 1=strong)")
        elif variant == "static_belief":
            # Add static opponent belief
            prompt_parts.append("\nOpponent belief (static): Assume opponents play moderately tight-aggressive.")
        elif variant == "fixed_prompt":
            # Use fixed prompt template
            prompt_parts = [
                "Texas Hold'em Decision",
                f"Hand: {_format_cards(hand)}, Board: {_format_cards(public_cards)}",
                f"Actions: {', '.join(legal_actions_str) if legal_actions_str else 'unknown'}",
                "Respond with action name only (e.g., 'call', 'raise', 'fold', 'check').",
            ]

        prompt_parts.append("\nRespond with the action name you choose (e.g., 'call', 'raise', 'fold', 'check').")
        return "\n".join(prompt_parts)

    def _estimate_hand_strength(self, hand: List[str], public_cards: List[str]) -> float:
        """Simple heuristic hand strength estimate."""
        if not hand or len(hand) < 2:
            return 0.5
        # Very simple: pairs and high cards are stronger
        ranks = [c[1] if len(c) >= 2 else "2" for c in hand]
        rank_order = "23456789TJQKA"
        vals = [rank_order.index(r) + 2 if r in rank_order else 2 for r in ranks]
        pair = 1.0 if ranks[0] == ranks[1] else 0.0
        high = max(vals) / 14.0
        return 0.3 * pair + 0.7 * high

    def act(self, obs: Any, legal_actions: List[int]) -> ActOutput:
        if not legal_actions:
            raise ValueError("No legal actions provided.")

        variant = getattr(self, "_variant", "only")
        prompt = self._build_prompt(obs, legal_actions, variant=variant)
        raw_legal_actions = obs.get("raw_legal_actions", []) if isinstance(obs, dict) else []
        response_format = self._get_response_format(raw_legal_actions) if raw_legal_actions else None

        # DEBUG: Print prompt length and first/last lines
        prompt_lines = prompt.split("\n")
        print(f"  [DEBUG LLM Call] Prompt length: {len(prompt)} chars, {len(prompt_lines)} lines")
        print(f"  [DEBUG LLM Call] Prompt first 3 lines: {prompt_lines[:3]}")
        print(f"  [DEBUG LLM Call] Prompt last 3 lines: {prompt_lines[-3:]}")
        print(f"  [DEBUG LLM Call] Calling LLM API with model: {self.model}")
        
        response = self._call_llm(prompt, response_format=response_format)
        
        # DEBUG: Check if response is empty
        if not response or len(response.strip()) == 0:
            print(f"  [DEBUG LLM Call] WARNING: LLM returned empty response!")
            print(f"  [DEBUG LLM Call] Response type: {type(response)}, value: {repr(response)}")

        action_id = self._parse_action_from_response(response, legal_actions, raw_legal_actions)

        return ActOutput(action=action_id, info={"policy": f"llm_{variant}", "llm_response": response[:100] if response else "EMPTY"})


@dataclass
class LLMOnlyAgent(LLMAgent):
    """LLM-only agent: Pure LLM decision without additional heuristics."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self._variant = "only"


@dataclass
class LLMHeuristicRangeAgent(LLMAgent):
    """LLM + Heuristic range agent: LLM with hand strength heuristic."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self._variant = "heuristic_range"


@dataclass
class LLMStaticBeliefAgent(LLMAgent):
    """LLM + Static belief agent: LLM with static opponent modeling."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self._variant = "static_belief"


@dataclass
class LLMFixedPromptAgent(LLMAgent):
    """LLM + Fixed prompt agent: LLM with fixed prompt template."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self._variant = "fixed_prompt"


@dataclass
class LLMFixedStyleOpponentAgent(LLMAgent):
    """LLM opponent agent with fixed player style (Nit/Rock, Calling Station, TAG, LAG).
    
    This agent:
    - Uses fixed style prompts to simulate different player archetypes
    - Observes complete game information (hand, street, board, stacks, pot, action history)
    - Makes decisions based on hand strength and playing style
    - Outputs actions within the same abstraction (fold/call/raise with discrete sizing buckets)
    """
    
    player_style: str = "tag"  # Options: "nit_rock", "calling_station", "tag", "lag"
    
    def __post_init__(self) -> None:
        super().__post_init__()
        self._variant = "fixed_style_opponent"
        # Normalize player style
        style_lower = self.player_style.lower().strip()
        if style_lower in ["nit", "rock", "nit_rock", "nit-rock"]:
            self._style = "nit_rock"
        elif style_lower in ["calling_station", "calling-station", "station"]:
            self._style = "calling_station"
        elif style_lower in ["tag", "tight_aggressive", "tight-aggressive"]:
            self._style = "tag"
        elif style_lower in ["lag", "loose_aggressive", "loose-aggressive"]:
            self._style = "lag"
        else:
            raise ValueError(f"Unknown player style: {self.player_style}. Must be one of: nit_rock, calling_station, tag, lag")

    def _get_response_format(self, raw_legal_actions: List[Any]) -> Optional[Dict[str, Any]]:
        """Return JSON-schema response_format so the model returns {\"action\": \"fold\"|\"call\"|...}."""
        names = _normalize_raw_legal_action_names(list(raw_legal_actions) if raw_legal_actions else [])
        if not names:
            return None
        enum_vals = sorted(set(names))
        schema_quick = {
            "type": "json_schema",
            "json_schema": {
                "name": "poker_action",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {"action": {"type": "string", "enum": enum_vals}},
                    "required": ["action"],
                    "additionalProperties": False,
                },
            },
        }
        return schema_quick
    
    def _get_style_prompt(self) -> str:
        """Get the fixed style prompt for this player type from markdown file."""
        # Map style to filename
        style_to_file = {
            "nit_rock": "nit_rock.md",
            "calling_station": "calling_station.md",
            "tag": "tag.md",
            "lag": "lag.md",
        }
        
        filename = style_to_file.get(self._style, "tag.md")
        
        # Get prompts directory (same directory as this file)
        current_file = Path(__file__)
        prompts_dir = current_file.parent / "prompts"
        prompt_file = prompts_dir / filename
        
        # Try to read from file
        if prompt_file.exists():
            try:
                return prompt_file.read_text(encoding="utf-8").strip()
            except Exception as e:
                print(f"Warning: Failed to read prompt file {prompt_file}: {e}")
                # Fallback to default TAG prompt
                return self._get_default_tag_prompt()
        else:
            print(f"Warning: Prompt file not found: {prompt_file}, using default TAG prompt")
            return self._get_default_tag_prompt()
    
    def _get_default_tag_prompt(self) -> str:
        """Fallback default TAG prompt if file reading fails."""
        return """You are a Tight-Aggressive (TAG) player in Texas Hold'em poker. Your playing style is:
- Tight: Only play strong starting hands (pairs, high cards, suited connectors)
- Aggressive: When you play, you bet and raise frequently
- Selective aggression: Only aggressive with strong hands
- Value-oriented: Bet for value when you have strong hands
- Fold weak hands pre-flop, but play aggressively post-flop with strong hands

Make decisions based ONLY on public information (board cards, pot size, stack sizes, action history).
You do NOT know your private cards - play as if you have a typical hand for your style."""
    
    def _build_prompt(self, obs: Any, legal_actions: List[int], variant: str = "fixed_style_opponent") -> str:
        """Build prompt for fixed-style opponent agent (with hand information)."""
        raw = _parse_rlcard_raw_obs(obs)
        
        # Extract all game information including hand
        hand = raw.get("hand", [])
        public_cards = raw.get("public_cards", [])
        my_chips = raw.get("my_chips", 0)
        all_chips = raw.get("all_chips", [])
        pot = raw.get("pot", 0)
        
        # Determine street based on number of public cards
        num_public = len(public_cards) if public_cards else 0
        if num_public == 0:
            street = "pre-flop"
        elif num_public == 3:
            street = "flop"
        elif num_public == 4:
            street = "turn"
        elif num_public == 5:
            street = "river"
        else:
            street = raw.get("street", "unknown")
        
        # Extract action history from observation if available
        # RLCard may not provide this directly, so we try multiple sources
        action_history = raw.get("action_history", [])
        if not action_history:
            # Try alternative field names
            action_history = raw.get("history", [])
            if not action_history:
                action_history = raw.get("actions", [])
        
        legal_action_names = obs.get("raw_legal_actions", []) if isinstance(obs, dict) else []
        
        # Convert Action enum objects to strings
        legal_actions_str = []
        for la in legal_action_names:
            if isinstance(la, str):
                legal_actions_str.append(la)
            elif hasattr(la, "name"):  # Action enum
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
        
        # Calculate pot odds information if possible
        pot_info = f"{pot}"
        if my_chips > 0 and pot > 0:
            # Calculate pot odds as percentage
            pot_odds_pct = (pot / (pot + my_chips)) * 100 if (pot + my_chips) > 0 else 0
            pot_info = f"{pot} (pot odds: {pot_odds_pct:.1f}%)"
        
        # Format stacks information
        # Prefer tracked chips (from eval_runner) over RLCard's internal chips
        # RLCard uses its own small chip system, but we track actual chips across episodes
        tracked_chips = raw.get("tracked_chips", {})
        tracked_chips_bb = raw.get("tracked_chips_bb", {})
        
        # Also check top-level obs for tracked chips
        if not tracked_chips and isinstance(obs, dict):
            tracked_chips = obs.get("tracked_chips", {})
            tracked_chips_bb = obs.get("tracked_chips_bb", {})
        
        if tracked_chips and tracked_chips_bb:
            # Use tracked chips (accurate across episodes)
            stack_parts = []
            for pid, chips in tracked_chips.items():
                chips_bb = tracked_chips_bb.get(pid, chips / 2)  # Fallback to chips/2 if BB not available
                stack_parts.append(f"Player {int(pid)}: {chips:.0f} chips ({chips_bb:.1f} BB)")
            stacks_info = ", ".join(stack_parts)
        else:
            # Fallback to RLCard's internal chips (less accurate, but better than nothing)
            big_blind_value = 2  # RLCard limit-holdem default big blind
            stacks_info = f"{all_chips}" if all_chips else f"Your stack: {my_chips}"
            if all_chips and isinstance(all_chips, list):
                # Format as both raw chips and Big Blinds for clarity
                stack_parts = []
                for i, chips in enumerate(all_chips):
                    chips_bb = chips / big_blind_value if big_blind_value > 0 else chips
                    stack_parts.append(f"Player {i}: {chips} chips ({chips_bb:.1f} BB)")
                stacks_info = ", ".join(stack_parts)
            elif my_chips > 0:
                my_chips_bb = my_chips / big_blind_value if big_blind_value > 0 else my_chips
                stacks_info = f"Your stack: {my_chips} chips ({my_chips_bb:.1f} BB)"
        
        # Build prompt with style instruction and complete game information
        prompt_parts = [
            self._get_style_prompt(),
            "",
            "=== Current Game Situation ===",
            f"Street: {street}",
            f"Your hand: {_format_cards(hand)}",
            f"Board cards: {_format_cards(public_cards)}",
            f"Pot size: {pot_info}",
            f"Stack sizes: {stacks_info}",
        ]
        
        if action_history:
            # Format action history nicely
            hist_str = ', '.join(str(a) for a in action_history[-10:])  # Last 10 actions
            prompt_parts.append(f"Recent action history: {hist_str}")
        else:
            # If no action history available, at least mention the street
            prompt_parts.append("Action history: Not available (early in hand)")
        
        # Add explicit instruction based on player style
        decision_instruction = "Based on your hand, the current board texture, pot size, stack sizes, and available actions, make a decision that reflects your playing style."
        if self._style == "calling_station":
            # Extra emphasis for Calling Station
            decision_instruction = "IMPORTANT: As a Calling Station, if 'call' is in the legal actions, you MUST choose 'call'. Only fold if 'call' is NOT available or you cannot afford to call."
        
        prompt_parts.extend([
            f"Legal actions: {', '.join(legal_actions_str) if legal_actions_str else 'unknown'}",
            "",
            decision_instruction,
            "Respond with a JSON object: {\"action\": \"<name>\"} where <name> is exactly one of the legal action names above.",
        ])
        
        prompt = "\n".join(prompt_parts)
        
        # DEBUG: Print prompt for LLM fixed-style opponent to verify hand information is included
        if hand:
            print(f"  [DEBUG LLM] Player hand received: {_format_cards(hand)}")
        else:
            print(f"  [DEBUG LLM] WARNING: Player hand is EMPTY or not found in observation!")
            print(f"  [DEBUG LLM] Raw observation keys: {list(raw.keys()) if isinstance(raw, dict) else 'N/A'}")
        
        # DEBUG: Print full prompt for Calling Station to debug
        if self._style == "calling_station":
            print(f"  [DEBUG LLM Calling Station] Full prompt:")
            print("  " + "\n  ".join(prompt.split("\n")))
        
        return prompt
