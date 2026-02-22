from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from tqdm import tqdm

from ma_poker.agents.base import Agent
from ma_poker.envs.base import MultiAgentEnv


@dataclass
class EvalConfig:
    episodes: int = 200
    seed: Optional[int] = 0
    show_progress: bool = True
    save_detailed_logs: bool = True  # Save detailed episode logs
    log_dir: Optional[Path] = None  # Directory to save episode logs
    # Chip management settings
    initial_chips_bb: int = 100  # Initial chips in Big Blinds (default 100 BB)
    rebuy_threshold_bb: int = 0  # Rebuy when chips < this threshold (0 = no rebuy)
    rebuy_amount_bb: int = 100  # Amount to rebuy (in BB)
    track_chips_across_episodes: bool = True  # Track chips across episodes (like real poker)


def _parse_rlcard_raw_obs(obs: Any) -> Dict[str, Any]:
    """Extract raw observation dict from RLCard obs."""
    if not isinstance(obs, dict):
        return {}
    raw = obs.get("raw_obs")
    return raw if isinstance(raw, dict) else {}


def _format_cards(cards: List[Any]) -> str:
    """Format card list to readable string.
    
    Handles both string cards and RLCard Card objects.
    """
    if not cards:
        return "None"
    
    # Convert Card objects to strings if needed
    card_strings = []
    for card in cards:
        if isinstance(card, str):
            card_strings.append(card)
        elif hasattr(card, '__str__'):
            # RLCard Card object - convert to string
            card_strings.append(str(card))
        elif hasattr(card, 'get_str'):
            # Alternative Card object method
            card_strings.append(card.get_str())
        else:
            # Fallback: try to convert to string
            card_strings.append(str(card))
    
    return ", ".join(card_strings)


def _get_player_name(agent: Agent, player_id: int) -> str:
    """Get a friendly name for a player based on their agent type."""
    agent_type = type(agent).__name__
    
    # Check for LLMFixedStyleOpponentAgent with player_style
    if agent_type == "LLMFixedStyleOpponentAgent":
        if hasattr(agent, "player_style"):
            style = agent.player_style.lower().strip()
            if style in ["nit_rock", "nit", "rock"]:
                return "Nit/Rock"
            elif style in ["calling_station", "calling-station", "station"]:
                return "Calling Station"
            elif style in ["tag", "tight_aggressive", "tight-aggressive"]:
                return "TAG"
            elif style in ["lag", "loose_aggressive", "loose-aggressive"]:
                return "LAG"
        return "LLM Opponent"
    
    # Check for CFRAgent
    if agent_type == "CFRAgent":
        if hasattr(agent, "use_deep_cfr") and agent.use_deep_cfr:
            return "DeepCFR"
        return "CFR"
    
    # Check for DeepCFRAgent
    if agent_type == "DeepCFRAgent":
        return "DeepCFR"
    
    # Check for OursAgent
    if agent_type == "OursAgent":
        return "Ours"
    
    # Check for other agent types
    if agent_type == "LLMOnlyAgent":
        return "LLM"
    if agent_type == "LLMHeuristicRangeAgent":
        return "LLM+Heuristic"
    if agent_type == "LLMStaticBeliefAgent":
        return "LLM+Belief"
    if agent_type == "LLMFixedPromptAgent":
        return "LLM+Fixed"
    if agent_type == "RuleBasedAgent":
        return "Rule-Based"
    if agent_type == "RandomAgent":
        return "Random"
    if agent_type == "RLAgent":
        return "RL"
    
    # Fallback: use class name without "Agent" suffix
    if agent_type.endswith("Agent"):
        return agent_type[:-5]  # Remove "Agent" suffix
    
    return agent_type


def _convert_action_enum_to_str(obj: Any) -> Any:
    """Recursively convert Action enum objects to strings for JSON serialization."""
    if hasattr(obj, "name"):  # Action enum
        enum_name = obj.name.lower()
        # Map common enum names
        if "fold" in enum_name:
            return "fold"
        elif "check" in enum_name and "call" in enum_name:
            return "check_call"
        elif "check" in enum_name:
            return "check"
        elif "call" in enum_name:
            return "call"
        elif "raise" in enum_name or "pot" in enum_name:
            return "raise"
        elif "all" in enum_name and "in" in enum_name:
            return "all_in"
        else:
            return enum_name
    elif isinstance(obj, dict):
        return {k: _convert_action_enum_to_str(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_convert_action_enum_to_str(item) for item in obj]
    else:
        return obj


def evaluate(env: MultiAgentEnv, agents: Dict[int, Agent], cfg: EvalConfig) -> Dict[str, object]:
    for a in agents.values():
        a.reset()

    # Auto-load memory for OursAgent instances (if memory files exist)
    if cfg.log_dir:
        memories_dir = cfg.log_dir / "memories"
        if memories_dir.exists():
            for pid, agent in agents.items():
                # Check if agent is OursAgent (by checking for load_memory method)
                if hasattr(agent, "load_memory"):
                    memory_file = memories_dir / f"agent_{pid}_memory.json"
                    if memory_file.exists():
                        try:
                            agent.load_memory(str(memory_file))
                            print(f"Loaded memory for agent {pid} from {memory_file}")
                        except Exception as e:
                            # Log error but don't fail the evaluation
                            print(f"Warning: Failed to load memory for agent {pid}: {e}")

    total_returns = np.zeros((env.num_players,), dtype=np.float64)
    
    # Create log directory if needed
    if cfg.save_detailed_logs and cfg.log_dir:
        cfg.log_dir.mkdir(parents=True, exist_ok=True)

    # Chip management: track chips across episodes (like real poker)
    # Get big blind from environment (default to 2 if not available)
    try:
        if hasattr(env, '_env') and hasattr(env._env, 'game'):
            big_blind = getattr(env._env.game, 'big_blind', 2)
        else:
            big_blind = 2
    except:
        big_blind = 2
    
    initial_chips = cfg.initial_chips_bb * big_blind
    player_chips = {pid: initial_chips for pid in range(env.num_players)}  # Current chips for each player
    rebuy_count = {pid: 0 for pid in range(env.num_players)}  # Number of rebuys per player
    episode_chips_log = []  # Log chips at start/end of each episode
    
    # Track metrics for statistics
    # Cumulative chips in BB (for CumBB@T calculation)
    cumulative_chips_bb = {pid: [cfg.initial_chips_bb] for pid in range(env.num_players)}  # Track chips after each episode
    # Minimum chips reached (for MDD calculation) - allow negative
    min_chips_reached = {pid: initial_chips for pid in range(env.num_players)}  # Track minimum chips
    # VPIP tracking: track if player voluntarily put money in pot per episode
    # VPIP excludes blind actions (small blind and big blind are forced)
    vpip_tracking = {pid: {"voluntarily_put": 0, "total_hands": 0} for pid in range(env.num_players)}
    # WVPIP tracking: track wins when voluntarily put money in pot
    wvpip_tracking = {pid: {"voluntarily_put_and_won": 0, "voluntarily_put_count": 0} for pid in range(env.num_players)}
    # Episode rewards in BB (for BB/10 calculation)
    episode_rewards_bb = {pid: [] for pid in range(env.num_players)}  # Store rewards per episode in BB
    
    # Belief accuracy tracking: for OursAgent, track if opponent's actual hand is in TOP 10
    # Structure: {ours_agent_id: {"correct": count, "total": count, "per_opponent": {opponent_id: {"correct": count, "total": count}}}}
    belief_accuracy_tracking = {}  # Will be initialized per OursAgent

    it = range(cfg.episodes)
    if cfg.show_progress:
        it = tqdm(it, desc="eval", ncols=100)

    # Initialize seed for the entire experiment
    # Use a fixed base seed - RLCard will handle blind rotation internally
    # We set numpy seed once at the start, then let RLCard manage its own randomness
    base_seed = cfg.seed if cfg.seed is not None else 0
    if base_seed is not None:
        np.random.seed(base_seed)
    
    for ep in it:
        # Don't pass seed to reset() - let RLCard handle its own randomness and blind rotation
        # RLCard should automatically rotate button/blind positions between hands
        # This ensures proper blind rotation while maintaining card randomness
        ep_seed = None  # Let RLCard manage its own state
        
        # Record chips at start of episode (before rebuy check)
        episode_start_chips = {str(pid): player_chips[pid] for pid in range(env.num_players)}
        
        # Check for rebuys before episode starts
        rebuy_this_episode = {pid: False for pid in range(env.num_players)}
        if cfg.track_chips_across_episodes and cfg.rebuy_threshold_bb >= 0:
            rebuy_amount = cfg.rebuy_amount_bb * big_blind
            for pid in range(env.num_players):
                # Rebuy when chips < threshold (e.g., < 50BB)
                # Convert current chips to BB for comparison
                current_chips_bb = player_chips[pid] / big_blind
                if current_chips_bb < cfg.rebuy_threshold_bb:
                    player_chips[pid] += rebuy_amount
                    rebuy_count[pid] += 1
                    rebuy_this_episode[pid] = True
        
        # Reset environment - RLCard will handle blind rotation internally
        obs_dict = env.reset(seed=ep_seed)
        done = False
        returns = np.zeros((env.num_players,), dtype=np.float64)
        
        # Track first player to act in this episode (for blind rotation check)
        first_player = env.current_player()
        if ep < 10:  # Debug: print first 10 episodes to verify blind rotation
            print(f"[DEBUG] Episode {ep}: First player to act = Player {first_player}, Seed = {ep_seed}")
        
        # Track initial chips for this episode (after rebuy, for calculating final chips)
        episode_initial_chips = player_chips.copy()
        
        # Episode log data
        episode_log = {
            "episode_id": ep,
            "seed": ep_seed,
            "num_players": env.num_players,
            "steps": [],
            "final_rewards": {},
            "winner": None,
        }
        
        # Track chips for each player (initialize from first observation)
        chips_history = {pid: [] for pid in range(env.num_players)}
        step_id = 0
        
        # Track final state for training data collection
        final_public_cards = []
        final_chips_state = {}
        
        # Track VPIP per episode: track if player voluntarily put money in pot this episode
        # VPIP excludes blind actions - only count voluntary actions (raise, or call after blinds)
        # In RLCard, blinds are typically handled automatically, so we need to detect them differently
        episode_vpip = {pid: {"voluntarily_put": False, "had_preflop_action": False, "is_blind": False} for pid in range(env.num_players)}
        # Track pre-flop actions for VPIP (VPIP is typically measured pre-flop)
        pre_flop_actions = {pid: [] for pid in range(env.num_players)}  # Track actions before flop
        # Track chips at start of pre-flop to detect blind actions (blinds reduce chips before first action)
        pre_flop_start_chips = {}  # Track chips at start of pre-flop for each player
        pre_flop_action_count = 0  # Count pre-flop actions
        pre_flop_started = False  # Track if we've started tracking pre-flop
        
        # Track actual hands for each player (for belief accuracy evaluation)
        # Record each player's hand when they first act in pre-flop
        episode_player_hands = {pid: None for pid in range(env.num_players)}  # Store (card1, card2) tuple
        
        # DEBUG: Print all players' hands at the start of the episode
        # Try to get all players' hands from RLCard internal state
        all_players_hands = {}
        
        # Method 1: Try to get state for all players from RLCard internal environment
        if hasattr(env, '_env') and hasattr(env._env, 'get_state'):
            for pid in range(env.num_players):
                try:
                    state = env._env.get_state(pid)
                    if isinstance(state, dict):
                        # RLCard stores hand in 'hand' key
                        hand = state.get('hand', [])
                        if hand:
                            all_players_hands[pid] = hand
                except Exception as e:
                    # If we can't get state, continue trying other methods
                    pass
        
        # Method 2: Try to get hands from the initial observation (only current player)
        if obs_dict:
            for pid, obs in obs_dict.items():
                raw_obs = _parse_rlcard_raw_obs(obs)
                hand = raw_obs.get("hand", [])
                if hand:
                    all_players_hands[pid] = hand
        
        # Method 3: Try to access RLCard's internal game state directly
        if hasattr(env, '_env') and hasattr(env._env, 'game'):
            try:
                game = env._env.game
                if hasattr(game, 'players'):
                    for pid in range(env.num_players):
                        if pid < len(game.players):
                            player = game.players[pid]
                            if hasattr(player, 'hand') and player.hand:
                                all_players_hands[pid] = player.hand
            except Exception as e:
                pass
        
        # Print episode start debug info
        print(f"\n{'='*80}")
        print(f"[DEBUG] Episode {ep} - Hand Start")
        print(f"{'='*80}")
        if all_players_hands:
            for pid in range(env.num_players):
                player_name = _get_player_name(agents[pid], pid)
                hand = all_players_hands.get(pid, [])
                hand_str = _format_cards(hand) if hand else "Unknown"
                print(f"  Player {pid} ({player_name}): Hand = {hand_str}")
        else:
            print(f"  (Hands will be printed as players act)")

        while not done:
            cur = env.current_player()
            obs = obs_dict.get(cur, None)
            legal = env.legal_actions(cur)
            
            # Extract current state information
            raw_obs = _parse_rlcard_raw_obs(obs)
            current_hand = raw_obs.get("hand", [])
            public_cards = raw_obs.get("public_cards", [])
            my_chips = raw_obs.get("my_chips", 0)
            all_chips = raw_obs.get("all_chips", [])
            legal_action_names = obs.get("raw_legal_actions", []) if isinstance(obs, dict) else []
            
            # IMPORTANT: Override RLCard's internal chips with our tracked chips
            # RLCard uses its own small chip system (1-2 units), but we track chips across episodes
            # Calculate current chips for this episode: initial chips + returns so far
            current_tracked_chips = {}
            for pid in range(env.num_players):
                # Current chips = initial chips for this episode + returns accumulated so far
                current_tracked_chips[pid] = episode_initial_chips[pid] + returns[pid]
            
            # Add tracked chips to observation so LLM agents can use them
            if isinstance(obs, dict):
                # Add tracked chips to raw_obs so LLM can access them
                if "raw_obs" in obs and isinstance(obs["raw_obs"], dict):
                    obs["raw_obs"]["tracked_chips"] = current_tracked_chips
                    obs["raw_obs"]["tracked_chips_bb"] = {pid: chips / big_blind for pid, chips in current_tracked_chips.items()}
                # Also add to top level for easier access
                obs["tracked_chips"] = current_tracked_chips
                obs["tracked_chips_bb"] = {pid: chips / big_blind for pid, chips in current_tracked_chips.items()}
            
            # DEBUG: Record player's hand when they first act (for display at episode end)
            if current_hand and episode_player_hands[cur] is None:
                episode_player_hands[cur] = current_hand
                # Also update all_players_hands for display
                all_players_hands[cur] = current_hand
            
            # Record chips before action
            chips_before = {}
            if all_chips:
                for pid in range(env.num_players):
                    if pid < len(all_chips):
                        chips_before[pid] = int(all_chips[pid])
                    else:
                        chips_before[pid] = my_chips if pid == cur else 0
            else:
                # Fallback: use my_chips for current player, estimate for others
                chips_before[cur] = int(my_chips)
                for pid in range(env.num_players):
                    if pid != cur:
                        chips_before[pid] = chips_before.get(pid, 0)
            
            # Record chips history
            for pid in range(env.num_players):
                chips_history[pid].append(chips_before.get(pid, 0))
            
            # Get action
            act = agents[cur].act(obs, legal)
            
            # DEBUG: Print current player's hand and action
            player_name = _get_player_name(agents[cur], cur)
            hand_str = _format_cards(current_hand)
            street_str = "Pre-flop" if len(public_cards) == 0 else f"{len(public_cards)} cards"  # Pre-flop, Flop, Turn, River
            public_cards_str = _format_cards(public_cards) if public_cards else "None"
            
            # Extract action name, handling both string and Action enum formats
            if act.action in legal and len(legal_action_names) > legal.index(act.action):
                raw_action_name = legal_action_names[legal.index(act.action)]
                # Convert Action enum to string if needed
                if isinstance(raw_action_name, str):
                    action_name = raw_action_name
                elif hasattr(raw_action_name, "name"):  # Action enum
                    enum_name = raw_action_name.name.lower()
                    # Map common enum names
                    if "fold" in enum_name:
                        action_name = "fold"
                    elif "check" in enum_name and "call" in enum_name:
                        action_name = "check_call"
                    elif "check" in enum_name:
                        action_name = "check"
                    elif "call" in enum_name:
                        action_name = "call"
                    elif "raise" in enum_name or "pot" in enum_name:
                        action_name = "raise"
                    elif "all" in enum_name and "in" in enum_name:
                        action_name = "all_in"
                    else:
                        action_name = enum_name
                else:
                    action_name = str(raw_action_name)
            else:
                action_name = f"action_{act.action}"
            
            # DEBUG: Print action details
            print(f"  [Step {step_id}] Player {cur} ({player_name}): Hand={hand_str}, Street={street_str}, Public={public_cards_str}, Action={action_name}")
            
            # Track VPIP: check if this is pre-flop (no public cards) and player voluntarily put money in pot
            is_pre_flop = len(public_cards) == 0
            if is_pre_flop:
                # Initialize pre-flop tracking on first pre-flop action
                if not pre_flop_started:
                    # Record chips at start of pre-flop for all players
                    for pid in range(env.num_players):
                        pre_flop_start_chips[pid] = chips_before.get(pid, 0)
                    pre_flop_started = True
                
                pre_flop_action_count += 1
                episode_vpip[cur]["had_preflop_action"] = True
                
                # Record player's hand for belief accuracy evaluation (record once per player)
                if episode_player_hands[cur] is None and len(current_hand) >= 2:
                    episode_player_hands[cur] = (current_hand[0], current_hand[1])
                
                # Detect if this is a blind action by checking chip reduction
                # In RLCard, blinds are typically posted automatically before first action
                # We detect blinds by checking if chips decreased without a visible action
                # OR if this is the first action and chips are already reduced
                # For simplicity, we'll use a heuristic: if this is one of the first 2 actions
                # AND the player's chips are less than initial, it's likely a blind
                initial_chips_for_player = pre_flop_start_chips.get(cur, chips_before.get(cur, 0))
                current_chips = chips_before.get(cur, 0)
                
                # Heuristic: first 2 actions in pre-flop are likely blinds (for 2-player: SB and BB)
                # But we also check if player had chips reduced (indicating blind was posted)
                is_blind_action = False
                is_small_blind = False
                is_big_blind = False
                
                if env.num_players == 2:
                    # In 2-player game: first action is small blind, second is big blind
                    if pre_flop_action_count == 1:
                        is_small_blind = True
                        is_blind_action = True
                        episode_vpip[cur]["is_blind"] = True
                    elif pre_flop_action_count == 2:
                        is_big_blind = True
                        is_blind_action = True
                        episode_vpip[cur]["is_blind"] = True
                else:
                    # For multi-player games, use the original heuristic
                    if pre_flop_action_count <= 2:
                        is_blind_action = True
                        episode_vpip[cur]["is_blind"] = True
                
                # Determine if this is a voluntary action
                # VPIP definition: Voluntarily Put money In Pot
                # - Small blind: call (complete blind) is NOT voluntary, raise is voluntary
                # - Big blind: check (if SB called) is NOT voluntary, call/raise (if SB raised) is voluntary
                # - After blinds: call/raise is voluntary, fold is not
                
                if is_small_blind:
                    # Small blind: only raise is voluntary
                    if action_name in ["raise", "all_in"]:
                        episode_vpip[cur]["voluntarily_put"] = True
                    # Call (complete blind) is NOT voluntary
                elif is_big_blind:
                    # Big blind: check if this is a free check (SB called) or voluntary action (SB raised)
                    # We need to check the previous action (SB's action)
                    if pre_flop_actions and len(pre_flop_actions) > 0:
                        # Get the previous player's action (should be SB)
                        prev_player_actions = []
                        for pid, actions in pre_flop_actions.items():
                            if pid != cur and len(actions) > 0:
                                prev_player_actions.extend(actions)
                        
                        # If SB called, BB's check is free (not voluntary)
                        # If SB raised, BB's call/raise is voluntary
                        if prev_player_actions and prev_player_actions[-1] == "call":
                            # SB called, so BB's check is free (not voluntary)
                            # Only raise would be voluntary
                            if action_name in ["raise", "all_in"]:
                                episode_vpip[cur]["voluntarily_put"] = True
                        else:
                            # SB raised or folded, so BB's action is voluntary (if not fold)
                            if action_name in ["raise", "all_in", "call"]:
                                episode_vpip[cur]["voluntarily_put"] = True
                    else:
                        # Fallback: if we can't determine, treat raise as voluntary
                        if action_name in ["raise", "all_in"]:
                            episode_vpip[cur]["voluntarily_put"] = True
                elif not is_blind_action:
                    # This is a voluntary action (after blinds)
                    # VPIP: voluntarily put money in pot (raise, or call after blinds, but not fold)
                    # Note: check is only voluntary if it requires matching a bet
                    if action_name in ["raise", "all_in", "call"]:
                        # Raise, all-in, and call are voluntary (they put money in pot)
                        episode_vpip[cur]["voluntarily_put"] = True
                    # Check and fold don't count as voluntarily putting money in pot
                    # (Check is only voluntary if matching a bet, which we can't easily detect here)
                
                pre_flop_actions[cur].append(action_name)
            
            # Execute step
            step_out = env.step({cur: act.action})
            
            # Extract state after action
            next_raw_obs = {}
            if step_out.obs:
                next_player = list(step_out.obs.keys())[0] if step_out.obs else cur
                next_obs = step_out.obs.get(next_player, {})
                next_raw_obs = _parse_rlcard_raw_obs(next_obs)
            
            # Record chips after action
            chips_after = {}
            if not done:
                next_all_chips = next_raw_obs.get("all_chips", [])
                if next_all_chips and isinstance(next_all_chips, list):
                    for pid in range(env.num_players):
                        if pid < len(next_all_chips):
                            chips_after[pid] = int(next_all_chips[pid])
                        else:
                            chips_after[pid] = chips_before.get(pid, 0)
                else:
                    # If no chips info, use chips_before as estimate
                    chips_after = chips_before.copy()
            else:
                # Game over: final chips = initial chips + rewards
                # We'll calculate this from final rewards
                for pid in range(env.num_players):
                    # Estimate: chips_after = chips_before + reward (simplified)
                    # In reality, we'd need to track initial chips
                    chips_after[pid] = chips_before.get(pid, 0)
            
            # Update opponent analysis for OursAgent instances
            # When an opponent acts, update their analysis
            for pid, agent in agents.items():
                if pid != cur and hasattr(agent, "update_opponent_action"):
                    # Extract context for opponent analysis
                    betting_round = 0  # pre-flop
                    if len(public_cards) >= 3:
                        betting_round = 1  # flop
                    if len(public_cards) >= 4:
                        betting_round = 2  # turn
                    if len(public_cards) >= 5:
                        betting_round = 3  # river
                    
                    pot_size = sum(chips_before.values()) if chips_before else 0
                    bet_size = abs(chips_before.get(cur, 0) - chips_after.get(cur, 0)) if chips_after else 0
                    
                    context = {
                        "public_cards": public_cards,
                        "pot_size": pot_size,
                        "betting_round": betting_round,
                        "position": cur,
                        "bet_size": bet_size,
                    }
                    agent.update_opponent_action(cur, action_name, context)
            
            # Convert legal_action_names to strings (handle Action enum objects)
            legal_actions_str = []
            for la in legal_action_names:
                if isinstance(la, str):
                    legal_actions_str.append(la)
                elif hasattr(la, "name"):  # Action enum
                    # Convert enum to readable string
                    enum_name = la.name.lower()
                    # Map common enum names
                    if "fold" in enum_name:
                        legal_actions_str.append("fold")
                    elif "check" in enum_name and "call" in enum_name:
                        legal_actions_str.append("check_call")
                    elif "check" in enum_name:
                        legal_actions_str.append("check")
                    elif "call" in enum_name:
                        legal_actions_str.append("call")
                    elif "raise" in enum_name or "pot" in enum_name:
                        legal_actions_str.append("raise")
                    elif "all" in enum_name and "in" in enum_name:
                        legal_actions_str.append("all_in")
                    else:
                        legal_actions_str.append(enum_name)
                else:
                    legal_actions_str.append(str(la))
            
            # Record step
            step_log = {
                "step_id": step_id,
                "current_player": int(cur),
                "hand": _format_cards(current_hand),
                "public_cards": _format_cards(public_cards),
                "chips_before": {str(pid): chips_before.get(pid, 0) for pid in range(env.num_players)},
                "legal_actions": legal_actions_str,
                "action_taken": action_name,
                "action_id": int(act.action),
                "chips_after": {str(pid): chips_after.get(pid, 0) for pid in range(env.num_players)},
                "agent_info": act.info if hasattr(act, "info") else {},
            }
            episode_log["steps"].append(step_log)
            
            # Update returns
            for pid, r in step_out.rewards.items():
                returns[int(pid)] += float(r)
            
            obs_dict = step_out.obs
            done = step_out.terminated
            step_id += 1
            
            # Update final state tracking
            if done:
                final_public_cards = public_cards if 'public_cards' in locals() else []
                final_chips_state = chips_after if 'chips_after' in locals() else {}
                
                # DEBUG: Print final results for this hand
                print(f"\n  [DEBUG] Episode {ep} - Hand End")
                print(f"  Final Public Cards: {_format_cards(final_public_cards)}")
                for pid in range(env.num_players):
                    player_name = _get_player_name(agents[pid], pid)
                    final_hand = episode_player_hands.get(pid)
                    hand_str = _format_cards(list(final_hand)) if final_hand else "Unknown"
                    reward = returns[pid]
                    print(f"  Player {pid} ({player_name}): Hand={hand_str}, Reward={reward:.2f} chips ({reward/big_blind:.2f} BB)")
                print(f"{'='*80}\n")

        # Record final rewards and winner
        for pid in range(env.num_players):
            episode_log["final_rewards"][str(pid)] = float(returns[pid])
        
        # Determine winner (player with highest reward)
        if len(returns) > 0:
            winner_id = int(np.argmax(returns))
            if returns[winner_id] > 0:
                episode_log["winner"] = winner_id
        
        # Add chips history summary
        episode_log["chips_history"] = {
            str(pid): chips_history[pid] for pid in range(env.num_players)
        }
        
        # Update player chips across episodes (like real poker)
        if cfg.track_chips_across_episodes:
            for pid in range(env.num_players):
                # Update chips: current = initial + reward (reward is in chips)
                player_chips[pid] = episode_initial_chips[pid] + returns[pid]
                # Track minimum chips reached (for MDD) - allow negative
                min_chips_reached[pid] = min(min_chips_reached[pid], player_chips[pid])
        
        # Record chips at end of episode (allow negative for MDD calculation)
        episode_end_chips = {str(pid): player_chips[pid] for pid in range(env.num_players)}
        
        # Update metrics tracking
        for pid in range(env.num_players):
            # Convert chips to BB and track cumulative
            chips_bb = player_chips[pid] / big_blind
            cumulative_chips_bb[pid].append(chips_bb)
            
            # Track episode reward in BB
            reward_bb = returns[pid] / big_blind
            episode_rewards_bb[pid].append(reward_bb)
            
            # Update VPIP tracking (exclude blind actions)
            # VPIP definition: Voluntarily Put money In Pot
            # - Denominator: all hands where player had a chance to act pre-flop (excluding forced blind posts)
            # - Numerator: hands where player voluntarily put money in pot pre-flop
            # Note: In RLCard, each episode is one hand, so we count each episode as one hand
            # 
            # Rules:
            # 1. If player is in blind position and only completes the blind (call), don't count in denominator
            # 2. If player is in blind position but raises, count in both denominator and numerator
            # 3. If player is not in blind position, count in denominator if they had action
            # 4. Only count voluntary actions (raise, call after blinds, but not fold)
            # 
            # IMPORTANT: We need to ensure denominator is not zero
            # Use a more lenient approach: count all hands where player had pre-flop action,
            # except for blind positions where player only completed the blind (call)
            if episode_vpip[pid]["had_preflop_action"]:
                # Check if this is a blind position with only call (forced blind)
                is_forced_blind = episode_vpip[pid]["is_blind"] and not episode_vpip[pid]["voluntarily_put"]
                
                if not is_forced_blind:
                    # Count in denominator (either not blind, or blind but raised)
                    vpip_tracking[pid]["total_hands"] += 1
                    
                    if episode_vpip[pid]["voluntarily_put"]:
                        # Voluntarily put money in pot
                        vpip_tracking[pid]["voluntarily_put"] += 1
                        wvpip_tracking[pid]["voluntarily_put_count"] += 1
                        if returns[pid] > 0:
                            wvpip_tracking[pid]["voluntarily_put_and_won"] += 1
                # If is_forced_blind, don't count in denominator (forced blind doesn't count)
            # If player did not have pre-flop action, don't count (rare case)
        
        # Add episode-level chip tracking
        episode_log["chips_at_start"] = episode_start_chips
        episode_log["chips_at_end"] = episode_end_chips
        episode_log["chips_initial_for_episode"] = {str(pid): episode_initial_chips[pid] for pid in range(env.num_players)}
        
        # Record rebuy information
        episode_log["rebuy_before_episode"] = {str(pid): (1 if rebuy_this_episode[pid] else 0) for pid in range(env.num_players)}
        # Record cumulative rebuy count (total number of rebuys for each player up to this episode)
        episode_log["total_rebuy_count"] = {str(pid): rebuy_count[pid] for pid in range(env.num_players)}
        
        # Log episode chips summary
        episode_chips_log.append({
            "episode": ep,
            "chips_at_start": episode_start_chips,
            "chips_at_end": episode_end_chips,
            "rewards": {str(pid): float(returns[pid]) for pid in range(env.num_players)},
        })
        
        # Save episode log
        if cfg.save_detailed_logs and cfg.log_dir:
            log_file = cfg.log_dir / f"episode_{ep:06d}.json"
            # Convert any Action enums to strings before JSON serialization
            episode_log_clean = _convert_action_enum_to_str(episode_log)
            log_file.write_text(json.dumps(episode_log_clean, indent=2), encoding="utf-8")

        total_returns += returns

        # Update memory for agents that support it (e.g., OursAgent)
        for pid, agent in agents.items():
            if hasattr(agent, "update_episode_outcome"):
                agent.update_episode_outcome(float(returns[pid]))
        
        # Evaluate belief accuracy for OursAgent instances
        # Check if opponent's actual hand is in belief TOP 10
        from ma_poker.agents.hand_utils import cards_to_hand_type, get_top_hands
        
        for ours_pid, agent in agents.items():
            # Check if this is an OursAgent (has update_opponent_action method)
            if hasattr(agent, "update_opponent_action") and hasattr(agent, "_opponent_analysis_agent"):
                # Initialize tracking for this OursAgent if not exists
                if ours_pid not in belief_accuracy_tracking:
                    belief_accuracy_tracking[ours_pid] = {
                        "correct": 0,
                        "total": 0,
                        "per_opponent": {opp_pid: {"correct": 0, "total": 0} for opp_pid in range(env.num_players) if opp_pid != ours_pid}
                    }
                
                # Check each opponent's actual hand against belief TOP 10
                for opp_pid in range(env.num_players):
                    if opp_pid == ours_pid:
                        continue  # Skip self
                    
                    # Get opponent's actual hand
                    if episode_player_hands[opp_pid] is not None:
                        card1, card2 = episode_player_hands[opp_pid]
                        # Convert to hand type index
                        actual_hand_type = cards_to_hand_type(card1, card2)
                        
                        # Get TOP 10 hands from belief (via opponent_analysis_agent)
                        analysis_agent = agent._opponent_analysis_agent
                        if opp_pid in analysis_agent._opponent_beliefs:
                            belief_obj = analysis_agent._opponent_beliefs[opp_pid]
                            top_hands = get_top_hands(belief_obj.belief.tolist(), top_k=10)
                            top_hand_indices = [idx for idx, _, _ in top_hands]
                            
                            # Check if actual hand is in TOP 10
                            is_correct = actual_hand_type in top_hand_indices
                            
                            # Update tracking
                            belief_accuracy_tracking[ours_pid]["total"] += 1
                            belief_accuracy_tracking[ours_pid]["per_opponent"][opp_pid]["total"] += 1
                            if is_correct:
                                belief_accuracy_tracking[ours_pid]["correct"] += 1
                                belief_accuracy_tracking[ours_pid]["per_opponent"][opp_pid]["correct"] += 1
        
        # Collect training data for belief network (for OursAgent instances)
        # At the end of episode, we know actual hands, so we can train the belief network
        for ours_pid, agent in agents.items():
            if hasattr(agent, "add_belief_training_example") and hasattr(agent, "_opponent_analysis_agent"):
                analysis_agent = agent._opponent_analysis_agent
                for opp_pid in range(env.num_players):
                    if opp_pid == ours_pid:
                        continue
                    
                    # Get opponent's actual hand
                    if episode_player_hands[opp_pid] is not None:
                        card1, card2 = episode_player_hands[opp_pid]
                        actual_hand_type = cards_to_hand_type(card1, card2)
                        
                        # Get action sequence for this opponent
                        action_sequence = analysis_agent._opponent_action_sequence.get(opp_pid, [])
                        
                        # Get final context features (use the final state from episode)
                        final_betting_round = 0
                        if len(final_public_cards) >= 3:
                            final_betting_round = 1
                        if len(final_public_cards) >= 4:
                            final_betting_round = 2
                        if len(final_public_cards) >= 5:
                            final_betting_round = 3
                        
                        final_pot_size = sum(final_chips_state.values()) if final_chips_state else 0
                        
                        final_context = {
                            "public_cards": final_public_cards,
                            "pot_size": final_pot_size,
                            "betting_round": final_betting_round,
                            "position": opp_pid,
                        }
                        context_features = analysis_agent._extract_context_features(final_context)
                        
                        # Add training example
                        if len(action_sequence) > 0:
                            agent.add_belief_training_example(actual_hand_type, action_sequence, context_features, opp_pid)

        # Train RL agents (online learning during evaluation)
        for pid, agent in agents.items():
            if hasattr(agent, "train_step") and hasattr(agent, "trainable") and agent.trainable:
                try:
                    from ma_poker.envs.gym_adapter import RLCardGymAdapter
                    gym_env = RLCardGymAdapter(env, player_id=pid)
                    # Train for a small number of steps per episode
                    agent.train_step(gym_env, total_timesteps=10)
                except Exception:
                    # Skip training if adapter not available or other error
                    pass

    avg_returns = (total_returns / float(cfg.episodes)).tolist()
    
    # Calculate advanced metrics
    T = cfg.episodes // 2  # Half of total episodes
    
    # Calculate metrics for each player
    player_metrics = {}
    for pid in range(env.num_players):
        # BB/100: Average BB per 100 hands
        # Formula: (total reward in BB / number of hands) * 100
        total_reward_bb = sum(episode_rewards_bb[pid])
        bb_per_100 = (total_reward_bb / cfg.episodes) * 100 if cfg.episodes > 0 else 0.0
        
        # MinBR: Minimum bankroll value along the match trajectory
        # MinBR = min_t C_t, where C_t is the bankroll after hand t
        # We use cumulative_chips_bb which tracks chips after each episode (hand)
        if len(cumulative_chips_bb[pid]) > 0:
            min_br_bb = min(cumulative_chips_bb[pid])  # Minimum chips in BB across all hands
        else:
            min_br_bb = cfg.initial_chips_bb  # Fallback to initial if no data
        
        # Debug: Print calculation details for verification
        print(f"\n[DEBUG] Player {pid} Metrics Calculation:")
        print(f"  Total episodes: {cfg.episodes}")
        print(f"  Total reward (chips): {sum(episode_rewards_bb[pid]) * big_blind:.2f}")
        print(f"  Total reward (BB): {total_reward_bb:.2f}")
        print(f"  Average reward per hand (BB): {total_reward_bb / cfg.episodes if cfg.episodes > 0 else 0:.2f}")
        print(f"  BB/100: {bb_per_100:.2f}")
        print(f"  Cumulative chips (BB) history: {cumulative_chips_bb[pid][:5]}..." if len(cumulative_chips_bb[pid]) > 5 else f"  Cumulative chips (BB) history: {cumulative_chips_bb[pid]}")
        print(f"  MinBR (BB): {min_br_bb:.2f}")
        print(f"  Initial chips (BB): {cfg.initial_chips_bb}")
        print(f"  Final chips (BB): {cumulative_chips_bb[pid][-1] if len(cumulative_chips_bb[pid]) > 0 else cfg.initial_chips_bb:.2f}")
        
        player_metrics[pid] = {
            "bb_per_100": float(bb_per_100),
            "min_br": float(min_br_bb),  # Minimum bankroll in BB
        }
    
    # Print metrics
    print("\n" + "="*80)
    print("EXPERIMENT METRICS SUMMARY")
    print("="*80)
    print(f"Total Episodes (Hands): {cfg.episodes}")
    print(f"Big Blind: {big_blind}")
    print(f"Initial Chips: {initial_chips} ({cfg.initial_chips_bb} BB)")
    print("\nPlayer Statistics:")
    print("-" * 80)
    print(f"{'Player':<20} {'BB/100':<12} {'MinBR (BB)':<15}")
    print("-" * 80)
    for pid in range(env.num_players):
        agent = agents.get(pid)
        player_name = _get_player_name(agent, pid) if agent else f"Player {pid}"
        metrics = player_metrics[pid]
        print(f"{player_name:<20} {metrics['bb_per_100']:>11.2f} {metrics['min_br']:>14.2f}")
    print("="*80)
    
    # Calculate and print belief accuracy for OursAgent instances
    belief_accuracy_metrics = {}
    for ours_pid, tracking in belief_accuracy_tracking.items():
        if tracking["total"] > 0:
            accuracy = tracking["correct"] / tracking["total"]
            belief_accuracy_metrics[ours_pid] = {
                "accuracy": float(accuracy),
                "correct": tracking["correct"],
                "total": tracking["total"],
                "per_opponent": {}
            }
            
            # Per-opponent accuracy
            for opp_pid, opp_tracking in tracking["per_opponent"].items():
                if opp_tracking["total"] > 0:
                    opp_accuracy = opp_tracking["correct"] / opp_tracking["total"]
                    belief_accuracy_metrics[ours_pid]["per_opponent"][opp_pid] = {
                        "accuracy": float(opp_accuracy),
                        "correct": opp_tracking["correct"],
                        "total": opp_tracking["total"]
                    }
            
            # Print belief accuracy
            print(f"\nOursAgent {ours_pid} Belief Accuracy (TOP 10):")
            print(f"  Overall: {accuracy:.2%} ({tracking['correct']}/{tracking['total']})")
            for opp_pid, opp_tracking in tracking["per_opponent"].items():
                if opp_tracking["total"] > 0:
                    opp_accuracy = opp_tracking["correct"] / opp_tracking["total"]
                    print(f"  vs Player {opp_pid}: {opp_accuracy:.2%} ({opp_tracking['correct']}/{opp_tracking['total']})")
    
    # Auto-save memory for OursAgent instances
    if cfg.log_dir:
        memories_dir = cfg.log_dir / "memories"
        memories_dir.mkdir(exist_ok=True)
        
        for pid, agent in agents.items():
            # Check if agent is OursAgent (by checking for save_memory method)
            if hasattr(agent, "save_memory"):
                memory_file = memories_dir / f"agent_{pid}_memory.json"
                try:
                    agent.save_memory(str(memory_file))
                except Exception as e:
                    # Log error but don't fail the evaluation
                    print(f"Warning: Failed to save memory for agent {pid}: {e}")
    
    # Auto-save models for RL agents (if trained during evaluation)
    if cfg.log_dir:
        models_dir = cfg.log_dir / "models"
        models_dir.mkdir(exist_ok=True)
        
        for pid, agent in agents.items():
            if hasattr(agent, "save_model") and hasattr(agent, "get_model"):
                model = agent.get_model()
                if model is not None:
                    model_file = models_dir / f"agent_{pid}_final.zip"
                    try:
                        agent.save_model(str(model_file))
                    except Exception as e:
                        print(f"Warning: Failed to save model for agent {pid}: {e}")
    
    # Prepare return dictionary with all metrics
    result = {
        "episodes": cfg.episodes,
        "avg_return_per_player": avg_returns,
        "total_rebuy_count": {str(pid): rebuy_count[pid] for pid in range(env.num_players)},
        "big_blind": big_blind,
        "initial_chips_bb": cfg.initial_chips_bb,
        "T": T,
        "player_metrics": {
            str(pid): metrics for pid, metrics in player_metrics.items()
        },
        "belief_accuracy_metrics": {
            str(pid): metrics for pid, metrics in belief_accuracy_metrics.items()
        },  # Belief accuracy for OursAgent instances
    }
    
    return result

