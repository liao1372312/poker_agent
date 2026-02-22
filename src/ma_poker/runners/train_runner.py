from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional

import numpy as np
from tqdm import tqdm

from ma_poker.agents.base import Agent
from ma_poker.envs.base import MultiAgentEnv


@dataclass
class TrainConfig:
    """Training configuration for multi-agent poker."""

    episodes: int = 10000
    eval_interval: int = 1000
    eval_episodes: int = 100
    seed: Optional[int] = 0
    show_progress: bool = True
    save_interval: Optional[int] = None  # Save checkpoint every N episodes
    output_dir: Optional[Path] = None  # Directory to save training outputs


def train(
    env: MultiAgentEnv,
    agents: Dict[int, Agent],
    cfg: TrainConfig,
    eval_fn: Optional[Callable[[MultiAgentEnv, Dict[int, Agent], int], Dict[str, object]]] = None,
) -> Dict[str, object]:
    """Main training loop for multi-agent poker.

    Args:
        env: Multi-agent poker environment
        agents: Dict mapping player_id -> Agent
        cfg: Training configuration
        eval_fn: Optional evaluation function (env, agents, episodes) -> metrics dict

    Returns:
        Training metrics and final evaluation results
    """
    for a in agents.values():
        a.reset()

    training_metrics = {
        "episodes": [],
        "returns": {pid: [] for pid in range(env.num_players)},
    }

    it = range(cfg.episodes)
    if cfg.show_progress:
        it = tqdm(it, desc="train", ncols=100)

    for ep in it:
        ep_seed = None if cfg.seed is None else int(cfg.seed + ep)
        obs_dict = env.reset(seed=ep_seed)
        done = False
        returns = np.zeros((env.num_players,), dtype=np.float64)

        # One episode
        while not done:
            cur = env.current_player()
            obs = obs_dict.get(cur, None)
            legal = env.legal_actions(cur)
            act = agents[cur].act(obs, legal)
            step_out = env.step({cur: act.action})
            for pid, r in step_out.rewards.items():
                returns[int(pid)] += float(r)
            obs_dict = step_out.obs
            done = step_out.terminated

        # Record episode returns
        training_metrics["episodes"].append(ep)
        for pid in range(env.num_players):
            training_metrics["returns"][pid].append(float(returns[pid]))

        # Update memory for agents that support it (e.g., OursAgent)
        for pid, agent in agents.items():
            if hasattr(agent, "update_episode_outcome"):
                agent.update_episode_outcome(float(returns[pid]))

        # Train RL agents (online learning)
        for pid, agent in agents.items():
            if hasattr(agent, "train_step") and hasattr(agent, "trainable") and agent.trainable:
                try:
                    from ma_poker.envs.gym_adapter import RLCardGymAdapter
                    gym_env = RLCardGymAdapter(env, player_id=pid)
                    # Train for a small number of steps per episode
                    agent.train_step(gym_env, total_timesteps=10)
                except Exception as e:
                    # Skip training if adapter not available or other error
                    if cfg.show_progress and ep == 0:
                        it.write(f"Warning: RL training skipped for agent {pid}: {e}")

        # Periodic evaluation
        if eval_fn is not None and (ep + 1) % cfg.eval_interval == 0:
            eval_results = eval_fn(env, agents, cfg.eval_episodes)
            if cfg.show_progress:
                avg_returns = eval_results.get("avg_return_per_player", [])
                it.write(f"Eval @ {ep+1}: {avg_returns}")

        # Checkpoint saving for RL agents
        if cfg.save_interval is not None and (ep + 1) % cfg.save_interval == 0:
            if cfg.output_dir:
                checkpoints_dir = cfg.output_dir / "checkpoints"
                checkpoints_dir.mkdir(exist_ok=True)
                
                for pid, agent in agents.items():
                    if hasattr(agent, "save_model") and hasattr(agent, "get_model"):
                        model = agent.get_model()
                        if model is not None:
                            checkpoint_file = checkpoints_dir / f"agent_{pid}_episode_{ep+1}.zip"
                            try:
                                agent.save_model(str(checkpoint_file))
                            except Exception as e:
                                if cfg.show_progress:
                                    it.write(f"Warning: Failed to save checkpoint for agent {pid}: {e}")

    # Final evaluation
    final_eval = {}
    if eval_fn is not None:
        final_eval = eval_fn(env, agents, cfg.eval_episodes)

    # Auto-save memory for OursAgent instances
    if cfg.output_dir:
        memories_dir = cfg.output_dir / "memories"
        memories_dir.mkdir(exist_ok=True)
        
        for pid, agent in agents.items():
            # Check if agent is OursAgent (by checking for save_memory method)
            if hasattr(agent, "save_memory"):
                memory_file = memories_dir / f"agent_{pid}_memory.json"
                try:
                    agent.save_memory(str(memory_file))
                except Exception as e:
                    # Log error but don't fail the training
                    print(f"Warning: Failed to save memory for agent {pid}: {e}")
    
    # Auto-save final models for RL agents
    if cfg.output_dir:
        models_dir = cfg.output_dir / "models"
        models_dir.mkdir(exist_ok=True)
        
        for pid, agent in agents.items():
            if hasattr(agent, "save_model") and hasattr(agent, "get_model"):
                model = agent.get_model()
                if model is not None:
                    model_file = models_dir / f"agent_{pid}_final.zip"
                    try:
                        agent.save_model(str(model_file))
                    except Exception as e:
                        print(f"Warning: Failed to save final model for agent {pid}: {e}")

    return {
        "training": training_metrics,
        "final_eval": final_eval,
    }
