from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from ma_poker.agents.registry import SeatSpec, build_seated_agents
from ma_poker.envs.rlcard_holdem import RLCardHoldemConfig, RLCardHoldemEnv
from ma_poker.runners.eval_runner import EvalConfig, evaluate
from ma_poker.utils.io import RunPaths, load_yaml
from ma_poker.utils.seed import set_global_seed


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True, help="Path to YAML config")
    args = p.parse_args()

    cfg = load_yaml(args.config)

    # Get base output root from config
    base_root = cfg.get("output", {}).get("root", "runs/eval")
    
    # Add timestamp to output directory name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = Path(f"{base_root}_{timestamp}")
    
    paths = RunPaths(out_root)
    paths.mkdirs()

    seed = cfg.get("eval", {}).get("seed", 0)
    set_global_seed(seed)

    env_cfg = cfg.get("env", {})
    backend = env_cfg.get("backend", "rlcard")
    if backend != "rlcard":
        raise ValueError(f"Only backend=rlcard supported in scaffold, got {backend}")

    env = RLCardHoldemEnv(
        RLCardHoldemConfig(
            env_id=str(env_cfg.get("env_id", "limit-holdem")),
            num_players=int(env_cfg.get("num_players", 2)),
        )
    )

    seats_cfg = cfg.get("seats", None)
    if seats_cfg is None:
        # backward-compat: old single-agent config
        agent_cfg = cfg.get("agents", {})
        seats_cfg = [{"type": agent_cfg.get("type", "random"), "params": agent_cfg}]

    if not isinstance(seats_cfg, list):
        raise ValueError("Config field `seats` must be a list.")

    if len(seats_cfg) != env.num_players:
        raise ValueError(f"`seats` length ({len(seats_cfg)}) must match env.num_players ({env.num_players}).")

    seats = [SeatSpec(type=str(s.get("type")), params=dict(s.get("params", {}))) for s in seats_cfg]
    agents = build_seated_agents(seats)

    eval_cfg = cfg.get("eval", {})
    res = evaluate(
        env=env,
        agents=agents,
        cfg=EvalConfig(
            episodes=int(eval_cfg.get("episodes", 200)),
            seed=None if eval_cfg.get("seed", None) is None else int(eval_cfg.get("seed")),
            show_progress=bool(eval_cfg.get("show_progress", True)),
            save_detailed_logs=bool(eval_cfg.get("save_detailed_logs", True)),
            log_dir=paths.results / "episodes",
            initial_chips_bb=int(eval_cfg.get("initial_chips_bb", 100)),
            rebuy_threshold_bb=int(eval_cfg.get("rebuy_threshold_bb", 0)),
            rebuy_amount_bb=int(eval_cfg.get("rebuy_amount_bb", 100)),
            track_chips_across_episodes=bool(eval_cfg.get("track_chips_across_episodes", True)),
        ),
    )

    # Save summary metrics
    (paths.results / "metrics.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    
    # Save experiment config for reference
    config_summary = {
        "env": env_cfg,
        "seats": [{"type": s.type, "params": s.params} for s in seats],
        "eval": eval_cfg,
    }
    (paths.results / "config.json").write_text(json.dumps(config_summary, indent=2), encoding="utf-8")
    
    print(json.dumps(res, indent=2))
    print(f"\nDetailed episode logs saved to: {paths.results / 'episodes'}")


if __name__ == "__main__":
    main()

