from __future__ import annotations

import argparse
import json
from pathlib import Path

from ma_poker.agents.registry import SeatSpec, build_seated_agents
from ma_poker.envs.rlcard_holdem import RLCardHoldemConfig, RLCardHoldemEnv
from ma_poker.runners.eval_runner import EvalConfig, evaluate
from ma_poker.runners.train_runner import TrainConfig, train
from ma_poker.utils.io import RunPaths, load_yaml
from ma_poker.utils.seed import set_global_seed


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True, help="Path to YAML config")
    args = p.parse_args()

    cfg = load_yaml(args.config)

    out_root = Path(cfg.get("output", {}).get("root", "runs/train"))
    paths = RunPaths(out_root)
    paths.mkdirs()

    seed = cfg.get("train", {}).get("seed", 0)
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
        agent_cfg = cfg.get("agents", {})
        seats_cfg = [{"type": agent_cfg.get("type", "random"), "params": agent_cfg}]

    if not isinstance(seats_cfg, list):
        raise ValueError("Config field `seats` must be a list.")

    if len(seats_cfg) != env.num_players:
        raise ValueError(f"`seats` length ({len(seats_cfg)}) must match env.num_players ({env.num_players}).")

    seats = [SeatSpec(type=str(s.get("type")), params=dict(s.get("params", {}))) for s in seats_cfg]
    agents = build_seated_agents(seats)

    train_cfg = cfg.get("train", {})
    eval_cfg = cfg.get("eval", {})

    # Define evaluation function
    def eval_fn(env, agents, episodes):
        return evaluate(
            env=env,
            agents=agents,
            cfg=EvalConfig(
                episodes=int(episodes),
                seed=None,
                show_progress=False,
            ),
        )

    res = train(
        env=env,
        agents=agents,
        cfg=TrainConfig(
            episodes=int(train_cfg.get("episodes", 10000)),
            eval_interval=int(train_cfg.get("eval_interval", 1000)),
            eval_episodes=int(eval_cfg.get("episodes", 100)),
            seed=None if train_cfg.get("seed", None) is None else int(train_cfg.get("seed")),
            show_progress=bool(train_cfg.get("show_progress", True)),
            save_interval=train_cfg.get("save_interval"),
            output_dir=paths.results,
        ),
        eval_fn=eval_fn,
    )

    (paths.results / "training_metrics.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print("Training completed!")
    print(f"Final evaluation: {json.dumps(res['final_eval'], indent=2)}")


if __name__ == "__main__":
    main()
