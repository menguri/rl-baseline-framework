# rl_baseline_framework/eval/simulate.py
# ────────────────────────────────────────────────────────────────────
"""
Simulate a trained RL policy on a Gymnasium environment
and optionally record the rollout as an MP4.

Example
-------
python -m rl_baseline_framework.eval.simulate \
       --env HalfCheetah-v4 \
       --model-path results/halfcheetah_trpo/seed_0/best_model.pth \
       --algorithm trpo \
       --episodes 2 \
       --record-dir videos/halfcheetah_trpo/seed_0
"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import Literal, Sequence

import numpy as np
import torch
import gymnasium as gym
from gymnasium.wrappers import RecordVideo

# ────────────────────────────────────────────────────────────────────
# Import your algorithms  (package layout: rl_baseline_framework.algorithms)
# This assumes algorithms/__init__.py exposes PPO, TRPO, DDPG symbols!
# Otherwise `from algorithms.ppo import PPO` 와 같이 바꿔 주세요.
# ────────────────────────────────────────────────────────────────────
try:
    from algorithms import PPO, TRPO, DDPG
except ModuleNotFoundError:
    # fallback to plain "algorithms" namespace typical in small repos
    from algorithms.ppo import PPO        # type: ignore
    from algorithms.trpo import TRPO       # type: ignore
    from algorithms.ddpg import DDPG       # type: ignore

ALGOS = {
    "ppo":  PPO,
    "trpo": TRPO,
    "ddpg": DDPG,
}

# ===================================================================
# Helpers
# ===================================================================

# Hidden layer 맞춰줘야 함.
def _load_agent(
    algo_name: Literal["ppo", "trpo", "ddpg"],
    model_path: Path,
    state_dim: int,
    action_dim: int,
    device: torch.device,
):
    """Instantiate the right algorithm class and load weights."""
    algo_cls = ALGOS.get(algo_name)
    if algo_cls is None:
        raise ValueError(f"Unsupported algorithm: {algo_name}")

    agent = algo_cls(state_dim, action_dim, hidden_dims=[256, 256], device=device)
    agent.load(model_path)  # all baseline classes expose .load(path)
    return agent


def run_episode(
    env: gym.Env,
    agent,
    render: bool = False,
    max_steps: int | None = None,
) -> float:
    """Run one evaluation episode and return total reward."""
    state, _ = env.reset()
    done = truncated = False
    ep_return = 0.0
    steps = 0

    while not (done or truncated):
        # agent expects np.ndarray; some envs give lists
        action = agent.select_action(np.asarray(state, dtype=np.float32))
        state, reward, done, truncated, _ = env.step(action)
        ep_return += reward
        steps += 1
        if render:
            env.render()
        if max_steps and steps >= max_steps:
            break
    return ep_return


# ===================================================================
# Main entry
# ===================================================================


def simulate(
    env_name: str,
    model_path: str | Path,
    algorithm: Literal["ppo", "trpo", "ddpg"],
    episodes: int = 5,
    render: bool = False,
    record_dir: str | Path | None = None,
    device: str = "cpu",
    max_steps: int | None = None,
):
    """Run `episodes` roll-outs of a saved policy and (optionally) record video."""
    model_path = Path(model_path).expanduser().resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")

    device_t = torch.device(device)
    env = gym.make(env_name, render_mode="rgb_array")

    # RecordVideo wrapper (if needed)
    if record_dir:
        record_dir = Path(record_dir).expanduser().resolve()
        record_dir.mkdir(parents=True, exist_ok=True)
        env = RecordVideo(
            env,
            video_folder=str(record_dir),
            episode_trigger=lambda _: True,  # record every episode
            name_prefix=f"{algorithm}_{model_path.stem}",
        )
        print(f"📹  Recording videos to: {record_dir}")

    state_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))

    agent = _load_agent(
        algo_name=algorithm,
        model_path=model_path,
        state_dim=state_dim,
        action_dim=action_dim,
        device=device_t,
    )

    print(
        f"\n🎯  Env: {env_name}\n"
        f"🤖  Algo: {algorithm.upper()} | Episodes: {episodes}\n"
        f"📁  Checkpoint: {model_path}\n"
    )

    returns: list[float] = []
    for ep in range(1, episodes + 1):
        ep_ret = run_episode(env, agent, render, max_steps=max_steps)
        returns.append(ep_ret)
        print(f"[Episode {ep}] return = {ep_ret:.2f}")

    env.close()
    print(f"\nAvg Return over {episodes} ep.: {np.mean(returns):.2f}")


# ===================================================================
# CLI
# ===================================================================

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rollout a trained policy.")
    parser.add_argument("--env", required=True, help="Gymnasium env id")
    parser.add_argument("--model-path", required=True, help="Path to .pth file")
    parser.add_argument(
        "--algorithm", required=True, choices=list(ALGOS.keys())
    )
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--render", action="store_true", help="Render live window")
    parser.add_argument("--record-dir", help="Folder to save mp4")
    parser.add_argument("--device", default="cpu", help="cpu | cuda:0 ...")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Early-terminate after N steps (optional)",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    simulate(
        env_name=args.env,
        model_path=args.model_path,
        algorithm=args.algorithm,
        episodes=args.episodes,
        render=args.render,
        record_dir=args.record_dir,
        device=args.device,
        max_steps=args.max_steps,
    )


