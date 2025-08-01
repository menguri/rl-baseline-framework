# rl_framework/eval/simulate.py

import os
import torch
import argparse
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from pathlib import Path
import numpy as np

from algorithms.ppo import PPO
from algorithms.trpo import TRPO
from algorithms.ddpg import DDPG


# 알고리즘별 로딩 함수 정의
def load_model(algorithm, model_path, state_dim, action_dim):
    """알고리즘 이름에 따라 모델 인스턴스 생성 및 로드"""
    if algorithm == "ddpg":
        model = DDPG(state_dim, action_dim)
    elif algorithm == "ppo":
        model = PPO(state_dim, action_dim)
    elif algorithm == "trpo":
        model = TRPO(state_dim, action_dim)
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    model.load(model_path)
    return model

# 시뮬레이션 실행
def simulate(env_name, model_path, algorithm, episodes=5, render=False, record_dir=None):
    # 에이전트를 시뮬레이션하고 영상 저장
    print(f"\n🎯 Environment: {env_name}")
    print(f"📁 Model path: {model_path}")
    print(f"🎥 Render: {render} | Record dir: {record_dir}")
    print(f"🤖 Algorithm: {algorithm.upper()} | Episodes: {episodes}\n")

    # env 설정
    env = gym.make(env_name, render_mode="rgb_array")

    # 영상 저장 래퍼 적용
    if record_dir:
        record_dir = Path(record_dir)
        record_dir.mkdir(parents=True, exist_ok=True)
        env = RecordVideo(
            env, 
            video_folder=str(record_dir), 
            episode_trigger=lambda i: True,
            name_prefix=f"{algorithm}_{Path(model_path).stem}")
        print(f"📹 Recording episodes to: {record_dir}")

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # 알고리즘 객체 생성 및 모델 불러오기
    model = load_model(algorithm, model_path, state_dim, action_dim)

    print(f"🔍 Simulating {algorithm.upper()} on {env_name} for {episodes} episodes...")
    for ep in range(1, episodes + 1):
        state, _ = env.reset()
        done = False
        total_reward = 0
        timestep = 0

        while not done:
            action = model.select_action(state)
            state, reward, done, truncated, _ = env.step(action)
            total_reward += reward
            timestep += 1
            if render:
                env.render()
            if timestep > 1000:
                done = True

        print(f"[Episode {ep}] Total Reward: {total_reward:.2f}")

    env.close()

# CLI 인터페이스
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, required=True, help="Gym environment name (e.g., LunarLanderContinuous-v3)")
    parser.add_argument("--model-path", type=str, required=True, help="Path to trained model file (.pth)")
    parser.add_argument("--algorithm", type=str, choices=["ddpg", "ppo", "trpo"], required=True)
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes to simulate")
    parser.add_argument("--render", action="store_true", help="Enable live rendering (WSL 환경에선 off)")
    parser.add_argument("--record-dir", type=str, default=None, help="Directory to save recorded videos (mp4)")
    args = parser.parse_args()

    simulate(
        env_name=args.env,
        model_path=args.model_path,
        algorithm=args.algorithm,
        episodes=args.episodes,
        render=args.render,
        record_dir=args.record_dir
    )

