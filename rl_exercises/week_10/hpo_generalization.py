from __future__ import annotations

from dataclasses import dataclass
from statistics import mean, pstdev
from typing import Dict, Iterable, List

import gymnasium as gym
import numpy as np

from rl_exercises.week_10.dqn import DQNAgent, set_seed


ENV_NAME = "CartPole-v1"
NUM_FRAMES = 12000
EVAL_EPISODES = 10
TRAIN_SEEDS = (0, 1, 2)
TEST_SEEDS = (10, 11, 12, 13, 14)


@dataclass(frozen=True)
class TrialResult:
    label: str
    config: Dict[str, float | int]
    train_scores: List[float]
    test_scores: List[float] | None = None


DEFAULT_CONFIG: Dict[str, float | int] = {
    "buffer_capacity": 10000,
    "batch_size": 32,
    "lr": 1e-3,
    "gamma": 0.99,
    "epsilon_decay": 500,
    "target_update_freq": 1000,
    "hidden_dim": 64,
    "num_layers": 2,
}

SEARCH_SPACE: List[Dict[str, float | int]] = [
    {
        "buffer_capacity": 10000,
        "batch_size": 32,
        "lr": 1e-3,
        "gamma": 0.99,
        "epsilon_decay": 500,
        "target_update_freq": 1000,
        "hidden_dim": 64,
        "num_layers": 2,
    },
    {
        "buffer_capacity": 10000,
        "batch_size": 64,
        "lr": 5e-4,
        "gamma": 0.99,
        "epsilon_decay": 1000,
        "target_update_freq": 1000,
        "hidden_dim": 64,
        "num_layers": 2,
    },
    {
        "buffer_capacity": 5000,
        "batch_size": 32,
        "lr": 1e-3,
        "gamma": 0.98,
        "epsilon_decay": 1000,
        "target_update_freq": 500,
        "hidden_dim": 64,
        "num_layers": 2,
    },
    {
        "buffer_capacity": 10000,
        "batch_size": 32,
        "lr": 2e-3,
        "gamma": 0.99,
        "epsilon_decay": 2000,
        "target_update_freq": 1000,
        "hidden_dim": 128,
        "num_layers": 2,
    },
    {
        "buffer_capacity": 20000,
        "batch_size": 64,
        "lr": 1e-3,
        "gamma": 0.995,
        "epsilon_decay": 2000,
        "target_update_freq": 1000,
        "hidden_dim": 128,
        "num_layers": 2,
    },
    {
        "buffer_capacity": 10000,
        "batch_size": 32,
        "lr": 5e-4,
        "gamma": 0.995,
        "epsilon_decay": 3000,
        "target_update_freq": 2000,
        "hidden_dim": 64,
        "num_layers": 3,
    },
]


def evaluate_agent(agent: DQNAgent, seed: int, episodes: int = EVAL_EPISODES) -> float:
    eval_env = gym.make(ENV_NAME)
    set_seed(eval_env, seed)
    rewards = []

    for episode in range(episodes):
        state, _ = eval_env.reset(seed=seed + episode)
        done = False
        truncated = False
        episode_reward = 0.0

        while not (done or truncated):
            action = agent.predict_action(state, evaluate=True)
            state, reward, done, truncated, _ = eval_env.step(action)
            episode_reward += reward

        rewards.append(episode_reward)

    eval_env.close()
    return float(mean(rewards))


def train_and_score(config: Dict[str, float | int], seed: int) -> float:
    env = gym.make(ENV_NAME)
    agent = DQNAgent(env=env, seed=seed, **config)
    agent.train(NUM_FRAMES)
    score = evaluate_agent(agent, seed=seed + 1000)
    env.close()
    return score


def run_config(label: str, config: Dict[str, float | int], seeds: Iterable[int]) -> TrialResult:
    scores = []
    for seed in seeds:
        score = train_and_score(config, seed)
        scores.append(score)
        print(f"{label}, seed {seed}: {score:.1f}")
    return TrialResult(label=label, config=config, train_scores=scores)


def format_scores(scores: List[float]) -> str:
    return f"{mean(scores):.1f} +/- {pstdev(scores):.1f} ({scores})"


def print_results(best: TrialResult, default_test_scores: List[float]) -> None:
    test_scores = best.test_scores or []
    print(f"algorithm: DQN")
    print(f"environment: {ENV_NAME}")
    print(f"optimizer: random search")
    print(f"train_seeds: {list(TRAIN_SEEDS)}")
    print(f"test_seeds: {list(TEST_SEEDS)}")
    print(f"num_frames: {NUM_FRAMES}")
    print(f"eval_episodes: {EVAL_EPISODES}")
    print(f"best_trial: {best.label}")
    print(f"best_config: {best.config}")
    print(f"best_train_scores: {format_scores(best.train_scores)}")
    print(f"best_test_scores: {format_scores(test_scores)}")
    print(f"default_test_scores: {format_scores(default_test_scores)}")


def main() -> None:
    trial_results = [
        run_config(f"trial_{idx}", config, TRAIN_SEEDS)
        for idx, config in enumerate(SEARCH_SPACE)
    ]
    best = max(trial_results, key=lambda result: mean(result.train_scores))

    print(f"\nBest config: {best.label} with train score {mean(best.train_scores):.1f}")
    best_test_scores = []
    for seed in TEST_SEEDS:
        score = train_and_score(best.config, seed)
        best_test_scores.append(score)
        print(f"best config test, seed {seed}: {score:.1f}")

    default_test_scores = []
    for seed in TEST_SEEDS:
        score = train_and_score(DEFAULT_CONFIG, seed)
        default_test_scores.append(score)
        print(f"default config test, seed {seed}: {score:.1f}")

    best = TrialResult(
        label=best.label,
        config=best.config,
        train_scores=best.train_scores,
        test_scores=best_test_scores,
    )
    print_results(best, default_test_scores)


if __name__ == "__main__":
    main()
