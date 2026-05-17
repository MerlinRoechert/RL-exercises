import os

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from rl_exercises.week_4.dqn import DQNAgent
from rliable import library as rly
from rliable import metrics

NUM_FRAMES = 100000
PLOT_DIR = "rl_exercises/week_4/plots"
os.makedirs(PLOT_DIR, exist_ok=True)

def run_config(cfg):
    env = gym.make("CartPole-v1")
    agent = DQNAgent(
        env=env,
        buffer_capacity=cfg["buffer_capacity"],
        batch_size=cfg["batch_size"],
        epsilon_decay=cfg["epsilon_decay"],
        hidden_dim=cfg["hidden_dim"],
        num_layers=cfg["num_layers"],
        seed=cfg.get("seed", 0),
    )
    agent.train(num_frames=NUM_FRAMES)
    return agent.frame_log, agent.reward_log


# Plot 1: Buffer size
buffer_configs = [
    {"label": "baseline (10k)",  "buffer_capacity": 10000, "batch_size": 32, "epsilon_decay": 5000, "hidden_dim": 64, "num_layers": 2},
    {"label": "small (1k)",      "buffer_capacity": 1000,  "batch_size": 32, "epsilon_decay": 5000, "hidden_dim": 64, "num_layers": 2},
    {"label": "large (50k)",     "buffer_capacity": 50000, "batch_size": 32, "epsilon_decay": 5000, "hidden_dim": 64, "num_layers": 2},
]

plt.figure(figsize=(10, 6))
for cfg in buffer_configs:
    frames, rewards = run_config(cfg)
    plt.plot(frames, rewards, label=cfg["label"])
plt.xlabel("Frames")
plt.ylabel("Avg Reward (10 episodes)")
plt.title("DQN — Varying Buffer Size")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "l1_buffer.png"), dpi=150)
print("Saved l1_buffer.png")


# Plot 2: Batch size
batch_configs = [
    {"label": "baseline (32)",  "buffer_capacity": 10000, "batch_size": 32,  "epsilon_decay": 5000, "hidden_dim": 64, "num_layers": 2},
    {"label": "small (16)",     "buffer_capacity": 10000, "batch_size": 16,  "epsilon_decay": 5000, "hidden_dim": 64, "num_layers": 2},
    {"label": "large (128)",    "buffer_capacity": 10000, "batch_size": 128, "epsilon_decay": 5000, "hidden_dim": 64, "num_layers": 2},
]

plt.figure(figsize=(10, 6))
for cfg in batch_configs:
    frames, rewards = run_config(cfg)
    plt.plot(frames, rewards, label=cfg["label"])
plt.xlabel("Frames")
plt.ylabel("Avg Reward (10 episodes)")
plt.title("DQN — Varying Batch Size")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "l1_batch.png"), dpi=150)
print("Saved l1_batch.png")


# Plot 3: Network width
width_configs = [
    {"label": "narrow (32)",    "buffer_capacity": 10000, "batch_size": 32, "epsilon_decay": 5000, "hidden_dim": 32,  "num_layers": 2},
    {"label": "baseline (64)",  "buffer_capacity": 10000, "batch_size": 32, "epsilon_decay": 5000, "hidden_dim": 64,  "num_layers": 2},
    {"label": "wide (128)",     "buffer_capacity": 10000, "batch_size": 32, "epsilon_decay": 5000, "hidden_dim": 128, "num_layers": 2},
]

plt.figure(figsize=(10, 6))
for cfg in width_configs:
    frames, rewards = run_config(cfg)
    plt.plot(frames, rewards, label=cfg["label"])
plt.xlabel("Frames")
plt.ylabel("Avg Reward (10 episodes)")
plt.title("DQN — Varying Network Width")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "l1_width.png"), dpi=150)
print("Saved l1_width.png")


# Plot 4
depth_configs = [
    {"label": "shallow (1 layer)", "buffer_capacity": 10000, "batch_size": 32, "epsilon_decay": 5000, "hidden_dim": 64, "num_layers": 1},
    {"label": "baseline (2 layers)","buffer_capacity": 10000, "batch_size": 32, "epsilon_decay": 5000, "hidden_dim": 64, "num_layers": 2},
    {"label": "deep (3 layers)",   "buffer_capacity": 10000, "batch_size": 32, "epsilon_decay": 5000, "hidden_dim": 64, "num_layers": 3},
]

plt.figure(figsize=(10, 6))
for cfg in depth_configs:
    frames, rewards = run_config(cfg)
    plt.plot(frames, rewards, label=cfg["label"])
plt.xlabel("Frames")
plt.ylabel("Avg Reward (10 episodes)")
plt.title("DQN — Varying Network Depth")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "l1_depth.png"), dpi=150)
print("Saved l1_depth.png")

# Level 2: Seeding question with RLiable

l2_configs = [
    {"label": "seed 0", "seed": 0, "buffer_capacity": 10000, "batch_size": 32, "epsilon_decay": 5000, "hidden_dim": 64, "num_layers": 2},
    {"label": "seed 1", "seed": 1, "buffer_capacity": 10000, "batch_size": 32, "epsilon_decay": 5000, "hidden_dim": 64, "num_layers": 2},
    {"label": "seed 2", "seed": 2, "buffer_capacity": 10000, "batch_size": 32, "epsilon_decay": 5000, "hidden_dim": 64, "num_layers": 2},
    {"label": "seed 3", "seed": 3, "buffer_capacity": 10000, "batch_size": 32, "epsilon_decay": 5000, "hidden_dim": 64, "num_layers": 2},
    {"label": "seed 4", "seed": 4, "buffer_capacity": 10000, "batch_size": 32, "epsilon_decay": 5000, "hidden_dim": 64, "num_layers": 2},
]

all_rewards = []
all_frames = []

plt.figure(figsize=(10, 6))

for cfg in l2_configs:
    frames, rewards = run_config(cfg)

    all_rewards.append(rewards)
    all_frames.append(frames)

    plt.plot(frames, rewards, alpha=0.6, label=cfg["label"])

plt.xlabel("Frames")
plt.ylabel("Avg Reward (10 episodes)")
plt.title("DQN CartPolev1 Training curves across seeds")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "l2_training_curves_seeds.png"), dpi=150)
print("Saved l2_training_curves_seeds.png")

min_len = min(len(r) for r in all_rewards)

all_rewards = [r[:min_len] for r in all_rewards]
all_frames = [f[:min_len] for f in all_frames]

frames = all_frames[0]

scores = np.stack(all_rewards, axis=0)
scores = scores[:, None, :]

score_dict = {"DQN": scores}


def curve(metric_fn):
    return lambda x: np.array([
        metric_fn(x[..., i])
        for i in range(x.shape[-1])
    ])


def optimality_gap_curve(x):
    return np.array([
        np.mean(np.maximum(500.0 - x[..., i], 0.0))
        for i in range(x.shape[-1])
    ])


plot_metrics = {
    "iqm": curve(metrics.aggregate_iqm),
    "median": curve(metrics.aggregate_median),
    "mean": curve(metrics.aggregate_mean),
    "optimality_gap": optimality_gap_curve,
}

for name, metric_fn in plot_metrics.items():
    estimates, intervals = rly.get_interval_estimates(
        score_dict,
        metric_fn,
        reps=200,
    )

    y = estimates["DQN"]
    lower = intervals["DQN"][0]
    upper = intervals["DQN"][1]

    plt.figure(figsize=(10, 6))
    plt.plot(frames, y, label=name)
    plt.fill_between(frames, lower, upper, alpha=0.2)

    plt.xlabel("Frames")
    plt.ylabel(name)
    plt.title(f"DQN CartPolev1 {name} across seeds")

    plt.legend()
    plt.tight_layout()

    plt.savefig(os.path.join(PLOT_DIR, f"l2_{name}.png"), dpi=150)
    print(f"Saved l2_{name}.png")

    plt.show()