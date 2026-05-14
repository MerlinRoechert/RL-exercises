import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from rl_exercises.week_4.dqn import DQNAgent

NUM_FRAMES = 100000

def run_config(cfg):
    env = gym.make("CartPole-v1")
    agent = DQNAgent(
        env=env,
        buffer_capacity=cfg["buffer_capacity"],
        batch_size=cfg["batch_size"],
        epsilon_decay=cfg["epsilon_decay"],
        hidden_dim=cfg["hidden_dim"],
        num_layers=cfg["num_layers"],
        seed=0,
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
plt.savefig("l1_buffer.png", dpi=150)
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
plt.savefig("l1_batch.png", dpi=150)
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
plt.savefig("l1_width.png", dpi=150)
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
plt.savefig("l1_depth.png", dpi=150)
print("Saved l1_depth.png")

plt.show()