import argparse
import os

import torch
from stable_baselines3 import PPO

from env_ultra import UltraKillWrapper


def main(steps: int) -> None:
    """Train the RL agent for the specified number of steps."""
    env = UltraKillWrapper(mouse=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = PPO("CnnPolicy", env, device=device, verbose=1, tensorboard_log="runs")

    if os.path.exists("bc.pt"):
        model.policy.load_state_dict(torch.load("bc.pt", map_location=device), strict=False)
        print("[RL] warm-started")

    model.learn(total_timesteps=steps)
    model.save("ultra_rl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1_000_000)
    main(parser.parse_args().steps)
