
from stable_baselines3 import PPO
from env_ultra import UltraKillWrapper
import argparse
import time
import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="ultra_rl.zip")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    env = UltraKillWrapper()
    model = PPO.load(args.weights, env=env, device=device)

    obs, _ = env.reset()
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, _, _ = env.step(action)
        if terminated:
            obs, _ = env.reset()
        time.sleep(0.016)


if __name__ == "__main__":
    main()
