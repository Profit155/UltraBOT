from stable_baselines3 import PPO
from env_ultra import UltraKillWrapper
import torch, argparse

p = argparse.ArgumentParser()
p.add_argument('--steps', type=int, default=1000000)
a = p.parse_args()

env = UltraKillWrapper(mouse=True)

dev = 'cuda' if torch.cuda.is_available() else 'cpu'
model = PPO('CnnPolicy', env, device=dev, verbose=1, tensorboard_log='runs')

try:
    import os
    if os.path.exists('bc.pt'):
        model.policy.load_state_dict(torch.load('bc.pt', map_location=dev), strict=False)
        print('[RL] warm-started')
except:
    pass

model.learn(total_timesteps=a.steps)
model.save('ultra_rl')
