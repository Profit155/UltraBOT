
from stable_baselines3 import PPO
from env_ultra import UltraKillWrapper
import torch,time,argparse
p=argparse.ArgumentParser();p.add_argument('--weights',default='ultra_rl.zip');a=p.parse_args()
env=UltraKillWrapper()
model=PPO.load(a.weights,env=env,device='cuda' if torch.cuda.is_available() else 'cpu')
obs,_=env.reset()
while True:
    action,_=model.predict(obs,deterministic=True)
    obs,_,done,_,_=env.step(action)
    if done: obs,_=env.reset()
    time.sleep(0.016)
