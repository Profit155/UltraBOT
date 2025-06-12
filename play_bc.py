import torch
import numpy as np
from env_ultra import UltraKillEnv

A_KEYS = [
    'w','a','s','d','space','shift','ctrl',
    'left','right','1','2','3','4','5','r','f','g'
]

class BCNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = torch.nn.Sequential(
            torch.nn.Conv2d(3,16,8,4), torch.nn.ReLU(),
            torch.nn.Conv2d(16,32,4,2), torch.nn.ReLU(),
            torch.nn.Conv2d(32,64,3,1), torch.nn.ReLU(),
            torch.nn.Flatten()
        )
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(64*7*7,256), torch.nn.ReLU(),
            torch.nn.Linear(256,len(A_KEYS)), torch.nn.Sigmoid()
        )
    def forward(self,x): return self.fc(self.cnn(x))

# ─── main ─────────────────────────────────────

def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = BCNet().to(device)
    net.load_state_dict(torch.load("bc.pt", map_location=device))
    net.eval()

    env = UltraKillEnv()
    obs, _ = env.reset()
    while True:
        with torch.no_grad():
            inp = (
                torch.tensor(obs / 255.0, dtype=torch.float32)
                .unsqueeze(0)
                .to(device)
            )
            out = net(inp)[0].cpu().numpy()
        action = (out > 0.5).astype(np.float32)  # 0/1
        obs, _, _, _, _ = env.step(action)


if __name__ == "__main__":
    main()
