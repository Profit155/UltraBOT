# train_bc.py
# ============  имитационное обучение для ULTRAKILL  ==================
# usage: python train_bc.py --data dataset --epochs 10

import glob, argparse, json, os, numpy as np, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

A_KEYS = [
    'w','a','s','d',
    'space','shift','ctrl',
    'left','right',
    '1','2','3','4','5',
    'r','f','g'
]                                  # 17 кнопок в том же порядке, что в env_ultra

# ───────────────── Dataset ────────────────────────────────────────────
class UltraDataset(Dataset):
    def __init__(self, folder):
        self.files = sorted(glob.glob(os.path.join(folder, "*.npz")))
        if not self.files:
            raise RuntimeError(f"Пустая папка {folder}")

    def __len__(self): return len(self.files)

    def __getitem__(self, idx):
        d = np.load(self.files[idx], allow_pickle=True)
        img  = d['img'].astype(np.float32) / 255.0        # (84,84,3)
        img  = np.transpose(img, (2,0,1))                 # (3,84,84)

        keys = d['keys'].item() if 'keys' in d else {}
        action = np.zeros(len(A_KEYS), dtype=np.float32)
        for i,k in enumerate(A_KEYS):
            action[i] = keys.get(f"Key.{k}", 0)            # клавиши
            if k in ('left','right'):
                action[i] = keys.get(f"Button.{k}", action[i])

        return torch.tensor(img), torch.tensor(action)

# ───────────────── Модель ─────────────────────────────────────────────
class BCNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, 8, 4), nn.ReLU(),            # 84→20
            nn.Conv2d(16,32,4,2), nn.ReLU(),              # 20→9
            nn.Conv2d(32,64,3,1), nn.ReLU(),              # 9→7
            nn.Flatten()
        )
        self.fc = nn.Sequential(
            nn.Linear(64*7*7, 256), nn.ReLU(),
            nn.Linear(256, len(A_KEYS)), nn.Sigmoid()     # 17 выходов 0..1
        )

    def forward(self, x):
        return self.fc(self.cnn(x))

# ───────────────── main ───────────────────────────────────────────────
def main():
    argp = argparse.ArgumentParser()
    argp.add_argument("--data",   default="dataset")
    argp.add_argument("--epochs", type=int, default=10)
    argp.add_argument("--bs",     type=int, default=128)
    argp.add_argument("--lr",     type=float, default=1e-4)
    args = argp.parse_args()

    ds  = UltraDataset(args.data)
    dl  = DataLoader(ds, batch_size=args.bs, shuffle=True, num_workers=4)

    net = BCNet().cuda()
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    loss_fn = nn.BCELoss()

    for epoch in range(1, args.epochs+1):
        net.train(); running = 0
        pbar = tqdm(dl, desc=f"Epoch {epoch}/{args.epochs}")
        for x,y in pbar:
            x,y = x.cuda(), y.cuda()
            out = net(x)
            loss = loss_fn(out, y)
            opt.zero_grad(); loss.backward(); opt.step()
            running += loss.item()
            pbar.set_postfix(loss=running/ (pbar.n+1))

    torch.save(net.state_dict(), "bc.pt")
    print("Saved bc.pt")

if __name__ == "__main__":
    main()
