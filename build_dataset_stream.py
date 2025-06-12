import argparse
import json
import os
import pathlib
import cv2
import numpy as np
from tqdm import tqdm

p = argparse.ArgumentParser()
p.add_argument("--video", required=True)
p.add_argument("--log",   required=True)
p.add_argument("--out",   default="dataset")
p.add_argument("--size",  type=int, default=84)
p.add_argument("--fps",   type=int, default=60)
a = p.parse_args()

# ---------- читаем и сразу сортируем события ----------
with open(a.log, "r", encoding="utf-8") as f:
    raw = json.load(f)
events = sorted((float(t), ev) for t, ev in raw.items())
ev_i   = 0                             # указатель на текущий эвент

SIZE = (a.size, a.size)
pathlib.Path(a.out).mkdir(exist_ok=True, parents=True)

cap   = cv2.VideoCapture(a.video)
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
state = {}
idx   = 0

with tqdm(total=total, unit="frame") as bar:
    while True:
        ok, frame = cap.read()
        if not ok: break

        t0 = idx / a.fps
        t1 = (idx+1) / a.fps

        # --- перемещаем указатель событий только вперёд ---
        while ev_i < len(events) and events[ev_i][0] < t1:
            ts, ev = events[ev_i]
            if ts >= t0:              # событие попало в интервал кадра
                state.update(ev)
            ev_i += 1

        small = cv2.resize(frame, SIZE, interpolation=cv2.INTER_AREA)
        np.savez_compressed(
            os.path.join(a.out, f"{idx:08d}.npz"),
            img=small,
            keys=state.copy()
        )
        idx += 1
        bar.update(1)

cap.release()
print("Saved", idx, "samples →", a.out)
