# locale guard
import locale, os
try:
    locale.setlocale(locale.LC_ALL, '')
except locale.Error:
    os.environ["LC_ALL"] = 'C'
    locale.setlocale(locale.LC_ALL, 'C')

import glob, numpy as np, matplotlib.pyplot as plt, cv2, os

FILES = sorted(glob.glob("dataset\\*.npz"))
if not FILES:
    raise SystemExit("dataset пуст")

idx = 0
def load(i):
    d = np.load(FILES[i], allow_pickle=True)
    rgb = cv2.cvtColor(d["img"], cv2.COLOR_BGR2RGB)
    return rgb, d["keys"].item()

fig, ax = plt.subplots(figsize=(3,3))
img, k = load(idx)
im  = ax.imshow(img); ax.axis("off")
ttl = ax.set_title(os.path.basename(FILES[idx]))
txt = fig.text(0.5, 0.02, str(k), ha="center", wrap=True, fontsize=8)

def refresh():
    global img, k
    img, k = load(idx)
    im.set_data(img)
    ttl.set_text(os.path.basename(FILES[idx]))
    txt.set_text(str(k))
    fig.canvas.draw_idle()

def on_key(e):
    global idx
    if e.key in ["right","d"]: idx = (idx+1) % len(FILES)
    if e.key in ["left","a"]:  idx = (idx-1) % len(FILES)
    refresh()

def on_scroll(e):
    global idx
    idx = (idx-1)%len(FILES) if e.button=="up" else (idx+1)%len(FILES)
    refresh()

fig.canvas.mpl_connect("key_press_event", on_key)
fig.canvas.mpl_connect("scroll_event",    on_scroll)
plt.show()

