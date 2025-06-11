import glob, numpy as np, pprint
f = sorted(glob.glob(r"dataset\\*.npz"))[0]
d = np.load(f, allow_pickle=True)
pprint.pp(d["keys"].item())
