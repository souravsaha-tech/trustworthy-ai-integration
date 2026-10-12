"""Run all experiments over several seeds and write results/ (JSON, CSV summary, series)."""
import json, sys, time
import numpy as np, pandas as pd
from aiarch.experiments import e1, e2, e3, e4

SEEDS = list(range(1, 11))
out = {}


def flat(d, prefix=""):
    r = {}
    for k, v in d.items():
        if isinstance(v, dict):
            r.update(flat(v, f"{prefix}{k}."))
        else:
            r[f"{prefix}{k}"] = v
    return r


t0 = time.time()
for name, fn in (("E1", e1), ("E2", e2), ("E3", e3), ("E4", e4)):
    rows = [flat(fn(s)) for s in SEEDS]
    df = pd.DataFrame(rows, index=SEEDS)
    df.to_csv(f"results/{name}_per_seed.csv", index_label="seed")
    out[name] = {c: dict(mean=float(df[c].astype(float).mean()), sd=float(df[c].astype(float).std()),
                         min=float(df[c].astype(float).min()), max=float(df[c].astype(float).max()),
                         n_nan=int(df[c].astype(float).isna().sum()))
                 for c in df.columns}
    print(name, "done", round(time.time() - t0), "s", flush=True)
json.dump(out, open("results/summary.json", "w"), indent=1)

# time series for the figure (seed 1)
_, s1 = e1(1, return_series=True)
_, s2 = e2(1, return_series=True)
np.savez("results/series_seed1.npz",
         **{f"e1_{m}_{k}": v for m, d in s1.items() for k, v in d.items()},
         **{f"e2_{k}": v for k, v in s2.items()})
print("all done", round(time.time() - t0), "s")
