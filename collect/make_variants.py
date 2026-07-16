"""Relabel a collected dataset's action channels without re-rendering anything.

Two relabels, applied to a LeRobot dataset produced by collect_demos.py:

  --binary   gripper channel snapped to two values: 0 (closed) or 45 (open).
             This was the single largest win in the repo's experiments: with
             quantile normalization the two values map exactly onto -1/+1, so the
             discrete action head faces a clean two-class problem instead of a
             smear of in-between angles.
  --delta    arm channels become per-step deltas (a_t - a_{t-1}; first step vs
             state). Measured to HURT on this task; kept so the ablation is
             reproducible.

The action column, meta/stats.json, and the per-episode stats (meta/episodes/*.parquet) are
rewritten; the videos are copied unchanged, so each variant is self-contained and uploads to
the Hub with its videos. Stats MUST be rewritten whenever actions change: training normalizes
with the stored quantiles, and stale ones mis-normalize silently.

Usage:
    python collect/make_variants.py --src data/cube500 --dst data/cube500_bin --binary
"""
import argparse
import glob
import json
import pathlib
import shutil

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as papq

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--src", required=True, help="source LeRobot dataset directory")
ap.add_argument("--dst", required=True, help="output directory for the relabeled variant")
ap.add_argument("--binary", action="store_true", help="snap gripper to {0, 45}")
ap.add_argument("--delta", action="store_true", help="relabel arm dims to per-step deltas")
ap.add_argument("--overwrite", action="store_true", help="delete an existing --dst first")
args = ap.parse_args()

SRC = str(pathlib.Path(args.src).resolve())
DST = str(pathlib.Path(args.dst).resolve())
if not (args.binary or args.delta):
    ap.error("nothing to do: pass --binary and/or --delta")
if pathlib.Path(DST).exists():
    if not args.overwrite:
        ap.error(f"{DST} exists; pass --overwrite to replace it")
    shutil.rmtree(DST)

# Copy everything, videos included, so the variant is self-contained and uploads to the Hub
# with its videos. Copy rather than symlink: push_to_hub does not follow directory symlinks,
# so a symlinked video tree would upload with no video files.
shutil.copytree(SRC, DST)

all_actions = []
for pq in sorted(glob.glob(f"{DST}/data/*/*.parquet")):
    df = pd.read_parquet(pq)
    A = np.stack(df["action"].to_numpy())
    S = np.stack(df["observation.state"].to_numpy())
    ep = df["episode_index"].to_numpy()
    D = A.copy()
    if args.binary:
        D[:, 5] = np.where(A[:, 5] > 22.5, 45.0, 0.0)
    if args.delta:
        for e in np.unique(ep):
            m = ep == e
            a, s = D[m], S[m]
            d = a.copy()
            d[0, :5] = a[0, :5] - s[0, :5]
            d[1:, :5] = a[1:, :5] - a[:-1, :5]
            D[m] = d
    df["action"] = list(D.astype(np.float32))
    df.to_parquet(pq)
    all_actions.append(D)

# Rewrite action stats so normalization matches the new labels.
D = np.concatenate(all_actions)
stats = json.load(open(f"{DST}/meta/stats.json"))
a = stats["action"]
a["min"] = D.min(0).tolist()
a["max"] = D.max(0).tolist()
a["mean"] = D.mean(0).tolist()
a["std"] = D.std(0).tolist()
for q in ("q01", "q10", "q50", "q90", "q99"):
    a[q] = np.quantile(D, float(q[1:]) / 100.0, axis=0).tolist()
json.dump(stats, open(f"{DST}/meta/stats.json", "w"))

# Rewrite the PER-EPISODE action stats too (LeRobot v3 meta/episodes/*.parquet). Editing only
# the data Parquet and the global stats.json leaves each per-episode record describing the
# pre-relabel data, so anything that aggregates or inspects per-episode stats reads stale values.
ep_files = glob.glob(f"{DST}/meta/episodes/**/*.parquet", recursive=True)
if ep_files:
    frames = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{DST}/data/*/*.parquet"))],
                       ignore_index=True)
    by_ep = {int(e): np.stack(g["action"].to_numpy()).astype(np.float64)
             for e, g in frames.groupby("episode_index")}
    STAT = {"min": lambda X: X.min(0), "max": lambda X: X.max(0), "mean": lambda X: X.mean(0),
            "std": lambda X: X.std(0), "count": lambda X: [len(X)],
            "q01": lambda X: np.quantile(X, .01, 0), "q10": lambda X: np.quantile(X, .10, 0),
            "q50": lambda X: np.quantile(X, .50, 0), "q90": lambda X: np.quantile(X, .90, 0),
            "q99": lambda X: np.quantile(X, .99, 0)}
    for epf in ep_files:
        t = papq.read_table(epf)
        sch = t.schema
        order = t.column("episode_index").to_pylist()
        for s, fn in STAT.items():
            col = f"stats/action/{s}"
            if col not in sch.names:
                continue
            vals = [([len(by_ep[int(e)])] if s == "count"
                     else list(np.asarray(fn(by_ep[int(e)])).tolist())) for e in order]
            j = sch.get_field_index(col)
            t = t.set_column(j, sch.field(j), pa.array(vals, type=sch.field(j).type))
        papq.write_table(t, epf)
print(f"OK {DST} frames={len(D)} binary={args.binary} delta={args.delta}")
