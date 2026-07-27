#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Incremental-information test reported in Results 3.11.

Asks whether Q carries information about realized future bodily compromise
that immediate pain, physical risk, and learned danger do not carry, on the
event types whose learning updates were withheld.

    python3 incremental_information_test.py 16_heldout_event_generalization.csv

In-sample R2 cannot decrease when a regressor is added, so the incremental
contribution is estimated out of sample by five-fold cross-validation within
each seed x held-out-event cell and is therefore free to be negative.

Note the asymmetry: Q is the only one of the four signals trained against a
quantity related to the target, so the comparison favours Q by construction.
The test establishes non-reducibility of Q to its strongest components, not
superiority over them.
"""
import argparse
import numpy as np
import pandas as pd

BASE_SIGNALS = ["pain", "physical_risk", "danger"]
TARGET = "future_compromise"


def cv_r2(y, columns, folds=5, seed=0):
    n = len(y)
    idx = np.random.default_rng(seed).permutation(n)
    X = np.column_stack([np.ones(n)] + [np.asarray(c, float) for c in columns])
    pred = np.empty(n)
    for k in range(folds):
        test = idx[k::folds]
        train = np.setdiff1d(idx, test)
        beta, *_ = np.linalg.lstsq(X[train], y[train], rcond=None)
        pred[test] = X[test] @ beta
    ss = ((y - y.mean()) ** 2).sum()
    return 1.0 - ((y - pred) ** 2).sum() / ss if ss > 0 else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", help="16_heldout_event_generalization.csv")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--permutations", type=int, default=20000)
    ap.add_argument("--bootstrap", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20260725)
    ap.add_argument("--out", default=None, help="optional CSV of per-cell results")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    h = pd.read_csv(args.csv)
    missing = [c for c in BASE_SIGNALS + ["q", TARGET, "seed", "heldout_event"]
               if c not in h.columns]
    if missing:
        raise SystemExit("missing columns: %s" % missing)

    rows = []
    for (seed, event), g in h.groupby(["seed", "heldout_event"]):
        y = g[TARGET].to_numpy(float)
        if y.std() < 1e-9 or len(g) < 5 * args.folds:
            continue
        base = [g[c] for c in BASE_SIGNALS]
        s = int(seed) % (2 ** 31)
        rows.append(dict(seed=seed, event=event,
                         cv_r2_base=cv_r2(y, base, args.folds, s),
                         cv_r2_full=cv_r2(y, base + [g["q"]], args.folds, s),
                         n=len(g)))
    cells = pd.DataFrame(rows)
    if cells.empty:
        print("no cell had enough held-out trials for %d-fold cross-validation; "
              "this analysis needs the main preset" % args.folds)
        return
    cells["incremental"] = cells.cv_r2_full - cells.cv_r2_base
    per_seed = cells.groupby("seed").incremental.mean()
    x = per_seed.to_numpy()

    dz = x.mean() / x.std(ddof=1)
    null = np.array([np.mean(x * rng.choice([-1.0, 1.0], size=len(x)))
                     for _ in range(args.permutations)])
    p = (np.sum(np.abs(null) >= abs(x.mean())) + 1) / (args.permutations + 1)
    boot = np.array([np.mean(rng.choice(x, size=len(x), replace=True))
                     for _ in range(args.bootstrap)])
    lo, hi = np.percentile(boot, [2.5, 97.5])

    print("Incremental information of Q over %s" % ", ".join(BASE_SIGNALS))
    print("target: %s, on parameter-update-held-out events" % TARGET)
    print("  cells                     %d (%d seeds x %d events)"
          % (len(cells), per_seed.size, cells.event.nunique()))
    print("  cross-validated R2 base   %.4f" % cells.cv_r2_base.mean())
    print("  cross-validated R2 + Q    %.4f" % cells.cv_r2_full.mean())
    print("  incremental               %+.5f \u00b1 %.5f (SD across seeds)"
          % (x.mean(), x.std(ddof=1)))
    print("  95%% bootstrap CI          [%+.5f, %+.5f]" % (lo, hi))
    print("  paired dz                 %+.3f" % dz)
    print("  permutation P             %.5f" % p)
    print("  positive seeds            %d / %d" % ((x > 0).sum(), len(x)))
    print()
    print("by held-out event:")
    summary = cells.groupby("event").incremental.agg(
        mean="mean", sd="std", positive=lambda s: int((s > 0).sum()))
    print(summary.round(5).to_string())

    if args.out:
        cells.to_csv(args.out, index=False)
        print("\nper-cell results written to %s" % args.out)


if __name__ == "__main__":
    main()
