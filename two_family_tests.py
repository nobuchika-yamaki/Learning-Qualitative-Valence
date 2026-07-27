#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Two-family inferential analysis.

Primary central-hypothesis family (six contrasts) and secondary
functional-benchmark family (five contrasts), Holm-corrected separately.
The base seed is the inferential unit throughout, as in Section 2.30.

    python3 two_family_tests.py <results_dir> [-o two_family_tests.csv]
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

PERM = 20000
BOOT = 10000
SEED = 20260725


def sign_flip_p(x, rng, n=PERM):
    obs = abs(np.mean(x))
    null = np.array([np.mean(x * rng.choice([-1.0, 1.0], size=len(x))) for _ in range(n)])
    return (np.sum(np.abs(null) >= obs) + 1) / (n + 1)


def boot_ci(x, rng, n=BOOT):
    b = np.array([np.mean(rng.choice(x, size=len(x), replace=True)) for _ in range(n)])
    return tuple(np.percentile(b, [2.5, 97.5]))


def holm(pvals):
    order = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj


def summarize(name, x, rng):
    x = np.asarray(x, float)
    return dict(contrast=name, mean=x.mean(), sd=x.std(ddof=1),
                dz=x.mean() / x.std(ddof=1), n_seeds=len(x),
                positive_seeds=int((x > 0).sum()),
                ci_low=boot_ci(x, rng)[0], ci_high=boot_ci(x, rng)[1],
                permutation_p=sign_flip_p(x, rng))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir")
    ap.add_argument("-o", "--output", default="two_family_tests.csv")
    args = ap.parse_args()
    d = Path(args.results_dir)
    rng = np.random.default_rng(SEED)

    # ---------------------------------------------------------- primary family
    loop = pd.read_csv(d / "10_closed_loop_probe.csv")
    closed = loop[loop.record_type == "jsd"].groupby("seed").js_divergence.mean()
    openl = loop[loop.record_type == "open_loop_control"].groupby("seed").js_divergence.mean()
    common = closed.index.intersection(openl.index)
    c1 = (closed[common] - openl[common]).to_numpy()

    pro = pd.read_csv(d / "04_prospective_metrics.csv")
    pro = pro[(pro.condition == "Full_Learned_Q") & (pro.predictor == "learned_Q_decoder")]
    pro = pro.dropna(subset=["mean_q_prediction_loss", "mean_null_prediction_loss"])
    if pro.empty:
        raise SystemExit("no learned_Q_decoder rows with prediction losses")
    per = pro.groupby("episode")[["mean_null_prediction_loss", "mean_q_prediction_loss"]].mean()
    c2 = (per.mean_null_prediction_loss - per.mean_q_prediction_loss).to_numpy()

    state = pd.read_csv(d / "11_same_event_state_probe.csv")
    piv = state.pivot_table(index=["seed", "pair"], columns="state", values="q").reset_index()
    c3 = piv.assign(diff=piv.vulnerable - piv.stable).groupby("seed")["diff"].mean().to_numpy()

    hist = pd.read_csv(d / "12_history_memory_probe.csv")
    intact = hist[hist.memory_mode == "intact"].pivot_table(index="seed", columns="history", values="q")
    contrast_intact = intact["harmful"] - intact["benign"]
    c4 = contrast_intact.to_numpy()
    lesion = hist[hist.memory_mode == "zero"].pivot_table(index="seed", columns="history", values="q")
    contrast_lesion = lesion["harmful"] - lesion["benign"]
    c5 = (contrast_intact - contrast_lesion).to_numpy()

    clone = pd.read_csv(d / "15_state_clone_interventions.csv")
    learned = clone[clone.intervention == "learned_q"].groupby("seed").damage_increment.mean()
    zero = clone[clone.intervention == "zero"].groupby("seed").damage_increment.mean()
    c6 = (learned - zero).to_numpy()

    primary = [
        summarize("Closed-loop JSD \u2212 action-open-loop JSD", c1, rng),
        summarize("Null prediction loss \u2212 learned-Q prediction loss", c2, rng),
        summarize("Vulnerable Q \u2212 stable Q", c3, rng),
        summarize("Harmful-history Q \u2212 benign-history Q", c4, rng),
        summarize("Intact history contrast \u2212 no-memory history contrast", c5, rng),
        summarize("Learned-Q clone damage \u2212 zero-Q clone damage", c6, rng),
    ]

    # -------------------------------------------------------- secondary family
    ep = pd.read_csv(d / "03_episode_summary.csv")
    cell = ep.groupby(["condition", "episode"]).cumulative_damage.mean().unstack(0)
    secondary = []
    for comparator in ["Fixed_Q", "Action_Open_Loop_Q", "Pain_Only", "Danger_Only",
                       "Physical_Risk_Only"]:
        x = (cell["Full_Learned_Q"] - cell[comparator]).to_numpy()
        secondary.append(summarize("Full Learned Q \u2212 %s" % comparator.replace("_", " "), x, rng))

    out = []
    for family, rows in (("primary_central_hypothesis", primary),
                         ("secondary_functional_benchmark", secondary)):
        adj = holm(np.array([r["permutation_p"] for r in rows]))
        for r, a in zip(rows, adj):
            r["family"] = family
            r["holm_p"] = a
            out.append(r)
    table = pd.DataFrame(out)[
        ["family", "contrast", "mean", "sd", "ci_low", "ci_high", "dz",
         "positive_seeds", "n_seeds", "permutation_p", "holm_p"]]
    table.to_csv(args.output, index=False)

    pd.set_option("display.width", 200)
    for family in table.family.unique():
        sub = table[table.family == family]
        print("\n=== %s ===" % family)
        for _, r in sub.iterrows():
            print("  %-52s %+0.5f \u00b1 %0.5f  [%+0.5f, %+0.5f]  dz=%+7.3f  %2d/%d  P=%.5f  Holm=%.5f"
                  % (r.contrast[:52], r["mean"], r.sd, r.ci_low, r.ci_high, r.dz,
                     r.positive_seeds, r.n_seeds, r.permutation_p, r.holm_p))
    print("\nwritten to", args.output)


if __name__ == "__main__":
    main()
