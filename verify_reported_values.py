#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Recompute every quantity quoted in the manuscript directly from the run
outputs and compare it with the value printed in the text.

    python3 verify_reported_values.py <results_dir> [--values manuscript_values.json]

Exit status is 0 only if every check passes. Each check states the file and
the aggregation it uses, so that a reader can follow the number from the text
back to the raw output without reading the simulation code.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS = None
_cache = {}


def load(name):
    if name not in _cache:
        _cache[name] = pd.read_csv(RESULTS / name)
    return _cache[name]


def auc(labels, scores):
    y = np.asarray(labels, float)
    s = np.asarray(scores, float)
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s))
    ranks[order] = np.arange(1, len(s) + 1)
    n1 = y.sum()
    n0 = len(y) - n1
    return (ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


# --------------------------------------------------------------- quantities
def q_closed_loop_jsd():
    d = load("10_closed_loop_probe.csv")
    return d[d.record_type == "jsd"].js_divergence.mean()


def q_open_loop_jsd():
    d = load("10_closed_loop_probe.csv")
    return d[d.record_type == "open_loop_control"].js_divergence.mean()


def q_weight_entropy():
    w = load("06_learned_q_weights.csv")
    w = w[w.condition == "Full_Learned_Q"]
    piv = w.pivot_table(index=["episode", "risk_label", "morphology"],
                        columns="component", values="weight")
    return float((-(piv * np.log(piv.clip(lower=1e-12))).sum(axis=1)).mean())


def q_weight_sd():
    w = load("06_learned_q_weights.csv")
    w = w[w.condition == "Full_Learned_Q"]
    piv = w.pivot_table(index=["episode", "risk_label", "morphology"],
                        columns="component", values="weight")
    return float(piv.std(axis=1, ddof=0).mean())


def q_pain_danger_share():
    w = load("06_learned_q_weights.csv")
    w = w[w.condition == "Full_Learned_Q"]
    piv = w.pivot_table(index=["episode", "risk_label", "morphology"],
                        columns="component", values="weight")
    return float(piv[["pain", "danger_memory"]].sum(axis=1).mean())


def _prospective():
    p = load("04_prospective_metrics.csv")
    p = p[(p.condition == "Full_Learned_Q") & (p.predictor == "learned_Q_decoder")]
    return p.dropna(subset=["mean_q_prediction_loss", "mean_null_prediction_loss"])


def q_learned_loss():
    return float(_prospective().groupby("episode").mean_q_prediction_loss.mean().mean())


def q_null_loss():
    return float(_prospective().groupby("episode").mean_null_prediction_loss.mean().mean())


def _predictor_metric(pred, col):
    p = load("04_prospective_metrics.csv")
    p = p[(p.condition == "Full_Learned_Q") & (p.predictor == pred)]
    return float(p.groupby("episode")[col].mean().mean())


def _state_contrast(col):
    s = load("11_same_event_state_probe.csv")
    piv = s.pivot_table(index=["seed", "pair"], columns="state", values=col).reset_index()
    per = piv.assign(diff=piv.vulnerable - piv.stable).groupby("seed")["diff"].mean()
    return float(per.mean()), float(per.std(ddof=1)), int((per > 0).sum()), len(per)


def _history_contrast(col, mode="intact"):
    h = load("12_history_memory_probe.csv")
    piv = h[h.memory_mode == mode].pivot_table(index="seed", columns="history", values=col)
    diff = piv["harmful"] - piv["benign"]
    return float(diff.mean()), float(diff.std(ddof=1)), int((diff > 0).sum()), len(diff)


def _clone(intervention):
    c = load("15_state_clone_interventions.csv")
    learned = c[c.intervention == "learned_q"].groupby("seed").damage_increment.mean()
    other = c[c.intervention == intervention].groupby("seed").damage_increment.mean()
    diff = learned - other
    return float(diff.mean()), float(diff.mean() / diff.std(ddof=1)), int((diff < 0).sum()), len(diff)


def _condition_contrast(comparator, col="cumulative_damage"):
    ep = load("03_episode_summary.csv")
    cell = ep.groupby(["condition", "episode"])[col].mean().unstack(0)
    diff = cell["Full_Learned_Q"] - cell[comparator]
    return float(diff.mean())


def q_agency_controlled_auc():
    b = load("14_agency_mismatch_battery.csv")
    vals = []
    for _, g in b.groupby("seed"):
        g = g[g.trial_type.isin(["matched_self", "external", "passive_replay"])]
        vals.append(auc((g.trial_type == "matched_self").astype(int), g.agency_score))
    return float(np.mean(vals))


def q_agency_trial(trial_type):
    b = load("14_agency_mismatch_battery.csv")
    return float(b[b.trial_type == trial_type].groupby("seed").agency_score.mean().mean())


def q_agency_natural(col):
    a = load("05_agency_episode_metrics.csv")
    return float(a[a.condition == "Full_Learned_Q"].groupby("episode")[col].mean().mean())


def _morphology(event, col):
    m = load("13_morphology_probe.csv")
    return float(m[m.event_type == event].groupby("seed")[col].mean().mean())


def _heldout(col):
    h = load("16_heldout_event_generalization.csv")
    per = h.groupby(["seed", "heldout_event"])[col].mean().groupby("seed").mean()
    return float(per.mean()), float(per.std(ddof=1))


CHECKS = [
    ("closed-loop mean JSD", "10_closed_loop_probe.csv", q_closed_loop_jsd),
    ("action-open-loop mean JSD", "10_closed_loop_probe.csv", q_open_loop_jsd),
    ("learned weight entropy", "06_learned_q_weights.csv", q_weight_entropy),
    ("within-model weight SD", "06_learned_q_weights.csv", q_weight_sd),
    ("pain + danger-memory weight share", "06_learned_q_weights.csv", q_pain_danger_share),
    ("learned-Q prediction loss", "04_prospective_metrics.csv", q_learned_loss),
    ("null decoder prediction loss", "04_prospective_metrics.csv", q_null_loss),
    ("high-compromise AUC, learned Q scalar", "04_prospective_metrics.csv",
     lambda: _predictor_metric("learned_Q_scalar", "high_compromise_auc")),
    ("high-compromise AUC, pain", "04_prospective_metrics.csv",
     lambda: _predictor_metric("pain", "high_compromise_auc")),
    ("high-compromise AUC, physical risk", "04_prospective_metrics.csv",
     lambda: _predictor_metric("physical_risk", "high_compromise_auc")),
    ("compromise correlation, learned Q scalar", "04_prospective_metrics.csv",
     lambda: _predictor_metric("learned_Q_scalar", "compromise_r")),
    ("compromise correlation, pain", "04_prospective_metrics.csv",
     lambda: _predictor_metric("pain", "compromise_r")),
    ("compromise correlation, physical risk", "04_prospective_metrics.csv",
     lambda: _predictor_metric("physical_risk", "compromise_r")),
    ("state contrast in Q", "11_same_event_state_probe.csv", lambda: _state_contrast("q")[0]),
    ("state contrast in pain", "11_same_event_state_probe.csv", lambda: _state_contrast("pain")[0]),
    ("state contrast in danger", "11_same_event_state_probe.csv", lambda: _state_contrast("danger")[0]),
    ("history contrast in Q, memories intact", "12_history_memory_probe.csv",
     lambda: _history_contrast("q")[0]),
    ("history contrast in Q, danger memory lesioned", "12_history_memory_probe.csv",
     lambda: _history_contrast("q", "danger_zero")[0]),
    ("history contrast in Q, pain memory lesioned", "12_history_memory_probe.csv",
     lambda: _history_contrast("q", "pain_zero")[0]),
    ("history contrast in Q, comfort memory lesioned", "12_history_memory_probe.csv",
     lambda: _history_contrast("q", "comfort_zero")[0]),
    ("history contrast in Q, all memories lesioned", "12_history_memory_probe.csv",
     lambda: _history_contrast("q", "zero")[0]),
    ("history contrast in pain", "12_history_memory_probe.csv",
     lambda: _history_contrast("pain")[0]),
    ("history contrast in fixed Q", "12_history_memory_probe.csv",
     lambda: _history_contrast("fixed_q")[0]),
    ("clone contrast, learned minus zero", "15_state_clone_interventions.csv",
     lambda: _clone("zero")[0]),
    ("clone contrast dz, learned minus zero", "15_state_clone_interventions.csv",
     lambda: _clone("zero")[1]),
    ("clone seeds favouring learned over shuffled", "15_state_clone_interventions.csv",
     lambda: float(_clone("shuffled_q")[2])),
    ("clone seeds favouring learned over danger", "15_state_clone_interventions.csv",
     lambda: float(_clone("danger")[2])),
    ("clone seeds favouring learned over pain", "15_state_clone_interventions.csv",
     lambda: float(_clone("pain")[2])),
    ("damage, Full Learned Q minus Fixed Q", "03_episode_summary.csv",
     lambda: _condition_contrast("Fixed_Q")),
    ("damage, Full Learned Q minus Action Open Loop Q", "03_episode_summary.csv",
     lambda: _condition_contrast("Action_Open_Loop_Q")),
    ("damage, Full Learned Q minus Pain Only", "03_episode_summary.csv",
     lambda: _condition_contrast("Pain_Only")),
    ("damage, Full Learned Q minus Danger Only", "03_episode_summary.csv",
     lambda: _condition_contrast("Danger_Only")),
    ("damage, Full Learned Q minus Physical Risk Only", "03_episode_summary.csv",
     lambda: _condition_contrast("Physical_Risk_Only")),
    ("failure, Full Learned Q minus Pain Only", "03_episode_summary.csv",
     lambda: _condition_contrast("Pain_Only", "viability_failure")),
    ("failure, Full Learned Q minus Physical Risk Only", "03_episode_summary.csv",
     lambda: _condition_contrast("Physical_Risk_Only", "viability_failure")),
    ("controlled agency AUC", "14_agency_mismatch_battery.csv", q_agency_controlled_auc),
    ("agency score, external coincidence", "14_agency_mismatch_battery.csv",
     lambda: q_agency_trial("external_coincidence")),
    ("natural-episode agency AUC", "05_agency_episode_metrics.csv",
     lambda: q_agency_natural("agency_auc")),
    ("natural-episode delay MAE", "05_agency_episode_metrics.csv",
     lambda: q_agency_natural("agency_delay_mae")),
    ("Q-deterioration rank correlation, collision", "13_morphology_probe.csv",
     lambda: _morphology("collision", "q_deterioration_spearman")),
    ("Q-deterioration rank correlation, landing", "13_morphology_probe.csv",
     lambda: _morphology("landing", "q_deterioration_spearman")),
    ("Q-deterioration rank correlation, slip", "13_morphology_probe.csv",
     lambda: _morphology("slip", "q_deterioration_spearman")),
    ("swap penalty, collision", "13_morphology_probe.csv",
     lambda: _morphology("collision", "morphology_swap_penalty")),
    ("held-out improvement over action-only", "16_heldout_event_generalization.csv",
     lambda: _heldout("q_improvement_vs_action_only")[0]),
    ("held-out improvement over null", "16_heldout_event_generalization.csv",
     lambda: _heldout("q_improvement_vs_null")[0]),
]


def main():
    global RESULTS
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir")
    ap.add_argument("--values", default=str(Path(__file__).with_name("manuscript_values.json")))
    args = ap.parse_args()
    RESULTS = Path(args.results_dir)
    expected = json.loads(Path(args.values).read_text(encoding="utf-8"))

    print("%-52s %12s %12s %9s  %s" % ("quantity", "manuscript", "recomputed", "abs diff", "source"))
    print("-" * 118)
    failures = 0
    for label, source, fn in CHECKS:
        got = float(fn())
        if label not in expected:
            print("%-52s %12s %12.5f %9s  %s" % (label[:52], "not listed", got, "", source))
            continue
        want, tol = expected[label]["value"], expected[label]["tolerance"]
        diff = abs(got - want)
        ok = diff <= tol
        failures += 0 if ok else 1
        print("%-52s %12.5f %12.5f %9.5f  %s%s"
              % (label[:52], want, got, diff, source, "" if ok else "   <== MISMATCH"))
    print("-" * 118)
    print("%d checks, %d mismatch(es)" % (len(CHECKS), failures))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
