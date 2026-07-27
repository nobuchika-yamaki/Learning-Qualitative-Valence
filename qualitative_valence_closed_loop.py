#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
04_QUALITATIVE_VALENCE_CLOSED_LOOP_V2_VALIDATION.py

V2 confirmatory implementation for:
"Learning Qualitative Valence in an Embodied Computational Agent".

The program tests the strong Path 1 hypothesis.  It preserves qualitative
valence as an integrated bodily state composed of pain, learned danger,
comfort, action possibility, memory, bodily condition, and self-attribution.
It does not replace valence with a single survival score.

V2 corrections relative to the previous implementation
---------------------------------------------------------
1. Adaptive calibration uses the confirmatory episode length, independent
   calibration seeds, and representative morphology corners.  It expands the
   search grid until ordered mild/moderate/challenging regimes satisfy the
   prespecified dynamic-range criteria, otherwise the run stops.
2. Q cannot be bypassed by a freely trainable decoder intercept or action
   offset.  Frozen calibration baselines, a scalar Q bottleneck, replay-based
   multi-outcome prediction, burden alignment, and pairwise ranking identify
   the integration weights.
3. The Q target anchors the current event by its pre-action continuation
   counterfactual burden and adds discounted future bodily costs.  The same
   target is used online, in morphology probes, and in held-out evaluation.
4. Morphology changes actual mechanical damage, energetic cost, traction,
   recovery, and event transitions, while morphology labels remain hidden from
   the learners.
5. Agency compares capacity-matched true-action and decoy-action predictors
   using pre-update errors.  Delay tests use adaptation blocks followed by
   held-out blocks, including exact passive sensory replay.
6. Confirmatory episodes, calibration candidates, robustness episodes, and
   seed-level probes are parallelized.  BLAS threads are restricted per worker
   and long pools recycle workers to prevent oversubscription and memory growth.
7. Reference-agent training is shared among state, history, and state-clone
   probes; vectorized closed-loop draws and signature-safe caches avoid repeated
   computation.
8. Inference uses the independent base seed as the permutation/effect-size
   unit, hierarchical seed-to-morphology bootstrap intervals, clustered GEE/OLS,
   and null/action-only held-out prediction baselines.

Typical commands
----------------
Smoke test:
    python3 -u 04_QUALITATIVE_VALENCE_CLOSED_LOOP_V2_VALIDATION.py \
      --preset smoke --workers 4 \
      --outdir ~/Desktop/qualitative_valence_v2_smoke

Methods-level confirmatory run:
    python3 -u 04_QUALITATIVE_VALENCE_CLOSED_LOOP_V2_VALIDATION.py \
      --preset main --workers 8 \
      --outdir ~/Desktop/qualitative_valence_v2_main \
      2>&1 | tee ~/Desktop/qualitative_valence_v2_main.log

Resume an interrupted run:
    python3 -u 04_QUALITATIVE_VALENCE_CLOSED_LOOP_V2_VALIDATION.py \
      --preset main --workers 8 --resume \
      --outdir ~/Desktop/qualitative_valence_v2_main \
      2>&1 | tee -a ~/Desktop/qualitative_valence_v2_main.log

The main and reviewer presets run the robustness battery automatically unless
--skip-robustness is supplied.  No post-hoc parameter selection based on model
comparisons is performed.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
import math
import multiprocessing as mp
import pickle
import os
import platform
import sys
import time
import traceback

# Each Python worker owns one simulation task.  Prevent NumPy/BLAS and
# numexpr from spawning additional thread pools inside every process.
for _thread_var in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS",
):
    os.environ.setdefault(_thread_var, "1")

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# Imported lazily after all multiprocessing stages. Importing statsmodels in
# every spawned worker creates joblib/loky semaphore resources and can stall a
# long-running pool on macOS/Python 3.13.
sm = None
smf = None
plt = None


def _ensure_matplotlib() -> None:
    global plt
    if plt is not None:
        return
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as _plt
    plt = _plt


def _ensure_statsmodels() -> None:
    global sm, smf
    if sm is not None and smf is not None:
        return
    try:
        import statsmodels.api as _sm
        import statsmodels.formula.api as _smf
        sm, smf = _sm, _smf
    except Exception:
        sm, smf = None, None


# =============================================================================
# Constants and utilities
# =============================================================================

CALIBRATION_VERSION = "QVALENCE_V2_COUNTERFACTUAL_DELAY_ALIGNED_20260725"

A_CONTINUE = 0
A_CAUTIOUS = 1
A_AVOID = 2
ACTIONS = (A_CONTINUE, A_CAUTIOUS, A_AVOID)
ACTION_NAMES = {
    A_CONTINUE: "CONTINUATION",
    A_CAUTIOUS: "CAUTIOUS_REGULATION",
    A_AVOID: "ACTIVE_AVOIDANCE",
}
MITIGATION = {A_CONTINUE: 0.00, A_CAUTIOUS: 0.30, A_AVOID: 0.60}
BASE_ENERGY_COST = {A_CONTINUE: 0.003, A_CAUTIOUS: 0.008, A_AVOID: 0.015}
BASE_FATIGUE_COST = {A_CONTINUE: 0.001, A_CAUTIOUS: 0.004, A_AVOID: 0.008}

EVENT_TYPES = ["rest", "walk", "slope", "slip", "jump", "landing", "collision", "brake"]
EVENT_INDEX = {name: i for i, name in enumerate(EVENT_TYPES)}
BASE_EVENT_PROBS = np.array([0.10, 0.15, 0.12, 0.13, 0.12, 0.13, 0.14, 0.11], dtype=float)
HAZARD_EVENTS = {"slope", "slip", "jump", "landing", "collision", "brake"}
HIGH_CONTACT_EVENTS = {"slip", "landing", "collision", "brake"}

Q_COMPONENT_NAMES = [
    "danger",
    "pain",
    "lack_comfort",
    "lack_action_possibility",
    "danger_memory",
    "pain_memory",
    "lack_comfort_memory",
    "fatigue",
    "instability",
    "integrity_loss",
    "energy_loss",
    "lack_controllability",
]
FUTURE_TARGET_NAMES = [
    "damage_increase",
    "energy_loss",
    "fatigue_increase",
    "stability_loss",
    "failure",
]


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def clip01(x: float) -> float:
    return float(np.clip(float(x), 0.0, 1.0))


def sigmoid(x: float | np.ndarray) -> float | np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


def softplus(x: float | np.ndarray) -> float | np.ndarray:
    x = np.asarray(x, dtype=float)
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


def softmax(x: Sequence[float]) -> np.ndarray:
    z = np.asarray(x, dtype=float)
    z = z - np.max(z)
    e = np.exp(np.clip(z, -60.0, 60.0))
    return e / max(float(np.sum(e)), 1e-12)


def action_onehot(action: int) -> np.ndarray:
    out = np.zeros(len(ACTIONS), dtype=float)
    out[int(action)] = 1.0
    return out


def safe_mean(x: Iterable[float]) -> float:
    a = np.asarray(list(x), dtype=float)
    a = a[np.isfinite(a)]
    return float(np.mean(a)) if a.size else float("nan")


def safe_std(x: Iterable[float]) -> float:
    a = np.asarray(list(x), dtype=float)
    a = a[np.isfinite(a)]
    return float(np.std(a, ddof=1)) if a.size > 1 else 0.0


def r2_score(y: Sequence[float], pred: Sequence[float]) -> float:
    y = np.asarray(y, dtype=float)
    p = np.asarray(pred, dtype=float)
    ok = np.isfinite(y) & np.isfinite(p)
    y, p = y[ok], p[ok]
    if len(y) < 2:
        return float("nan")
    den = float(np.sum((y - np.mean(y)) ** 2))
    return float(1.0 - np.sum((y - p) ** 2) / den) if den > 1e-12 else float("nan")


def pearson_r(x: Sequence[float], y: Sequence[float]) -> float:
    a = np.asarray(x, dtype=float)
    b = np.asarray(y, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if len(a) < 3 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _average_ranks(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    sorted_values = values[order]
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and sorted_values[j] == sorted_values[i]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + 1 + j)
        i = j
    return ranks


def spearman_r(x: Sequence[float], y: Sequence[float]) -> float:
    a = np.asarray(x, dtype=float)
    b = np.asarray(y, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if len(a) < 3:
        return float("nan")
    ra, rb = _average_ranks(a), _average_ranks(b)
    if np.std(ra) < 1e-12 or np.std(rb) < 1e-12:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def auc_score(y_true: Sequence[int], score: Sequence[float]) -> float:
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(score, dtype=float)
    ok = np.isfinite(s)
    y, s = y[ok], s[ok]
    pos = s[y == 1]
    neg = s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    scores = np.concatenate([pos, neg])
    labels = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=float)
    for value in np.unique(scores):
        idx = np.flatnonzero(scores == value)
        if len(idx) > 1:
            ranks[idx] = np.mean(ranks[idx])
    rank_sum = float(np.sum(ranks[labels == 1]))
    return float((rank_sum - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg)))


def brier_score(y_true: Sequence[float], score: Sequence[float]) -> float:
    y = np.asarray(y_true, dtype=float)
    s = np.asarray(score, dtype=float)
    ok = np.isfinite(y) & np.isfinite(s)
    return float(np.mean((y[ok] - s[ok]) ** 2)) if np.any(ok) else float("nan")


def balanced_accuracy(y_true: Sequence[int], score: Sequence[float], threshold: float = 0.5) -> float:
    y = np.asarray(y_true, dtype=int)
    pred = (np.asarray(score, dtype=float) >= threshold).astype(int)
    tpr = np.mean(pred[y == 1] == 1) if np.any(y == 1) else np.nan
    tnr = np.mean(pred[y == 0] == 0) if np.any(y == 0) else np.nan
    return float(np.nanmean([tpr, tnr]))


def js_divergence(p: Sequence[float], q: Sequence[float]) -> float:
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    p = p / max(np.sum(p), 1e-12)
    q = q / max(np.sum(q), 1e-12)
    m = 0.5 * (p + q)
    def kl(a: np.ndarray, b: np.ndarray) -> float:
        ok = a > 0
        return float(np.sum(a[ok] * np.log(a[ok] / np.maximum(b[ok], 1e-12))))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def deterministic_seed(*parts: Any) -> int:
    text = "|".join(str(x) for x in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(text).digest()[:8], "little") & 0x7FFFFFFF


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def paired_permutation_p(
    diff: Sequence[float],
    n_perm: int = 10000,
    seed: int = 71023,
    chunk_size: int = 1000,
) -> float:
    """Two-sided paired sign-permutation test evaluated in vectorized chunks."""
    d = np.asarray(diff, dtype=np.float64)
    d = d[np.isfinite(d)]
    if d.size == 0:
        return float("nan")
    obs = abs(float(np.mean(d)))
    rng = np.random.default_rng(seed)
    extreme = 1
    done = 0
    chunk_size = max(1, int(chunk_size))
    while done < int(n_perm):
        batch = min(chunk_size, int(n_perm) - done)
        signs = rng.integers(0, 2, size=(batch, d.size), dtype=np.int8)
        signs = signs * 2 - 1
        perm_means = np.abs((signs @ d) / float(d.size))
        extreme += int(np.count_nonzero(perm_means >= obs))
        done += batch
    return float(extreme / (int(n_perm) + 1))


def hierarchical_bootstrap_ci(
    df: pd.DataFrame,
    value_col: str,
    seed_col: str = "episode",
    morphology_col: str = "morphology",
    n_boot: int = 10000,
    rng_seed: int = 8211,
    chunk_size: int = 512,
) -> Tuple[float, float]:
    """Hierarchical seed-then-morphology bootstrap using NumPy arrays.

    Values within a seed-by-morphology cell are averaged across challenge
    levels. Each replicate samples seeds with replacement and then samples
    morphologies with replacement independently within each sampled seed.
    """
    d = df[[seed_col, morphology_col, value_col]].dropna().copy()
    if d.empty:
        return float("nan"), float("nan")
    table = (
        d.groupby([seed_col, morphology_col], sort=True)[value_col]
        .mean()
        .unstack(morphology_col)
    )
    matrix = table.to_numpy(dtype=np.float64)
    n_seeds, n_morphs = matrix.shape
    if n_seeds == 0 or n_morphs == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(rng_seed)
    estimates = np.empty(int(n_boot), dtype=np.float64)
    done = 0
    chunk_size = max(1, int(chunk_size))
    while done < int(n_boot):
        batch = min(chunk_size, int(n_boot) - done)
        seed_idx = rng.integers(0, n_seeds, size=(batch, n_seeds))
        sampled_seed_rows = matrix[seed_idx, :]
        morph_idx = rng.integers(0, n_morphs, size=(batch, n_seeds, n_morphs))
        sampled = np.take_along_axis(sampled_seed_rows, morph_idx, axis=2)
        estimates[done:done + batch] = np.nanmean(sampled, axis=(1, 2))
        done += batch
    finite = estimates[np.isfinite(estimates)]
    if finite.size == 0:
        return float("nan"), float("nan")
    return float(np.quantile(finite, 0.025)), float(np.quantile(finite, 0.975))




def read_csv_allow_empty(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()

def holm_adjust(p_values: Sequence[float]) -> List[float]:
    p = np.asarray(p_values, dtype=float)
    out = np.full(len(p), np.nan)
    valid = np.flatnonzero(np.isfinite(p))
    if len(valid) == 0:
        return out.tolist()
    order = valid[np.argsort(p[valid])]
    running = 0.0
    m = len(order)
    for rank, idx in enumerate(order):
        adj = min(1.0, (m - rank) * p[idx])
        running = max(running, adj)
        out[idx] = running
    return out.tolist()


class Logger:
    def __init__(self, outdir: Path):
        self.outdir = outdir
        outdir.mkdir(parents=True, exist_ok=True)
        self.path = outdir / "run_progress.log"
        self.t0 = time.time()
        if not self.path.exists():
            self.path.write_text(f"[{now()}] started\n", encoding="utf-8")

    def log(self, text: str) -> None:
        line = f"[{time.time() - self.t0:10.2f}s] {text}"
        print(line, flush=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


# =============================================================================
# Configuration
# =============================================================================


# Hand-specified coefficients that the reviewers asked to be probed.  They are
# module-level so that the defaults remain single-sourced and the robustness
# battery perturbs them explicitly rather than by editing the model.
VULNERABILITY_WEIGHTS: Tuple[float, float, float, float] = (0.35, 0.25, 0.25, 0.15)
AVOIDANCE_COEFFICIENTS: Dict[str, float] = {
    "danger": 1.65, "pain": 1.10, "comfort": -0.85, "controllability": -0.55,
    "danger_memory": 0.55, "integrity_loss": 0.25, "bias": -0.05,
}
AGENCY_PRIOR_WEIGHT: float = 0.55
# Held separately rather than derived as 1 - prior: in IEEE double
# 1.0 - 0.55 != 0.45, and deriving it would perturb every agency score.
AGENCY_CURRENT_WEIGHT: float = 0.45
AGENCY_GAIN: float = 6.0


@dataclass(frozen=True)
class Morphology:
    name: str
    mass: float
    energy_capacity: float
    traction_tolerance: float
    damage_tolerance: float


def build_morphologies() -> Dict[str, Morphology]:
    out: Dict[str, Morphology] = {}
    for mass, energy, traction, damage in itertools.product([0.80, 1.20], repeat=4):
        name = f"m{mass:.1f}_e{energy:.1f}_t{traction:.1f}_d{damage:.1f}"
        out[name] = Morphology(name, mass, energy, traction, damage)
    return out


MORPHOLOGIES = build_morphologies()
REFERENCE_MORPHOLOGY = Morphology("reference", 1.0, 1.0, 1.0, 1.0)


@dataclass(frozen=True)
class EnvironmentConfig:
    risk_label: str
    risk_scale: float
    damage_scale: float
    hazard_multiplier: float
    recovery_multiplier: float
    damage_normalizer: float
    energy_normalizer: float
    fatigue_normalizer: float
    stability_normalizer: float
    target_mean_damage: float
    target_mean_energy: float
    target_mean_fatigue: float
    target_mean_stability: float
    target_mean_failure: float
    damage_target_threshold: float
    true_delay: int
    horizon: int
    morphology: str
    calibration_passed: bool = True

    def target_normalizers(self) -> np.ndarray:
        return np.array([
            self.damage_normalizer,
            self.energy_normalizer,
            self.fatigue_normalizer,
            self.stability_normalizer,
            1.0,
        ], dtype=float)

    def target_means(self) -> np.ndarray:
        return np.array([
            self.target_mean_damage,
            self.target_mean_energy,
            self.target_mean_fatigue,
            self.target_mean_stability,
            self.target_mean_failure,
        ], dtype=float)


@dataclass(frozen=True)
class Condition:
    name: str
    event_loop_mode: str = "closed"  # closed | action_open | exogenous
    signal_mode: str = "learned_q"  # learned_q | fixed_q | pain | danger | physical_risk | body | shuffled_q | zero
    agency_mode: str = "intact"  # intact | zero | time_shuffle
    memory_mode: str = "intact"  # intact | zero | pain_zero | danger_zero | comfort_zero | time_shuffle
    train_q: bool = True


CONDITIONS: Dict[str, Condition] = {
    "Full_Learned_Q": Condition("Full_Learned_Q"),
    "Action_Open_Loop_Q": Condition("Action_Open_Loop_Q", event_loop_mode="action_open"),
    "Fully_Exogenous_Q": Condition("Fully_Exogenous_Q", event_loop_mode="exogenous"),
    "Fixed_Q": Condition("Fixed_Q", signal_mode="fixed_q", train_q=False),
    "Pain_Only": Condition("Pain_Only", signal_mode="pain", train_q=False),
    "Danger_Only": Condition("Danger_Only", signal_mode="danger", train_q=False),
    "Physical_Risk_Only": Condition("Physical_Risk_Only", signal_mode="physical_risk", train_q=False),
    "Body_State_Only": Condition("Body_State_Only", signal_mode="body", train_q=False),
    "Shuffled_Q": Condition("Shuffled_Q", signal_mode="shuffled_q"),
    "No_Q": Condition("No_Q", signal_mode="zero", train_q=False),
    "No_Agency": Condition("No_Agency", agency_mode="zero"),
    "No_Memory": Condition("No_Memory", memory_mode="zero"),
    "Agency_Time_Shuffle": Condition("Agency_Time_Shuffle", agency_mode="time_shuffle"),
    "Pain_Memory_Lesion": Condition("Pain_Memory_Lesion", memory_mode="pain_zero"),
    "Danger_Memory_Lesion": Condition("Danger_Memory_Lesion", memory_mode="danger_zero"),
    "Comfort_Memory_Lesion": Condition("Comfort_Memory_Lesion", memory_mode="comfort_zero"),
    "Memory_Time_Shuffle": Condition("Memory_Time_Shuffle", memory_mode="time_shuffle"),
}

CORE_CONDITIONS = [
    "Full_Learned_Q",
    "Action_Open_Loop_Q",
    "Fully_Exogenous_Q",
    "Fixed_Q",
    "Pain_Only",
    "Danger_Only",
    "Physical_Risk_Only",
    "Body_State_Only",
    "Shuffled_Q",
    "No_Q",
    "No_Agency",
    "No_Memory",
]


@dataclass(frozen=True)
class RunPreset:
    episodes: int
    steps: int
    burn_in: int
    calibration_seeds: int
    calibration_steps: int
    probe_seeds: int
    probe_train_steps: int
    clone_snapshots: int
    clone_horizon: int
    history_steps: int
    loop_states: int
    loop_draws: int
    state_pairs: int
    agency_trials: int
    generalization_trials: int
    morphology_names: Tuple[str, ...]
    challenge_labels: Tuple[str, ...]
    conditions: Tuple[str, ...]


_MORPH_NAMES = tuple(MORPHOLOGIES.keys())
PRESETS: Dict[str, RunPreset] = {
    "smoke": RunPreset(
        episodes=1, steps=320, burn_in=60,
        calibration_seeds=2, calibration_steps=220,
        probe_seeds=1, probe_train_steps=260,
        clone_snapshots=2, clone_horizon=12, history_steps=20,
        loop_states=8, loop_draws=30, state_pairs=20, agency_trials=20,
        generalization_trials=25,
        morphology_names=(_MORPH_NAMES[0], _MORPH_NAMES[-1]),
        challenge_labels=("moderate",),
        conditions=tuple(CORE_CONDITIONS),
    ),
    "compact": RunPreset(
        episodes=4, steps=1500, burn_in=300,
        calibration_seeds=4, calibration_steps=800,
        probe_seeds=4, probe_train_steps=1000,
        clone_snapshots=5, clone_horizon=25, history_steps=50,
        loop_states=60, loop_draws=120, state_pairs=150, agency_trials=150,
        generalization_trials=150,
        morphology_names=(_MORPH_NAMES[0], _MORPH_NAMES[5], _MORPH_NAMES[10], _MORPH_NAMES[15]),
        challenge_labels=("mild", "moderate", "challenging"),
        conditions=tuple(CORE_CONDITIONS),
    ),
    "main": RunPreset(
        episodes=32, steps=5000, burn_in=1000,
        calibration_seeds=8, calibration_steps=5000,
        probe_seeds=32, probe_train_steps=3500,
        clone_snapshots=10, clone_horizon=25, history_steps=40,
        loop_states=500, loop_draws=1000, state_pairs=400, agency_trials=500,
        generalization_trials=500,
        morphology_names=_MORPH_NAMES,
        challenge_labels=("mild", "moderate", "challenging"),
        conditions=tuple(CORE_CONDITIONS),
    ),
    "reviewer": RunPreset(
        episodes=64, steps=5000, burn_in=1000,
        calibration_seeds=12, calibration_steps=5000,
        probe_seeds=64, probe_train_steps=4000,
        clone_snapshots=12, clone_horizon=25, history_steps=40,
        loop_states=750, loop_draws=1000, state_pairs=600, agency_trials=750,
        generalization_trials=750,
        morphology_names=_MORPH_NAMES,
        challenge_labels=("mild", "moderate", "challenging"),
        conditions=tuple(CORE_CONDITIONS),
    ),
}


# =============================================================================
# State, events, and learners
# =============================================================================


@dataclass
class AgentState:
    integrity: float = 1.0
    energy: float = 1.0
    fatigue: float = 0.0
    stability: float = 1.0
    damage: float = 0.0
    pain_memory: float = 0.0
    danger_memory: float = 0.0
    comfort_memory: float = 0.55
    failure_counter: int = 0
    ever_failed: bool = False

    def viability(self) -> float:
        return clip01(min(self.integrity, self.energy, 1.0 - self.fatigue, self.stability))

    def update_failure(self) -> bool:
        boundary = self.integrity <= 0.20 or self.energy <= 0.05 or self.stability <= 0.05
        self.failure_counter = self.failure_counter + 1 if boundary else 0
        if self.failure_counter >= 5:
            self.ever_failed = True
        return self.ever_failed


@dataclass
class Event:
    event_type: str
    baseline_intensity: float
    intensity: float
    friction: float
    slope: float
    affordance: float
    physical_risk: float
    true_delay: int
    source_class: str
    self_generated: int
    mismatch: int
    motor_contribution: float
    exogenous_contribution: float
    reafferent_signal: float


class OnlineLogistic:
    def __init__(self, n_features: int, lr: float, l2: float, init_bias: float, rng: np.random.Generator):
        self.lr = float(lr)
        self.l2 = float(l2)
        self.w = rng.normal(0.0, 0.02, n_features + 1)
        self.w[0] = float(init_bias)
        self.count = 0

    def predict(self, x: Sequence[float]) -> float:
        x = np.asarray(x, dtype=float)
        return float(sigmoid(self.w[0] + np.dot(self.w[1:], x)))

    def update(self, x: Sequence[float], target: float, weight: float = 1.0) -> None:
        if weight <= 0.0:
            return
        x = np.asarray(x, dtype=float)
        p = self.predict(x)
        err = p - clip01(target)
        self.w[0] -= self.lr * weight * err
        self.w[1:] -= self.lr * weight * (err * x + self.l2 * self.w[1:])
        self.count += 1


class AdamArray:
    def __init__(self, shape: Tuple[int, ...]):
        self.m = np.zeros(shape, dtype=float)
        self.v = np.zeros(shape, dtype=float)

    def step(self, param: np.ndarray, grad: np.ndarray, t: int, lr: float, beta1: float, beta2: float) -> np.ndarray:
        self.m = beta1 * self.m + (1.0 - beta1) * grad
        self.v = beta2 * self.v + (1.0 - beta2) * (grad * grad)
        mhat = self.m / (1.0 - beta1 ** t)
        vhat = self.v / (1.0 - beta2 ** t)
        return param - lr * mhat / (np.sqrt(vhat) + 1e-8)


class QReplayBuffer:
    """Fixed-capacity replay without Python-object growth."""

    def __init__(self, capacity: int = 2048):
        self.capacity = int(capacity)
        self.z = np.zeros((self.capacity, len(Q_COMPONENT_NAMES)), dtype=np.float32)
        self.y = np.zeros((self.capacity, len(FUTURE_TARGET_NAMES)), dtype=np.float32)
        self.size = 0
        self.pos = 0

    def add(self, z: Sequence[float], target: Sequence[float]) -> None:
        self.z[self.pos] = np.asarray(z, dtype=np.float32)
        self.y[self.pos] = np.asarray(target, dtype=np.float32)
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, rng: np.random.Generator, batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
        if self.size <= 0:
            raise ValueError("empty replay buffer")
        n = min(int(batch_size), self.size)
        idx = rng.integers(0, self.size, size=n)
        return self.z[idx].astype(float), self.y[idx].astype(float)


class QualitativeValenceIntegrator:
    """Identifiable scalar bottleneck for multidimensional bodily consequences.

    V1 allowed a trainable decoder bias and action offset to predict the almost
    constant targets without using Q.  V2 freezes the target-specific baseline
    from independent calibration, removes the action offset, initializes
    non-zero positive gains, and adds a pairwise ranking term.  Consequently,
    state-dependent variation must pass through Q.
    """

    def __init__(
        self,
        target_means: Sequence[float],
        lr: float = 0.004,
        l2: float = 1e-5,
        ranking_weight: float = 0.15,
        burden_weight: float = 0.50,
        ranking_margin: float = 0.10,
        ranking_temperature: float = 0.20,
    ):
        self.lr = float(lr)
        self.l2 = float(l2)
        self.ranking_weight = float(ranking_weight)
        self.burden_weight = float(burden_weight)
        self.ranking_margin = float(ranking_margin)
        self.ranking_temperature = float(ranking_temperature)
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.omega = np.zeros(len(Q_COMPONENT_NAMES), dtype=float)
        means = np.clip(np.asarray(target_means, dtype=float), 0.03, 0.97)
        self.baseline_logits = np.log(means / (1.0 - means))
        # softplus(1.25) ~= 1.50: Q has a usable gradient from the first update.
        self.decoder_g = np.full(len(FUTURE_TARGET_NAMES), 1.25, dtype=float)
        self.opt_omega = AdamArray(self.omega.shape)
        self.opt_g = AdamArray(self.decoder_g.shape)
        self.update_count = 0
        self.loss_ema = 0.0
        self.prediction_loss_ema = 0.0
        self.null_loss_ema = 0.0
        self.rank_loss_ema = 0.0
        self.burden_loss_ema = 0.0

    def weights(self) -> np.ndarray:
        return softmax(self.omega)

    def q_batch(self, z: np.ndarray) -> np.ndarray:
        z = np.asarray(z, dtype=float)
        return np.clip(z @ self.weights(), 0.0, 1.0)

    def q(self, z: Sequence[float]) -> float:
        return float(self.q_batch(np.asarray(z, dtype=float)[None, :])[0])

    def predict_batch(self, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        z = np.asarray(z, dtype=float)
        q = self.q_batch(z)
        gains = np.asarray(softplus(self.decoder_g), dtype=float) + 0.10
        logits = self.baseline_logits[None, :] + (q[:, None] - 0.50) * gains[None, :]
        return q, np.asarray(sigmoid(logits), dtype=float)

    def predict(self, z: Sequence[float], action: int = A_CONTINUE) -> Tuple[float, np.ndarray]:
        q, pred = self.predict_batch(np.asarray(z, dtype=float)[None, :])
        return float(q[0]), pred[0]

    @staticmethod
    def _prediction_loss_vector(pred: np.ndarray, y: np.ndarray) -> np.ndarray:
        mse = np.mean((pred[:, :4] - y[:, :4]) ** 2, axis=1)
        p = np.clip(pred[:, 4], 1e-8, 1.0 - 1e-8)
        bce = -(y[:, 4] * np.log(p) + (1.0 - y[:, 4]) * np.log(1.0 - p))
        return (4.0 * mse + bce) / 5.0

    def null_loss(self, target: Sequence[float] | np.ndarray) -> float:
        y = np.asarray(target, dtype=float)
        if y.ndim == 1:
            y = y[None, :]
        p = np.asarray(sigmoid(self.baseline_logits), dtype=float)[None, :]
        p = np.repeat(p, len(y), axis=0)
        return float(np.mean(self._prediction_loss_vector(p, y)))

    def loss(self, z: Sequence[float], action: int, target: Sequence[float]) -> float:
        _, pred = self.predict(z, action)
        return float(self._prediction_loss_vector(pred[None, :], np.asarray(target, dtype=float)[None, :])[0])

    def batch_update(self, z: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        z = np.asarray(z, dtype=float)
        y = np.asarray(y, dtype=float)
        n = len(z)
        if n < 4:
            return {"loss": float("nan"), "null_loss": float("nan"), "rank_loss": 0.0}
        alpha = self.weights()
        q, pred = self.predict_batch(z)
        gains = np.asarray(softplus(self.decoder_g), dtype=float) + 0.10
        gain_deriv = np.asarray(sigmoid(self.decoder_g), dtype=float)

        dlogit = np.zeros_like(pred)
        dlogit[:, :4] = (2.0 / (5.0 * n)) * (pred[:, :4] - y[:, :4]) * pred[:, :4] * (1.0 - pred[:, :4])
        dlogit[:, 4] = (1.0 / (5.0 * n)) * (pred[:, 4] - y[:, 4])
        d_q = np.sum(dlogit * gains[None, :], axis=1)
        burden = np.mean(y, axis=1)
        burden_loss = float(np.mean((q - burden) ** 2))
        d_q += (2.0 * self.burden_weight / n) * (q - burden)

        # Deterministic half-batch pairing avoids O(n^2) ranking cost.
        half = n // 2
        rank_loss = 0.0
        if half > 0:
            i1 = np.arange(half)
            i2 = np.arange(n - half, n)
            delta_b = burden[i1] - burden[i2]
            mask = np.abs(delta_b) >= self.ranking_margin
            if np.any(mask):
                a = i1[mask]
                b = i2[mask]
                sign = np.sign(delta_b[mask])
                margin = sign * (q[a] - q[b]) / self.ranking_temperature
                pair_loss = np.logaddexp(0.0, -margin)
                rank_loss = float(np.mean(pair_loss))
                coeff = (
                    -sign * sigmoid(-margin)
                    / self.ranking_temperature
                    * self.ranking_weight
                    / max(len(a), 1)
                )
                np.add.at(d_q, a, coeff)
                np.add.at(d_q, b, -coeff)

        d_omega = np.sum(d_q[:, None] * alpha[None, :] * (z - q[:, None]), axis=0)
        d_omega += 2.0 * self.l2 * self.omega
        d_g = np.sum(dlogit * (q[:, None] - 0.50) * gain_deriv[None, :], axis=0)
        d_omega = np.clip(d_omega, -2.0, 2.0)
        d_g = np.clip(d_g, -2.0, 2.0)

        self.update_count += 1
        t = self.update_count
        self.omega = self.opt_omega.step(self.omega, d_omega, t, self.lr, self.beta1, self.beta2)
        self.decoder_g = self.opt_g.step(self.decoder_g, d_g, t, self.lr, self.beta1, self.beta2)
        self.omega = np.clip(self.omega, -6.0, 6.0)
        self.decoder_g = np.clip(self.decoder_g, -3.0, 5.0)

        pred_loss = float(np.mean(self._prediction_loss_vector(pred, y)))
        null_loss = self.null_loss(y)
        total = pred_loss + self.ranking_weight * rank_loss + self.burden_weight * burden_loss
        self.loss_ema = 0.99 * self.loss_ema + 0.01 * total
        self.prediction_loss_ema = 0.99 * self.prediction_loss_ema + 0.01 * pred_loss
        self.null_loss_ema = 0.99 * self.null_loss_ema + 0.01 * null_loss
        self.rank_loss_ema = 0.99 * self.rank_loss_ema + 0.01 * rank_loss
        self.burden_loss_ema = 0.99 * self.burden_loss_ema + 0.01 * burden_loss
        return {"loss": total, "prediction_loss": pred_loss, "null_loss": null_loss, "rank_loss": rank_loss, "burden_loss": burden_loss}


@dataclass
class PendingQSample:
    start_step: int
    z: np.ndarray
    action: int
    row_index: int
    start_damage: float
    start_energy: float
    start_fatigue: float
    start_stability: float
    start_failed: bool
    future_failed: bool
    learn_allowed: bool
    prediction: np.ndarray
    discounted_outcomes: np.ndarray


class DelayedActionAgencyEstimator:
    """Capacity-matched prequential agency inference.

    For every candidate delay, a predictor receiving the true delayed action is
    compared with a predictor receiving an independently generated decoy action.
    Both models have exactly the same dimensionality and update rule.  Evidence
    is computed from pre-update prediction errors, preventing training leakage.
    """

    def __init__(
        self,
        max_delay: int = 7,
        lr: float = 0.018,
        decay: float = 0.9995,
        error_alpha: float = 0.025,
        seed: int = 0,
        prior_weight: float = AGENCY_PRIOR_WEIGHT,
        current_weight: float = AGENCY_CURRENT_WEIGHT,
        gain: float = AGENCY_GAIN,
    ):
        self.max_delay = int(max_delay)
        self.lr = float(lr)
        self.decay = float(decay)
        self.error_alpha = float(error_alpha)
        # Mixture of prequential (prior) and current predictive advantage, and
        # the logistic gain converting evidence into the agency score.  Exposed
        # so that the robustness battery can vary them; defaults reproduce the
        # confirmatory model exactly.
        self.prior_weight = float(prior_weight)
        self.current_weight = float(current_weight)
        self.gain = float(gain)
        self.output_dim = 6
        # Agency is defined primarily by predicted reafference.  Remaining
        # sensory dimensions constrain the model but must not numerically bury
        # the action-linked channel in a uniform mean-squared error.
        self.output_error_weights = np.array([0.35, 0.25, 0.10, 0.10, 0.10, 0.10], dtype=float)
        self.context_dim = len(EVENT_TYPES) + 4
        self.common_dim = self.output_dim + 4 + self.context_dim
        self.model_dim = self.common_dim + len(ACTIONS) + 1
        self.w_true = {d: np.zeros((self.output_dim, self.model_dim), dtype=float) for d in range(1, self.max_delay + 1)}
        self.w_decoy = {d: np.zeros((self.output_dim, self.model_dim), dtype=float) for d in range(1, self.max_delay + 1)}
        self.E_true = {d: 1.0 for d in range(1, self.max_delay + 1)}
        self.E_decoy = {d: 1.0 for d in range(1, self.max_delay + 1)}
        self.previous_consequence = np.zeros(self.output_dim, dtype=float)
        self.action_history: List[int] = [A_CONTINUE for _ in range(self.max_delay + 32)]
        self.decoy_history: List[int] = [A_AVOID for _ in range(self.max_delay + 32)]
        self.action_counts = np.ones(len(ACTIONS), dtype=float)
        self.rng = np.random.default_rng(int(seed))
        self.score_history: List[float] = []
        self.best_delay = 1
        self.ready_count = 0

    def push_action(self, action: int) -> None:
        action = int(action)
        self.action_counts[action] += 1.0
        probs = self.action_counts / np.sum(self.action_counts)
        decoy = int(self.rng.choice(ACTIONS, p=probs))
        self.action_history.append(action)
        self.decoy_history.append(decoy)
        if len(self.action_history) > self.max_delay + 64:
            self.action_history.pop(0)
            self.decoy_history.pop(0)

    def reset_evidence(self) -> None:
        self.E_true = {d: 1.0 for d in range(1, self.max_delay + 1)}
        self.E_decoy = {d: 1.0 for d in range(1, self.max_delay + 1)}
        self.score_history.clear()
        self.ready_count = 0

    def observe(
        self,
        consequence: Sequence[float],
        body: Sequence[float],
        learn: bool = True,
        action_history_includes_current: bool = True,
        context: Optional[Sequence[float]] = None,
    ) -> Dict[str, float]:
        y = np.asarray(consequence, dtype=float)
        b = np.asarray(body, dtype=float)
        c = np.zeros(self.context_dim, dtype=float) if context is None else np.asarray(context, dtype=float)
        if c.shape != (self.context_dim,):
            raise ValueError(f"agency context must have shape {(self.context_dim,)}, got {c.shape}")
        common = np.concatenate([self.previous_consequence, b, c])
        current: Dict[int, Tuple[float, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
        # Synthetic blocks push action[t] before observing y[t], whereas the
        # embodied loop observes event[t] before choosing action[t].  Explicitly
        # account for that call-order difference so delay d has one meaning.
        offset = 1 if action_history_includes_current else 0
        for d in range(1, self.max_delay + 1):
            a_true = self.action_history[-(d + offset)]
            a_decoy = self.decoy_history[-(d + offset)]
            x_true = np.concatenate([common, action_onehot(a_true), np.ones(1)])
            x_decoy = np.concatenate([common, action_onehot(a_decoy), np.ones(1)])
            pred_true = self.w_true[d] @ x_true
            pred_decoy = self.w_decoy[d] @ x_decoy
            eps_true_vec = y - pred_true
            eps_decoy_vec = y - pred_decoy
            err_true = float(np.dot(self.output_error_weights, eps_true_vec * eps_true_vec))
            err_decoy = float(np.dot(self.output_error_weights, eps_decoy_vec * eps_decoy_vec))
            current[d] = (err_true, err_decoy, eps_true_vec, eps_decoy_vec, x_true, x_decoy)

        # Delay selection uses only prior prequential error summaries.
        prior_adv = {
            d: (self.E_decoy[d] - self.E_true[d]) / (self.E_decoy[d] + self.E_true[d] + 1e-9)
            for d in range(1, self.max_delay + 1)
        }
        best_d = int(max(prior_adv, key=prior_adv.get))
        err_true, err_decoy, _, _, _, _ = current[best_d]
        current_adv = (err_decoy - err_true) / (err_decoy + err_true + 1e-9)
        evidence = self.prior_weight * prior_adv[best_d] + self.current_weight * current_adv
        score = float(sigmoid(self.gain * evidence))

        if learn:
            for d, (et, ed, eps_t, eps_d, xt, xd) in current.items():
                self.w_true[d] = self.decay * (self.w_true[d] + self.lr * np.outer(eps_t, xt))
                self.w_decoy[d] = self.decay * (self.w_decoy[d] + self.lr * np.outer(eps_d, xd))
                self.E_true[d] = (1.0 - self.error_alpha) * self.E_true[d] + self.error_alpha * et
                self.E_decoy[d] = (1.0 - self.error_alpha) * self.E_decoy[d] + self.error_alpha * ed

        self.best_delay = int(max(
            range(1, self.max_delay + 1),
            key=lambda d: (self.E_decoy[d] - self.E_true[d]) / (self.E_decoy[d] + self.E_true[d] + 1e-9),
        ))
        self.previous_consequence = y.copy()
        self.score_history.append(score)
        if len(self.score_history) > 1024:
            self.score_history.pop(0)
        self.ready_count += 1
        return {
            "agency_score": score,
            "agency_evidence": float(evidence),
            "best_delay": float(self.best_delay),
            "true_error": float(err_true),
            "decoy_error": float(err_decoy),
            # Backward-compatible names used by the existing output code.
            "exo_error": float(err_decoy),
            "action_error": float(err_true),
        }


# =============================================================================
# Closed-loop environment
# =============================================================================


class ClosedLoopEnvironment:
    def __init__(
        self,
        cfg: EnvironmentConfig,
        morphology: Morphology,
        mode: str,
        rng: np.random.Generator,
    ):
        self.cfg = cfg
        self.morph = morphology
        self.mode = mode
        self.rng = rng
        self.previous_event = "rest"
        self.action_history: List[int] = [A_CONTINUE for _ in range(max(20, cfg.true_delay + 12))]
        self.latent_noise = 0.0
        self.t = 0

    def push_action(self, action: int) -> None:
        self.action_history.append(int(action))
        if len(self.action_history) > 64:
            self.action_history.pop(0)

    def delayed_action(self, delay_override: Optional[int] = None) -> int:
        delay = self.cfg.true_delay if delay_override is None else int(delay_override)
        delay = int(np.clip(delay, 1, len(self.action_history)))
        # The current action has not yet been appended when an event is drawn;
        # delay=1 therefore refers to the most recently completed action.
        return int(self.action_history[-delay])

    @staticmethod
    def preceding_modifier(previous: str, subsequent: str) -> float:
        special = {
            ("jump", "landing"): 6.00,
            ("slip", "rest"): 2.00,
            ("collision", "rest"): 2.00,
            ("slip", "brake"): 1.50,
            ("collision", "brake"): 1.50,
            ("rest", "walk"): 1.40,
            ("rest", "rest"): 0.80,
        }
        return float(special.get((previous, subsequent), 1.0))

    @staticmethod
    def action_modifier(action: int, event_type: str) -> float:
        table = {
            A_CONTINUE: {
                "rest": 0.75, "walk": 1.25, "slope": 1.10, "slip": 1.30,
                "jump": 1.10, "landing": 1.15, "collision": 1.30, "brake": 0.60,
            },
            A_CAUTIOUS: {
                "rest": 1.00, "walk": 1.10, "slope": 1.00, "slip": 0.65,
                "jump": 0.75, "landing": 0.80, "collision": 0.70, "brake": 1.15,
            },
            A_AVOID: {
                "rest": 1.60, "walk": 0.75, "slope": 0.65, "slip": 0.35,
                "jump": 0.35, "landing": 0.60, "collision": 0.40, "brake": 2.20,
            },
        }
        return float(table[int(action)][event_type])

    def body_modifier(self, event_type: str, state: AgentState) -> float:
        if self.mode == "exogenous":
            return 1.0
        if event_type == "slip":
            return float(math.exp(
                1.20 * state.fatigue
                + 1.40 * (1.0 - state.stability)
                + 0.60 * (1.0 - state.energy)
                - 2.00 * (self.morph.traction_tolerance - 1.0)
            ))
        if event_type == "collision":
            return float(math.exp(
                0.80 * state.fatigue
                + 0.80 * (1.0 - state.stability)
                + 0.50 * (1.0 - state.integrity)
                + 0.50 * (self.morph.mass - 1.0)
            ))
        if event_type == "landing":
            return float(math.exp(
                0.60 * state.fatigue
                + 0.90 * (1.0 - state.stability)
                + 0.50 * (self.morph.mass - 1.0)
            ))
        if event_type == "rest":
            return float(math.exp(0.50 * (1.0 - state.energy) + 0.50 * state.fatigue))
        return 1.0

    def event_probabilities(self, state: AgentState, forced_delayed_action: Optional[int] = None) -> np.ndarray:
        action = self.delayed_action() if forced_delayed_action is None else int(forced_delayed_action)
        weights = []
        for event_type, base_p in zip(EVENT_TYPES, BASE_EVENT_PROBS):
            transition = self.preceding_modifier(self.previous_event, event_type)
            action_mult = 1.0 if self.mode in {"action_open", "exogenous"} else self.action_modifier(action, event_type)
            body_mult = self.body_modifier(event_type, state)
            hazard_mult = self.cfg.hazard_multiplier if event_type in HAZARD_EVENTS else 1.0
            weights.append(base_p * transition * action_mult * body_mult * hazard_mult)
        w = np.asarray(weights, dtype=float)
        w = np.clip(w, 1e-12, 1e12)
        return w / np.sum(w)

    def _baseline_intensity(self, event_type: str, friction: float, slope: float) -> float:
        r = self.rng
        if event_type == "rest":
            value = r.uniform(0.00, 0.08)
        elif event_type == "walk":
            value = r.uniform(0.18, 1.10)
        elif event_type == "slope":
            value = r.uniform(0.45, 2.20) + 1.80 * slope
        elif event_type == "slip":
            value = r.uniform(0.85, 3.65) * (1.18 - 0.55 * friction)
        elif event_type == "jump":
            value = r.uniform(0.75, 2.80)
        elif event_type == "landing":
            value = r.uniform(1.05, 4.40)
        elif event_type == "collision":
            value = r.uniform(0.90, 4.85)
        elif event_type == "brake":
            value = r.uniform(0.45, 3.10) * (1.0 - 0.25 * friction)
        else:
            value = r.uniform(0.1, 1.0)
        return float(value)

    @staticmethod
    def action_intensity_effect(action: int) -> float:
        return {A_CONTINUE: 0.10, A_CAUTIOUS: -0.18, A_AVOID: -0.35}[int(action)]

    def generate_event(
        self,
        state: AgentState,
        forced_source: Optional[str] = None,
        forced_delay: Optional[int] = None,
        forced_mismatch_factor: Optional[float] = None,
        fixed_event_type: Optional[str] = None,
        fixed_intensity: Optional[float] = None,
        fixed_friction: Optional[float] = None,
        fixed_slope: Optional[float] = None,
        fixed_affordance: Optional[float] = None,
    ) -> Event:
        probs = self.event_probabilities(state)
        event_type = fixed_event_type or str(self.rng.choice(EVENT_TYPES, p=probs))
        friction = float(fixed_friction if fixed_friction is not None else self.rng.choice(
            [0.06, 0.26, 0.42, 0.72], p=[0.22, 0.18, 0.42, 0.18]
        ))
        slope = float(fixed_slope if fixed_slope is not None else self.rng.uniform(0.0, 0.35))
        affordance = float(fixed_affordance if fixed_affordance is not None else self.rng.uniform(0.18, 1.00))
        baseline = self._baseline_intensity(event_type, friction, slope)
        if fixed_intensity is not None:
            baseline = float(fixed_intensity / max(self.cfg.risk_scale, 1e-9))

        source = forced_source or str(self.rng.choice(
            ["matched_self", "mismatched_self", "external", "external_coincidence"],
            p=[0.42, 0.18, 0.28, 0.12],
        ))
        delay = self.cfg.true_delay if forced_delay is None else int(forced_delay)
        delayed_action = self.delayed_action(delay)
        motor_effect = self.action_intensity_effect(delayed_action) * affordance * baseline
        motor_contribution = 0.0
        exogenous_contribution = 0.0
        self_generated = 0
        mismatch = 0
        mismatch_factor = 0.0
        # A noisy reafferent channel is an observed sensory consequence, not a
        # source label.  Its relation to delayed action must be learned.  The
        # same channel can be mimicked by an external coincidence.
        expected_reafference = self.action_intensity_effect(delayed_action) * affordance
        reafferent_noise = float(self.rng.normal(0.0, 0.06))
        reafferent_signal = reafferent_noise
        if source == "matched_self":
            self_generated = 1
            motor_contribution = motor_effect
            reafferent_signal = expected_reafference + reafferent_noise
        elif source == "mismatched_self":
            self_generated = 1
            mismatch = 1
            factor = forced_mismatch_factor
            if factor is None:
                factor = float(self.rng.choice([0.0, -0.50, 0.50, 1.50], p=[0.25, 0.20, 0.25, 0.30]))
            mismatch_factor = float(factor)
            motor_contribution = motor_effect * mismatch_factor
            reafferent_signal = expected_reafference * mismatch_factor + reafferent_noise
        elif source == "external_coincidence":
            coincidence_factor = float(self.rng.choice([-1.0, 1.0])) * float(self.rng.uniform(0.70, 1.30))
            exogenous_contribution = coincidence_factor * abs(motor_effect)
            reafferent_signal = coincidence_factor * abs(expected_reafference) + reafferent_noise

        self.latent_noise = 0.90 * self.latent_noise + float(self.rng.normal(0.0, 0.02))
        noise = float(self.rng.normal(0.0, 0.02) + 0.05 * self.latent_noise)
        intensity = max(0.0, self.cfg.risk_scale * baseline + motor_contribution + exogenous_contribution + noise)
        if fixed_intensity is not None:
            intensity = max(0.0, float(fixed_intensity) + motor_contribution + exogenous_contribution + noise)
        physical_risk = float(sigmoid(intensity - 2.35 + 0.55 * (0.28 - friction) + 0.35 * slope))
        self.previous_event = event_type
        self.t += 1
        return Event(
            event_type=event_type,
            baseline_intensity=baseline,
            intensity=float(intensity),
            friction=friction,
            slope=slope,
            affordance=affordance,
            physical_risk=physical_risk,
            true_delay=int(delay),
            source_class=source,
            self_generated=int(self_generated),
            mismatch=int(mismatch),
            motor_contribution=float(motor_contribution),
            exogenous_contribution=float(exogenous_contribution),
            reafferent_signal=float(reafferent_signal),
        )


# =============================================================================
# Embodied agent
# =============================================================================


class EmbodiedAgent:
    def __init__(
        self,
        cfg: EnvironmentConfig,
        morphology: Morphology,
        condition: Condition,
        seed: int,
        beta: float = 8.0,
        q_lr_multiplier: float = 1.0,
        predictor_lr_multiplier: float = 1.0,
        memory_decay_multiplier: float = 1.0,
        mitigation_multiplier: float = 1.0,
        vulnerability_weights: Optional[Sequence[float]] = None,
        avoidance_coefficients: Optional[Dict[str, float]] = None,
        agency_prior_weight: float = AGENCY_PRIOR_WEIGHT,
        agency_current_weight: float = AGENCY_CURRENT_WEIGHT,
        agency_gain: float = AGENCY_GAIN,
    ):
        self.cfg = cfg
        self.morph = morphology
        self.condition = condition
        self.rng = np.random.default_rng(seed)
        self.state = AgentState()
        self.beta = float(beta)
        self.mitigation_multiplier = float(mitigation_multiplier)
        self.memory_decay_multiplier = float(memory_decay_multiplier)
        self.vulnerability_weights = np.array(
            VULNERABILITY_WEIGHTS if vulnerability_weights is None else vulnerability_weights,
            dtype=float,
        )
        self.avoidance_coefficients = dict(
            AVOIDANCE_COEFFICIENTS if avoidance_coefficients is None else avoidance_coefficients
        )
        n_features = 21
        self.danger_model = OnlineLogistic(n_features, 0.040 * predictor_lr_multiplier, 1e-4, -0.2, self.rng)
        self.comfort_model = OnlineLogistic(n_features, 0.035 * predictor_lr_multiplier, 1e-4, 0.2, self.rng)
        self.action_model = OnlineLogistic(n_features, 0.035 * predictor_lr_multiplier, 1e-4, 0.1, self.rng)
        self.q_integrator = QualitativeValenceIntegrator(
            target_means=cfg.target_means(), lr=0.004 * q_lr_multiplier, l2=1e-5
        )
        self.q_replay = QReplayBuffer(capacity=2048)
        self.q_update_interval = 4
        self.q_batch_size = 64
        self._matured_since_update = 0
        self.agency = DelayedActionAgencyEstimator(
            max_delay=7, seed=seed ^ 0xA631,
            prior_weight=agency_prior_weight, current_weight=agency_current_weight,
            gain=agency_gain,
        )
        self.pending_q: List[PendingQSample] = []
        self.q_history: List[float] = []
        self.memory_history: List[Tuple[float, float, float]] = []
        self.agency_history: List[float] = []
        self.last_deltas = np.zeros(4, dtype=float)  # energy, fatigue, stability, damage
        # Only the most recent horizon is retained.  V1 updated every pending
        # Q sample at every step (O(H) Python work per step).  V2 records one
        # four-vector per step and evaluates only the sample that matures.
        self.outcome_cost_history: Dict[int, np.ndarray] = {}
        self.current_step = 0

    def memory_view(self) -> Tuple[float, float, float]:
        pm, dm, cm = self.state.pain_memory, self.state.danger_memory, self.state.comfort_memory
        mode = self.condition.memory_mode
        if mode == "zero":
            return 0.0, 0.0, 0.55
        if mode == "pain_zero":
            return 0.0, dm, cm
        if mode == "danger_zero":
            return pm, 0.0, cm
        if mode == "comfort_zero":
            return pm, dm, 0.55
        if mode == "time_shuffle" and self.memory_history:
            return self.memory_history[int(self.rng.integers(0, len(self.memory_history)))]
        return pm, dm, cm

    def raw_impact(self, event: Event, effective_intensity: Optional[float] = None) -> float:
        u = event.intensity if effective_intensity is None else float(effective_intensity)
        if event.event_type in {"collision", "landing", "slip"}:
            return u
        if event.event_type in {"slope", "brake"}:
            return 0.55 * u
        if event.event_type == "jump":
            return 0.35 * u
        return 0.12 * u

    def vulnerability(self, state: Optional[AgentState] = None) -> float:
        s = self.state if state is None else state
        w = self.vulnerability_weights
        return float(
            w[0] * (1.0 - s.integrity)
            + w[1] * s.fatigue
            + w[2] * (1.0 - s.stability)
            + w[3] * (1.0 - s.energy)
        )

    def immediate_pain(self, event: Event, effective_intensity: Optional[float] = None) -> float:
        impact = self.morph.mass * self.raw_impact(event, effective_intensity)
        return float(sigmoid(1.25 * (impact - 2.20) + 1.25 * self.vulnerability()))

    def expected_event_intensity(self, event: Event) -> float:
        if event.event_type == "rest":
            base = 0.04
        elif event.event_type == "walk":
            base = 0.64
        elif event.event_type == "slope":
            base = 1.325 + 1.80 * event.slope
        elif event.event_type == "slip":
            base = 2.25 * (1.18 - 0.55 * event.friction)
        elif event.event_type == "jump":
            base = 1.775
        elif event.event_type == "landing":
            base = 2.725
        elif event.event_type == "collision":
            base = 2.875
        elif event.event_type == "brake":
            base = 1.775 * (1.0 - 0.25 * event.friction)
        else:
            base = 0.5
        return float(self.cfg.risk_scale * base)

    def agency_context(self, event: Event) -> np.ndarray:
        onehot = np.zeros(len(EVENT_TYPES), dtype=float)
        onehot[EVENT_INDEX[event.event_type]] = 1.0
        return np.concatenate([
            onehot,
            np.array([
                np.clip(self.expected_event_intensity(event) / 5.5, 0.0, 1.5),
                event.friction,
                event.slope / 0.35,
                event.affordance,
            ], dtype=float),
        ])

    def agency_consequence_vector(self, event: Event, pre_action_pain: float) -> np.ndarray:
        """Current sensory consequence only.

        V1 mixed the previous event's body deltas into the consequence vector
        while scoring the current event's causal source.  This temporal mismatch
        made trial-level agency unidentifiable.  External context is supplied
        separately and identically to the true-action and decoy models.
        """
        expected = self.expected_event_intensity(event)
        residual = float(event.intensity - expected)
        signed_residual = float(np.tanh(residual / 0.75))
        return np.array([
            np.clip(event.reafferent_signal, -1.0, 1.0),
            np.clip(abs(event.reafferent_signal) / 0.50, 0.0, 1.5),
            signed_residual,
            np.clip(event.intensity / 5.5, 0.0, 1.5),
            event.physical_risk,
            pre_action_pain,
        ], dtype=float)

    def agency_step(self, event: Event, pre_action_pain: float, learn: bool) -> Dict[str, float]:
        body = [self.state.integrity, self.state.energy, self.state.fatigue, self.state.stability]
        raw = self.agency.observe(
            self.agency_consequence_vector(event, pre_action_pain), body,
            learn=learn, action_history_includes_current=False,
            context=self.agency_context(event),
        )
        used = float(raw["agency_score"])
        if self.condition.agency_mode == "zero":
            used = 0.0
        elif self.condition.agency_mode == "time_shuffle" and self.agency_history:
            used = float(self.agency_history[int(self.rng.integers(0, len(self.agency_history)))])
        self.agency_history.append(float(raw["agency_score"]))
        if len(self.agency_history) > 1024:
            self.agency_history.pop(0)
        raw["agency_score_used"] = used
        return raw

    def predictor_features(self, event: Event, agency_score: float) -> np.ndarray:
        pm, dm, cm = self.memory_view()
        onehot = np.zeros(len(EVENT_TYPES), dtype=float)
        onehot[EVENT_INDEX[event.event_type]] = 1.0
        return np.concatenate([
            onehot,
            np.array([
                event.intensity / 5.5,
                event.friction,
                event.slope / 0.35,
                event.affordance,
                event.physical_risk,
                self.state.integrity,
                self.state.energy,
                self.state.fatigue,
                self.state.stability,
                pm,
                dm,
                cm,
                agency_score,
            ], dtype=float),
        ])

    def components(self, event: Event, pain: float, agency_score: float) -> Dict[str, Any]:
        x = self.predictor_features(event, agency_score)
        danger = self.danger_model.predict(x)
        comfort = self.comfort_model.predict(x)
        action_possibility = self.action_model.predict(x)
        controllability = action_possibility * (0.65 + 0.35 * agency_score)
        pm, dm, cm = self.memory_view()
        z = np.array([
            danger,
            pain,
            1.0 - comfort,
            1.0 - action_possibility,
            dm,
            pm,
            1.0 - cm,
            self.state.fatigue,
            1.0 - self.state.stability,
            1.0 - self.state.integrity,
            1.0 - self.state.energy,
            1.0 - controllability,
        ], dtype=float)
        q, q_prediction = self.q_integrator.predict(z, A_CONTINUE)
        c = self.avoidance_coefficients
        avoidance_pressure = float(sigmoid(
            c["danger"] * danger + c["pain"] * pain + c["comfort"] * comfort
            + c["controllability"] * controllability + c["danger_memory"] * dm
            + c["integrity_loss"] * (1.0 - self.state.integrity) + c["bias"]
        ))
        fixed_q = clip01(
            0.23 * danger + 0.20 * pain + 0.20 * avoidance_pressure
            + 0.12 * (1.0 - action_possibility) + 0.07 * (1.0 - comfort)
            + 0.08 * dm + 0.05 * pm + 0.03 * self.state.fatigue
            + 0.02 * (1.0 - self.state.stability)
        )
        return {
            "x": x,
            "danger": danger,
            "comfort": comfort,
            "action_possibility": action_possibility,
            "controllability": controllability,
            "avoidance_pressure": avoidance_pressure,
            "z": z,
            "q": q,
            "fixed_q": fixed_q,
            "q_prediction_continue": q_prediction,
        }

    def policy_signal(self, comp: Dict[str, Any], event: Event, override: Optional[str] = None) -> float:
        mode = self.condition.signal_mode if override is None else override
        if mode in {"learned_q", "full_q"}:
            return float(comp["q"])
        if mode == "fixed_q":
            return float(comp["fixed_q"])
        if mode == "pain":
            return float(comp["pain"])
        if mode == "danger":
            return float(comp["danger"])
        if mode == "physical_risk":
            return float(event.physical_risk)
        if mode == "body":
            return float(1.0 - self.state.viability())
        if mode == "shuffled_q":
            if self.q_history:
                return float(self.q_history[int(self.rng.integers(0, len(self.q_history)))])
            return float(comp["q"])
        if mode in {"zero", "no_q"}:
            return 0.0
        raise ValueError(f"Unknown policy signal: {mode}")

    def prospective_action_cost(self, event: Event, action: int) -> float:
        energy = BASE_ENERGY_COST[action] + 0.008 * event.intensity / max(self.morph.energy_capacity, 0.2)
        fatigue = BASE_FATIGUE_COST[action] + 0.004 * event.intensity
        return clip01((energy + fatigue) / 0.10)

    def choose_action(
        self,
        event: Event,
        comp: Dict[str, Any],
        signal_override: Optional[str] = None,
        forced_action: Optional[int] = None,
    ) -> Tuple[int, np.ndarray, float]:
        signal = self.policy_signal(comp, event, override=signal_override)
        costs = []
        for action in ACTIONS:
            mitigation = np.clip(MITIGATION[action] * self.mitigation_multiplier, 0.0, 0.95)
            j = self.prospective_action_cost(event, action) + signal * (1.0 - mitigation * event.affordance)
            costs.append(j)
        probabilities = softmax(-self.beta * np.asarray(costs, dtype=float))
        action = int(forced_action) if forced_action is not None else int(self.rng.choice(ACTIONS, p=probabilities))
        return action, probabilities, signal

    def counterfactual_continuation_costs(self, event: Event) -> np.ndarray:
        """Pre-action bodily burden under no regulatory mitigation.

        This anchors Q to the significance of the current event before the
        selected policy can suppress its observed consequence.
        """
        s = self.state
        action = A_CONTINUE
        effective_intensity = event.intensity
        effective_risk = float(sigmoid(
            effective_intensity - 2.45 + 0.55 * (0.30 - event.friction) + 0.28 * event.slope
        ))
        vulnerability = self.vulnerability(s)
        event_load_factor = {
            "rest": 0.05, "walk": 0.12, "slope": 0.35, "slip": 1.00,
            "jump": 0.45, "landing": 1.15, "collision": 1.25, "brake": 0.55,
        }[event.event_type]
        intensity_factor = 0.55 + 0.45 * min(effective_intensity / 3.5, 1.75)
        mechanical_load = event_load_factor * effective_risk * intensity_factor * self.morph.mass
        pressure = max(0.0, mechanical_load + 0.75 * vulnerability - 0.32)
        damage_cost = self.cfg.damage_scale * pressure * pressure / max(self.morph.damage_tolerance, 0.2)
        if event.event_type == "rest":
            energy_cost = 0.0
            fatigue_cost = 0.0
            stability_cost = 0.0
        else:
            energy_cost = max(0.0, 0.20 * (BASE_ENERGY_COST[action] + 0.008 * effective_intensity)
                              / max(self.morph.energy_capacity, 0.2) - 0.003)
            fatigue_cost = max(0.0, 0.20 * (BASE_FATIGUE_COST[action] + 0.004 * effective_intensity) - 0.002)
            stability_cost = max(0.0, 0.20 * (0.007 + 0.025 * effective_risk))
        return np.array([damage_cost, energy_cost, fatigue_cost, stability_cost], dtype=float)

    def apply_event(self, event: Event, action: int) -> Dict[str, float]:
        s = self.state
        pre = copy.deepcopy(s)
        mitigation = np.clip(MITIGATION[action] * self.mitigation_multiplier, 0.0, 0.95)
        effective_intensity = event.intensity * (1.0 - mitigation * event.affordance)
        effective_risk = float(sigmoid(effective_intensity - 2.45 + 0.55 * (0.30 - event.friction) + 0.28 * event.slope))
        vulnerability = self.vulnerability(pre)
        # V1 allowed mass to change pain and event probability but not actual
        # tissue damage.  V2 couples morphology to mechanical consequences.
        event_load_factor = {
            "rest": 0.05, "walk": 0.12, "slope": 0.35, "slip": 1.00,
            "jump": 0.45, "landing": 1.15, "collision": 1.25, "brake": 0.55,
        }[event.event_type]
        intensity_factor = 0.55 + 0.45 * min(effective_intensity / 3.5, 1.75)
        mechanical_load = event_load_factor * effective_risk * intensity_factor * self.morph.mass
        raw_damage_pressure = max(0.0, mechanical_load + 0.75 * vulnerability - 0.32)
        damage_increment = self.cfg.damage_scale * (raw_damage_pressure ** 2) / max(self.morph.damage_tolerance, 0.2)
        pain_after = self.immediate_pain(event, effective_intensity=effective_intensity)

        if event.event_type == "rest":
            s.energy = clip01(s.energy + 0.030 * self.morph.energy_capacity * self.cfg.recovery_multiplier)
            s.fatigue = clip01(s.fatigue - 0.026 * self.cfg.recovery_multiplier)
            s.stability = clip01(s.stability + 0.023 * self.cfg.recovery_multiplier)
        else:
            # Physiological state changes are slower than the discrete event clock.
            # The 0.20 scale prevents deterministic energy/stability collapse while
            # preserving the specified relative costs among the three actions.
            energy_cost = 0.20 * (BASE_ENERGY_COST[action] + 0.008 * effective_intensity) / max(self.morph.energy_capacity, 0.2)
            fatigue_gain = 0.20 * (BASE_FATIGUE_COST[action] + 0.004 * effective_intensity)
            stability_loss = 0.20 * (0.007 + 0.025 * effective_risk)
            s.energy = clip01(s.energy - energy_cost + 0.003)
            s.fatigue = clip01(s.fatigue + fatigue_gain - 0.002)
            s.stability = clip01(s.stability - stability_loss + 0.012 * mitigation)

        s.damage = clip01(s.damage + damage_increment)
        s.integrity = clip01(1.0 - s.damage)
        pm_decay = np.clip(0.940 * self.memory_decay_multiplier, 0.80, 0.999)
        dm_decay = np.clip(0.945 * self.memory_decay_multiplier, 0.80, 0.999)
        cm_decay = np.clip(0.945 * self.memory_decay_multiplier, 0.80, 0.999)
        s.pain_memory = clip01(pm_decay * s.pain_memory + (1.0 - pm_decay) * pain_after)
        s.danger_memory = clip01(dm_decay * s.danger_memory + (1.0 - dm_decay) * effective_risk)
        comfort_observed = float(
            damage_increment <= self.cfg.damage_target_threshold
            and pain_after < 0.35
            and s.energy > 0.45
            and s.stability > 0.55
            and s.fatigue < 0.60
        )
        s.comfort_memory = clip01(cm_decay * s.comfort_memory + (1.0 - cm_decay) * comfort_observed)
        if self.condition.memory_mode == "zero":
            s.pain_memory, s.danger_memory, s.comfort_memory = 0.0, 0.0, 0.55
        elif self.condition.memory_mode == "pain_zero":
            s.pain_memory = 0.0
        elif self.condition.memory_mode == "danger_zero":
            s.danger_memory = 0.0
        elif self.condition.memory_mode == "comfort_zero":
            s.comfort_memory = 0.55
        s.update_failure()

        self.last_deltas = np.array([
            s.energy - pre.energy,
            s.fatigue - pre.fatigue,
            s.stability - pre.stability,
            s.damage - pre.damage,
        ], dtype=float)
        self.memory_history.append((s.pain_memory, s.danger_memory, s.comfort_memory))
        if len(self.memory_history) > 1024:
            self.memory_history.pop(0)
        return {
            "effective_intensity": float(effective_intensity),
            "effective_risk": effective_risk,
            "damage_increment": float(damage_increment),
            "pain_after": pain_after,
            "energy_change": float(s.energy - pre.energy),
            "fatigue_change": float(s.fatigue - pre.fatigue),
            "stability_change": float(s.stability - pre.stability),
            "post_integrity": s.integrity,
            "post_energy": s.energy,
            "post_fatigue": s.fatigue,
            "post_stability": s.stability,
            "post_damage": s.damage,
            "post_viability": s.viability(),
            "failure": float(s.ever_failed),
        }

    def update_component_predictors(
        self,
        comp: Dict[str, Any],
        event: Event,
        outcome: Dict[str, float],
        action: int,
        learn: bool,
    ) -> None:
        if not learn:
            return
        x = comp["x"]
        danger_target = float(
            outcome["damage_increment"] > self.cfg.damage_target_threshold
            or outcome["pain_after"] > 0.66
        )
        comfort_target = float(
            outcome["damage_increment"] <= self.cfg.damage_target_threshold
            and outcome["pain_after"] < 0.35
            and outcome["post_energy"] > 0.45
            and outcome["post_stability"] > 0.55
            and outcome["post_fatigue"] < 0.60
        )
        unmitigated = event.physical_risk
        action_target = float(
            outcome["failure"] < 0.5
            and (
                outcome["effective_risk"] < unmitigated
                or outcome["effective_risk"] < 0.55
            )
        )
        self.danger_model.update(x, danger_target)
        self.comfort_model.update(x, comfort_target)
        self.action_model.update(x, action_target)

    def enqueue_q_sample(
        self,
        step: int,
        comp: Dict[str, Any],
        event: Event,
        action: int,
        row_index: int,
        learn_allowed: bool,
    ) -> None:
        _, pred = self.q_integrator.predict(comp["z"], action)
        self.pending_q.append(PendingQSample(
            start_step=step,
            z=np.asarray(comp["z"], dtype=float).copy(),
            action=int(action),
            row_index=int(row_index),
            start_damage=float(self.state.damage),
            start_energy=float(self.state.energy),
            start_fatigue=float(self.state.fatigue),
            start_stability=float(self.state.stability),
            start_failed=bool(self.state.ever_failed),
            future_failed=False,
            learn_allowed=bool(learn_allowed),
            prediction=pred.copy(),
            discounted_outcomes=self.counterfactual_continuation_costs(event),
        ))

    def _normalize_future_target(self, raw: np.ndarray) -> np.ndarray:
        norms = np.maximum(self.cfg.target_normalizers(), 1e-9)
        out = np.asarray(raw, dtype=float).copy()
        out[:4] = np.clip(out[:4] / norms[:4], 0.0, 1.0)
        out[4] = clip01(out[4])
        return out

    def mature_q_samples(
        self,
        step: int,
        rows: List[Dict[str, Any]],
        outcome: Dict[str, float],
        learn: bool,
    ) -> None:
        """Mature Q samples without repeatedly updating the whole horizon.

        The previous implementation added the current outcome to every pending
        sample at every step.  The numerical target was correct but the Python
        loop scaled as steps × horizon × episodes.  Here the current cost is
        stored once and only newly mature samples are evaluated.
        """
        actual_cost = np.array([
            max(0.0, outcome["damage_increment"]),
            max(0.0, -outcome["energy_change"]),
            max(0.0, outcome["fatigue_change"]),
            max(0.0, -outcome["stability_change"]),
        ], dtype=float)
        step = int(step)
        self.outcome_cost_history[step] = actual_cost
        discount = 0.92
        horizon = int(self.cfg.horizon)

        remaining: List[PendingQSample] = []
        for sample in self.pending_q:
            age = step - int(sample.start_step)
            if age < horizon - 1:
                remaining.append(sample)
                continue

            future_steps = range(int(sample.start_step) + 1, step + 1)
            costs = [self.outcome_cost_history.get(k) for k in future_steps]
            valid = [(k, c) for k, c in zip(future_steps, costs) if c is not None]
            discounted = sample.discounted_outcomes.astype(float, copy=True)
            if valid:
                ages = np.asarray([k - sample.start_step for k, _ in valid], dtype=float)
                matrix = np.vstack([c for _, c in valid])
                discounted += np.sum((discount ** ages)[:, None] * matrix, axis=0)

            raw_target = np.r_[
                discounted,
                float(self.state.ever_failed and not sample.start_failed),
            ]
            target = self._normalize_future_target(raw_target)
            loss_before = self.q_integrator.loss(sample.z, sample.action, target)
            null_loss = self.q_integrator.null_loss(target)
            if learn and self.condition.train_q and sample.learn_allowed:
                self.q_replay.add(sample.z, target)
                self._matured_since_update += 1
                if self._matured_since_update >= self.q_update_interval and self.q_replay.size >= 32:
                    bz, by = self.q_replay.sample(self.rng, self.q_batch_size)
                    self.q_integrator.batch_update(bz, by)
                    self._matured_since_update = 0
            if 0 <= sample.row_index < len(rows):
                rows[sample.row_index].update({
                    "future_damage_increase": target[0],
                    "future_energy_loss": target[1],
                    "future_fatigue_increase": target[2],
                    "future_stability_loss": target[3],
                    "future_failure": target[4],
                    "future_compromise": float(np.mean(target)),
                    "q_pred_damage": sample.prediction[0],
                    "q_pred_energy": sample.prediction[1],
                    "q_pred_fatigue": sample.prediction[2],
                    "q_pred_stability": sample.prediction[3],
                    "q_pred_failure": sample.prediction[4],
                    "q_prediction_loss": loss_before,
                    "q_null_prediction_loss": null_loss,
                    "q_prediction_improvement": null_loss - loss_before,
                })
        self.pending_q = remaining

        # No future sample can require costs older than this cutoff.
        cutoff = step - horizon - 2
        if cutoff >= 0:
            for old_step in [k for k in self.outcome_cost_history if k < cutoff]:
                del self.outcome_cost_history[old_step]

    def step(
        self,
        env: ClosedLoopEnvironment,
        step: int,
        rows: List[Dict[str, Any]],
        learn: bool = True,
        forced_action: Optional[int] = None,
        signal_override: Optional[str] = None,
        heldout_event: Optional[str] = None,
        event_override: Optional[Dict[str, Any]] = None,
        track_q_target: bool = True,
    ) -> Dict[str, Any]:
        event_override = event_override or {}
        event = env.generate_event(self.state, **event_override)
        pre_pain = self.immediate_pain(event)
        agency = self.agency_step(event, pre_pain, learn=learn)
        comp = self.components(event, pre_pain, float(agency["agency_score_used"]))
        comp["pain"] = pre_pain
        action, action_probs, signal = self.choose_action(
            event, comp, signal_override=signal_override, forced_action=forced_action
        )
        row_index = len(rows)
        learn_allowed = heldout_event is None or event.event_type != heldout_event
        if track_q_target:
            self.enqueue_q_sample(step, comp, event, action, row_index, learn_allowed=learn_allowed)
        outcome = self.apply_event(event, action)
        self.update_component_predictors(comp, event, outcome, action, learn=learn and learn_allowed)
        env.push_action(action)
        self.agency.push_action(action)
        self.q_history.append(float(comp["q"]))
        if len(self.q_history) > 2048:
            self.q_history.pop(0)

        row: Dict[str, Any] = {
            "step": int(step),
            "event_type": event.event_type,
            "intensity": event.intensity,
            "friction": event.friction,
            "slope": event.slope,
            "affordance": event.affordance,
            "physical_risk": event.physical_risk,
            "source_class": event.source_class,
            "self_generated": event.self_generated,
            "mismatch": event.mismatch,
            "motor_contribution": event.motor_contribution,
            "exogenous_contribution": event.exogenous_contribution,
            "reafferent_signal": event.reafferent_signal,
            "true_delay": event.true_delay,
            "agency_score": agency["agency_score"],
            "agency_score_used": agency["agency_score_used"],
            "agency_best_delay": agency["best_delay"],
            "agency_exo_error": agency["exo_error"],
            "agency_action_error": agency["action_error"],
            "pain": pre_pain,
            "danger": comp["danger"],
            "comfort": comp["comfort"],
            "action_possibility": comp["action_possibility"],
            "controllability": comp["controllability"],
            "avoidance_pressure": comp["avoidance_pressure"],
            "q": comp["q"],
            "fixed_q": comp["fixed_q"],
            "policy_signal": signal,
            "action": action,
            "action_name": ACTION_NAMES[action],
            "p_continue": action_probs[A_CONTINUE],
            "p_cautious": action_probs[A_CAUTIOUS],
            "p_avoid": action_probs[A_AVOID],
                "post_integrity": outcome["post_integrity"],
            "post_energy": outcome["post_energy"],
            "post_fatigue": outcome["post_fatigue"],
            "post_stability": outcome["post_stability"],
            **outcome,
        }
        rows.append(row)
        if track_q_target:
            self.mature_q_samples(step, rows, outcome=outcome, learn=learn)
        self.current_step = max(self.current_step, int(step) + 1)
        return row


# =============================================================================
# Calibration
# =============================================================================


def _representative_calibration_morphologies() -> List[Morphology]:
    """Four corners spanning the morphology factorial without using all 16 bodies."""
    names = [
        "m0.8_e0.8_t0.8_d0.8",
        "m0.8_e1.2_t1.2_d0.8",
        "m1.2_e0.8_t0.8_d1.2",
        "m1.2_e1.2_t1.2_d1.2",
    ]
    return [MORPHOLOGIES[n] for n in names]


def _categorical_from_uniform(prob: np.ndarray, u: np.ndarray) -> np.ndarray:
    c = np.cumsum(prob, axis=1)
    return np.sum(u[:, None] > c, axis=1).astype(int)


def _vectorized_calibration_candidate(task: Dict[str, Any]) -> Dict[str, Any]:
    """Random-policy calibration over independent seeds and representative bodies.

    All trajectories for one candidate are simulated in one NumPy batch.  This
    replaces thousands of Python agent objects and ensures calibration uses the
    same episode duration as confirmatory execution.
    """
    ds = float(task["damage_scale"])
    hm = float(task["hazard_multiplier"])
    rm = float(task["recovery_multiplier"])
    steps = int(task["steps"])
    horizon = int(task["horizon"])
    true_delay = int(task.get("true_delay", 1))
    seed_count = int(task["seed_count"])
    morphs = _representative_calibration_morphologies()
    masses = np.tile(np.array([m.mass for m in morphs], dtype=float), seed_count)
    ecaps = np.tile(np.array([m.energy_capacity for m in morphs], dtype=float), seed_count)
    tractions = np.tile(np.array([m.traction_tolerance for m in morphs], dtype=float), seed_count)
    dtols = np.tile(np.array([m.damage_tolerance for m in morphs], dtype=float), seed_count)
    morph_idx = np.tile(np.arange(len(morphs)), seed_count)
    n = len(masses)
    rng = np.random.default_rng(int(task["seed"]))

    integrity = np.ones(n, dtype=float)
    energy = np.ones(n, dtype=float)
    fatigue = np.zeros(n, dtype=float)
    stability = np.ones(n, dtype=float)
    damage = np.zeros(n, dtype=float)
    fail_count = np.zeros(n, dtype=np.int16)
    ever_failed = np.zeros(n, dtype=bool)
    previous_event = np.zeros(n, dtype=np.int16)
    action_history = np.zeros((steps, n), dtype=np.int16)
    latent_noise = np.zeros(n, dtype=float)

    hist_damage = np.empty((steps + 1, n), dtype=np.float32)
    hist_energy = np.empty((steps + 1, n), dtype=np.float32)
    hist_fatigue = np.empty((steps + 1, n), dtype=np.float32)
    hist_stability = np.empty((steps + 1, n), dtype=np.float32)
    hist_failed = np.empty((steps + 1, n), dtype=np.bool_)
    # Current-event continuation counterfactual costs, matching the online Q target.
    cf_damage = np.empty((steps, n), dtype=np.float32)
    cf_energy = np.empty((steps, n), dtype=np.float32)
    cf_fatigue = np.empty((steps, n), dtype=np.float32)
    cf_stability = np.empty((steps, n), dtype=np.float32)
    hist_damage[0] = damage
    hist_energy[0] = energy
    hist_fatigue[0] = fatigue
    hist_stability[0] = stability
    hist_failed[0] = ever_failed
    positive_damage: List[np.ndarray] = []

    transition = np.ones((len(EVENT_TYPES), len(EVENT_TYPES)), dtype=float)
    for prev, nxt, value in [
        ("jump", "landing", 6.0), ("slip", "rest", 2.0),
        ("collision", "rest", 2.0), ("slip", "brake", 1.5),
        ("collision", "brake", 1.5), ("rest", "walk", 1.4),
        ("rest", "rest", 0.8),
    ]:
        transition[EVENT_INDEX[prev], EVENT_INDEX[nxt]] = value
    action_table = np.array([
        [0.75, 1.25, 1.10, 1.30, 1.10, 1.15, 1.30, 0.60],
        [1.00, 1.10, 1.00, 0.65, 0.75, 0.80, 0.70, 1.15],
        [1.60, 0.75, 0.65, 0.35, 0.35, 0.60, 0.40, 2.20],
    ], dtype=float)
    hazard_vec = np.array([1.0 if e not in HAZARD_EVENTS else hm for e in EVENT_TYPES], dtype=float)
    event_load = np.array([0.05, 0.12, 0.35, 1.00, 0.45, 1.15, 1.25, 0.55], dtype=float)

    for t in range(steps):
        action = rng.integers(0, len(ACTIONS), size=n, dtype=np.int16)
        delayed_action = action_history[t - true_delay] if t >= true_delay else np.zeros(n, dtype=np.int16)
        body = np.ones((n, len(EVENT_TYPES)), dtype=float)
        body[:, EVENT_INDEX["slip"]] = np.exp(
            1.20 * fatigue + 1.40 * (1.0 - stability) + 0.60 * (1.0 - energy)
            - 2.00 * (tractions - 1.0)
        )
        body[:, EVENT_INDEX["collision"]] = np.exp(
            0.80 * fatigue + 0.80 * (1.0 - stability) + 0.50 * (1.0 - integrity)
            + 0.50 * (masses - 1.0)
        )
        body[:, EVENT_INDEX["landing"]] = np.exp(
            0.60 * fatigue + 0.90 * (1.0 - stability) + 0.50 * (masses - 1.0)
        )
        body[:, EVENT_INDEX["rest"]] = np.exp(0.50 * (1.0 - energy) + 0.50 * fatigue)
        weights = (
            BASE_EVENT_PROBS[None, :] * transition[previous_event]
            * action_table[delayed_action] * body * hazard_vec[None, :]
        )
        weights = np.clip(weights, 1e-12, 1e12)
        probs = weights / np.sum(weights, axis=1, keepdims=True)
        event_idx = _categorical_from_uniform(probs, rng.random(n))

        uf = rng.random(n)
        friction = np.select(
            [uf < 0.22, uf < 0.40, uf < 0.82], [0.06, 0.26, 0.42], default=0.72
        ).astype(float)
        slope = rng.uniform(0.0, 0.35, size=n)
        affordance = rng.uniform(0.18, 1.00, size=n)
        ubase = rng.random(n)
        baseline = np.empty(n, dtype=float)
        for idx, event_type in enumerate(EVENT_TYPES):
            mask = event_idx == idx
            if not np.any(mask):
                continue
            u = ubase[mask]
            if event_type == "rest":
                val = 0.08 * u
            elif event_type == "walk":
                val = 0.18 + (1.10 - 0.18) * u
            elif event_type == "slope":
                val = 0.45 + (2.20 - 0.45) * u + 1.80 * slope[mask]
            elif event_type == "slip":
                val = (0.85 + (3.65 - 0.85) * u) * (1.18 - 0.55 * friction[mask])
            elif event_type == "jump":
                val = 0.75 + (2.80 - 0.75) * u
            elif event_type == "landing":
                val = 1.05 + (4.40 - 1.05) * u
            elif event_type == "collision":
                val = 0.90 + (4.85 - 0.90) * u
            else:
                val = (0.45 + (3.10 - 0.45) * u) * (1.0 - 0.25 * friction[mask])
            baseline[mask] = val

        source_u = rng.random(n)
        source = np.select(
            [source_u < 0.42, source_u < 0.60, source_u < 0.88], [0, 1, 2], default=3
        )
        action_effect = np.choose(delayed_action, [0.10, -0.18, -0.35])
        motor = action_effect * affordance * baseline
        motor_contrib = np.where(source == 0, motor, 0.0)
        mismatch_factor = rng.choice(np.array([0.0, -0.5, 0.5, 1.5]), size=n, p=[0.25, 0.20, 0.25, 0.30])
        motor_contrib = np.where(source == 1, motor * mismatch_factor, motor_contrib)
        external = np.where(
            source == 3,
            rng.choice(np.array([-1.0, 1.0]), size=n) * np.abs(motor) * rng.uniform(0.70, 1.30, size=n),
            0.0,
        )
        latent_noise = 0.90 * latent_noise + rng.normal(0.0, 0.02, size=n)
        noise = rng.normal(0.0, 0.02, size=n) + 0.05 * latent_noise
        intensity = np.maximum(0.0, baseline + motor_contrib + external + noise)

        mitigation = np.choose(action, [0.00, 0.30, 0.60])
        vulnerability = 0.35 * (1.0 - integrity) + 0.25 * fatigue + 0.25 * (1.0 - stability) + 0.15 * (1.0 - energy)

        # Age-zero target term: current event under continuation/no mitigation.
        cf_risk = sigmoid(intensity - 2.45 + 0.55 * (0.30 - friction) + 0.28 * slope)
        cf_intensity_factor = 0.55 + 0.45 * np.minimum(intensity / 3.5, 1.75)
        cf_mechanical = event_load[event_idx] * cf_risk * cf_intensity_factor * masses
        cf_pressure = np.maximum(0.0, cf_mechanical + 0.75 * vulnerability - 0.32)
        cf_damage[t] = ds * cf_pressure * cf_pressure / np.maximum(dtols, 0.2)
        rest_now = event_idx == EVENT_INDEX["rest"]
        cf_energy[t] = np.where(
            rest_now, 0.0,
            np.maximum(0.0, 0.20 * (0.003 + 0.008 * intensity) / np.maximum(ecaps, 0.2) - 0.003),
        )
        cf_fatigue[t] = np.where(
            rest_now, 0.0,
            np.maximum(0.0, 0.20 * (0.001 + 0.004 * intensity) - 0.002),
        )
        cf_stability[t] = np.where(
            rest_now, 0.0,
            np.maximum(0.0, 0.20 * (0.007 + 0.025 * cf_risk)),
        )

        effective_intensity = intensity * (1.0 - mitigation * affordance)
        effective_risk = sigmoid(effective_intensity - 2.45 + 0.55 * (0.30 - friction) + 0.28 * slope)
        intensity_factor = 0.55 + 0.45 * np.minimum(effective_intensity / 3.5, 1.75)
        mechanical_load = event_load[event_idx] * effective_risk * intensity_factor * masses
        pressure = np.maximum(0.0, mechanical_load + 0.75 * vulnerability - 0.32)
        d_damage = ds * pressure * pressure / np.maximum(dtols, 0.2)
        positive_damage.append(d_damage[d_damage > 0])

        rest = event_idx == EVENT_INDEX["rest"]
        energy = np.where(
            rest,
            np.clip(energy + 0.030 * ecaps * rm, 0.0, 1.0),
            np.clip(
                energy - 0.20 * (np.choose(action, [0.003, 0.008, 0.015]) + 0.008 * effective_intensity)
                / np.maximum(ecaps, 0.2) + 0.003,
                0.0, 1.0,
            ),
        )
        fatigue = np.where(
            rest,
            np.clip(fatigue - 0.026 * rm, 0.0, 1.0),
            np.clip(fatigue + 0.20 * (np.choose(action, [0.001, 0.004, 0.008]) + 0.004 * effective_intensity) - 0.002, 0.0, 1.0),
        )
        stability = np.where(
            rest,
            np.clip(stability + 0.023 * rm, 0.0, 1.0),
            np.clip(stability - 0.20 * (0.007 + 0.025 * effective_risk) + 0.012 * mitigation, 0.0, 1.0),
        )
        damage = np.clip(damage + d_damage, 0.0, 1.0)
        integrity = np.clip(1.0 - damage, 0.0, 1.0)
        boundary = (integrity <= 0.20) | (energy <= 0.05) | (stability <= 0.05)
        fail_count = np.where(boundary, fail_count + 1, 0)
        ever_failed |= fail_count >= 5
        previous_event = event_idx.astype(np.int16)
        action_history[t] = action.astype(np.int16)

        hist_damage[t + 1] = damage
        hist_energy[t + 1] = energy
        hist_fatigue[t + 1] = fatigue
        hist_stability[t + 1] = stability
        hist_failed[t + 1] = ever_failed

    stride = max(1, horizon // 5)
    starts = np.arange(0, max(1, steps - horizon), stride, dtype=int)
    ends = starts + horizon
    step_damage = np.maximum(0.0, hist_damage[1:] - hist_damage[:-1]).astype(float)
    step_energy = np.maximum(0.0, hist_energy[:-1] - hist_energy[1:]).astype(float)
    step_fatigue = np.maximum(0.0, hist_fatigue[1:] - hist_fatigue[:-1]).astype(float)
    step_stability = np.maximum(0.0, hist_stability[:-1] - hist_stability[1:]).astype(float)
    discount = 0.92
    # Current event uses the continuation counterfactual; only later ages use
    # actually experienced costs.  This is identical to mature_q_samples().
    disc_damage = cf_damage[starts].astype(float)
    disc_energy = cf_energy[starts].astype(float)
    disc_fatigue = cf_fatigue[starts].astype(float)
    disc_stability = cf_stability[starts].astype(float)
    for k in range(1, horizon):
        w = discount ** k
        idx = starts + k
        disc_damage += w * step_damage[idx]
        disc_energy += w * step_energy[idx]
        disc_fatigue += w * step_fatigue[idx]
        disc_stability += w * step_stability[idx]
    raw_damage = disc_damage.ravel()
    raw_energy = disc_energy.ravel()
    raw_fatigue = disc_fatigue.ravel()
    raw_stability = disc_stability.ravel()
    raw_failure = (hist_failed[ends] & ~hist_failed[starts]).ravel().astype(float)

    def norm_and_mean(x: np.ndarray, floor: float) -> Tuple[float, float]:
        q95 = max(float(np.quantile(x, 0.95)), float(floor))
        return q95, float(np.mean(np.clip(x / q95, 0.0, 1.0)))

    nd, md = norm_and_mean(raw_damage, ds * 0.02)
    ne, me = norm_and_mean(raw_energy, 0.02)
    nf, mf = norm_and_mean(raw_fatigue, 0.02)
    ns, ms = norm_and_mean(raw_stability, 0.02)
    pos = np.concatenate(positive_damage) if positive_damage else np.array([], dtype=float)
    morph_failure = [float(np.mean(ever_failed[morph_idx == i])) for i in range(len(morphs))]
    return {
        "damage_scale": ds,
        "hazard_multiplier": hm,
        "recovery_multiplier": rm,
        "median_final_integrity": float(np.median(integrity)),
        "mean_final_integrity": float(np.mean(integrity)),
        "failure_rate": float(np.mean(ever_failed)),
        "failure_rate_min_morph": float(np.min(morph_failure)),
        "failure_rate_max_morph": float(np.max(morph_failure)),
        "damage_normalizer": nd,
        "energy_normalizer": ne,
        "fatigue_normalizer": nf,
        "stability_normalizer": ns,
        "target_mean_damage": md,
        "target_mean_energy": me,
        "target_mean_fatigue": mf,
        "target_mean_stability": ms,
        "target_mean_failure": float(np.mean(raw_failure)),
        "positive_damage_p75": float(np.quantile(pos, 0.75)) if len(pos) else ds * 1e-3,
        "steps": steps,
        "trajectory_count": n,
        "calibration_seed_count": seed_count,
        "calibration_morphology_count": len(morphs),
    }


def _calibration_candidate_cache_path(outdir: Path, task: Dict[str, Any]) -> Path:
    d = outdir / "calibration_candidate_cache"
    d.mkdir(parents=True, exist_ok=True)
    key = deterministic_seed(
        "v2cal_counterfactual_delay_20260725E", task["damage_scale"], task["hazard_multiplier"],
        task["recovery_multiplier"], task["steps"], task["seed_count"], task["horizon"],
        task.get("true_delay", 1), task["seed"],
    )
    return d / f"candidate_{key}.json"


def _run_calibration_tasks(
    tasks: List[Dict[str, Any]], outdir: Path, workers: int, resume: bool, logger: Logger,
) -> List[Dict[str, Any]]:
    done: List[Dict[str, Any]] = []
    pending: List[Dict[str, Any]] = []
    for task in tasks:
        path = _calibration_candidate_cache_path(outdir, task)
        if resume and path.exists():
            done.append(json.loads(path.read_text(encoding="utf-8")))
        else:
            task = dict(task)
            task["cache_path"] = str(path)
            pending.append(task)
    if pending:
        logger.log(f"calibration candidate batch: {len(pending)} candidates")
        if workers > 1:
            with mp.get_context("spawn").Pool(workers, maxtasksperchild=8) as pool:
                iterator = pool.imap_unordered(_vectorized_calibration_candidate, pending, chunksize=1)
                for i, row in enumerate(iterator, 1):
                    task = next(t for t in pending if float(t["damage_scale"]) == float(row["damage_scale"])
                                and float(t["hazard_multiplier"]) == float(row["hazard_multiplier"])
                                and float(t["recovery_multiplier"]) == float(row["recovery_multiplier"]))
                    Path(task["cache_path"]).write_text(json.dumps(row, indent=2), encoding="utf-8")
                    done.append(row)
                    if i % max(1, len(pending) // 10) == 0 or i == len(pending):
                        logger.log(f"calibration progress {i}/{len(pending)}")
        else:
            for i, task in enumerate(pending, 1):
                row = _vectorized_calibration_candidate(task)
                Path(task["cache_path"]).write_text(json.dumps(row, indent=2), encoding="utf-8")
                done.append(row)
                if i % max(1, len(pending) // 10) == 0 or i == len(pending):
                    logger.log(f"calibration progress {i}/{len(pending)}")
    return done


def _select_calibration_levels(grouped: pd.DataFrame, strict: bool = True) -> Dict[str, pd.Series]:
    d = grouped.copy()
    d["severity"] = d["damage_scale"] * d["hazard_multiplier"] / d["recovery_multiplier"]
    if strict:
        moderate_pool = d[
            d["median_final_integrity"].between(0.45, 0.85)
            & d["failure_rate"].between(0.05, 0.60)
            & (d["failure_rate_max_morph"] < 0.95)
        ].copy()
    else:
        moderate_pool = d[d["median_final_integrity"].between(0.20, 0.92)].copy()
    if moderate_pool.empty:
        raise RuntimeError("No calibration candidate reached the required non-ceiling dynamic range.")
    moderate_pool["score"] = (
        abs(moderate_pool["median_final_integrity"] - 0.68)
        + 0.80 * abs(moderate_pool["failure_rate"] - 0.25)
        + 0.20 * (moderate_pool["failure_rate_max_morph"] - moderate_pool["failure_rate_min_morph"])
    )
    moderate = moderate_pool.sort_values(["score", "severity"]).iloc[0]
    mild_pool = d[
        (d["median_final_integrity"] >= float(moderate["median_final_integrity"]) + 0.08)
        & (d["failure_rate"] <= float(moderate["failure_rate"]))
    ].copy()
    challenging_pool = d[
        (d["median_final_integrity"] <= float(moderate["median_final_integrity"]) - 0.08)
        & (d["failure_rate"] >= float(moderate["failure_rate"]))
    ].copy()
    if mild_pool.empty or challenging_pool.empty:
        raise RuntimeError("Calibration found a moderate regime but not ordered mild and challenging regimes.")
    mild_pool["score"] = abs(mild_pool["median_final_integrity"] - 0.88) + 0.5 * abs(mild_pool["failure_rate"] - 0.03)
    challenging_pool["score"] = abs(challenging_pool["median_final_integrity"] - 0.38) + 0.5 * abs(challenging_pool["failure_rate"] - 0.65)
    mild = mild_pool.sort_values(["score", "severity"]).iloc[0]
    challenging = challenging_pool.sort_values(["score", "severity"]).iloc[0]
    if not (
        float(mild["median_final_integrity"]) > float(moderate["median_final_integrity"])
        > float(challenging["median_final_integrity"])
    ):
        raise RuntimeError("Selected calibration regimes are not ordered by realized integrity.")
    return {"mild": mild, "moderate": moderate, "challenging": challenging}


def calibrate_environments(
    outdir: Path,
    preset: RunPreset,
    base_seed: int,
    workers: int,
    horizon: int,
    true_delay: int,
    resume: bool,
    logger: Logger,
    strict: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, EnvironmentConfig]]:
    result_path = outdir / "01_calibration_results.csv"
    selected_path = outdir / "02_selected_environments.csv"
    if resume and result_path.exists() and selected_path.exists():
        raw = pd.read_csv(result_path)
        sel = pd.read_csv(selected_path)
        configs = {str(r["risk_label"]): EnvironmentConfig(**{
            k: (bool(r[k]) if k == "calibration_passed" else r[k])
            for k in EnvironmentConfig.__dataclass_fields__ if k in r
        }) for _, r in sel.iterrows()}
        version_ok = (
            "calibration_version" in sel.columns
            and bool((sel["calibration_version"].astype(str) == CALIBRATION_VERSION).all())
        )
        if set(configs) == {"mild", "moderate", "challenging"} and version_ok:
            logger.log("loading existing V2 calibration results")
            return raw, configs
        logger.log("existing calibration is from an incompatible V2 revision; recalibrating")

    steps = int(preset.steps)  # deliberately identical to confirmatory episodes
    seed_count = int(preset.calibration_seeds)
    scale_factor = 5000.0 / max(steps, 1)
    damage_scales = sorted(set(float(x * scale_factor) for x in [0.005, 0.01, 0.02, 0.04, 0.08, 0.16, 0.32]))
    hazard_mults = [0.80, 1.20, 1.60]
    recovery_mults = [0.75, 1.00]
    all_rows: List[Dict[str, Any]] = []
    attempted: set[Tuple[float, float, float]] = set()
    selected: Optional[Dict[str, pd.Series]] = None

    for round_idx in range(4):
        tasks: List[Dict[str, Any]] = []
        for ds, hm, rm in itertools.product(damage_scales, hazard_mults, recovery_mults):
            key = (round(float(ds), 12), float(hm), float(rm))
            if key in attempted:
                continue
            attempted.add(key)
            tasks.append({
                "damage_scale": ds, "hazard_multiplier": hm, "recovery_multiplier": rm,
                "steps": steps, "horizon": int(horizon), "true_delay": int(true_delay),
                "seed_count": seed_count,
                "seed": deterministic_seed(base_seed, "v2_calibration", ds, hm, rm, true_delay),
            })
        all_rows.extend(_run_calibration_tasks(tasks, outdir, workers, resume, logger))
        grouped = pd.DataFrame(all_rows).drop_duplicates(
            ["damage_scale", "hazard_multiplier", "recovery_multiplier"], keep="last"
        )
        try:
            selected = _select_calibration_levels(grouped, strict=strict)
            break
        except RuntimeError as exc:
            logger.log(f"calibration round {round_idx + 1} did not bracket all regimes: {exc}")
            lo, hi = min(damage_scales), max(damage_scales)
            damage_scales = sorted(set(damage_scales + [lo / 2.0, hi * 2.0]))

    grouped = pd.DataFrame(all_rows).drop_duplicates(
        ["damage_scale", "hazard_multiplier", "recovery_multiplier"], keep="last"
    ).sort_values(["damage_scale", "hazard_multiplier", "recovery_multiplier"])
    grouped.to_csv(result_path, index=False)
    if selected is None:
        raise RuntimeError(
            "V2 calibration failed after adaptive grid expansion. Confirmatory execution was not started; "
            "inspect 01_calibration_results.csv rather than accepting a ceiling regime."
        )

    configs: Dict[str, EnvironmentConfig] = {}
    selected_rows: List[Dict[str, Any]] = []
    for label in ["mild", "moderate", "challenging"]:
        r = selected[label]
        cfg = EnvironmentConfig(
            risk_label=label,
            risk_scale=1.0,
            damage_scale=float(r["damage_scale"]),
            hazard_multiplier=float(r["hazard_multiplier"]),
            recovery_multiplier=float(r["recovery_multiplier"]),
            damage_normalizer=max(float(r["damage_normalizer"]), 1e-8),
            energy_normalizer=max(float(r["energy_normalizer"]), 1e-5),
            fatigue_normalizer=max(float(r["fatigue_normalizer"]), 1e-5),
            stability_normalizer=max(float(r["stability_normalizer"]), 1e-5),
            target_mean_damage=float(np.clip(r["target_mean_damage"], 0.03, 0.97)),
            target_mean_energy=float(np.clip(r["target_mean_energy"], 0.03, 0.97)),
            target_mean_fatigue=float(np.clip(r["target_mean_fatigue"], 0.03, 0.97)),
            target_mean_stability=float(np.clip(r["target_mean_stability"], 0.03, 0.97)),
            target_mean_failure=float(np.clip(r["target_mean_failure"], 0.03, 0.97)),
            damage_target_threshold=max(float(r["positive_damage_p75"]), 1e-10),
            true_delay=int(true_delay),
            horizon=int(horizon),
            morphology="reference",
            calibration_passed=True,
        )
        configs[label] = cfg
        row = asdict(cfg)
        row.update({
            "calibration_version": CALIBRATION_VERSION,
            "calibration_median_final_integrity": float(r["median_final_integrity"]),
            "calibration_failure_rate": float(r["failure_rate"]),
            "calibration_failure_rate_min_morph": float(r["failure_rate_min_morph"]),
            "calibration_failure_rate_max_morph": float(r["failure_rate_max_morph"]),
            "calibration_steps": steps,
            "calibration_seed_count": seed_count,
            "calibration_morphology_count": 4,
        })
        selected_rows.append(row)
    pd.DataFrame(selected_rows).to_csv(selected_path, index=False)
    logger.log("V2 calibration passed and parameters were frozen before confirmatory execution")
    return grouped, configs


# =============================================================================
# Episode execution and summaries
# =============================================================================


def _task_extra_parameters(task: Dict[str, Any]) -> Dict[str, Any]:
    """Non-default hand-coefficient overrides carried by a task.

    Returns an empty dict when the task uses the confirmatory defaults, so that
    the cache key and the model are byte-identical to the pre-existing run.
    """
    extra: Dict[str, Any] = {}
    vw = task.get("vulnerability_weights")
    if vw is not None and tuple(float(v) for v in vw) != tuple(VULNERABILITY_WEIGHTS):
        extra["vulnerability_weights"] = tuple(round(float(v), 6) for v in vw)
    ac = task.get("avoidance_coefficients")
    if ac is not None and {k: float(v) for k, v in ac.items()} != dict(AVOIDANCE_COEFFICIENTS):
        extra["avoidance_coefficients"] = tuple(sorted((k, round(float(v), 6)) for k, v in ac.items()))
    apw = float(task.get("agency_prior_weight", AGENCY_PRIOR_WEIGHT))
    if apw != AGENCY_PRIOR_WEIGHT:
        extra["agency_prior_weight"] = apw
    acw = float(task.get("agency_current_weight", AGENCY_CURRENT_WEIGHT))
    if acw != AGENCY_CURRENT_WEIGHT:
        extra["agency_current_weight"] = acw
    ag = float(task.get("agency_gain", AGENCY_GAIN))
    if ag != AGENCY_GAIN:
        extra["agency_gain"] = ag
    return extra


def episode_task_cache_path(task: Dict[str, Any]) -> Path:
    outdir = Path(task["outdir"])
    cache = outdir / "task_cache_v2"
    cache.mkdir(parents=True, exist_ok=True)
    cfg_items = tuple(sorted((str(k), str(v)) for k, v in task["cfg"].items()))
    key = deterministic_seed(
        "QVALENCE_V2_20260725O",
        cfg_items,
        task["condition"], task["episode"], task["steps"], task["burn_in"],
        task["beta"], task["q_lr_multiplier"], task["predictor_lr_multiplier"],
        task["memory_decay_multiplier"], task["mitigation_multiplier"],
        task.get("robustness_setting", ""), task.get("common_seed", ""),
        *([] if not _task_extra_parameters(task)
          else [tuple(sorted(_task_extra_parameters(task).items()))]),
    )
    return cache / f"episode_{key}.json"


def summarize_prospective(rows: List[Dict[str, Any]], burn_in: int = 0) -> List[Dict[str, Any]]:
    d = pd.DataFrame(rows)
    if not d.empty and "step" in d.columns:
        d = d[d["step"] >= int(burn_in)].copy()
    if d.empty or "future_compromise" not in d.columns:
        return []
    d = d[np.isfinite(d["future_compromise"])].copy()
    if d.empty:
        return []
    q_pred_comp = d[[
        "q_pred_damage", "q_pred_energy", "q_pred_fatigue", "q_pred_stability", "q_pred_failure"
    ]].mean(axis=1).to_numpy()
    predictor_map = {
        "learned_Q_decoder": (d["q_pred_failure"].to_numpy(), q_pred_comp),
        "learned_Q_scalar": (d["q"].to_numpy(), d["q"].to_numpy()),
        "pain": (d["pain"].to_numpy(), d["pain"].to_numpy()),
        "danger": (d["danger"].to_numpy(), d["danger"].to_numpy()),
        "physical_risk": (d["physical_risk"].to_numpy(), d["physical_risk"].to_numpy()),
        "body_state": ((1.0 - d["post_viability"]).to_numpy(), (1.0 - d["post_viability"]).to_numpy()),
        "fixed_Q": (d["fixed_q"].to_numpy(), d["fixed_q"].to_numpy()),
    }
    out = []
    y_failure = d["future_failure"].astype(int).to_numpy()
    y_comp = d["future_compromise"].to_numpy()
    high_comp = (y_comp >= np.quantile(y_comp, 0.75)).astype(int) if len(y_comp) >= 8 else y_failure
    for name, (failure_score, comp_score) in predictor_map.items():
        out.append({
            "predictor": name,
            "failure_auc": auc_score(y_failure, failure_score),
            "failure_brier": brier_score(y_failure, failure_score),
            "high_compromise_auc": auc_score(high_comp, comp_score),
            "compromise_r": pearson_r(y_comp, comp_score),
            "compromise_r2_identity": r2_score(y_comp, comp_score),
            "mean_q_prediction_loss": safe_mean(d["q_prediction_loss"]) if name == "learned_Q_decoder" else np.nan,
            "mean_null_prediction_loss": safe_mean(d["q_null_prediction_loss"]) if name == "learned_Q_decoder" else np.nan,
            "mean_prediction_improvement": safe_mean(d["q_prediction_improvement"]) if name == "learned_Q_decoder" else np.nan,
            "n": int(len(d)),
        })
    return out


def summarize_agency(rows: List[Dict[str, Any]], burn_in: int = 0) -> Dict[str, float]:
    d = pd.DataFrame(rows)
    if not d.empty and "step" in d.columns:
        d = d[d["step"] >= int(burn_in)].copy()
    if d.empty:
        return {}

    matched = d[d["source_class"] == "matched_self"]
    mismatch = d[d["source_class"] == "mismatched_self"]
    external = d[d["source_class"] == "external"]
    coincidence = d[d["source_class"] == "external_coincidence"]

    # Primary source discrimination compares matched self-caused outcomes with
    # passive external outcomes.  V1 incorrectly labelled mismatched/omitted
    # self trials as positive even though the hypothesis predicts reduced
    # agency for them, making the AUC internally contradictory.
    discrim = pd.concat([matched, external], ignore_index=True)
    labels = (discrim["source_class"] == "matched_self").astype(int).to_numpy()
    scores = discrim["agency_score_used"].to_numpy(dtype=float)

    matched_scores = matched["agency_score_used"].to_numpy(dtype=float)
    mismatch_scores = mismatch["agency_score_used"].to_numpy(dtype=float)
    external_scores = external["agency_score_used"].to_numpy(dtype=float)
    coincidence_scores = coincidence["agency_score_used"].to_numpy(dtype=float)
    delay_mae = safe_mean(np.abs(
        matched["true_delay"].to_numpy(dtype=float)
        - matched["agency_best_delay"].to_numpy(dtype=float)
    )) if not matched.empty else float("nan")
    delay_hit = safe_mean(np.abs(
        matched["true_delay"].to_numpy(dtype=float)
        - matched["agency_best_delay"].to_numpy(dtype=float)
    ) <= 1) if not matched.empty else float("nan")
    return {
        "agency_auc": auc_score(labels, scores),
        "agency_balanced_accuracy": balanced_accuracy(labels, scores),
        "agency_brier": brier_score(labels, scores),
        "agency_matched_mean": safe_mean(matched_scores),
        "agency_external_mean": safe_mean(external_scores),
        "agency_external_coincidence_mean": safe_mean(coincidence_scores),
        "agency_mismatch_mean": safe_mean(mismatch_scores),
        "agency_mismatch_drop": safe_mean(matched_scores) - safe_mean(mismatch_scores),
        "agency_matched_external_contrast": safe_mean(matched_scores) - safe_mean(external_scores),
        "agency_delay_mae": delay_mae,
        "agency_near_delay_hit": delay_hit,
    }


def summarize_episode(rows: List[Dict[str, Any]], agent: EmbodiedAgent, burn_in: int) -> Dict[str, float]:
    d = pd.DataFrame(rows)
    eval_d = d[d["step"] >= burn_in] if not d.empty else d
    return {
        "final_integrity": agent.state.integrity,
        "final_energy": agent.state.energy,
        "final_fatigue": agent.state.fatigue,
        "final_stability": agent.state.stability,
        "final_viability": agent.state.viability(),
        "cumulative_damage": agent.state.damage,
        "viability_failure": float(agent.state.ever_failed),
        "time_to_failure": float(d.loc[d["failure"] > 0.5, "step"].min()) if np.any(d["failure"] > 0.5) else float(len(d)),
        "mean_q": safe_mean(eval_d["q"]),
        "mean_fixed_q": safe_mean(eval_d["fixed_q"]),
        "mean_pain": safe_mean(eval_d["pain"]),
        "mean_danger": safe_mean(eval_d["danger"]),
        "mean_physical_risk": safe_mean(eval_d["physical_risk"]),
        "continuation_rate": safe_mean(eval_d["action"] == A_CONTINUE),
        "cautious_rate": safe_mean(eval_d["action"] == A_CAUTIOUS),
        "avoidance_rate": safe_mean(eval_d["action"] == A_AVOID),
        "regulation_rate": safe_mean(eval_d["action"] != A_CONTINUE),
        "cumulative_energy_deficit": float(np.sum(np.maximum(0.0, -eval_d["energy_change"].to_numpy()))) if not eval_d.empty else 0.0,
        "cumulative_stability_loss": float(np.sum(np.maximum(0.0, -eval_d["stability_change"].to_numpy()))) if not eval_d.empty else 0.0,
        "mean_fatigue": safe_mean(eval_d["post_fatigue"]),
        "action_cost": safe_mean([
            BASE_ENERGY_COST[int(a)] + BASE_FATIGUE_COST[int(a)] for a in eval_d["action"].to_numpy()
        ]) if not eval_d.empty else float("nan"),
        "q_learning_updates": float(agent.q_integrator.update_count),
        "q_replay_size": float(agent.q_replay.size),
        "q_total_loss_ema": float(agent.q_integrator.loss_ema),
        "q_loss_ema": float(agent.q_integrator.prediction_loss_ema),
        "q_null_loss_ema": float(agent.q_integrator.null_loss_ema),
        "q_prediction_improvement_ema": float(agent.q_integrator.null_loss_ema - agent.q_integrator.prediction_loss_ema),
        "q_rank_loss_ema": float(agent.q_integrator.rank_loss_ema),
        "q_burden_loss_ema": float(agent.q_integrator.burden_loss_ema),
        "q_weight_sd": float(np.std(agent.q_integrator.weights())),
        "q_weight_entropy": float(-np.sum(agent.q_integrator.weights() * np.log(agent.q_integrator.weights() + 1e-12))),
    }


def run_episode_task(task: Dict[str, Any]) -> Dict[str, Any]:
    cache_path = episode_task_cache_path(task)
    if task.get("resume", False) and cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))
    cfg = EnvironmentConfig(**task["cfg"])
    condition = CONDITIONS[task["condition"]]
    morph = MORPHOLOGIES[cfg.morphology]
    common_seed = int(task["common_seed"])
    env = ClosedLoopEnvironment(cfg, morph, condition.event_loop_mode, np.random.default_rng(common_seed))
    agent = EmbodiedAgent(
        cfg, morph, condition, common_seed,
        beta=float(task["beta"]),
        q_lr_multiplier=float(task["q_lr_multiplier"]),
        predictor_lr_multiplier=float(task["predictor_lr_multiplier"]),
        memory_decay_multiplier=float(task["memory_decay_multiplier"]),
        mitigation_multiplier=float(task["mitigation_multiplier"]),
        vulnerability_weights=task.get("vulnerability_weights"),
        avoidance_coefficients=task.get("avoidance_coefficients"),
        agency_prior_weight=float(task.get("agency_prior_weight", AGENCY_PRIOR_WEIGHT)),
        agency_current_weight=float(task.get("agency_current_weight", AGENCY_CURRENT_WEIGHT)),
        agency_gain=float(task.get("agency_gain", AGENCY_GAIN)),
    )
    rows: List[Dict[str, Any]] = []
    for step in range(int(task["steps"])):
        agent.step(env, step, rows, learn=True)
    summary = summarize_episode(rows, agent, int(task["burn_in"]))
    prospective = summarize_prospective(rows, int(task["burn_in"]))
    agency = summarize_agency(rows, int(task["burn_in"]))
    weights = agent.q_integrator.weights()
    result = {
        "meta": {
            "condition": condition.name,
            "risk_label": cfg.risk_label,
            "morphology": cfg.morphology,
            "episode": int(task["episode"]),
            "episode_seed": common_seed,
            "true_delay": cfg.true_delay,
            "horizon": cfg.horizon,
            "beta": float(task["beta"]),
            "q_lr_multiplier": float(task["q_lr_multiplier"]),
            "predictor_lr_multiplier": float(task["predictor_lr_multiplier"]),
            "memory_decay_multiplier": float(task["memory_decay_multiplier"]),
            "mitigation_multiplier": float(task["mitigation_multiplier"]),
            "robustness_setting": str(task.get("robustness_setting", "")),
            "agency_prior_weight": float(task.get("agency_prior_weight", AGENCY_PRIOR_WEIGHT)),
            "agency_gain": float(task.get("agency_gain", AGENCY_GAIN)),
            "vulnerability_weights": ",".join(
                "%.4f" % float(v) for v in (task.get("vulnerability_weights") or VULNERABILITY_WEIGHTS)),
        },
        "summary": summary,
        "prospective": prospective,
        "agency": agency,
        "weights": {name: float(value) for name, value in zip(Q_COMPONENT_NAMES, weights)},
    }
    if task.get("save_step_timeseries", False):
        stepdir = Path(task["outdir"]) / "step_timeseries"
        stepdir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(stepdir / f"{cache_path.stem}.csv.gz", index=False, compression="gzip")
    cache_path.write_text(json.dumps(result, indent=2, default=lambda o: o.item() if isinstance(o, np.generic) else str(o)), encoding="utf-8")
    return result


# =============================================================================
# Independent probes
# =============================================================================


def train_probe_agent(
    seed: int,
    cfg: EnvironmentConfig,
    morphology: Morphology,
    steps: int,
    condition_name: str = "Full_Learned_Q",
    heldout_event: Optional[str] = None,
) -> Tuple[EmbodiedAgent, ClosedLoopEnvironment, List[Dict[str, Any]]]:
    condition = CONDITIONS[condition_name]
    env = ClosedLoopEnvironment(cfg, morphology, condition.event_loop_mode, np.random.default_rng(seed))
    agent = EmbodiedAgent(cfg, morphology, condition, seed)
    rows: List[Dict[str, Any]] = []
    for t in range(steps):
        agent.step(env, t, rows, learn=True, heldout_event=heldout_event)
    return agent, env, rows


def random_probe_state(rng: np.random.Generator) -> AgentState:
    integrity = float(rng.uniform(0.35, 1.0))
    return AgentState(
        integrity=integrity,
        energy=float(rng.uniform(0.20, 1.0)),
        fatigue=float(rng.uniform(0.0, 0.85)),
        stability=float(rng.uniform(0.20, 1.0)),
        damage=float(1.0 - integrity),
        pain_memory=float(rng.uniform(0.0, 0.75)),
        danger_memory=float(rng.uniform(0.0, 0.75)),
        comfort_memory=float(rng.uniform(0.25, 0.90)),
    )


def _vectorized_closed_loop_draw_summary(
    seed: int,
    state_id: int,
    action: int,
    cfg: EnvironmentConfig,
    probs: np.ndarray,
    n_draws: int,
) -> Dict[str, float]:
    """Vectorized independent next-event draws for the closed-loop probe.

    The original implementation deep-copied the complete environment and
    recreated a NumPy generator for every draw.  Under the main preset this
    produced 48,000,000 deep copies.  This function preserves the same
    stochastic event-generation equations but draws each random variable as
    an array.  The same seed is deliberately reused across actions within a
    state so that the intervention comparison uses common random numbers.
    """
    n = int(max(1, n_draws))
    rng = np.random.default_rng(deterministic_seed(seed, state_id, "loop_vectorized"))

    # Common random numbers across action interventions.
    u_event = rng.random(n)
    event_idx = np.searchsorted(np.cumsum(probs), u_event, side="right")
    event_idx = np.clip(event_idx, 0, len(EVENT_TYPES) - 1)
    friction = rng.choice(np.array([0.06, 0.26, 0.42, 0.72]), size=n,
                          p=np.array([0.22, 0.18, 0.42, 0.18]))
    slope = rng.uniform(0.0, 0.35, size=n)
    affordance = rng.uniform(0.18, 1.00, size=n)

    baseline = np.zeros(n, dtype=float)
    for idx, event_type in enumerate(EVENT_TYPES):
        mask = event_idx == idx
        count = int(np.sum(mask))
        if count == 0:
            continue
        if event_type == "rest":
            value = rng.uniform(0.00, 0.08, size=count)
        elif event_type == "walk":
            value = rng.uniform(0.18, 1.10, size=count)
        elif event_type == "slope":
            value = rng.uniform(0.45, 2.20, size=count) + 1.80 * slope[mask]
        elif event_type == "slip":
            value = rng.uniform(0.85, 3.65, size=count) * (1.18 - 0.55 * friction[mask])
        elif event_type == "jump":
            value = rng.uniform(0.75, 2.80, size=count)
        elif event_type == "landing":
            value = rng.uniform(1.05, 4.40, size=count)
        elif event_type == "collision":
            value = rng.uniform(0.90, 4.85, size=count)
        elif event_type == "brake":
            value = rng.uniform(0.45, 3.10, size=count) * (1.0 - 0.25 * friction[mask])
        else:
            value = rng.uniform(0.10, 1.00, size=count)
        baseline[mask] = value

    source_idx = rng.choice(4, size=n, p=np.array([0.42, 0.18, 0.28, 0.12]))
    motor_effect = ClosedLoopEnvironment.action_intensity_effect(action) * affordance * baseline
    motor_contribution = np.zeros(n, dtype=float)
    external_contribution = np.zeros(n, dtype=float)

    matched_mask = source_idx == 0
    mismatch_mask = source_idx == 1
    coincidence_mask = source_idx == 3
    motor_contribution[matched_mask] = motor_effect[matched_mask]
    mismatch_count = int(np.sum(mismatch_mask))
    if mismatch_count:
        factors = rng.choice(np.array([0.0, -0.50, 0.50, 1.50]), size=mismatch_count,
                             p=np.array([0.25, 0.20, 0.25, 0.30]))
        motor_contribution[mismatch_mask] = motor_effect[mismatch_mask] * factors
    coincidence_count = int(np.sum(coincidence_mask))
    if coincidence_count:
        signs = rng.choice(np.array([-1.0, 1.0]), size=coincidence_count)
        multipliers = rng.uniform(0.70, 1.30, size=coincidence_count)
        external_contribution[coincidence_mask] = (
            signs * np.abs(motor_effect[coincidence_mask]) * multipliers
        )

    # Each original draw began from latent_noise == 0 because the environment
    # was deep-copied before one generate_event call.
    latent_noise = rng.normal(0.0, 0.02, size=n)
    noise = rng.normal(0.0, 0.02, size=n) + 0.05 * latent_noise
    intensity = np.maximum(
        0.0,
        cfg.risk_scale * baseline + motor_contribution + external_contribution + noise,
    )
    risk_linear = intensity - 2.35 + 0.55 * (0.28 - friction) + 0.35 * slope
    risk_linear = np.clip(risk_linear, -60.0, 60.0)
    physical_risk = 1.0 / (1.0 + np.exp(-risk_linear))
    counts = np.bincount(event_idx, minlength=len(EVENT_TYPES))

    return {
        "mean_intensity": float(np.mean(intensity)),
        "mean_physical_risk": float(np.mean(physical_risk)),
        "slip_probability": float(counts[EVENT_INDEX["slip"]] / n),
        "collision_probability": float(counts[EVENT_INDEX["collision"]] / n),
        "landing_probability": float(counts[EVENT_INDEX["landing"]] / n),
    }


def closed_loop_probe(
    seed: int,
    cfg: EnvironmentConfig,
    n_states: int,
    n_draws: int,
) -> List[Dict[str, Any]]:
    rng = np.random.default_rng(seed)
    rows: List[Dict[str, Any]] = []
    for state_id in range(n_states):
        state = random_probe_state(rng)
        base_env = ClosedLoopEnvironment(
            cfg,
            REFERENCE_MORPHOLOGY,
            "closed",
            np.random.default_rng(seed + state_id + 1),
        )
        base_env.previous_event = str(rng.choice(EVENT_TYPES))
        probs_by_action = {
            a: base_env.event_probabilities(state, forced_delayed_action=a)
            for a in ACTIONS
        }
        for a1, a2 in itertools.combinations(ACTIONS, 2):
            rows.append({
                "seed": seed,
                "state_id": state_id,
                "record_type": "jsd",
                "action_1": ACTION_NAMES[a1],
                "action_2": ACTION_NAMES[a2],
                "js_divergence": js_divergence(probs_by_action[a1], probs_by_action[a2]),
            })
        for action in ACTIONS:
            summary = _vectorized_closed_loop_draw_summary(
                seed=seed,
                state_id=state_id,
                action=action,
                cfg=cfg,
                probs=probs_by_action[action],
                n_draws=n_draws,
            )
            rows.append({
                "seed": seed,
                "state_id": state_id,
                "record_type": "action_distribution",
                "action": ACTION_NAMES[action],
                **summary,
            })

    # Open-loop negative control.
    control_env = ClosedLoopEnvironment(
        cfg,
        REFERENCE_MORPHOLOGY,
        "action_open",
        np.random.default_rng(seed + 999),
    )
    state = random_probe_state(rng)
    p = [control_env.event_probabilities(state, forced_delayed_action=a) for a in ACTIONS]
    rows.append({
        "seed": seed,
        "state_id": -1,
        "record_type": "open_loop_control",
        "action_1": ACTION_NAMES[A_CONTINUE],
        "action_2": ACTION_NAMES[A_AVOID],
        "js_divergence": js_divergence(p[0], p[2]),
    })
    return rows


def fixed_event(
    env: ClosedLoopEnvironment,
    agent: EmbodiedAgent,
    event_type: str,
    intensity: float,
    friction: float = 0.42,
    slope: float = 0.0,
    affordance: float = 0.55,
    source: str = "external",
) -> Event:
    return env.generate_event(
        agent.state,
        forced_source=source,
        fixed_event_type=event_type,
        fixed_intensity=intensity,
        fixed_friction=friction,
        fixed_slope=slope,
        fixed_affordance=affordance,
    )


def controlled_history_exposure(agent: EmbodiedAgent, env: ClosedLoopEnvironment, harmful: bool, steps: int) -> None:
    event_cycle = ["collision", "landing", "slip"] if harmful else ["rest", "walk", "slope"]
    intensity_cycle = [4.0, 3.6, 3.4] if harmful else [0.05, 0.5, 0.6]
    rows: List[Dict[str, Any]] = []
    for t in range(steps):
        event_type = event_cycle[t % len(event_cycle)]
        intensity = intensity_cycle[t % len(intensity_cycle)]
        forced = A_CONTINUE if harmful else A_CAUTIOUS
        agent.step(
            env, agent.current_step, rows, learn=False, forced_action=forced,
            track_q_target=False,
            event_override={
                "forced_source": "external", "fixed_event_type": event_type,
                "fixed_intensity": intensity, "fixed_friction": 0.42,
                "fixed_slope": 0.05, "fixed_affordance": 0.60,
            },
        )


def _copy_generator(rng: np.random.Generator) -> np.random.Generator:
    new_rng = np.random.default_rng()
    new_rng.bit_generator.state = copy.deepcopy(rng.bit_generator.state)
    return new_rng


def lightweight_environment_clone(env: ClosedLoopEnvironment) -> ClosedLoopEnvironment:
    """Clone only mutable rollout state; configuration and morphology are immutable."""
    clone = copy.copy(env)
    clone.rng = _copy_generator(env.rng)
    clone.action_history = list(env.action_history)
    return clone


def lightweight_agent_clone(agent: EmbodiedAgent) -> EmbodiedAgent:
    """Create a no-learning rollout clone without copying replay/training arrays."""
    clone = copy.copy(agent)
    clone.rng = _copy_generator(agent.rng)
    clone.state = copy.deepcopy(agent.state)
    clone.agency = copy.deepcopy(agent.agency)
    clone.pending_q = []
    clone.outcome_cost_history = {}
    clone.q_history = list(agent.q_history)
    clone.memory_history = list(agent.memory_history)
    clone.agency_history = list(agent.agency_history)
    # These objects are read-only during controlled rollouts.
    clone.q_replay = agent.q_replay
    return clone


def controlled_future_target(
    agent: EmbodiedAgent,
    env: ClosedLoopEnvironment,
    event: Event,
    action: int,
    horizon: int,
) -> np.ndarray:
    """Return the same target used by online Q learning.

    The first term is the current event's pre-action burden under continuation,
    so a successful regulating action cannot erase the event's significance.
    Later terms are the actually experienced bodily costs under the frozen
    policy, discounted exactly as in ``mature_q_samples``.  V1 instead used
    net endpoint differences, which were dominated by unrelated later events
    and could cancel opposing changes.
    """
    clone = lightweight_agent_clone(agent)
    eclone = lightweight_environment_clone(env)
    clone._matured_since_update = 0

    raw = clone.counterfactual_continuation_costs(event).astype(float, copy=True)
    start_failed = bool(clone.state.ever_failed)

    # Apply the controlled current event.  Its actual mitigated costs are not
    # added because age zero is represented by the continuation counterfactual.
    clone.apply_event(event, int(action))
    eclone.push_action(int(action))
    clone.agency.push_action(int(action))

    discount = 0.92
    for age in range(1, max(1, int(horizon))):
        future_event = eclone.generate_event(clone.state)
        pain = clone.immediate_pain(future_event)
        agency = clone.agency_step(future_event, pain, learn=False)
        comp = clone.components(future_event, pain, float(agency["agency_score_used"]))
        future_action, _, _ = clone.choose_action(future_event, comp)
        outcome = clone.apply_event(future_event, future_action)
        eclone.push_action(future_action)
        clone.agency.push_action(future_action)
        actual_cost = np.array([
            max(0.0, outcome["damage_increment"]),
            max(0.0, -outcome["energy_change"]),
            max(0.0, outcome["fatigue_change"]),
            max(0.0, -outcome["stability_change"]),
        ], dtype=float)
        raw += (discount ** age) * actual_cost

    target = np.r_[raw, float(clone.state.ever_failed and not start_failed)]
    return clone._normalize_future_target(target)


def morphology_probe(
    seed: int,
    base_cfg: EnvironmentConfig,
    train_steps: int,
) -> List[Dict[str, Any]]:
    trained: Dict[str, Tuple[EmbodiedAgent, ClosedLoopEnvironment]] = {}
    common_training_seed = deterministic_seed(seed, "morphology_common_training")
    for morph_name, morph in MORPHOLOGIES.items():
        cfg = EnvironmentConfig(**{**asdict(base_cfg), "morphology": morph_name})
        # Identical initial learner weights and common exogenous random numbers;
        # trajectories may diverge only through morphology-dependent physics and
        # the resulting closed-loop actions.
        agent, env, _ = train_probe_agent(common_training_seed, cfg, morph, train_steps)
        trained[morph_name] = (agent, env)
    rows: List[Dict[str, Any]] = []
    for event_type, intensity in [("collision", 3.20), ("slip", 3.00), ("landing", 3.00)]:
        native_records = []
        for morph_name, (agent, env) in trained.items():
            clone = lightweight_agent_clone(agent)
            clone.state = AgentState(
                integrity=0.80, energy=0.60, fatigue=0.40, stability=0.60, damage=0.20,
                pain_memory=0.25, danger_memory=0.25, comfort_memory=0.65,
            )
            eclone = lightweight_environment_clone(env)
            eclone.rng = np.random.default_rng(deterministic_seed(seed, event_type, "common_probe"))
            event = fixed_event(eclone, clone, event_type, intensity, source="external")
            pain = clone.immediate_pain(event)
            comp = clone.components(event, pain, agency_score=0.0)
            target = controlled_future_target(clone, eclone, event, A_CAUTIOUS, base_cfg.horizon)
            native_loss = clone.q_integrator.loss(comp["z"], A_CAUTIOUS, target)
            foreign_losses = []
            for other_name, (other_agent, _) in trained.items():
                if other_name == morph_name:
                    continue
                foreign_losses.append(other_agent.q_integrator.loss(comp["z"], A_CAUTIOUS, target))
            deterioration = float(np.mean(target))
            native_records.append((comp["q"], deterioration))
            rows.append({
                "seed": seed, "event_type": event_type, "morphology": morph_name,
                "mass": clone.morph.mass, "energy_capacity": clone.morph.energy_capacity,
                "traction_tolerance": clone.morph.traction_tolerance,
                "damage_tolerance": clone.morph.damage_tolerance,
                "q": comp["q"], "fixed_q": comp["fixed_q"], "pain": pain,
                "danger": comp["danger"], "physical_risk": event.physical_risk,
                "realized_deterioration": deterioration,
                "native_prediction_loss": native_loss,
                "foreign_prediction_loss_mean": safe_mean(foreign_losses),
                "morphology_swap_penalty": safe_mean(foreign_losses) - native_loss,
            })
        rho = spearman_r([x[0] for x in native_records], [x[1] for x in native_records])
        for row in rows:
            if row["seed"] == seed and row["event_type"] == event_type:
                row["q_deterioration_spearman"] = rho
    return rows


def _agency_synthetic_sequence(
    rng: np.random.Generator,
    actions: np.ndarray,
    delay: int,
    factor: float,
    external: bool,
    total: int,
) -> np.ndarray:
    y = np.zeros(total, dtype=float)
    exo = 0.0
    effect = np.array([0.22, -0.16, -0.34], dtype=float)
    for t in range(total):
        exo = 0.80 * exo + float(rng.normal(0.0, 0.10))
        motor = 0.0 if external or t < delay else factor * effect[int(actions[t - delay])]
        y[t] = 0.55 * (y[t - 1] if t > 0 else 0.0) + motor + 0.25 * exo + float(rng.normal(0.0, 0.035))
    return y


def _agency_consequence_from_scalar(y: float, prev_y: float) -> np.ndarray:
    return np.array([
        np.tanh(y),
        np.clip((y + 1.5) / 3.0, 0.0, 1.0),
        np.clip(abs(y), 0.0, 1.0),
        np.clip(0.5 - 0.20 * y, 0.0, 1.0),
        np.clip(0.5 + 0.20 * (y - prev_y), 0.0, 1.0),
        np.clip(abs(y - prev_y), 0.0, 1.0),
    ], dtype=float)


def _pretrain_agency_estimator(seed: int, steps: int = 1200) -> DelayedActionAgencyEstimator:
    rng = np.random.default_rng(seed)
    est = DelayedActionAgencyEstimator(max_delay=7, seed=seed ^ 0xAC31)
    actions = rng.integers(0, len(ACTIONS), size=steps + 16)
    y = _agency_synthetic_sequence(rng, actions, delay=3, factor=1.0, external=False, total=steps)
    for t in range(steps):
        est.push_action(int(actions[t]))
        prev = y[t - 1] if t > 0 else 0.0
        est.observe(_agency_consequence_from_scalar(y[t], prev), [0.8, 0.7, 0.2, 0.8], learn=True)
    return est


def agency_mismatch_battery(
    seed: int,
    cfg: EnvironmentConfig,
    train_steps: int,
    n_trials: int,
) -> List[Dict[str, Any]]:
    """Block adaptation followed by held-out agency evaluation.

    V1 deep-copied a pretrained estimator for one observation per condition, so
    it could not adapt to a new delay.  V2 presents sustained blocks, separates
    adaptation from evaluation, and uses an exactly replayed sensory sequence
    for the passive-replay control.
    """
    base = _pretrain_agency_estimator(seed, steps=max(600, min(1600, train_steps // 2)))
    adapt_n = max(80, int(n_trials * 0.60))
    eval_n = max(40, int(n_trials) - adapt_n)
    total = adapt_n + eval_n
    trial_specs = [
        ("matched_self", cfg.true_delay, 1.0, False, 1),
        ("delayed_self_1", 1, 1.0, False, 1),
        ("delayed_self_3", 3, 1.0, False, 1),
        ("delayed_self_5", 5, 1.0, False, 1),
        ("delayed_self_7", 7, 1.0, False, 1),
        ("weak_self", cfg.true_delay, 0.50, False, 1),
        ("strong_self", cfg.true_delay, 1.50, False, 1),
        ("omitted_self", cfg.true_delay, 0.0, False, 1),
        ("direction_mismatch", cfg.true_delay, -0.50, False, 1),
        ("external", cfg.true_delay, 0.0, True, 0),
        ("external_coincidence", max(1, cfg.true_delay + 2), 0.55, False, 0),
    ]
    rows: List[Dict[str, Any]] = []
    matched_actions: Optional[np.ndarray] = None
    matched_y: Optional[np.ndarray] = None
    for block_id, (label, delay, factor, external, true_self) in enumerate(trial_specs):
        rng = np.random.default_rng(deterministic_seed(seed, "agency_block", label))
        actions = rng.integers(0, len(ACTIONS), size=total + 16)
        y = _agency_synthetic_sequence(rng, actions, delay, factor, external, total)
        if label == "matched_self":
            matched_actions, matched_y = actions.copy(), y.copy()
        est = copy.deepcopy(base)
        est.reset_evidence()
        for t in range(total):
            est.push_action(int(actions[t]))
            prev = y[t - 1] if t > 0 else 0.0
            out = est.observe(
                _agency_consequence_from_scalar(y[t], prev),
                [0.8, 0.7, 0.2, 0.8],
                learn=t < adapt_n,
            )
            if t >= adapt_n:
                rows.append({
                    "seed": seed,
                    "block_id": block_id,
                    "trial": t - adapt_n,
                    "phase": "heldout",
                    "trial_type": label,
                    "true_self_generated": true_self,
                    "true_delay": delay,
                    "agency_score": out["agency_score"],
                    "agency_evidence": out["agency_evidence"],
                    "inferred_delay": out["best_delay"],
                    "true_error": out["true_error"],
                    "decoy_error": out["decoy_error"],
                    "q": np.nan,
                    "fixed_q": np.nan,
                    "intensity": y[t],
                    "adaptation_steps": adapt_n,
                })

    if matched_actions is None or matched_y is None:
        raise RuntimeError("matched agency block was not generated")
    # Exact sensory replay with an independently permuted action sequence.
    rng = np.random.default_rng(deterministic_seed(seed, "passive_replay_actions"))
    replay_actions = matched_actions.copy()
    rng.shuffle(replay_actions)
    est = copy.deepcopy(base)
    est.reset_evidence()
    for t in range(total):
        est.push_action(int(replay_actions[t]))
        prev = matched_y[t - 1] if t > 0 else 0.0
        out = est.observe(
            _agency_consequence_from_scalar(matched_y[t], prev),
            [0.8, 0.7, 0.2, 0.8],
            learn=t < adapt_n,
        )
        if t >= adapt_n:
            rows.append({
                "seed": seed,
                "block_id": len(trial_specs),
                "trial": t - adapt_n,
                "phase": "heldout",
                "trial_type": "passive_replay",
                "true_self_generated": 0,
                "true_delay": cfg.true_delay,
                "agency_score": out["agency_score"],
                "agency_evidence": out["agency_evidence"],
                "inferred_delay": out["best_delay"],
                "true_error": out["true_error"],
                "decoy_error": out["decoy_error"],
                "q": np.nan,
                "fixed_q": np.nan,
                "intensity": matched_y[t],
                "adaptation_steps": adapt_n,
            })
    return rows


def heldout_event_generalization(
    seed: int,
    cfg: EnvironmentConfig,
    train_steps: int,
    n_trials: int,
    event_types: Sequence[str],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    target_cols = [
        "future_damage_increase", "future_energy_loss", "future_fatigue_increase",
        "future_stability_loss", "future_failure",
    ]
    for event_type in event_types:
        agent, env, train_rows = train_probe_agent(
            deterministic_seed(seed, "heldout", event_type), cfg, REFERENCE_MORPHOLOGY,
            train_steps, heldout_event=event_type,
        )
        train_df = pd.DataFrame(train_rows)
        train_df = train_df.dropna(subset=[c for c in target_cols if c in train_df.columns])
        if not train_df.empty and "event_type" in train_df.columns:
            train_df = train_df[train_df["event_type"] != event_type].copy()
        global_base = agent.cfg.target_means()
        action_base: Dict[int, np.ndarray] = {}
        if not train_df.empty and set(target_cols).issubset(train_df.columns):
            global_base = train_df[target_cols].mean().to_numpy(dtype=float)
            for action, d in train_df.groupby("action"):
                action_base[int(action)] = d[target_cols].mean().to_numpy(dtype=float)
        rng = np.random.default_rng(deterministic_seed(seed, "heldout_eval", event_type))
        for trial in range(n_trials):
            agent.state = random_probe_state(rng)
            eclone = lightweight_environment_clone(env)
            eclone.rng = np.random.default_rng(deterministic_seed(seed, event_type, trial))
            intensity = float(rng.uniform(0.5, 4.5))
            event = fixed_event(eclone, agent, event_type, intensity, source="external")
            pain = agent.immediate_pain(event)
            comp = agent.components(event, pain, agency_score=0.5)
            action = A_CAUTIOUS
            target = controlled_future_target(agent, eclone, event, action, cfg.horizon)
            _, q_pred = agent.q_integrator.predict(comp["z"], action)
            q_loss = agent.q_integrator.loss(comp["z"], action, target)
            null_loss = agent.q_integrator.null_loss(target)
            action_pred = np.clip(action_base.get(action, global_base), 1e-6, 1.0 - 1e-6)[None, :]
            action_loss = float(QualitativeValenceIntegrator._prediction_loss_vector(
                action_pred, target[None, :]
            )[0])
            rows.append({
                "seed": seed, "heldout_event": event_type, "trial": trial,
                "q": comp["q"], "fixed_q": comp["fixed_q"], "pain": pain,
                "danger": comp["danger"], "physical_risk": event.physical_risk,
                "future_compromise": float(np.mean(target)), "future_failure": target[4],
                "q_prediction_loss": q_loss,
                "null_prediction_loss": null_loss,
                "action_only_prediction_loss": action_loss,
                "q_improvement_vs_null": null_loss - q_loss,
                "q_improvement_vs_action_only": action_loss - q_loss,
                "q_pred_failure": q_pred[4],
            })
    return rows


def _same_event_state_from_trained(
    seed: int, agent: EmbodiedAgent, env: ClosedLoopEnvironment, n_pairs: int,
) -> List[Dict[str, Any]]:
    rng = np.random.default_rng(seed + 101)
    rows: List[Dict[str, Any]] = []
    for pair in range(n_pairs):
        common_memory = (
            float(rng.uniform(0.10, 0.50)), float(rng.uniform(0.10, 0.50)),
            float(rng.uniform(0.45, 0.85)),
        )
        states = {
            "stable": AgentState(
                integrity=float(rng.uniform(0.85, 1.00)), energy=float(rng.uniform(0.70, 1.00)),
                fatigue=float(rng.uniform(0.00, 0.25)), stability=float(rng.uniform(0.75, 1.00)),
                damage=0.0, pain_memory=common_memory[0], danger_memory=common_memory[1], comfort_memory=common_memory[2],
            ),
            "vulnerable": AgentState(
                integrity=float(rng.uniform(0.35, 0.65)), energy=float(rng.uniform(0.20, 0.50)),
                fatigue=float(rng.uniform(0.55, 0.85)), stability=float(rng.uniform(0.20, 0.50)),
                damage=0.0, pain_memory=common_memory[0], danger_memory=common_memory[1], comfort_memory=common_memory[2],
            ),
        }
        for state in states.values():
            state.damage = float(1.0 - state.integrity)
        for state_name, state in states.items():
            clone = lightweight_agent_clone(agent)
            clone.state = state
            eclone = lightweight_environment_clone(env)
            eclone.rng = np.random.default_rng(deterministic_seed(seed, pair, state_name))
            event = fixed_event(eclone, clone, "collision", 3.20, source="external")
            pain = clone.immediate_pain(event)
            comp = clone.components(event, pain, agency_score=0.5)
            rows.append({
                "seed": seed, "pair": pair, "state": state_name,
                "q": comp["q"], "fixed_q": comp["fixed_q"], "pain": pain,
                "danger": comp["danger"], "comfort": comp["comfort"],
                "action_possibility": comp["action_possibility"],
            })
    return rows


def _history_from_trained(
    seed: int, trained: EmbodiedAgent, env: ClosedLoopEnvironment, history_steps: int,
) -> List[Dict[str, Any]]:
    benign = lightweight_agent_clone(trained)
    harmful = lightweight_agent_clone(trained)
    env_b = lightweight_environment_clone(env)
    env_h = lightweight_environment_clone(env)
    env_b.rng = np.random.default_rng(seed + 301)
    env_h.rng = np.random.default_rng(seed + 302)
    controlled_history_exposure(benign, env_b, harmful=False, steps=history_steps)
    controlled_history_exposure(harmful, env_h, harmful=True, steps=history_steps)
    benign_memory = (benign.state.pain_memory, benign.state.danger_memory, benign.state.comfort_memory)
    harmful_memory = (harmful.state.pain_memory, harmful.state.danger_memory, harmful.state.comfort_memory)
    common = AgentState(integrity=0.75, energy=0.58, fatigue=0.42, stability=0.62, damage=0.25)
    rows: List[Dict[str, Any]] = []
    for history_name, base_agent, memory in [
        ("benign", benign, benign_memory), ("harmful", harmful, harmful_memory),
    ]:
        for memory_mode in ["intact", "zero", "pain_zero", "danger_zero", "comfort_zero"]:
            clone = lightweight_agent_clone(base_agent)
            clone.state = copy.deepcopy(common)
            clone.state.pain_memory, clone.state.danger_memory, clone.state.comfort_memory = memory
            clone.condition = Condition("probe", memory_mode=memory_mode)
            eclone = lightweight_environment_clone(env)
            eclone.rng = np.random.default_rng(deterministic_seed(seed, history_name, memory_mode))
            event = fixed_event(eclone, clone, "landing", 3.00, affordance=0.60, source="external")
            pain = clone.immediate_pain(event)
            comp = clone.components(event, pain, agency_score=0.5)
            rows.append({
                "seed": seed, "history": history_name, "memory_mode": memory_mode,
                "q": comp["q"], "fixed_q": comp["fixed_q"], "pain": pain,
                "danger": comp["danger"], "comfort": comp["comfort"],
                "pain_memory": memory[0], "danger_memory": memory[1], "comfort_memory": memory[2],
            })
    return rows


def _state_clone_from_trained(
    seed: int,
    trained: EmbodiedAgent,
    env: ClosedLoopEnvironment,
    train_rows: List[Dict[str, Any]],
    snapshots: int,
    horizon: int,
) -> List[Dict[str, Any]]:
    agent_base = copy.deepcopy(trained)
    env_base = copy.deepcopy(env)
    rows_base = list(train_rows)
    rows: List[Dict[str, Any]] = []
    spacing = max(1, agent_base.current_step // max(snapshots, 1) // 4)
    for snap in range(snapshots):
        for _ in range(spacing):
            agent_base.step(env_base, agent_base.current_step, rows_base, learn=True)
        for intervention in ["learned_q", "zero", "shuffled_q", "pain", "danger"]:
            agent = lightweight_agent_clone(agent_base)
            eclone = lightweight_environment_clone(env_base)
            common = deterministic_seed(seed, "clone", snap)
            agent.rng = np.random.default_rng(common)
            eclone.rng = np.random.default_rng(common ^ 0x9911)
            start = copy.deepcopy(agent.state)
            future_rows: List[Dict[str, Any]] = []
            for _ in range(horizon):
                agent.step(
                    eclone, agent.current_step, future_rows, learn=False,
                    signal_override=intervention, track_q_target=False,
                )
            rows.append({
                "seed": seed, "snapshot": snap, "intervention": intervention,
                "damage_increment": agent.state.damage - start.damage,
                "integrity_change": agent.state.integrity - start.integrity,
                "energy_change": agent.state.energy - start.energy,
                "fatigue_change": agent.state.fatigue - start.fatigue,
                "stability_change": agent.state.stability - start.stability,
                "failure": float(agent.state.ever_failed and not start.ever_failed),
                "action_sequence": ",".join(str(int(x["action"])) for x in future_rows),
                "regulation_rate": safe_mean([int(x["action"] != A_CONTINUE) for x in future_rows]),
            })
    return rows


def reference_probe_bundle_worker(task: Dict[str, Any]) -> Dict[str, Any]:
    cache = Path(task["cache_path"])
    if task.get("resume", False) and cache.exists():
        with cache.open("rb") as f:
            return pickle.load(f)
    cfg = EnvironmentConfig(**task["cfg"])
    seed = int(task["seed"])
    trained, env, train_rows = train_probe_agent(seed, cfg, REFERENCE_MORPHOLOGY, int(task["train_steps"]))
    result = {
        "state": _same_event_state_from_trained(seed, trained, env, int(task["state_pairs"])),
        "history": _history_from_trained(seed, trained, env, int(task["history_steps"])),
        "clone": _state_clone_from_trained(
            seed, trained, env, train_rows, int(task["snapshots"]), int(task["clone_horizon"])
        ),
    }
    cache.parent.mkdir(parents=True, exist_ok=True)
    with cache.open("wb") as f:
        pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
    return result


def closed_loop_probe_worker(task: Dict[str, Any]) -> List[Dict[str, Any]]:
    cache = Path(task["cache_path"])
    if task.get("resume", False) and cache.exists():
        with cache.open("rb") as f:
            return pickle.load(f)
    rows = closed_loop_probe(int(task["seed"]), EnvironmentConfig(**task["cfg"]), int(task["states"]), int(task["draws"]))
    cache.parent.mkdir(parents=True, exist_ok=True)
    with cache.open("wb") as f:
        pickle.dump(rows, f, protocol=pickle.HIGHEST_PROTOCOL)
    return rows


def morphology_probe_worker(task: Dict[str, Any]) -> List[Dict[str, Any]]:
    cache = Path(task["cache_path"])
    if task.get("resume", False) and cache.exists():
        with cache.open("rb") as f:
            return pickle.load(f)
    rows = morphology_probe(int(task["seed"]), EnvironmentConfig(**task["cfg"]), int(task["train_steps"]))
    cache.parent.mkdir(parents=True, exist_ok=True)
    with cache.open("wb") as f:
        pickle.dump(rows, f, protocol=pickle.HIGHEST_PROTOCOL)
    return rows


def agency_battery_worker(task: Dict[str, Any]) -> List[Dict[str, Any]]:
    cache = Path(task["cache_path"])
    if task.get("resume", False) and cache.exists():
        with cache.open("rb") as f:
            return pickle.load(f)
    rows = agency_mismatch_battery(
        int(task["seed"]), EnvironmentConfig(**task["cfg"]),
        int(task["train_steps"]), int(task["trials"]),
    )
    cache.parent.mkdir(parents=True, exist_ok=True)
    with cache.open("wb") as f:
        pickle.dump(rows, f, protocol=pickle.HIGHEST_PROTOCOL)
    return rows


def heldout_worker(task: Dict[str, Any]) -> List[Dict[str, Any]]:
    cache = Path(task["cache_path"])
    if task.get("resume", False) and cache.exists():
        with cache.open("rb") as f:
            return pickle.load(f)
    rows = heldout_event_generalization(
        int(task["seed"]), EnvironmentConfig(**task["cfg"]), int(task["train_steps"]),
        int(task["trials"]), list(task["event_types"]),
    )
    cache.parent.mkdir(parents=True, exist_ok=True)
    with cache.open("wb") as f:
        pickle.dump(rows, f, protocol=pickle.HIGHEST_PROTOCOL)
    return rows


def _run_parallel_stage(
    tasks: List[Dict[str, Any]], worker_fn: Any, workers: int, logger: Logger, label: str,
) -> List[Any]:
    out: List[Any] = []
    if workers > 1 and len(tasks) > 1:
        with mp.get_context("spawn").Pool(min(workers, len(tasks)), maxtasksperchild=8) as pool:
            iterator = pool.imap_unordered(worker_fn, tasks, chunksize=1)
            for i, result in enumerate(iterator, 1):
                out.append(result)
                logger.log(f"{label} progress {i}/{len(tasks)}")
    else:
        for i, task in enumerate(tasks, 1):
            out.append(worker_fn(task))
            logger.log(f"{label} progress {i}/{len(tasks)}")
    return out


# =============================================================================
# Statistics, figures, and report
# =============================================================================


def primary_paired_tests(
    ep: pd.DataFrame,
    n_boot: int,
    n_perm: int,
    logger: Optional[Logger] = None,
    checkpoint_path: Optional[Path] = None,
) -> pd.DataFrame:
    comparisons = [
        ("Fixed_Q", "learned_vs_fixed"),
        ("Action_Open_Loop_Q", "learned_vs_action_open_loop"),
        ("Pain_Only", "learned_vs_pain"),
        ("Danger_Only", "learned_vs_danger"),
        ("Physical_Risk_Only", "learned_vs_physical_risk"),
    ]
    outcomes = ["cumulative_damage", "final_integrity", "viability_failure", "action_cost"]
    # episode is the base seed index shared across challenge, morphology, and condition.
    keys = ["risk_label", "morphology", "episode"]
    rows: List[Dict[str, Any]] = []
    total_tests = len(comparisons) * len(outcomes)
    completed = 0
    for comparator, label in comparisons:
        a = ep[ep["condition"] == "Full_Learned_Q"]
        b = ep[ep["condition"] == comparator]
        merged = a.merge(b, on=keys, suffixes=("_full", "_comp"))
        for outcome in outcomes:
            cell_diff = merged[f"{outcome}_full"] - merged[f"{outcome}_comp"]
            temp = merged[["episode", "morphology"]].copy()
            temp["difference"] = cell_diff.to_numpy(dtype=float)
            # Inferential unit is the independent base seed, not 1,536 cells.
            seed_diff = temp.groupby("episode", sort=True)["difference"].mean().to_numpy(dtype=float)
            dz = float(np.mean(seed_diff) / np.std(seed_diff, ddof=1)) if len(seed_diff) > 1 and np.std(seed_diff, ddof=1) > 1e-12 else float("nan")
            low, high = hierarchical_bootstrap_ci(
                temp, "difference", seed_col="episode", n_boot=n_boot,
                rng_seed=deterministic_seed("v2_bootstrap", label, outcome),
            )
            rows.append({
                "comparison": label,
                "comparator": comparator,
                "outcome": outcome,
                "mean_difference_full_minus_comparator": safe_mean(seed_diff),
                "ci95_low": low,
                "ci95_high": high,
                "paired_dz": dz,
                "permutation_p": paired_permutation_p(
                    seed_diff, n_perm=n_perm, seed=deterministic_seed("v2_permutation", label, outcome)
                ),
                "n_pairs": int(len(seed_diff)),
                "n_seed_morphology_environment_cells": int(len(cell_diff)),
                "inferential_unit": "base_seed",
            })
            completed += 1
            if logger is not None:
                logger.log(f"paired statistics progress {completed}/{total_tests}: {label}, {outcome}")
            if checkpoint_path is not None:
                pd.DataFrame(rows).to_csv(checkpoint_path, index=False)
    out = pd.DataFrame(rows)
    if not out.empty:
        mask = out["outcome"] == "cumulative_damage"
        out.loc[mask, "holm_p"] = holm_adjust(out.loc[mask, "permutation_p"].to_numpy())
    return out


def append_clone_primary_test(tests: pd.DataFrame, clones: pd.DataFrame, n_perm: int) -> pd.DataFrame:
    if clones.empty:
        return tests
    pivot = clones.pivot_table(index=["seed", "snapshot"], columns="intervention", values="damage_increment")
    if not {"learned_q", "zero"}.issubset(pivot.columns):
        return tests
    snapshot_diff = (pivot["learned_q"] - pivot["zero"]).rename("difference").reset_index()
    seed_diff = snapshot_diff.groupby("seed", sort=True)["difference"].mean().to_numpy(dtype=float)
    rng = np.random.default_rng(deterministic_seed("clone_bootstrap"))
    boot = np.mean(seed_diff[rng.integers(0, len(seed_diff), size=(10000, len(seed_diff)))], axis=1) if len(seed_diff) else np.array([])
    row = pd.DataFrame([{
        "comparison": "intact_vs_zero_q_clone",
        "comparator": "zero_q_clone",
        "outcome": "cumulative_damage",
        "mean_difference_full_minus_comparator": safe_mean(seed_diff),
        "ci95_low": float(np.quantile(boot, 0.025)) if len(boot) else np.nan,
        "ci95_high": float(np.quantile(boot, 0.975)) if len(boot) else np.nan,
        "paired_dz": float(np.mean(seed_diff) / np.std(seed_diff, ddof=1)) if len(seed_diff) > 1 and np.std(seed_diff, ddof=1) > 1e-12 else np.nan,
        "permutation_p": paired_permutation_p(seed_diff, n_perm=n_perm),
        "n_pairs": int(len(seed_diff)),
        "n_seed_morphology_environment_cells": int(len(snapshot_diff)),
        "inferential_unit": "base_seed",
    }])
    out = pd.concat([tests, row], ignore_index=True)
    mask = out["outcome"] == "cumulative_damage"
    out.loc[mask, "holm_p"] = holm_adjust(out.loc[mask, "permutation_p"].to_numpy())
    return out


def failure_gee(ep: pd.DataFrame) -> pd.DataFrame:
    _ensure_statsmodels()
    if sm is None or ep.empty or ep["viability_failure"].nunique() < 2:
        return pd.DataFrame(columns=["term", "coefficient", "std_error", "p_value", "ci95_low", "ci95_high"])
    d = ep.copy()
    for factor in ["mass", "energy_capacity", "traction_tolerance", "damage_tolerance"]:
        if factor not in d.columns:
            return pd.DataFrame()
    try:
        model = smf.gee(
            "viability_failure ~ C(condition) + C(risk_label) + mass + energy_capacity + traction_tolerance + damage_tolerance",
            groups="episode", data=d, family=sm.families.Binomial(),
        ).fit()
        return pd.DataFrame({
            "term": model.params.index,
            "coefficient": model.params.values,
            "std_error": model.bse.values,
            "p_value": model.pvalues.values,
            "ci95_low": model.conf_int()[0].values,
            "ci95_high": model.conf_int()[1].values,
        })
    except Exception as exc:
        return pd.DataFrame([{"term": "ERROR", "coefficient": np.nan, "std_error": np.nan, "p_value": np.nan, "message": str(exc)}])


def morphology_factorial(ep: pd.DataFrame) -> pd.DataFrame:
    _ensure_statsmodels()
    if smf is None or ep.empty:
        return pd.DataFrame(columns=["term", "coefficient", "std_error", "p_value", "ci95_low", "ci95_high"])
    try:
        d = ep.copy()
        formula = (
            "cumulative_damage ~ C(condition) + C(risk_label) + mass + energy_capacity + "
            "traction_tolerance + damage_tolerance + C(condition):mass + "
            "C(condition):energy_capacity + C(condition):traction_tolerance + C(condition):damage_tolerance"
        )
        model = smf.ols(formula, data=d).fit(cov_type="cluster", cov_kwds={"groups": d["episode"]})
        ci = model.conf_int()
        return pd.DataFrame({
            "term": model.params.index, "coefficient": model.params.values,
            "std_error": model.bse.values, "p_value": model.pvalues.values,
            "ci95_low": ci[0].values, "ci95_high": ci[1].values,
        })
    except Exception as exc:
        return pd.DataFrame([{"term": "ERROR", "coefficient": np.nan, "std_error": np.nan, "p_value": np.nan, "message": str(exc)}])



def morphology_stratified_audit(
    ep: pd.DataFrame,
    prospective: pd.DataFrame,
    agency: pd.DataFrame,
) -> pd.DataFrame:
    if ep.empty:
        return pd.DataFrame(columns=[
            "record_type", "condition", "risk_label", "morphology",
            "failure_rate", "q_failure_brier", "agency_brier",
            "component_advantage_damage",
        ])
    base = ep.groupby(["condition", "risk_label", "morphology"], as_index=False).agg(
        failure_rate=("viability_failure", "mean"),
        cumulative_damage=("cumulative_damage", "mean"),
    )
    if not prospective.empty:
        qpred = prospective[prospective["predictor"] == "learned_Q_decoder"].groupby(
            ["condition", "risk_label", "morphology"], as_index=False
        )["failure_brier"].mean().rename(columns={"failure_brier": "q_failure_brier"})
        base = base.merge(qpred, on=["condition", "risk_label", "morphology"], how="left")
    else:
        base["q_failure_brier"] = np.nan
    if not agency.empty:
        ag = agency.groupby(["condition", "risk_label", "morphology"], as_index=False)["agency_brier"].mean()
        base = base.merge(ag, on=["condition", "risk_label", "morphology"], how="left")
    else:
        base["agency_brier"] = np.nan

    singles = ["Pain_Only", "Danger_Only", "Physical_Risk_Only", "Body_State_Only"]
    full = base[base["condition"] == "Full_Learned_Q"][["risk_label", "morphology", "cumulative_damage"]].rename(
        columns={"cumulative_damage": "full_damage"}
    )
    single_best = base[base["condition"].isin(singles)].groupby(
        ["risk_label", "morphology"], as_index=False
    )["cumulative_damage"].min().rename(columns={"cumulative_damage": "best_single_damage"})
    advantage = full.merge(single_best, on=["risk_label", "morphology"], how="left")
    advantage["component_advantage_damage"] = advantage["best_single_damage"] - advantage["full_damage"]
    base = base.merge(
        advantage[["risk_label", "morphology", "component_advantage_damage"]],
        on=["risk_label", "morphology"], how="left",
    )
    base["record_type"] = "morphology_level"

    disparity_rows = []
    for (condition, risk), d in base.groupby(["condition", "risk_label"]):
        disparity_rows.append({
            "record_type": "max_minus_min_disparity",
            "condition": condition,
            "risk_label": risk,
            "morphology": "__DISPARITY__",
            "failure_rate": float(d["failure_rate"].max() - d["failure_rate"].min()),
            "q_failure_brier": float(d["q_failure_brier"].max() - d["q_failure_brier"].min()) if d["q_failure_brier"].notna().any() else np.nan,
            "agency_brier": float(d["agency_brier"].max() - d["agency_brier"].min()) if d["agency_brier"].notna().any() else np.nan,
            "component_advantage_damage": float(d["component_advantage_damage"].max() - d["component_advantage_damage"].min()) if d["component_advantage_damage"].notna().any() else np.nan,
            "cumulative_damage": float(d["cumulative_damage"].max() - d["cumulative_damage"].min()),
        })
    return pd.concat([base, pd.DataFrame(disparity_rows)], ignore_index=True, sort=False)


# Vulnerability profiles all sum to 1.0, so a profile change is a reweighting
# rather than a change of scale; the gain factor changes scale separately.
VULNERABILITY_PROFILES: Dict[str, Tuple[float, float, float, float]] = {
    "uniform": (0.25, 0.25, 0.25, 0.25),
    "integrity_heavy": (0.55, 0.15, 0.20, 0.10),
    "fatigue_heavy": (0.20, 0.45, 0.20, 0.15),
}

_Q_CONDITIONS = ("Full_Learned_Q",)
# The vulnerability term feeds immediate pain and damage pressure, so it is
# probed under the learned integrator and under the comparator that beat it.
_VULNERABILITY_CONDITIONS = ("Full_Learned_Q", "Pain_Only")
# Avoidance pressure enters Q_fixed only (it is not one of the twelve learned
# components), so it can only be probed in the Fixed_Q condition.
_AVOIDANCE_CONDITIONS = ("Fixed_Q",)


def robustness_settings() -> List[Dict[str, Any]]:
    baseline: Dict[str, Any] = {
        "setting": "baseline", "risk_scale_multiplier": 1.0, "horizon": 25,
        "beta": 8.0, "delay": 1, "predictor_lr_multiplier": 1.0,
        "q_lr_multiplier": 1.0, "memory_decay_multiplier": 1.0,
        "mitigation_multiplier": 1.0,
        "vulnerability_weights": tuple(VULNERABILITY_WEIGHTS),
        "avoidance_coefficients": dict(AVOIDANCE_COEFFICIENTS),
        "agency_prior_weight": AGENCY_PRIOR_WEIGHT,
        "agency_current_weight": AGENCY_CURRENT_WEIGHT,
        "agency_gain": AGENCY_GAIN,
        "conditions": _Q_CONDITIONS,
    }
    settings: List[Dict[str, Any]] = [dict(baseline)]

    factors = {
        "risk_scale_multiplier": [0.75, 1.25],
        "horizon": [10, 50],
        "beta": [4.0, 12.0],
        "delay": [3, 5, 7],
        "predictor_lr_multiplier": [0.5, 1.5],
        "q_lr_multiplier": [0.5, 1.5],
        "memory_decay_multiplier": [0.9, 1.1],
        "mitigation_multiplier": [0.8, 1.2],
    }
    for factor, values in factors.items():
        for value in values:
            row = dict(baseline)
            row[factor] = value
            row["setting"] = f"{factor}={value}"
            settings.append(row)

    # --- vulnerability term, Eq. 21 -----------------------------------------
    base_v = dict(baseline)
    base_v["conditions"] = _VULNERABILITY_CONDITIONS
    base_v["setting"] = "vulnerability_baseline"
    settings.append(base_v)
    for name, weights in VULNERABILITY_PROFILES.items():
        row = dict(base_v)
        row["vulnerability_weights"] = tuple(weights)
        row["setting"] = f"vulnerability_profile={name}"
        settings.append(row)
    for gain in (0.8, 1.2):
        row = dict(base_v)
        row["vulnerability_weights"] = tuple(gain * w for w in VULNERABILITY_WEIGHTS)
        row["setting"] = f"vulnerability_gain={gain}"
        settings.append(row)

    # --- avoidance pressure, Eq. 64 (Fixed_Q only) --------------------------
    base_a = dict(baseline)
    base_a["conditions"] = _AVOIDANCE_CONDITIONS
    base_a["setting"] = "avoidance_baseline"
    settings.append(base_a)
    for key, values in {"danger": [1.15, 2.15], "pain": [0.70, 1.50],
                        "comfort": [-1.25, -0.45], "controllability": [-0.85, -0.25]}.items():
        for value in values:
            row = dict(base_a)
            coef = dict(AVOIDANCE_COEFFICIENTS)
            coef[key] = value
            row["avoidance_coefficients"] = coef
            row["setting"] = f"avoidance_{key}={value}"
            settings.append(row)
    for gain in (0.8, 1.2):
        row = dict(base_a)
        coef = {k: (v if k == "bias" else gain * v) for k, v in AVOIDANCE_COEFFICIENTS.items()}
        row["avoidance_coefficients"] = coef
        row["setting"] = f"avoidance_gain={gain}"
        settings.append(row)

    # --- agency evidence mixture, Eq. 38-39 ---------------------------------
    for value in (0.0, 0.35, 0.75, 1.0):
        row = dict(baseline)
        row["agency_prior_weight"] = value
        row["agency_current_weight"] = 1.0 - value
        row["setting"] = f"agency_prior_weight={value}"
        settings.append(row)
    for value in (3.0, 12.0):
        row = dict(baseline)
        row["agency_gain"] = value
        row["setting"] = f"agency_gain={value}"
        settings.append(row)
    return settings


def run_robustness_battery(
    outdir: Path,
    base_cfg: EnvironmentConfig,
    morphologies: Sequence[str],
    base_seed: int,
    workers: int,
    episodes: int,
    steps: int,
    burn_in: int,
    resume: bool,
    logger: Logger,
    common_seeds: bool = False,
) -> pd.DataFrame:
    tasks: List[Dict[str, Any]] = []
    for setting in robustness_settings():
      for condition_name in setting.get("conditions", ("Full_Learned_Q",)):
        for morph_name, episode in itertools.product(morphologies, range(episodes)):
            cfg = EnvironmentConfig(**{
                **asdict(base_cfg),
                "risk_label": "robustness",
                "risk_scale": base_cfg.risk_scale * float(setting["risk_scale_multiplier"]),
                "true_delay": int(setting["delay"]),
                "horizon": int(setting["horizon"]),
                "morphology": morph_name,
            })
            common_seed = (
                deterministic_seed(base_seed, "robustness", morph_name, episode)
                if common_seeds else
                deterministic_seed(base_seed, "robustness", setting["setting"], morph_name, episode)
            )
            tasks.append({
                "cfg": asdict(cfg), "condition": condition_name, "episode": episode,
                "common_seed": common_seed, "steps": int(steps), "burn_in": int(min(burn_in, steps // 3)),
                "beta": float(setting["beta"]),
                "q_lr_multiplier": float(setting["q_lr_multiplier"]),
                "predictor_lr_multiplier": float(setting["predictor_lr_multiplier"]),
                "memory_decay_multiplier": float(setting["memory_decay_multiplier"]),
                "mitigation_multiplier": float(setting["mitigation_multiplier"]),
                "vulnerability_weights": tuple(setting.get("vulnerability_weights", VULNERABILITY_WEIGHTS)),
                "avoidance_coefficients": dict(setting.get("avoidance_coefficients", AVOIDANCE_COEFFICIENTS)),
                "agency_prior_weight": float(setting.get("agency_prior_weight", AGENCY_PRIOR_WEIGHT)),
                "agency_current_weight": float(setting.get("agency_current_weight", AGENCY_CURRENT_WEIGHT)),
                "agency_gain": float(setting.get("agency_gain", AGENCY_GAIN)),
                "outdir": str(outdir / "robustness_cache"), "resume": resume,
                "save_step_timeseries": False, "robustness_setting": setting["setting"],
            })
    logger.log(f"robustness battery: {len(tasks)} Full-Q episodes")
    results = []
    if workers > 1:
        with mp.get_context("spawn").Pool(workers, maxtasksperchild=8) as pool:
            for i, result in enumerate(pool.imap_unordered(run_episode_task, tasks, chunksize=1), start=1):
                results.append(result)
                if i % max(1, len(tasks) // 10) == 0 or i == len(tasks):
                    logger.log(f"robustness progress {i}/{len(tasks)}")
    else:
        for i, task in enumerate(tasks, start=1):
            results.append(run_episode_task(task))
            if i % max(1, len(tasks) // 10) == 0 or i == len(tasks):
                logger.log(f"robustness progress {i}/{len(tasks)}")
    rows = []
    for result in results:
        rows.append({**result["meta"], **result["summary"], **result.get("agency", {})})
    raw = pd.DataFrame(rows)
    if raw.empty:
        return raw
    for column, default in (("agency_auc", float("nan")), ("agency_delay_mae", float("nan")),
                            ("agency_mismatch_drop", float("nan"))):
        if column not in raw.columns:
            raw[column] = default
    return raw.groupby(["robustness_setting", "condition"], as_index=False).agg(
        final_integrity_mean=("final_integrity", "mean"),
        final_integrity_sd=("final_integrity", "std"),
        cumulative_damage_mean=("cumulative_damage", "mean"),
        cumulative_damage_sd=("cumulative_damage", "std"),
        failure_rate=("viability_failure", "mean"),
        regulation_rate=("regulation_rate", "mean"),
        q_loss_ema=("q_loss_ema", "mean"),
        agency_auc=("agency_auc", "mean"),
        agency_delay_mae=("agency_delay_mae", "mean"),
        agency_mismatch_drop=("agency_mismatch_drop", "mean"),
        n=("episode", "size"),
    )


def save_figures(
    outdir: Path,
    ep: pd.DataFrame,
    weights: pd.DataFrame,
    loop: pd.DataFrame,
    state_probe: pd.DataFrame,
    morph_probe: pd.DataFrame,
    agency_battery: pd.DataFrame,
    clones: pd.DataFrame,
) -> None:
    _ensure_matplotlib()
    figdir = outdir / "figures"
    figdir.mkdir(exist_ok=True)
    if not ep.empty:
        core = ep.groupby("condition")["cumulative_damage"].agg(["mean", "std"]).sort_values("mean")
        plt.figure(figsize=(11, 6))
        plt.bar(np.arange(len(core)), core["mean"], yerr=core["std"], capsize=3)
        plt.xticks(np.arange(len(core)), core.index, rotation=45, ha="right")
        plt.ylabel("Cumulative damage")
        plt.title("Figure 1. Component and structural controls")
        plt.tight_layout()
        plt.savefig(figdir / "figure1_component_controls.png", dpi=300)
        plt.close()
    if not weights.empty:
        w = weights[weights["condition"] == "Full_Learned_Q"].groupby("component")["weight"].agg(["mean", "std"]).sort_values("mean")
        plt.figure(figsize=(9, 6))
        plt.barh(np.arange(len(w)), w["mean"], xerr=w["std"], capsize=3)
        plt.yticks(np.arange(len(w)), w.index)
        plt.xlabel("Learned integration weight")
        plt.title("Figure 2. Learned Q integration")
        plt.tight_layout()
        plt.savefig(figdir / "figure2_q_weights.png", dpi=300)
        plt.close()
    if not loop.empty:
        d = loop[loop["record_type"] == "action_distribution"].groupby("action")["mean_physical_risk"].agg(["mean", "std"])
        if not d.empty:
            plt.figure(figsize=(7, 5))
            plt.bar(np.arange(len(d)), d["mean"], yerr=d["std"], capsize=3)
            plt.xticks(np.arange(len(d)), d.index, rotation=20)
            plt.ylabel("Expected next-event physical risk")
            plt.title("Figure 3. Closed-loop action intervention")
            plt.tight_layout()
            plt.savefig(figdir / "figure3_closed_loop.png", dpi=300)
            plt.close()
    if not state_probe.empty:
        d = state_probe.groupby("state")["q"].agg(["mean", "std"])
        plt.figure(figsize=(6, 5))
        plt.bar(np.arange(len(d)), d["mean"], yerr=d["std"], capsize=3)
        plt.xticks(np.arange(len(d)), d.index)
        plt.ylabel("Learned Q")
        plt.title("Figure 4. Same-event bodily-state dependence")
        plt.tight_layout()
        plt.savefig(figdir / "figure4_state_dependence.png", dpi=300)
        plt.close()
    if not morph_probe.empty:
        d = morph_probe.groupby("morphology")["q"].mean().sort_values()
        plt.figure(figsize=(10, 6))
        plt.bar(np.arange(len(d)), d.values)
        plt.xticks(np.arange(len(d)), d.index, rotation=70, ha="right")
        plt.ylabel("Learned Q")
        plt.title("Figure 5. Morphology-dependent valence")
        plt.tight_layout()
        plt.savefig(figdir / "figure5_morphology.png", dpi=300)
        plt.close()
    if not agency_battery.empty:
        d = agency_battery.groupby("trial_type")["agency_score"].agg(["mean", "std"]).sort_values("mean")
        plt.figure(figsize=(10, 6))
        plt.bar(np.arange(len(d)), d["mean"], yerr=d["std"], capsize=3)
        plt.xticks(np.arange(len(d)), d.index, rotation=45, ha="right")
        plt.ylabel("Agency score")
        plt.title("Figure 6. Agency mismatch battery")
        plt.tight_layout()
        plt.savefig(figdir / "figure6_agency_mismatch.png", dpi=300)
        plt.close()
    if not clones.empty:
        d = clones.groupby("intervention")["damage_increment"].agg(["mean", "std"]).sort_values("mean")
        plt.figure(figsize=(7, 5))
        plt.bar(np.arange(len(d)), d["mean"], yerr=d["std"], capsize=3)
        plt.xticks(np.arange(len(d)), d.index, rotation=25)
        plt.ylabel("Damage over cloned future")
        plt.title("Figure 7. State-clone Q interventions")
        plt.tight_layout()
        plt.savefig(figdir / "figure7_state_clone.png", dpi=300)
        plt.close()


def write_report(
    outdir: Path,
    selected: pd.DataFrame,
    ep: pd.DataFrame,
    prospective: pd.DataFrame,
    agency: pd.DataFrame,
    tests: pd.DataFrame,
    loop: pd.DataFrame,
    state_probe: pd.DataFrame,
    history: pd.DataFrame,
    morphology: pd.DataFrame,
    battery: pd.DataFrame,
    clones: pd.DataFrame,
    generalization: pd.DataFrame,
) -> None:
    lines: List[str] = []
    lines.append("Closed-loop qualitative-valence V2 validation\n")
    lines.append("=" * 88 + "\n")
    lines.append(f"Generated: {now()}\n\n")
    lines.append("MODEL STATUS\n")
    lines.append("The code tests the strong Path 1 model. It does not presuppose that the claims must be reduced.\n")
    lines.append("Interpretation is to be determined only from the confirmatory outputs.\n\n")
    lines.append("1. FROZEN ENVIRONMENT CALIBRATION\n")
    lines.append(selected.to_string(index=False, float_format=lambda x: f"{x:.6g}") + "\n\n")
    lines.append("2. CONFIRMATORY REGULATORY OUTCOMES\n")
    if not ep.empty:
        summary = ep.groupby("condition")[["cumulative_damage", "final_integrity", "viability_failure", "regulation_rate", "action_cost"]].agg(["mean", "std"])
        lines.append(summary.to_string(float_format=lambda x: f"{x:.6f}") + "\n\n")
    lines.append("3. PROSPECTIVE VALIDITY\n")
    if not prospective.empty:
        lines.append(prospective.groupby("predictor")[["failure_auc", "failure_brier", "high_compromise_auc", "compromise_r"]].mean().to_string(float_format=lambda x: f"{x:.6f}") + "\n\n")
    lines.append("4. AGENCY DURING EPISODES\n")
    if not agency.empty:
        lines.append(agency.groupby("condition")[["agency_auc", "agency_brier", "agency_mismatch_drop", "agency_delay_mae"]].mean().to_string(float_format=lambda x: f"{x:.6f}") + "\n\n")
    lines.append("5. PRIMARY PAIRED TESTS\n")
    if not tests.empty:
        lines.append(tests.to_string(index=False, float_format=lambda x: f"{x:.6g}") + "\n\n")
    lines.append("6. SENSORIMOTOR-LOOP INTERVENTION\n")
    if not loop.empty:
        action_rows = loop[loop["record_type"] == "action_distribution"]
        lines.append(action_rows.groupby("action")[["mean_intensity", "mean_physical_risk", "slip_probability", "collision_probability"]].mean().to_string(float_format=lambda x: f"{x:.6f}") + "\n")
        lines.append(f"Mean pairwise JSD: {safe_mean(loop.loc[loop['record_type']=='jsd', 'js_divergence']):.6f}\n")
        lines.append(f"Action-open-loop control JSD: {safe_mean(loop.loc[loop['record_type']=='open_loop_control', 'js_divergence']):.6f}\n\n")
    lines.append("7. SAME-EVENT BODY-STATE PROBE\n")
    if not state_probe.empty:
        pivot = state_probe.groupby("state")[["q", "fixed_q", "pain", "danger"]].mean()
        lines.append(pivot.to_string(float_format=lambda x: f"{x:.6f}") + "\n\n")
    lines.append("8. EXPERIENCED-HISTORY MEMORY PROBE\n")
    if not history.empty:
        lines.append(history.groupby(["memory_mode", "history"])["q"].mean().unstack("history").to_string(float_format=lambda x: f"{x:.6f}") + "\n\n")
    lines.append("9. MORPHOLOGY PROBE\n")
    if not morphology.empty:
        lines.append(morphology.groupby("event_type")[["q_deterioration_spearman", "morphology_swap_penalty"]].mean().to_string(float_format=lambda x: f"{x:.6f}") + "\n\n")
    lines.append("10. AGENCY MISMATCH BATTERY\n")
    if not battery.empty:
        battery_cols = [c for c in ["agency_score", "agency_evidence", "inferred_delay", "true_error", "decoy_error"] if c in battery.columns]
        lines.append(battery.groupby("trial_type")[battery_cols].mean().to_string(float_format=lambda x: f"{x:.6f}") + "\n\n")
    lines.append("11. STATE-CLONE CAUSAL INTERVENTIONS\n")
    if not clones.empty:
        lines.append(clones.groupby("intervention")[["damage_increment", "integrity_change", "regulation_rate", "failure"]].mean().to_string(float_format=lambda x: f"{x:.6f}") + "\n\n")
    lines.append("12. HELD-OUT EVENT GENERALIZATION\n")
    if not generalization.empty:
        gen_cols = [c for c in [
            "q_prediction_loss", "null_prediction_loss", "action_only_prediction_loss",
            "q_improvement_vs_null", "q_improvement_vs_action_only", "q", "future_compromise",
        ] if c in generalization.columns]
        lines.append(generalization.groupby("heldout_event")[gen_cols].mean().to_string(float_format=lambda x: f"{x:.6f}") + "\n\n")
    lines.append("INTERPRETIVE RULE\n")
    lines.append("The outputs do not automatically support either Path 1 or Path 2. The final theoretical framing must be selected from the complete results, including negative findings and component baselines.\n")
    (outdir / "19_consolidated_report.txt").write_text("".join(lines), encoding="utf-8")


# =============================================================================
# Main
# =============================================================================


def parse_csv(text: str) -> List[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def build_manifest(args: argparse.Namespace, preset: RunPreset, outdir: Path) -> Dict[str, Any]:
    script = Path(__file__).resolve()
    return {
        "generated": now(),
        "script": str(script),
        "script_sha256": sha256_file(script),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "preset": args.preset,
        "preset_values": asdict(preset),
        "arguments": vars(args),
        "morphologies": {k: asdict(v) for k, v in MORPHOLOGIES.items()},
        "conditions": {k: asdict(v) for k, v in CONDITIONS.items()},
        "q_components": Q_COMPONENT_NAMES,
        "future_targets": FUTURE_TARGET_NAMES,
    }


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--outdir", type=str, default=str(Path.home() / "Desktop" / "qualitative_valence_closed_loop_main"))
    parser.add_argument("--preset", choices=PRESETS.keys(), default="main")
    parser.add_argument("--workers", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--burn-in", type=int, default=None)
    parser.add_argument("--horizon", type=int, default=25)
    parser.add_argument("--true-delay", type=int, default=1)
    parser.add_argument("--beta", type=float, default=8.0)
    parser.add_argument("--conditions", type=str, default="")
    parser.add_argument("--morphologies", type=str, default="")
    parser.add_argument("--challenge-levels", type=str, default="")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--save-step-timeseries", action="store_true")
    parser.add_argument("--skip-generalization", action="store_true")
    parser.add_argument("--run-robustness-grid", action="store_true")
    parser.add_argument("--skip-robustness", action="store_true")
    parser.add_argument("--robustness-seeds", type=int, default=None)
    parser.add_argument("--robustness-steps", type=int, default=None)
    parser.add_argument("--robustness-common-seeds", action="store_true",
                        help="pair robustness settings on common random numbers "
                             "(changes seeds; the published battery used per-setting seeds)")
    parser.add_argument("--risk-scale-multiplier", type=float, default=1.0)
    parser.add_argument("--q-lr-multiplier", type=float, default=1.0)
    parser.add_argument("--predictor-lr-multiplier", type=float, default=1.0)
    parser.add_argument("--memory-decay-multiplier", type=float, default=1.0)
    parser.add_argument("--mitigation-multiplier", type=float, default=1.0)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--permutations", type=int, default=10000)
    args = parser.parse_args()

    outdir = Path(os.path.expanduser(args.outdir)).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    logger = Logger(outdir)
    preset = PRESETS[args.preset]
    if args.episodes is not None:
        preset = RunPreset(**{**asdict(preset), "episodes": int(args.episodes)})
    if args.steps is not None:
        preset = RunPreset(**{**asdict(preset), "steps": int(args.steps)})
    if args.burn_in is not None:
        preset = RunPreset(**{**asdict(preset), "burn_in": int(args.burn_in)})
    conditions = parse_csv(args.conditions) if args.conditions else list(preset.conditions)
    morphologies = parse_csv(args.morphologies) if args.morphologies else list(preset.morphology_names)
    challenges = parse_csv(args.challenge_levels) if args.challenge_levels else list(preset.challenge_labels)
    for c in conditions:
        if c not in CONDITIONS:
            raise ValueError(f"Unknown condition: {c}")
    for m in morphologies:
        if m not in MORPHOLOGIES:
            raise ValueError(f"Unknown morphology: {m}")

    (outdir / "00_manifest.json").write_text(
        json.dumps(build_manifest(args, preset, outdir), indent=2, default=str), encoding="utf-8"
    )
    logger.log("starting independent environment calibration")
    calibration, selected_configs = calibrate_environments(
        outdir, preset, args.seed, args.workers, args.horizon, args.true_delay,
        args.resume, logger
    )
    selected_df = pd.DataFrame([asdict(selected_configs[k]) for k in ["mild", "moderate", "challenging"]])

    confirmatory_paths = {
        "episode": outdir / "03_episode_summary.csv",
        "prospective": outdir / "04_prospective_metrics.csv",
        "agency": outdir / "05_agency_episode_metrics.csv",
        "weights": outdir / "06_learned_q_weights.csv",
    }
    if args.resume and all(path.exists() for path in confirmatory_paths.values()):
        logger.log("resume: loading completed confirmatory output tables")
        ep_df = read_csv_allow_empty(confirmatory_paths["episode"])
        prospective_df = read_csv_allow_empty(confirmatory_paths["prospective"])
        agency_df = read_csv_allow_empty(confirmatory_paths["agency"])
        weights_df = read_csv_allow_empty(confirmatory_paths["weights"])
    else:
        tasks: List[Dict[str, Any]] = []
        for challenge, morph_name, episode, condition_name in itertools.product(
            challenges, morphologies, range(preset.episodes), conditions
        ):
            base = selected_configs[challenge]
            cfg = EnvironmentConfig(**{
                **asdict(base), "risk_scale": base.risk_scale * float(args.risk_scale_multiplier),
                "true_delay": int(args.true_delay), "horizon": int(args.horizon),
                "morphology": morph_name,
            })
            common_seed = deterministic_seed(args.seed, "confirmatory", challenge, morph_name, episode)
            tasks.append({
                "cfg": asdict(cfg), "condition": condition_name, "episode": episode,
                "common_seed": common_seed, "steps": preset.steps, "burn_in": preset.burn_in,
                "beta": args.beta, "q_lr_multiplier": args.q_lr_multiplier,
                "predictor_lr_multiplier": args.predictor_lr_multiplier,
                "memory_decay_multiplier": args.memory_decay_multiplier,
                "mitigation_multiplier": args.mitigation_multiplier,
                "outdir": str(outdir), "resume": args.resume,
                "save_step_timeseries": args.save_step_timeseries,
            })
        logger.log(f"confirmatory episodes: {len(tasks)}")
        results: List[Dict[str, Any]] = []
        if args.workers > 1:
            with mp.get_context("spawn").Pool(args.workers, maxtasksperchild=16) as pool:
                for i, result in enumerate(pool.imap_unordered(run_episode_task, tasks, chunksize=1), start=1):
                    results.append(result)
                    if i % max(1, len(tasks) // 20) == 0 or i == len(tasks):
                        logger.log(f"confirmatory progress {i}/{len(tasks)}")
        else:
            for i, task in enumerate(tasks, start=1):
                results.append(run_episode_task(task))
                if i % max(1, len(tasks) // 20) == 0 or i == len(tasks):
                    logger.log(f"confirmatory progress {i}/{len(tasks)}")

        episode_rows: List[Dict[str, Any]] = []
        prospective_rows: List[Dict[str, Any]] = []
        agency_rows: List[Dict[str, Any]] = []
        weight_rows: List[Dict[str, Any]] = []
        for result in results:
            meta = result["meta"]
            morph = MORPHOLOGIES[meta["morphology"]]
            common = {**meta, **asdict(morph)}
            episode_rows.append({**common, **result["summary"]})
            for row in result["prospective"]:
                prospective_rows.append({**common, **row})
            agency_rows.append({**common, **result["agency"]})
            for component, weight in result["weights"].items():
                weight_rows.append({**common, "component": component, "weight": weight})
        def _ordered(frame: pd.DataFrame) -> pd.DataFrame:
            keys = [c for c in ("condition", "risk_label", "morphology", "episode", "component")
                    if c in frame.columns]
            if not keys or frame.empty:
                return frame
            return frame.sort_values(keys, kind="mergesort").reset_index(drop=True)

        ep_df = _ordered(pd.DataFrame(episode_rows))
        prospective_df = _ordered(pd.DataFrame(prospective_rows))
        agency_df = _ordered(pd.DataFrame(agency_rows))
        weights_df = _ordered(pd.DataFrame(weight_rows))
        ep_df.to_csv(confirmatory_paths["episode"], index=False)
        prospective_df.to_csv(confirmatory_paths["prospective"], index=False)
        agency_df.to_csv(confirmatory_paths["agency"], index=False)
        weights_df.to_csv(confirmatory_paths["weights"], index=False)

    moderate = selected_configs["moderate"]
    probe_cfg = EnvironmentConfig(**{
        **asdict(moderate), "risk_scale": moderate.risk_scale * float(args.risk_scale_multiplier),
        "true_delay": int(args.true_delay), "horizon": int(args.horizon),
        "morphology": "reference",
    })

    loop_output = outdir / "10_closed_loop_probe.csv"
    if args.resume and loop_output.exists():
        logger.log("resume: loading completed sensorimotor-loop intervention probe")
        loop_df = read_csv_allow_empty(loop_output)
    else:
        logger.log("running sensorimotor-loop intervention probe in parallel")
        cache_dir = outdir / "probe_cache" / "closed_loop_v2"
        tasks = [{
            "seed": deterministic_seed(args.seed, "loop", i),
            "cfg": asdict(probe_cfg),
            "states": preset.loop_states,
            "draws": preset.loop_draws,
            "resume": args.resume,
            "cache_path": str(cache_dir / f"seed_{i:04d}_{deterministic_seed('V2L', asdict(probe_cfg), preset.loop_states, preset.loop_draws)}.pkl"),
        } for i in range(preset.probe_seeds)]
        results = _run_parallel_stage(tasks, closed_loop_probe_worker, args.workers, logger, "sensorimotor-loop probe")
        loop_df = pd.DataFrame([row for part in results for row in part])
        loop_df.to_csv(loop_output, index=False)

    state_output = outdir / "11_same_event_state_probe.csv"
    history_output = outdir / "12_history_memory_probe.csv"
    clone_output = outdir / "15_state_clone_interventions.csv"
    if args.resume and state_output.exists() and history_output.exists() and clone_output.exists():
        logger.log("resume: loading completed shared reference-agent probe bundle")
        state_df = read_csv_allow_empty(state_output)
        history_df = read_csv_allow_empty(history_output)
        clone_df = read_csv_allow_empty(clone_output)
    else:
        logger.log("training one reference agent per seed and reusing it across state, history, and clone probes")
        cache_dir = outdir / "probe_cache" / "reference_bundle_v2"
        tasks = [{
            "seed": deterministic_seed(args.seed, "reference_bundle", i),
            "cfg": asdict(probe_cfg),
            "train_steps": preset.probe_train_steps,
            "state_pairs": preset.state_pairs,
            "history_steps": preset.history_steps,
            "snapshots": preset.clone_snapshots,
            "clone_horizon": preset.clone_horizon,
            "resume": args.resume,
            "cache_path": str(cache_dir / f"seed_{i:04d}_{deterministic_seed('V2L', asdict(probe_cfg), preset.probe_train_steps, preset.state_pairs, preset.history_steps, preset.clone_snapshots, preset.clone_horizon)}.pkl"),
        } for i in range(preset.probe_seeds)]
        bundles = _run_parallel_stage(tasks, reference_probe_bundle_worker, args.workers, logger, "reference probe bundle")
        state_df = pd.DataFrame([row for bundle in bundles for row in bundle["state"]])
        history_df = pd.DataFrame([row for bundle in bundles for row in bundle["history"]])
        clone_df = pd.DataFrame([row for bundle in bundles for row in bundle["clone"]])
        state_df.to_csv(state_output, index=False)
        history_df.to_csv(history_output, index=False)
        clone_df.to_csv(clone_output, index=False)

    morph_output = outdir / "13_morphology_probe.csv"
    if args.resume and morph_output.exists():
        logger.log("resume: loading completed morphology-dependent valence probe")
        morph_df = read_csv_allow_empty(morph_output)
    else:
        logger.log("running morphology-dependent valence probe in parallel by seed")
        cache_dir = outdir / "probe_cache" / "morphology_v2"
        tasks = [{
            "seed": deterministic_seed(args.seed, "morphology", i),
            "cfg": asdict(probe_cfg),
            "train_steps": preset.probe_train_steps,
            "resume": args.resume,
            "cache_path": str(cache_dir / f"seed_{i:04d}_{deterministic_seed('V2L', asdict(probe_cfg), preset.probe_train_steps, 'morphology_v2')}.pkl"),
        } for i in range(preset.probe_seeds)]
        results = _run_parallel_stage(tasks, morphology_probe_worker, args.workers, logger, "morphology probe")
        morph_df = pd.DataFrame([row for part in results for row in part])
        morph_df.to_csv(morph_output, index=False)

    battery_output = outdir / "14_agency_mismatch_battery.csv"
    if args.resume and battery_output.exists():
        logger.log("resume: loading completed agency mismatch battery")
        battery_df = read_csv_allow_empty(battery_output)
    else:
        logger.log("running block-adaptation agency battery in parallel by seed")
        cache_dir = outdir / "probe_cache" / "agency_v2"
        tasks = [{
            "seed": deterministic_seed(args.seed, "agency_battery", i),
            "cfg": asdict(probe_cfg),
            "train_steps": preset.probe_train_steps,
            "trials": preset.agency_trials,
            "resume": args.resume,
            "cache_path": str(cache_dir / f"seed_{i:04d}_{deterministic_seed('V2L', asdict(probe_cfg), preset.probe_train_steps, preset.agency_trials, 'agency_v2')}.pkl"),
        } for i in range(preset.probe_seeds)]
        results = _run_parallel_stage(tasks, agency_battery_worker, args.workers, logger, "agency battery")
        battery_df = pd.DataFrame([row for part in results for row in part])
        battery_df.to_csv(battery_output, index=False)

    generalization_output = outdir / "16_heldout_event_generalization.csv"
    generalization_df = pd.DataFrame()
    if args.resume and generalization_output.exists():
        logger.log("resume: loading completed held-out event generalization")
        generalization_df = read_csv_allow_empty(generalization_output)
    elif not args.skip_generalization:
        logger.log("running held-out event generalization in parallel by seed")
        event_subset = EVENT_TYPES if args.preset != "smoke" else ["collision", "slip"]
        cache_dir = outdir / "probe_cache" / "heldout_v2"
        tasks = [{
            "seed": deterministic_seed(args.seed, "generalization", i),
            "cfg": asdict(probe_cfg),
            "train_steps": preset.probe_train_steps,
            "trials": preset.generalization_trials,
            "event_types": event_subset,
            "resume": args.resume,
            "cache_path": str(cache_dir / f"seed_{i:04d}_{deterministic_seed('V2L', asdict(probe_cfg), preset.probe_train_steps, preset.generalization_trials, tuple(event_subset), 'heldout_v2')}.pkl"),
        } for i in range(preset.probe_seeds)]
        results = _run_parallel_stage(tasks, heldout_worker, args.workers, logger, "held-out generalization")
        generalization_df = pd.DataFrame([row for part in results for row in part])
        generalization_df.to_csv(generalization_output, index=False)
    else:
        generalization_df.to_csv(generalization_output, index=False)


    stats_paths = {
        "tests": outdir / "07_primary_paired_tests.csv",
        "gee": outdir / "08_failure_gee.csv",
        "factorial": outdir / "09_morphology_factorial.csv",
        "audit": outdir / "17_morphology_stratified_audit.csv",
    }
    if args.resume and all(path.exists() for path in stats_paths.values()):
        logger.log("resume: loading completed paired and model-based statistics")
        tests_df = read_csv_allow_empty(stats_paths["tests"])
        gee_df = read_csv_allow_empty(stats_paths["gee"])
        morph_factor_df = read_csv_allow_empty(stats_paths["factorial"])
        morphology_audit_df = read_csv_allow_empty(stats_paths["audit"])
    else:
        logger.log("running paired and model-based statistics")
        n_boot = min(args.bootstrap, 500 if args.preset == "smoke" else args.bootstrap)
        n_perm = min(args.permutations, 500 if args.preset == "smoke" else args.permutations)
        partial_stats = outdir / "07_primary_paired_tests.partial.csv"

        logger.log(f"statistics stage 1/4: paired tests ({n_boot} bootstrap, {n_perm} permutations)")
        tests_df = primary_paired_tests(
            ep_df, n_boot=n_boot, n_perm=n_perm, logger=logger, checkpoint_path=partial_stats
        )
        tests_df = append_clone_primary_test(tests_df, clone_df, n_perm=n_perm)
        tests_df.to_csv(stats_paths["tests"], index=False)
        if partial_stats.exists():
            partial_stats.unlink()
        logger.log("statistics stage 1/4 completed")

        logger.log("statistics stage 2/4: binomial GEE")
        gee_df = failure_gee(ep_df)
        gee_df.to_csv(stats_paths["gee"], index=False)
        logger.log("statistics stage 2/4 completed")

        logger.log("statistics stage 3/4: morphology factorial regression")
        morph_factor_df = morphology_factorial(ep_df)
        morph_factor_df.to_csv(stats_paths["factorial"], index=False)
        logger.log("statistics stage 3/4 completed")

        logger.log("statistics stage 4/4: morphology-stratified audit")
        morphology_audit_df = morphology_stratified_audit(ep_df, prospective_df, agency_df)
        morphology_audit_df.to_csv(stats_paths["audit"], index=False)
        logger.log("statistics stage 4/4 completed")

    robustness_df = pd.DataFrame(columns=[
        "robustness_setting", "final_integrity_mean", "final_integrity_sd",
        "cumulative_damage_mean", "cumulative_damage_sd", "failure_rate",
        "regulation_rate", "q_loss_ema", "n",
    ])
    run_robustness = args.run_robustness_grid or (args.preset in {"main", "reviewer"} and not args.skip_robustness)
    if run_robustness:
        robustness_seeds = args.robustness_seeds or (2 if args.preset == "smoke" else 8)
        robustness_steps = args.robustness_steps or min(preset.steps, 2000)
        robustness_morphs = morphologies if len(morphologies) <= 4 else morphologies[::max(1, len(morphologies)//4)][:4]
        robustness_df = run_robustness_battery(
            outdir, probe_cfg, robustness_morphs, args.seed, args.workers,
            robustness_seeds, robustness_steps, preset.burn_in, args.resume, logger,
            common_seeds=bool(args.robustness_common_seeds),
        )
    robustness_df.to_csv(outdir / "18_robustness_summary.csv", index=False)

    logger.log("writing figures and consolidated report")
    save_figures(outdir, ep_df, weights_df, loop_df, state_df, morph_df, battery_df, clone_df)
    write_report(
        outdir, selected_df, ep_df, prospective_df, agency_df, tests_df,
        loop_df, state_df, history_df, morph_df, battery_df, clone_df, generalization_df,
    )
    logger.log("completed")
    print(f"Output directory: {outdir}", flush=True)


if __name__ == "__main__":
    mp.freeze_support()
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
