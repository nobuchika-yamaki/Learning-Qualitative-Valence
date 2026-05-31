#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
qualitative_valence_robustness_audit_v7.py

Robustness / sensitivity audit for the learned qualitative-valence + agency model.

Purpose
-------
This script freezes the v6 model logic and tests whether the core claims remain
stable when we vary:

1. seed count / independent episodes
2. environmental risk: mild / moderate / harsh
3. damage scaling: 0.75x / 1.00x / 1.25x by default
4. agency delay: multiple true sensorimotor delays
5. lesion type: none / Q lesion / agency lesion / memory lesion

No Gemini.
No LMM.
No semantic module.

Primary outputs
---------------
- consolidated_report.txt
- scenario_summary.csv
- support_matrix.csv
- lesion_summary.csv
- behavior_prediction_metrics.csv
- irreducibility_metrics.csv
- probe_metrics.csv
- agency_metrics.csv
- figures/*.png
- episode_csv/*.csv
- run_progress.log

Default run
-----------
python3 -u qualitative_valence_robustness_audit_v7.py \
  --outdir ~/Desktop/qualitative_valence_v7_robustness_audit \
  --steps 2500 \
  --episodes 6 \
  2>&1 | tee ~/Desktop/qualitative_valence_v7_robustness_audit.log

Quick run
---------
python3 -u qualitative_valence_robustness_audit_v7.py \
  --outdir ~/Desktop/qualitative_valence_v7_quick \
  --steps 600 \
  --episodes 3 \
  --quick \
  2>&1 | tee ~/Desktop/qualitative_valence_v7_quick.log

Resume
------
python3 -u qualitative_valence_robustness_audit_v7.py \
  --outdir ~/Desktop/qualitative_valence_v7_robustness_audit \
  --steps 2500 \
  --episodes 6 \
  --resume \
  2>&1 | tee -a ~/Desktop/qualitative_valence_v7_robustness_audit.log
"""

import argparse
import math
import os
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =============================================================================
# Utility
# =============================================================================

def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


class Logger:
    def __init__(self, outdir: Path):
        self.outdir = outdir
        self.outdir.mkdir(parents=True, exist_ok=True)
        self.path = self.outdir / "run_progress.log"
        with open(self.path, "w", encoding="utf-8") as f:
            f.write(f"[{now()}] started\n")

    def log(self, msg: str):
        line = f"[{now()}] {msg}"
        print(line, flush=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


def clip01(x):
    return float(np.clip(x, 0.0, 1.0))


def safe_mean(x):
    x = np.asarray(x, dtype=float)
    return float(np.mean(x)) if len(x) else np.nan


def safe_std(x):
    x = np.asarray(x, dtype=float)
    if len(x) <= 1:
        return 0.0
    return float(np.std(x, ddof=1))


def cohen_d(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) < 2 or len(b) < 2:
        return np.nan
    s1 = np.var(a, ddof=1)
    s2 = np.var(b, ddof=1)
    sp = math.sqrt(((len(a) - 1) * s1 + (len(b) - 1) * s2) / max(1, len(a) + len(b) - 2))
    if sp < 1e-12:
        return np.nan
    return float((np.mean(a) - np.mean(b)) / sp)


def auc_score(y_true, y_score):
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    scores = np.concatenate([pos, neg])
    labels = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    for us in np.unique(scores):
        idx = np.where(scores == us)[0]
        if len(idx) > 1:
            ranks[idx] = np.mean(ranks[idx])
    rank_sum_pos = np.sum(ranks[labels == 1])
    return float((rank_sum_pos - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg)))


def r2_score(y, pred):
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    ss_res = np.sum((y - pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    if ss_tot < 1e-12:
        return np.nan
    return float(1.0 - ss_res / ss_tot)


def standardize_train_apply(X_train, X_test):
    X_train = np.asarray(X_train, dtype=float)
    X_test = np.asarray(X_test, dtype=float)
    mu = np.mean(X_train, axis=0)
    sd = np.std(X_train, axis=0)
    sd[sd < 1e-8] = 1.0
    return (X_train - mu) / sd, (X_test - mu) / sd


def logistic_fit_predict(X, y, X_test, lr=0.06, n_iter=450, l2=1e-3):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    X_test = np.asarray(X_test, dtype=float)
    X_aug = np.column_stack([np.ones(len(X)), X])
    X_test_aug = np.column_stack([np.ones(len(X_test)), X_test])
    w = np.zeros(X_aug.shape[1], dtype=float)
    for _ in range(n_iter):
        p = sigmoid(X_aug @ w)
        grad = (X_aug.T @ (p - y)) / len(y)
        grad[1:] += l2 * w[1:]
        w -= lr * grad
    return sigmoid(X_test_aug @ w), w


def ridge_fit_predict(X, y, X_test, alpha=1e-3):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    X_test = np.asarray(X_test, dtype=float)
    X_aug = np.column_stack([np.ones(len(X)), X])
    X_test_aug = np.column_stack([np.ones(len(X_test)), X_test])
    I = np.eye(X_aug.shape[1])
    I[0, 0] = 0.0
    beta = np.linalg.pinv(X_aug.T @ X_aug + alpha * I) @ X_aug.T @ y
    return X_test_aug @ beta, beta


def parse_csv_list(text: str, cast=float):
    if text is None or str(text).strip() == "":
        return []
    return [cast(x.strip()) for x in str(text).split(",") if x.strip() != ""]


# =============================================================================
# State and model
# =============================================================================

EVENT_TYPES = ["rest", "walk", "slope", "slip", "jump", "landing", "collision", "brake"]
RISK_SCALES = {"mild": 0.75, "moderate": 1.0, "harsh": 1.25}
LESION_MODES = ["none", "q", "agency", "memory"]


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


@dataclass
class Event:
    event_type: str
    intensity: float
    mass: float
    friction: float
    slope: float
    affordance: float
    physical_risk: float
    self_generated: int
    efference_intensity: float
    true_delay: int


@dataclass
class EnvConfig:
    risk_label: str = "moderate"
    risk_scale: float = 1.0
    damage_multiplier: float = 1.0
    damage_scale: float = 0.000010
    noise_scale: float = 0.02
    true_delay: int = 3


class EfferenceBuffer:
    def __init__(self, max_lag=12):
        self.max_lag = max_lag
        self.buffer = [(0.0, 0) for _ in range(max_lag + 1)]

    def push(self, efference_intensity: float, self_generated: int):
        self.buffer.insert(0, (float(efference_intensity), int(self_generated)))
        self.buffer = self.buffer[:self.max_lag + 1]

    def get(self, lag: int):
        lag = int(np.clip(lag, 0, self.max_lag))
        return self.buffer[lag]


class OnlineLogistic:
    def __init__(self, n_features: int, lr=0.035, l2=1e-4, init_bias=0.0, rng=None):
        self.n_features = n_features
        self.lr = lr
        self.l2 = l2
        self.rng = np.random.default_rng(0) if rng is None else rng
        self.w = self.rng.normal(0.0, 0.03, size=n_features + 1)
        self.w[0] = init_bias
        self.count = 0

    def predict(self, x):
        x = np.asarray(x, dtype=float)
        return float(sigmoid(self.w[0] + np.dot(self.w[1:], x)))

    def update(self, x, y, weight=1.0):
        x = np.asarray(x, dtype=float)
        y = float(y)
        p = self.predict(x)
        err = p - y
        self.w[0] -= self.lr * weight * err
        self.w[1:] -= self.lr * weight * (err * x + self.l2 * self.w[1:])
        self.count += 1
        return p


class LearnedValenceModel:
    def __init__(self, rng, true_delay=3):
        self.rng = rng
        self.true_delay = true_delay
        self.max_lag = 12
        self.danger_model = OnlineLogistic(13, lr=0.040, init_bias=-0.2, rng=rng)
        self.comfort_model = OnlineLogistic(13, lr=0.035, init_bias=0.2, rng=rng)
        self.action_model = OnlineLogistic(13, lr=0.035, init_bias=0.1, rng=rng)
        self.agency_model = OnlineLogistic(6, lr=0.045, init_bias=-0.2, rng=rng)

    def features(self, event: Event, state: AgentState, agency_score: float):
        high_contact = 1.0 if event.event_type in ["slip", "landing", "collision", "brake"] else 0.0
        return np.array([
            event.intensity / 5.5,
            event.friction,
            event.slope / 0.35,
            event.affordance,
            event.physical_risk,
            state.integrity,
            state.energy,
            state.fatigue,
            state.stability,
            state.pain_memory,
            state.danger_memory,
            high_contact,
            agency_score,
        ], dtype=float)

    def agency_features_for_lag(self, event: Event, eff_intensity: float, eff_flag: int, lag: int):
        alignment = math.exp(-abs(event.intensity - eff_intensity) / 1.35)
        delay_prior = math.exp(-abs(lag - self.true_delay) / 2.0)
        high_contact = 1.0 if event.event_type in ["landing", "collision", "slip"] else 0.0
        return np.array([
            eff_intensity / 5.5,
            float(eff_flag),
            event.intensity / 5.5,
            alignment,
            delay_prior,
            high_contact,
        ], dtype=float)

    def estimate_agency(self, event: Event, buffer: EfferenceBuffer, lesion_mode="none"):
        if lesion_mode == "agency":
            return {
                "agency_score": 0.5,
                "agency_best_lag": -1,
                "agency_best_efference": 0.0,
                "agency_best_flag": 0,
                "agency_x": np.zeros(6),
            }
        scores = []
        for lag in range(self.max_lag + 1):
            eff, flag = buffer.get(lag)
            x = self.agency_features_for_lag(event, eff, flag, lag)
            p_learned = self.agency_model.predict(x)
            alignment = math.exp(-abs(event.intensity - eff) / 1.35)
            delay_prior = math.exp(-abs(lag - self.true_delay) / 1.4)
            p_analytic = sigmoid(3.8 * float(flag) + 2.2 * alignment + 2.0 * delay_prior - 4.2)
            p = 0.45 * p_learned + 0.55 * p_analytic
            scores.append((lag, p, eff, flag, x))
        best = max(scores, key=lambda z: z[1])
        return {
            "agency_score": float(best[1]),
            "agency_best_lag": int(best[0]),
            "agency_best_efference": float(best[2]),
            "agency_best_flag": int(best[3]),
            "agency_x": best[4],
        }

    def compute_components(self, event: Event, state: AgentState, buffer: EfferenceBuffer, pain: float, lesion_mode="none"):
        # Memory lesion removes history from the state visible to Q computation.
        if lesion_mode == "memory":
            state_view = AgentState(
                integrity=state.integrity,
                energy=state.energy,
                fatigue=state.fatigue,
                stability=state.stability,
                damage=state.damage,
                pain_memory=0.0,
                danger_memory=0.0,
                comfort_memory=0.55,
            )
        else:
            state_view = state

        ag = self.estimate_agency(event, buffer, lesion_mode=lesion_mode)
        agency_score = ag["agency_score"]
        x = self.features(event, state_view, agency_score)
        danger = self.danger_model.predict(x)
        comfort = self.comfort_model.predict(x)
        action_possibility = self.action_model.predict(x)

        if lesion_mode == "q":
            # Q lesion removes learned integrative valence while preserving weak physical reactivity.
            danger = clip01(0.24 + 0.28 * event.physical_risk)
            comfort = clip01(0.62 - 0.10 * event.physical_risk)
            action_possibility = clip01(0.04 + 0.06 * event.affordance)
            agency_score = 0.5
            x = self.features(event, state_view, agency_score)

        controllability = action_possibility * (0.65 + 0.35 * agency_score)
        avoidance_pressure = sigmoid(
            1.65 * danger
            + 1.10 * pain
            - 0.85 * comfort
            - 0.55 * controllability
            + 0.55 * state_view.danger_memory
            + 0.25 * (1.0 - state_view.integrity)
            - 0.05
        )
        q_aversive_index = clip01(
            0.23 * danger
            + 0.20 * pain
            + 0.20 * avoidance_pressure
            + 0.12 * (1.0 - action_possibility)
            + 0.07 * (1.0 - comfort)
            + 0.08 * state_view.danger_memory
            + 0.05 * state_view.pain_memory
            + 0.03 * state_view.fatigue
            + 0.02 * (1.0 - state_view.stability)
        )
        return {
            "comfort": float(comfort),
            "pain": float(pain),
            "danger": float(danger),
            "avoidance_pressure": float(avoidance_pressure),
            "action_possibility": float(action_possibility),
            "q_aversive_index": float(q_aversive_index),
            "agency_score": float(agency_score),
            "agency_best_lag": int(ag["agency_best_lag"]),
            "agency_best_efference": float(ag["agency_best_efference"]),
            "agency_best_flag": int(ag["agency_best_flag"]),
            "feature_vector": x,
            "agency_x": ag["agency_x"],
        }

    def update_from_outcome(self, comp: Dict, event: Event, outcome: Dict, buffer: EfferenceBuffer, lesion_mode="none"):
        if lesion_mode == "q":
            # Q-lesioned runs are intentionally not trained as a full learned-valence model.
            return
        x = comp["feature_vector"]
        damage_target = 1.0 if (outcome["damage_increment"] > 0.000010 or outcome["pain_after"] > 0.66) else 0.0
        comfort_target = 1.0 if (
            outcome["damage_increment"] <= 0.000010
            and outcome["pain_after"] < 0.52
            and outcome["post_energy"] > 0.45
            and outcome["post_stability"] > 0.45
        ) else 0.0
        if outcome["action_avoid"] == 1:
            action_target = 1.0 if (
                outcome["damage_increment"] <= 0.000010
                and outcome["effective_risk"] < max(0.58, event.physical_risk + 0.02)
            ) else 0.0
        else:
            action_target = 1.0 if outcome["damage_increment"] <= 0.000010 and outcome["effective_risk"] < 0.55 else 0.0

        self.danger_model.update(x, damage_target, weight=1.0)
        self.comfort_model.update(x, comfort_target, weight=1.0)
        self.action_model.update(x, action_target, weight=0.85)

        if lesion_mode != "agency":
            for lag in range(self.max_lag + 1):
                eff, flag = buffer.get(lag)
                ax = self.agency_features_for_lag(event, eff, flag, lag)
                target = 1.0 if (event.self_generated == 1 and flag == 1 and abs(lag - event.true_delay) <= 1) else 0.0
                if event.self_generated == 0:
                    target = 0.0
                self.agency_model.update(ax, target, weight=0.45)


# =============================================================================
# Generative process
# =============================================================================

def generate_event(rng: np.random.Generator, state: AgentState, cfg: EnvConfig, buffer: EfferenceBuffer) -> Event:
    self_generated = int(rng.random() < 0.58)
    et = rng.choice(EVENT_TYPES, p=np.array([0.10, 0.15, 0.12, 0.13, 0.12, 0.13, 0.14, 0.11]))
    mass = float(rng.uniform(48.0, 88.0))
    friction = float(rng.choice([0.06, 0.26, 0.42, 0.72], p=[0.22, 0.18, 0.42, 0.18]))
    slope = float(rng.uniform(0.0, 0.35))
    affordance = float(rng.uniform(0.18, 1.0))

    if et == "rest":
        intensity = float(rng.uniform(0.0, 0.08))
    elif et == "walk":
        intensity = float(rng.uniform(0.18, 1.10))
    elif et == "slope":
        intensity = float(rng.uniform(0.45, 2.20) + 1.80 * slope)
    elif et == "slip":
        intensity = float(rng.uniform(0.85, 3.65) * (1.0 - 0.55 * friction + 0.18))
    elif et == "jump":
        intensity = float(rng.uniform(0.75, 2.80))
    elif et == "landing":
        intensity = float(rng.uniform(1.05, 4.40))
    elif et == "collision":
        intensity = float(rng.uniform(0.90, 4.85))
    elif et == "brake":
        intensity = float(rng.uniform(0.45, 3.10) * (1.0 - 0.25 * friction))
    else:
        intensity = float(rng.uniform(0.1, 1.0))

    intensity = float(max(0.0, intensity * cfg.risk_scale + rng.normal(0, cfg.noise_scale)))
    physical_risk = sigmoid(1.00 * (intensity - 2.35) + 0.55 * (0.28 - friction) + 0.35 * slope)

    efference_intensity = intensity if self_generated else 0.0
    buffer.push(0.0, 0)
    delay = int(np.clip(cfg.true_delay, 0, buffer.max_lag))
    buffer.buffer[delay] = (efference_intensity if self_generated else 0.0, self_generated)

    return Event(
        event_type=str(et),
        intensity=float(intensity),
        mass=mass,
        friction=friction,
        slope=slope,
        affordance=affordance,
        physical_risk=float(physical_risk),
        self_generated=self_generated,
        efference_intensity=float(efference_intensity),
        true_delay=delay,
    )


def immediate_pain(event: Event, state: AgentState):
    if event.event_type in ["collision", "landing", "slip"]:
        impact = event.intensity
    elif event.event_type in ["slope", "brake"]:
        impact = 0.55 * event.intensity
    elif event.event_type == "jump":
        impact = 0.35 * event.intensity
    else:
        impact = 0.12 * event.intensity
    vulnerability = 0.35 * (1.0 - state.integrity) + 0.25 * state.fatigue + 0.25 * (1.0 - state.stability)
    return float(sigmoid(1.25 * (impact - 2.20) + 1.25 * vulnerability))


def choose_action(event: Event, state: AgentState, q: Dict, rng: np.random.Generator, lesion_mode="none"):
    if lesion_mode == "q":
        p = sigmoid(0.65 * event.physical_risk - 1.25 + rng.normal(0, 0.08))
    else:
        p = sigmoid(
            1.55 * q["danger"]
            + 1.25 * q["avoidance_pressure"]
            + 0.60 * q["pain"]
            - 0.72 * q["comfort"]
            - 0.35 * q["action_possibility"]
            - 0.48
        )
    p = clip01(p + rng.normal(0.0, 0.025))
    return int(rng.random() < p), float(p)


def apply_event(event: Event, state: AgentState, action: int, q: Dict, cfg: EnvConfig, rng: np.random.Generator, lesion_mode="none"):
    pre = asdict(state)
    if action == 1:
        mitigation = 0.18 + 0.58 * q["action_possibility"] + 0.10 * q.get("agency_score", 0.5)
        mitigation = clip01(mitigation)
        energy_cost = 0.006 + 0.014 * event.intensity * (1.0 - 0.45 * q["action_possibility"])
        fatigue_gain = 0.002 + 0.006 * event.intensity
    else:
        mitigation = 0.0
        energy_cost = 0.003 + 0.010 * event.intensity
        fatigue_gain = 0.001 + 0.004 * event.intensity

    effective_intensity = event.intensity * (1.0 - mitigation)
    effective_risk = sigmoid(1.00 * (effective_intensity - 2.45) + 0.55 * (0.30 - event.friction) + 0.28 * event.slope)
    vulnerability = 0.32 * (1.0 - state.integrity) + 0.28 * state.fatigue + 0.24 * (1.0 - state.energy) + 0.22 * (1.0 - state.stability)
    raw_damage = max(0.0, effective_risk + vulnerability - 0.75)
    damage_increment = cfg.damage_scale * (raw_damage ** 2)
    pain_after = float(sigmoid(1.12 * (effective_intensity - 2.30) + 1.08 * vulnerability))

    if event.event_type == "rest":
        state.energy = clip01(state.energy + 0.030)
        state.fatigue = clip01(state.fatigue - 0.026)
        state.stability = clip01(state.stability + 0.023)
    else:
        state.energy = clip01(state.energy - energy_cost + 0.003)
        state.fatigue = clip01(state.fatigue + fatigue_gain - 0.002)
        state.stability = clip01(state.stability - 0.007 - 0.025 * effective_risk + 0.010 * action)

    state.damage = clip01(state.damage + damage_increment)
    state.integrity = clip01(1.0 - state.damage)

    if lesion_mode == "memory":
        state.pain_memory = 0.0
        state.danger_memory = 0.0
        state.comfort_memory = 0.55
    else:
        state.pain_memory = clip01(0.940 * state.pain_memory + 0.060 * pain_after)
        state.danger_memory = clip01(0.945 * state.danger_memory + 0.055 * effective_risk)
        state.comfort_memory = clip01(0.945 * state.comfort_memory + 0.055 * q["comfort"])

    if state.energy > 0.38 and state.stability > 0.45:
        state.fatigue = clip01(state.fatigue - 0.0035)
        state.stability = clip01(state.stability + 0.0030)

    return {
        **{f"pre_{k}": v for k, v in pre.items()},
        "effective_intensity": float(effective_intensity),
        "effective_risk": float(effective_risk),
        "pain_after": float(pain_after),
        "damage_increment": float(damage_increment),
        "energy_cost": float(energy_cost),
        "post_integrity": state.integrity,
        "post_energy": state.energy,
        "post_fatigue": state.fatigue,
        "post_stability": state.stability,
        "post_damage": state.damage,
        "post_pain_memory": state.pain_memory,
        "post_danger_memory": state.danger_memory,
        "post_comfort_memory": state.comfort_memory,
    }


# =============================================================================
# Episode and probes
# =============================================================================

def scenario_id(cfg: EnvConfig):
    return f"risk-{cfg.risk_label}_dmg-{cfg.damage_multiplier:g}_delay-{cfg.true_delay}"


def run_episode(seed: int, steps: int, cfg: EnvConfig, lesion_mode: str, logger: Logger, prefix: str):
    rng = np.random.default_rng(seed)
    state = AgentState()
    buffer = EfferenceBuffer(max_lag=12)
    model = LearnedValenceModel(rng, true_delay=cfg.true_delay)
    rows = []
    marks = set([int(steps * k / 10) for k in range(1, 11)])

    for t in range(steps):
        event = generate_event(rng, state, cfg, buffer)
        pain = immediate_pain(event, state)
        comp = model.compute_components(event, state, buffer, pain, lesion_mode=lesion_mode)
        action, p_avoid = choose_action(event, state, comp, rng, lesion_mode=lesion_mode)
        outcome = apply_event(event, state, action, comp, cfg, rng, lesion_mode=lesion_mode)
        outcome["action_avoid"] = action
        model.update_from_outcome(comp, event, outcome, buffer, lesion_mode=lesion_mode)

        rows.append({
            "scenario": scenario_id(cfg),
            "risk_label": cfg.risk_label,
            "risk_scale": cfg.risk_scale,
            "damage_multiplier": cfg.damage_multiplier,
            "damage_scale": cfg.damage_scale,
            "true_delay": cfg.true_delay,
            "lesion_mode": lesion_mode,
            "seed": seed,
            "step": t,
            "event_type": event.event_type,
            "intensity": event.intensity,
            "mass": event.mass,
            "friction": event.friction,
            "slope": event.slope,
            "affordance": event.affordance,
            "physical_risk": event.physical_risk,
            "self_generated": event.self_generated,
            "efference_intensity": event.efference_intensity,
            "action_avoid": action,
            "p_avoid": p_avoid,
            "comfort": comp["comfort"],
            "pain": comp["pain"],
            "danger": comp["danger"],
            "avoidance_pressure": comp["avoidance_pressure"],
            "action_possibility": comp["action_possibility"],
            "q_aversive_index": comp["q_aversive_index"],
            "agency_score": comp["agency_score"],
            "agency_best_lag": comp["agency_best_lag"],
            "agency_best_efference": comp["agency_best_efference"],
            "agency_best_flag": comp["agency_best_flag"],
            **outcome,
        })
        if t in marks:
            logger.log(f"{prefix} step {t}/{steps}")
    return pd.DataFrame(rows)


def train_probe_model(seed: int, cfg: EnvConfig, steps=900, lesion_mode="none"):
    rng = np.random.default_rng(seed)
    st = AgentState()
    buffer = EfferenceBuffer(max_lag=12)
    model = LearnedValenceModel(rng, true_delay=cfg.true_delay)
    for _ in range(steps):
        ev = generate_event(rng, st, cfg, buffer)
        pn = immediate_pain(ev, st)
        comp = model.compute_components(ev, st, buffer, pn, lesion_mode=lesion_mode)
        act, _ = choose_action(ev, st, comp, rng, lesion_mode=lesion_mode)
        out = apply_event(ev, st, act, comp, cfg, rng, lesion_mode=lesion_mode)
        out["action_avoid"] = act
        model.update_from_outcome(comp, ev, out, buffer, lesion_mode=lesion_mode)
    return rng, model, buffer


def fixed_event_probe(seed: int, cfg: EnvConfig, lesion_mode="none", n=400):
    rng, model, buffer = train_probe_model(seed, cfg, steps=700, lesion_mode=lesion_mode)
    fixed = Event("collision", 3.20 * cfg.risk_scale, 70.0, 0.42, 0.0, 0.55,
                  float(sigmoid(1.0 * (3.20 * cfg.risk_scale - 2.35))), 0, 0.0, cfg.true_delay)
    rows = []
    for i in range(n):
        st = AgentState(
            integrity=float(rng.uniform(0.72, 1.0)),
            energy=float(rng.uniform(0.25, 1.0)),
            fatigue=float(rng.uniform(0.0, 0.75)),
            stability=float(rng.uniform(0.35, 1.0)),
            damage=0.0,
            pain_memory=float(rng.uniform(0.0, 0.75)),
            danger_memory=float(rng.uniform(0.0, 0.75)),
            comfort_memory=float(rng.uniform(0.15, 0.90)),
        )
        st.damage = 1.0 - st.integrity
        pn = immediate_pain(fixed, st)
        comp = model.compute_components(fixed, st, buffer, pn, lesion_mode=lesion_mode)
        rows.append({
            "scenario": scenario_id(cfg),
            "risk_label": cfg.risk_label,
            "damage_multiplier": cfg.damage_multiplier,
            "true_delay": cfg.true_delay,
            "lesion_mode": lesion_mode,
            "probe": "same_stimulus_state_dependence",
            "i": i,
            "history": "not_applicable",
            **{k: v for k, v in comp.items() if not isinstance(v, np.ndarray)},
        })
    return pd.DataFrame(rows)


def history_probe(seed: int, cfg: EnvConfig, lesion_mode="none", n=300):
    rng, model, buffer = train_probe_model(seed, cfg, steps=700, lesion_mode=lesion_mode)
    fixed = Event("landing", 3.00 * cfg.risk_scale, 70.0, 0.42, 0.0, 0.60,
                  float(sigmoid(1.0 * (3.00 * cfg.risk_scale - 2.35))), 0, 0.0, cfg.true_delay)
    rows = []
    for i in range(n):
        base_integrity = float(rng.uniform(0.78, 0.98))
        base_energy = float(rng.uniform(0.42, 0.88))
        base_stability = float(rng.uniform(0.45, 0.92))
        base_fatigue = float(rng.uniform(0.08, 0.56))
        for hist in ["benign_history", "harmful_history"]:
            if hist == "benign_history":
                pm = float(rng.uniform(0.00, 0.18)); dm = float(rng.uniform(0.00, 0.18)); cm = float(rng.uniform(0.64, 0.95))
            else:
                pm = float(rng.uniform(0.45, 0.84)); dm = float(rng.uniform(0.45, 0.84)); cm = float(rng.uniform(0.06, 0.35))
            st = AgentState(base_integrity, base_energy, base_fatigue, base_stability, 1 - base_integrity, pm, dm, cm)
            pn = immediate_pain(fixed, st)
            comp = model.compute_components(fixed, st, buffer, pn, lesion_mode=lesion_mode)
            rows.append({
                "scenario": scenario_id(cfg),
                "risk_label": cfg.risk_label,
                "damage_multiplier": cfg.damage_multiplier,
                "true_delay": cfg.true_delay,
                "lesion_mode": lesion_mode,
                "probe": "same_stimulus_history_dependence",
                "i": i,
                "history": hist,
                **{k: v for k, v in comp.items() if not isinstance(v, np.ndarray)},
            })
    return pd.DataFrame(rows)


# =============================================================================
# Analysis
# =============================================================================

def summarize_episodes(df):
    rows = []
    group_cols = ["scenario", "risk_label", "damage_multiplier", "true_delay", "lesion_mode", "seed"]
    for keys, d in df.groupby(group_cols):
        scenario, risk_label, dmg, delay, lesion_mode, seed = keys
        rows.append({
            "scenario": scenario,
            "risk_label": risk_label,
            "damage_multiplier": float(dmg),
            "true_delay": int(delay),
            "lesion_mode": lesion_mode,
            "seed": int(seed),
            "n_steps": len(d),
            "final_integrity": float(d["post_integrity"].iloc[-1]),
            "cumulative_damage": float(d["damage_increment"].sum()),
            "mean_comfort": float(d["comfort"].mean()),
            "mean_pain": float(d["pain"].mean()),
            "mean_danger": float(d["danger"].mean()),
            "mean_avoidance_pressure": float(d["avoidance_pressure"].mean()),
            "mean_action_possibility": float(d["action_possibility"].mean()),
            "mean_q_aversive_index": float(d["q_aversive_index"].mean()),
            "avoidance_rate": float(d["action_avoid"].mean()),
            "agency_auc_episode": auc_score(d["self_generated"], d["agency_score"]),
            "true_lag_hit_rate": float(np.mean(d["agency_best_lag"] == d["true_delay"])),
            "near_lag_hit_rate": float(np.mean(np.abs(d["agency_best_lag"] - d["true_delay"]) <= 1)),
        })
    return pd.DataFrame(rows)


def analyze_behavior_prediction(df):
    rows = []
    feats = {
        "physical_only": ["intensity", "friction", "slope", "affordance", "physical_risk", "self_generated"],
        "physical_plus_body": ["intensity", "friction", "slope", "affordance", "physical_risk", "self_generated",
                               "pre_integrity", "pre_energy", "pre_fatigue", "pre_stability", "pre_pain_memory", "pre_danger_memory", "pre_comfort_memory"],
        "physical_plus_q": ["intensity", "friction", "slope", "affordance", "physical_risk", "self_generated",
                            "comfort", "pain", "danger", "avoidance_pressure", "action_possibility", "q_aversive_index", "agency_score"],
        "full": ["intensity", "friction", "slope", "affordance", "physical_risk", "self_generated",
                 "pre_integrity", "pre_energy", "pre_fatigue", "pre_stability", "pre_pain_memory", "pre_danger_memory", "pre_comfort_memory",
                 "comfort", "pain", "danger", "avoidance_pressure", "action_possibility", "q_aversive_index", "agency_score"],
    }
    base = df[df["lesion_mode"] == "none"].copy()
    for scenario, d in base.groupby("scenario"):
        if len(d) < 300 or d["action_avoid"].nunique() < 2:
            continue
        rng = np.random.default_rng(123)
        idx = np.arange(len(d)); rng.shuffle(idx)
        split = int(0.70 * len(idx))
        tr, te = idx[:split], idx[split:]
        y = d["action_avoid"].values.astype(float)
        prevalence = float(np.mean(y))
        meta = d[["risk_label", "damage_multiplier", "true_delay"]].iloc[0].to_dict()
        for name, cols in feats.items():
            X = d[cols].values.astype(float)
            Xtr, Xte = standardize_train_apply(X[tr], X[te])
            pred, _ = logistic_fit_predict(Xtr, y[tr], Xte)
            yy = y[te]
            pred_label = (pred >= 0.5).astype(int)
            acc = float(np.mean(pred_label == yy.astype(int)))
            tpr = np.mean(pred_label[yy == 1] == 1) if np.any(yy == 1) else np.nan
            tnr = np.mean(pred_label[yy == 0] == 0) if np.any(yy == 0) else np.nan
            rows.append({
                "scenario": scenario,
                **meta,
                "feature_set": name,
                "prevalence": prevalence,
                "auc": auc_score(yy, pred),
                "accuracy": acc,
                "balanced_accuracy": float(np.nanmean([tpr, tnr])),
                "brier": float(np.mean((pred - yy) ** 2)),
                "n_train": len(tr),
                "n_test": len(te),
            })
    return pd.DataFrame(rows)


def analyze_irreducibility(df):
    rows = []
    base = df[df["lesion_mode"] == "none"].copy()
    physical = ["intensity", "friction", "slope", "affordance", "physical_risk", "self_generated"]
    body = ["pre_integrity", "pre_energy", "pre_fatigue", "pre_stability", "pre_pain_memory", "pre_danger_memory", "pre_comfort_memory"]
    for scenario, d in base.groupby("scenario"):
        if len(d) < 300:
            continue
        rng = np.random.default_rng(456)
        idx = np.arange(len(d)); rng.shuffle(idx)
        split = int(0.70 * len(idx)); tr, te = idx[:split], idx[split:]
        meta = d[["risk_label", "damage_multiplier", "true_delay"]].iloc[0].to_dict()
        for target in ["q_aversive_index", "agency_score"]:
            y = d[target].values.astype(float)
            for fs_name, cols in {"physical_only": physical, "physical_plus_body": physical + body}.items():
                X = d[cols].values.astype(float)
                Xtr, Xte = standardize_train_apply(X[tr], X[te])
                pred, _ = ridge_fit_predict(Xtr, y[tr], Xte)
                rows.append({
                    "scenario": scenario,
                    **meta,
                    "target": target,
                    "feature_set": fs_name,
                    "r2": r2_score(y[te], pred),
                    "mae": float(np.mean(np.abs(y[te] - pred))),
                    "n_train": len(tr),
                    "n_test": len(te),
                })
    return pd.DataFrame(rows)


def analyze_probes(probes):
    rows = []
    for (scenario, lesion_mode), d in probes.groupby(["scenario", "lesion_mode"]):
        meta = d[["risk_label", "damage_multiplier", "true_delay"]].iloc[0].to_dict()
        same = d[d["probe"] == "same_stimulus_state_dependence"]
        if len(same):
            for target in ["comfort", "pain", "danger", "avoidance_pressure", "action_possibility", "q_aversive_index"]:
                rows.append({"scenario": scenario, **meta, "lesion_mode": lesion_mode, "probe": "same_stimulus_state_dependence", "metric": f"sd_{target}", "value": safe_std(same[target])})
        hist = d[d["probe"] == "same_stimulus_history_dependence"]
        if len(hist):
            benign = hist[hist["history"] == "benign_history"]
            harmful = hist[hist["history"] == "harmful_history"]
            for target in ["comfort", "pain", "danger", "avoidance_pressure", "action_possibility", "q_aversive_index"]:
                rows.append({"scenario": scenario, **meta, "lesion_mode": lesion_mode, "probe": "same_stimulus_history_dependence", "metric": f"delta_harmful_minus_benign_{target}", "value": safe_mean(harmful[target]) - safe_mean(benign[target])})
                rows.append({"scenario": scenario, **meta, "lesion_mode": lesion_mode, "probe": "same_stimulus_history_dependence", "metric": f"d_harmful_vs_benign_{target}", "value": cohen_d(harmful[target], benign[target])})
    return pd.DataFrame(rows)


def build_scenario_summary(ep, behavior, irr, probe_metrics):
    rows = []
    scenarios = sorted(ep["scenario"].unique())
    for sc in scenarios:
        e = ep[ep["scenario"] == sc]
        meta = e[["risk_label", "damage_multiplier", "true_delay"]].iloc[0].to_dict()
        none = e[e["lesion_mode"] == "none"]
        qles = e[e["lesion_mode"] == "q"]
        agles = e[e["lesion_mode"] == "agency"]
        memles = e[e["lesion_mode"] == "memory"]

        base_integrity = safe_mean(none["final_integrity"])
        base_damage = safe_mean(none["cumulative_damage"])
        q_damage_delta = safe_mean(qles["cumulative_damage"]) - base_damage
        agency_damage_delta = safe_mean(agles["cumulative_damage"]) - base_damage
        memory_damage_delta = safe_mean(memles["cumulative_damage"]) - base_damage

        agency_auc = safe_mean(none["agency_auc_episode"])
        agency_auc_lesion = safe_mean(agles["agency_auc_episode"])
        near_hit = safe_mean(none["near_lag_hit_rate"])

        if behavior is None or behavior.empty or "scenario" not in behavior.columns:
            b = pd.DataFrame()
        else:
            b = behavior[behavior["scenario"] == sc]
        def get_auc(fs):
            if b.empty or "feature_set" not in b.columns:
                return np.nan
            vals = b[b["feature_set"] == fs]["auc"].values
            return float(vals[0]) if len(vals) else np.nan
        auc_phys = get_auc("physical_only")
        auc_q = get_auc("physical_plus_q")
        auc_delta = auc_q - auc_phys

        if irr is None or irr.empty or "scenario" not in irr.columns:
            ir = pd.DataFrame()
        else:
            ir = irr[(irr["scenario"] == sc) & (irr["target"] == "q_aversive_index")]
        def get_r2(fs):
            if ir.empty or "feature_set" not in ir.columns:
                return np.nan
            vals = ir[ir["feature_set"] == fs]["r2"].values
            return float(vals[0]) if len(vals) else np.nan
        r2_phys = get_r2("physical_only")
        r2_body = get_r2("physical_plus_body")
        r2_delta = r2_body - r2_phys

        pm = probe_metrics[probe_metrics["scenario"] == sc]
        def metric(lesion, name):
            vals = pm[(pm["lesion_mode"] == lesion) & (pm["metric"] == name)]["value"].values
            return float(vals[0]) if len(vals) else np.nan
        state_dep = metric("none", "sd_q_aversive_index")
        hist_dep = metric("none", "delta_harmful_minus_benign_q_aversive_index")
        hist_dep_mem = metric("memory", "delta_harmful_minus_benign_q_aversive_index")
        memory_reduction = hist_dep - hist_dep_mem

        rows.append({
            "scenario": sc,
            **meta,
            "final_integrity": base_integrity,
            "base_cumulative_damage": base_damage,
            "q_lesion_damage_delta": q_damage_delta,
            "agency_lesion_damage_delta": agency_damage_delta,
            "memory_lesion_damage_delta": memory_damage_delta,
            "agency_auc": agency_auc,
            "agency_auc_lesion": agency_auc_lesion,
            "agency_auc_drop": agency_auc - agency_auc_lesion,
            "near_lag_hit_rate": near_hit,
            "auc_physical_only": auc_phys,
            "auc_physical_plus_q": auc_q,
            "delta_auc_q_minus_physical": auc_delta,
            "r2_q_physical_only": r2_phys,
            "r2_q_physical_plus_body": r2_body,
            "delta_r2_body_minus_physical": r2_delta,
            "same_stimulus_state_dependence": state_dep,
            "history_dependence_q": hist_dep,
            "history_dependence_q_memory_lesion": hist_dep_mem,
            "memory_lesion_reduces_history": memory_reduction,
        })
    return pd.DataFrame(rows)


def build_support_matrix(summary):
    criteria = [
        ("viability_preserved", "final_integrity", ">= 0.85", lambda x: x >= 0.85),
        ("q_lesion_increases_damage", "q_lesion_damage_delta", "> 0", lambda x: x > 0),
        ("same_stimulus_state_dependence", "same_stimulus_state_dependence", "> 0", lambda x: x > 0),
        ("history_dependence_q", "history_dependence_q", "> 0", lambda x: x > 0),
        ("q_improves_avoidance_prediction_auc", "delta_auc_q_minus_physical", "> 0", lambda x: x > 0),
        ("q_not_reducible_to_external_physics", "delta_r2_body_minus_physical", "> 0.03", lambda x: x > 0.03),
        ("agency_attribution_above_chance", "agency_auc", "> 0.60", lambda x: x > 0.60),
        ("agency_temporal_alignment", "near_lag_hit_rate", "> 0.20", lambda x: x > 0.20),
        ("agency_lesion_reduces_agency", "agency_auc_drop", "> 0.20", lambda x: x > 0.20),
        ("memory_lesion_reduces_history", "memory_lesion_reduces_history", "> 0", lambda x: x > 0),
    ]
    rows = []
    for _, row in summary.iterrows():
        for crit, col, target, fn in criteria:
            val = row.get(col, np.nan)
            try:
                passed = bool(fn(val)) if not np.isnan(val) else False
            except Exception:
                passed = False
            rows.append({
                "scenario": row["scenario"],
                "risk_label": row["risk_label"],
                "damage_multiplier": row["damage_multiplier"],
                "true_delay": row["true_delay"],
                "criterion": crit,
                "value": val,
                "target": target,
                "passed": passed,
            })
    return pd.DataFrame(rows)


# =============================================================================
# Figures and report
# =============================================================================

def save_figures(outdir: Path, summary: pd.DataFrame, support: pd.DataFrame, ep: pd.DataFrame):
    figdir = outdir / "figures"
    figdir.mkdir(exist_ok=True)

    pass_rate = support.groupby("criterion")["passed"].mean().sort_values()
    plt.figure(figsize=(10, 5))
    plt.bar(np.arange(len(pass_rate)), pass_rate.values)
    plt.xticks(np.arange(len(pass_rate)), pass_rate.index, rotation=45, ha="right")
    plt.ylim(0, 1.05)
    plt.ylabel("Pass rate across scenarios")
    plt.title("Figure 1. Robustness pass rate by criterion")
    plt.tight_layout()
    plt.savefig(figdir / "figure1_pass_rate_by_criterion.png", dpi=300)
    plt.close()

    plt.figure(figsize=(7, 5))
    labels = []
    vals = []
    errs = []
    for risk in ["mild", "moderate", "harsh"]:
        d = summary[summary["risk_label"] == risk]
        labels.append(risk)
        vals.append(d["final_integrity"].mean())
        errs.append(d["final_integrity"].std())
    plt.bar(np.arange(len(labels)), vals, yerr=errs, capsize=4)
    plt.xticks(np.arange(len(labels)), labels)
    plt.ylim(0, 1.02)
    plt.ylabel("Final integrity")
    plt.title("Figure 2. Viability preservation by environmental risk")
    plt.tight_layout()
    plt.savefig(figdir / "figure2_integrity_by_risk.png", dpi=300)
    plt.close()

    lesion_cols = ["q_lesion_damage_delta", "agency_lesion_damage_delta", "memory_lesion_damage_delta"]
    plt.figure(figsize=(8, 5))
    vals = [summary[c].mean() for c in lesion_cols]
    errs = [summary[c].std() for c in lesion_cols]
    plt.bar(np.arange(len(lesion_cols)), vals, yerr=errs, capsize=4)
    plt.xticks(np.arange(len(lesion_cols)), ["Q lesion", "Agency lesion", "Memory lesion"], rotation=15)
    plt.ylabel("Damage delta vs full model")
    plt.title("Figure 3. Lesion-specific cumulative damage effects")
    plt.tight_layout()
    plt.savefig(figdir / "figure3_lesion_damage_deltas.png", dpi=300)
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.boxplot([summary["delta_auc_q_minus_physical"].dropna(), summary["delta_r2_body_minus_physical"].dropna()], labels=["Delta AUC", "Delta R2"])
    plt.axhline(0, linestyle="--", linewidth=1)
    plt.ylabel("Effect size")
    plt.title("Figure 4. Predictive and irreducibility effects across scenarios")
    plt.tight_layout()
    plt.savefig(figdir / "figure4_delta_auc_delta_r2.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 5))
    delays = sorted(summary["true_delay"].unique())
    vals = [summary[summary["true_delay"] == d]["agency_auc"].mean() for d in delays]
    errs = [summary[summary["true_delay"] == d]["agency_auc"].std() for d in delays]
    plt.bar(np.arange(len(delays)), vals, yerr=errs, capsize=4)
    plt.xticks(np.arange(len(delays)), [str(d) for d in delays])
    plt.ylim(0, 1.05)
    plt.xlabel("True sensorimotor delay")
    plt.ylabel("Agency AUC")
    plt.title("Figure 5. Self-attribution across agency delays")
    plt.tight_layout()
    plt.savefig(figdir / "figure5_agency_auc_by_delay.png", dpi=300)
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.boxplot([summary["history_dependence_q"].dropna(), summary["history_dependence_q_memory_lesion"].dropna()], labels=["Full memory", "Memory lesion"])
    plt.ylabel("History dependence of Q")
    plt.title("Figure 6. Memory lesion effect on history dependence")
    plt.tight_layout()
    plt.savefig(figdir / "figure6_memory_lesion_history_dependence.png", dpi=300)
    plt.close()


def write_report(outdir: Path, summary: pd.DataFrame, support: pd.DataFrame, ep: pd.DataFrame,
                 behavior: pd.DataFrame, irr: pd.DataFrame, probes: pd.DataFrame):
    path = outdir / "consolidated_report.txt"
    lines = []
    lines.append("Qualitative Valence Learning + Agency Robustness Audit, v7\n")
    lines.append(f"Generated: {now()}\n")
    lines.append("Gemini/LLM/semantic module: not used.\n")
    lines.append("Model status: v6 logic frozen. This script varies seeds, environmental risk, damage scaling, agency delay, and lesion mode.\n")

    lines.append("\n1. Scenario-level summary\n-------------------------\n")
    cols = [
        "scenario", "final_integrity", "q_lesion_damage_delta", "agency_lesion_damage_delta", "memory_lesion_damage_delta",
        "delta_auc_q_minus_physical", "delta_r2_body_minus_physical", "agency_auc", "agency_auc_drop",
        "near_lag_hit_rate", "history_dependence_q", "history_dependence_q_memory_lesion", "memory_lesion_reduces_history",
    ]
    lines.append(summary[cols].to_string(index=False))

    lines.append("\n\n2. Pass-rate by support criterion\n---------------------------------\n")
    pass_rates = support.groupby("criterion")["passed"].agg(["mean", "sum", "count"]).reset_index()
    pass_rates = pass_rates.rename(columns={"mean": "pass_rate", "sum": "n_pass", "count": "n_total"})
    lines.append(pass_rates.to_string(index=False))

    lines.append("\n\n3. Aggregate descriptive statistics\n-----------------------------------\n")
    agg_cols = [
        "final_integrity", "q_lesion_damage_delta", "agency_lesion_damage_delta", "memory_lesion_damage_delta",
        "delta_auc_q_minus_physical", "delta_r2_body_minus_physical", "agency_auc", "agency_auc_drop",
        "near_lag_hit_rate", "history_dependence_q", "memory_lesion_reduces_history",
    ]
    lines.append(summary[agg_cols].agg(["mean", "std", "min", "max"]).to_string())

    lines.append("\n\n4. Lesion-level episode summary\n-------------------------------\n")
    lines.append(ep.groupby("lesion_mode").agg({
        "final_integrity": ["mean", "std"],
        "cumulative_damage": ["mean", "std"],
        "avoidance_rate": ["mean", "std"],
        "agency_auc_episode": ["mean", "std"],
        "near_lag_hit_rate": ["mean", "std"],
    }).to_string())

    lines.append("\n\n5. Interpretation\n-----------------\n")
    lines.append(
        "The audit is supportive if the full model preserves viability across scenarios, Q lesion increases damage, "
        "Q remains state- and history-dependent, Q variables improve avoidance-action prediction, bodily variables improve "
        "the explanation of Q beyond external physical variables, agency attribution remains above chance across true delays, "
        "agency lesion reduces agency attribution, and memory lesion reduces history dependence. The appropriate claim remains "
        "limited: the system learns viability-weighted qualitative-valence predictors and self-attribution structure, not subjective consciousness itself.\n"
    )

    lines.append("\n6. Files\n--------\n")
    lines.append("CSV outputs: scenario_summary.csv, support_matrix.csv, lesion_summary.csv, behavior_prediction_metrics.csv, irreducibility_metrics.csv, probe_metrics.csv.\n")
    lines.append("Figures are saved in figures/. Per-episode CSV files are saved in episode_csv/.\n")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=str, default=str(Path.home() / "Desktop" / "qualitative_valence_v7_robustness_audit"))
    parser.add_argument("--steps", type=int, default=2500)
    parser.add_argument("--episodes", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20260531)
    parser.add_argument("--risk-levels", type=str, default="mild,moderate,harsh")
    parser.add_argument("--damage-mults", type=str, default="0.75,1.0,1.25")
    parser.add_argument("--delays", type=str, default="1,3,5,7")
    parser.add_argument("--base-damage-scale", type=float, default=0.000010)
    parser.add_argument("--probe-n-state", type=int, default=400)
    parser.add_argument("--probe-n-history", type=int, default=300)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    outdir = Path(os.path.expanduser(args.outdir))
    outdir.mkdir(parents=True, exist_ok=True)
    logger = Logger(outdir)
    t0 = time.time()

    risk_levels = [x.strip() for x in args.risk_levels.split(",") if x.strip()]
    damage_mults = parse_csv_list(args.damage_mults, float)
    delays = parse_csv_list(args.delays, int)

    if args.quick:
        # Keep the full logic but reduce the grid for a fast diagnostic run unless user explicitly overrides.
        if args.risk_levels == "mild,moderate,harsh":
            risk_levels = ["mild", "moderate", "harsh"]
        if args.damage_mults == "0.75,1.0,1.25":
            damage_mults = [0.75, 1.0, 1.25]
        if args.delays == "1,3,5,7":
            delays = [1, 3, 5]
        args.probe_n_state = min(args.probe_n_state, 250)
        args.probe_n_history = min(args.probe_n_history, 180)

    configs = []
    for risk in risk_levels:
        if risk not in RISK_SCALES:
            raise ValueError(f"Unknown risk level: {risk}. Valid: {list(RISK_SCALES)}")
        for dmg in damage_mults:
            for delay in delays:
                configs.append(EnvConfig(
                    risk_label=risk,
                    risk_scale=RISK_SCALES[risk],
                    damage_multiplier=float(dmg),
                    damage_scale=float(args.base_damage_scale) * float(dmg),
                    true_delay=int(delay),
                ))

    logger.log("v7 robustness audit started")
    logger.log(f"outdir={outdir}")
    logger.log(f"steps={args.steps}, episodes={args.episodes}, seed={args.seed}")
    logger.log(f"risk_levels={risk_levels}, damage_mults={damage_mults}, delays={delays}")
    logger.log(f"lesion_modes={LESION_MODES}")
    logger.log("Gemini/LLM/semantic module: not used")

    epdir = outdir / "episode_csv"
    epdir.mkdir(exist_ok=True)

    all_dfs = []
    total_jobs = len(configs) * len(LESION_MODES) * args.episodes
    job = 0

    for ci, cfg in enumerate(configs):
        sc = scenario_id(cfg)
        logger.log(f"Scenario {ci + 1}/{len(configs)}: {sc}")
        for lesion_mode in LESION_MODES:
            for ep_i in range(args.episodes):
                seed = args.seed + ci * 100000 + ep_i * 1000 + LESION_MODES.index(lesion_mode) * 71
                csv_path = epdir / f"episode_{sc}_lesion-{lesion_mode}_seed-{seed}.csv"
                job += 1
                logger.log(f"Job {job}/{total_jobs}: {sc}, lesion={lesion_mode}, seed={seed}")
                if args.resume and csv_path.exists():
                    logger.log(f"Resume: loading {csv_path.name}")
                    d = pd.read_csv(csv_path)
                else:
                    d = run_episode(seed, args.steps, cfg, lesion_mode, logger, f"{sc}/{lesion_mode}/seed{seed}")
                    d.to_csv(csv_path, index=False)
                all_dfs.append(d)

    logger.log("Combining episode data")
    df = pd.concat(all_dfs, ignore_index=True)
    sample_n = min(len(df), 120000)
    df.sample(n=sample_n, random_state=1).sort_values(["scenario", "lesion_mode", "seed", "step"]).to_csv(outdir / "all_step_data_downsampled.csv", index=False)

    logger.log("Running independent state/history probes")
    probe_dfs = []
    total_probe_jobs = len(configs) * 2  # full and memory lesion are sufficient for history-dependence lesion audit.
    pjob = 0
    for ci, cfg in enumerate(configs):
        for lesion_mode in ["none", "memory"]:
            pjob += 1
            logger.log(f"Probe job {pjob}/{total_probe_jobs}: {scenario_id(cfg)}, lesion={lesion_mode}")
            probe_dfs.append(fixed_event_probe(args.seed + 700000 + ci * 200 + LESION_MODES.index(lesion_mode), cfg, lesion_mode=lesion_mode, n=args.probe_n_state))
            probe_dfs.append(history_probe(args.seed + 710000 + ci * 200 + LESION_MODES.index(lesion_mode), cfg, lesion_mode=lesion_mode, n=args.probe_n_history))
    probes = pd.concat(probe_dfs, ignore_index=True)
    probes.to_csv(outdir / "probe_data.csv", index=False)

    logger.log("Computing episode summaries")
    ep = summarize_episodes(df)
    ep.to_csv(outdir / "lesion_summary.csv", index=False)

    logger.log("Analyzing behavior prediction")
    behavior = analyze_behavior_prediction(df)
    behavior.to_csv(outdir / "behavior_prediction_metrics.csv", index=False)

    logger.log("Analyzing irreducibility")
    irr = analyze_irreducibility(df)
    irr.to_csv(outdir / "irreducibility_metrics.csv", index=False)

    logger.log("Analyzing probes")
    pm = analyze_probes(probes)
    pm.to_csv(outdir / "probe_metrics.csv", index=False)

    logger.log("Building scenario summary and support matrix")
    summary = build_scenario_summary(ep, behavior, irr, pm)
    summary.to_csv(outdir / "scenario_summary.csv", index=False)
    support = build_support_matrix(summary)
    support.to_csv(outdir / "support_matrix.csv", index=False)

    logger.log("Saving figures")
    save_figures(outdir, summary, support, ep)

    logger.log("Writing consolidated report")
    report = write_report(outdir, summary, support, ep, behavior, irr, pm)

    elapsed = time.time() - t0
    logger.log(f"Finished in {elapsed:.1f} sec")
    logger.log(f"Report: {report}")
    logger.log(f"Scenario summary: {outdir / 'scenario_summary.csv'}")
    logger.log(f"Support matrix: {outdir / 'support_matrix.csv'}")
    logger.log(f"Figures: {outdir / 'figures'}")


if __name__ == "__main__":
    main()
