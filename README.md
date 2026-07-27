# Learning History-Dependent Bodily Valence in a Closed Sensorimotor Loop

This repository contains the simulation and analysis code used to test whether an embodied agent can learn an internal bodily-valence signal from its current bodily state, experienced history, and future bodily consequences.

The term *bodily valence* is used operationally. The model does not make claims about consciousness, sentience, or subjective experience.

## Files

### `qualitative_valence_closed_loop.py`

Main simulation and validation program.

It implements:

* a discrete-time embodied agent in a closed sensorimotor loop;
* bodily states including integrity, energy, fatigue, stability, and memory;
* learned danger, comfort, action possibility, agency, and bodily valence;
* a scalar valence bottleneck trained from future bodily consequences;
* multiple control and lesion conditions;
* state, history, morphology, agency, causal-intervention, generalization, and robustness tests;
* independently calibrated mild, moderate, and challenging environments.

### `run_all.py`

Runs the complete simulation and analysis pipeline.

It executes the main simulation, inferential analyses, additional information analysis, figure generation, and verification of manuscript values. Existing outputs are reused so that interrupted runs can be resumed.

### `two_family_tests.py`

Performs the confirmatory statistical analyses.

Tests are separated into:

1. the primary central-hypothesis family; and
2. the secondary functional-benchmark family.

Permutation tests, bootstrap confidence intervals, standardized effect sizes, and Holm corrections are calculated using the independent base seed as the inferential unit.

### `incremental_information_test.py`

Tests whether the learned valence signal contains information about future bodily compromise beyond immediate pain, physical risk, and learned danger.

Incremental prediction is evaluated out of sample using cross-validation on event types whose learning updates were withheld.

### `verify_reported_values.py`

Recomputes the numerical values reported in the manuscript directly from the simulation outputs.

The script compares each recomputed value with its expected manuscript value and reports any mismatch.

## Repository structure

```text
.
├── run_all.py
├── simulation/
│   └── qualitative_valence_closed_loop.py
└── analysis/
    ├── two_family_tests.py
    ├── incremental_information_test.py
    └── verify_reported_values.py
```

## Requirements

* Python 3
* NumPy
* pandas
* matplotlib
* statsmodels

Install the required packages with:

```bash
python3 -m pip install numpy pandas matplotlib statsmodels
```

## Running the complete analysis

From the repository root:

```bash
python3 run_all.py --preset main --workers 8 --outdir results_main
```

Available presets are:

* `smoke`: short execution test;
* `compact`: reduced analysis;
* `main`: confirmatory analysis reported in the manuscript;
* `reviewer`: larger replication analysis.

An interrupted run can be resumed by repeating the same command.

## Running only the main simulation

```bash
python3 simulation/qualitative_valence_closed_loop.py \
  --preset main \
  --workers 8 \
  --outdir results_main \
  --resume
```

## Main outputs

The simulation produces CSV files containing:

* episode-level bodily and behavioural outcomes;
* prospective prediction metrics;
* learned valence-component weights;
* closed-loop intervention results;
* same-event bodily-state contrasts;
* experienced-history and memory-lesion contrasts;
* morphology tests;
* agency tests;
* state-clone causal interventions;
* held-out-event generalization tests;
* robustness analyses.

## Reproducibility

The `main` preset reproduces the confirmatory design used in the manuscript. Random seeds, environmental calibration, model conditions, probes, and statistical procedures are specified directly in the code.
