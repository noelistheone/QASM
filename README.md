# QASM — Query-Aligned Spatial Moments

Implementation of **Query-Aligned Spatial Moments for Unseen Imaging Responses**.

QASM learns a shared spatial state from sparse imaging measurements and reuses it
through known propagation operators to predict responses at new wavelengths and
detector settings. A frozen image estimate supplies the mean field; learned
spatial factors retain response-relevant second-moment information. For a query
operator B, the readout is the energy of the propagated mean plus the summed
energies of the propagated factors.

This repository contains source code, experiment configurations, dependency files
and this README. Datasets, checkpoints, generated results and manuscript files are
not included. Download source data or provide the corresponding local inputs
before running data-dependent commands.

## Install

Use Python 3.11. The reference environment uses PyTorch 2.11.0; install the CPU or
CUDA build appropriate to your machine, then install the remaining dependencies.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.11.0
python -m pip install -r requirements-training.txt
```

`requirements.txt` is sufficient for the model and replay code; the training
requirements also install LMDB for the public scattering dataset. The original
GPU experiments used CUDA 12.8. CPU execution is supported by the mathematical
checks and inference examples.

## Start with the physical readout

These commands need no datasets or trained checkpoints:

```bash
python examples/query_response.py
python scripts/check_response_moments.py
python scripts/check_point_moment_risk.py
python scripts/check_nppc_normalization_estimand.py
```

The example demonstrates a state shared across multiple query operators. The
checks verify the quadratic-moment identity, the distinction between a point
estimate and a second-moment state, and the effect of normalization in the
image-moment comparison. Their outputs are written locally and ignored by Git.

## Train the scattering model

Download the official training archive with checksum verification:

```bash
python scripts/fetch_scattering_data.py --split train
python scripts/prepare_response_learning.py --device cuda:0
python scripts/prepare_gaussian_actions.py --device cuda:0
python scripts/prepare_mask_features.py --device cuda:0
```

The preparation code uses the fixed object split and observation protocol in
`protocols/`. Mean-image and spatial-factor training use disjoint portions of the
training objects. The Gaussian-action preparation supplies the common
observation realization used by the original experiment interfaces.

Train the frozen mean-image estimator, then QASM's factor network:

```bash
python scripts/train_mask_conditioned.py --mode image --seed 20261117 \
  --device cuda:0 --out runs/m2_mask_learning_20261101/image_seed20261117
python scripts/train_current_random.py --seed 20261126 --device cuda:0 \
  --out runs/qasm_seed20261126
```

The factor network uses eight fields, 4,000 Adam updates and a batch size of 128
by default. The paper's factor-training seeds are `20261126`, `20261127` and
`20261128`. Use separate output directories for additional seeds. The frozen mean
estimate is shared across them. In the code, budgets `[0, 2, 5, 10]` count added
receiver groups beyond the base acquisition and correspond to 60, 84, 120 and
180 receivers.

The matched image-moment controls use the same mean estimate and factor network:

```bash
python scripts/train_mask_components.py --normalization sample --seed 20261126 \
  --device cuda:0 --out runs/nppc_sample_seed20261126
python scripts/train_mask_components.py --normalization fixed --seed 20261126 \
  --device cuda:0 --out runs/nppc_fixed_seed20261126
```

`train_mask_conditioned.py --mode joint` implements the joint image/response
control. `train_controlled_rank.py` contains the matched rank and coherent-state
ablations. Historical experiment drivers retain their frozen run paths and
checks; the full-study evaluation also expects the corresponding fitted runs and
prepared caches.

## Inference and evaluation entry points

| Task | Entry points |
| --- | --- |
| Shared-state quadratic response | `src/controlled_query_learning.py`, `src/cached_query_readouts.py` |
| Physical scattering operator | `src/born_scattering.py`, `src/scattering_queries.py` |
| Mean and factor networks | `src/mask_conditioned_learning.py`, `src/posterior_components.py` |
| Independent scattering assessment | `prepare_official_scattering.py`, `cache_official_scattering_features.py`, `predict_official_scattering.py`, `score_official_scattering.py` |
| Image-moment objective comparison | `predict_m7_nppc.py`, `score_m7_nppc.py` |
| Morphology and acquired-power controls | `evaluate_m7_morphology.py`, `evaluate_m7_visible_calibration.py` |
| Real-measurement extension | `src/global_ptycho_response.py`, `src/query_aware_refinement.py`, `run_query_aware_pilot.py` |

Script names in the table are under `scripts/`. Training wavelengths are 4, 6 and
8; held-out queries use wavelengths 5, 7, 10 and 12, each with five angular bands.
`prepare_transfer_queries.py` generates the latter reference responses for
assessment. Independent-test prediction and scoring are separate commands.

For the separately supplied reproduction fixtures, the frozen-model entry points
are:

```bash
python scripts/replay_submission_core.py --case born --device cpu \
  --output reproduction/replay_new/born/report.json
python scripts/replay_submission_core.py --case broadband --device cpu \
  --output reproduction/replay_new/broadband/report.json
python scripts/replay_m7_objective.py --device cpu \
  --output reproduction/replay_new/nppc/report.json
```

These commands expect `reproduction/fixtures/` or `reproduction/m7_fixtures/`,
including their checksum manifests and checkpoints. They are not part of this
code-only repository. The Born replay begins with observation-only reconstruction
features; full multi-seed paper scores additionally require the complete
assessment records. The real-data replay uses the archived physical
initialization and a three-step learned update.

## Figures and tables

With the original result directories available locally, run:

```bash
python scripts/build_submission_figures.py --part all
python scripts/build_submission_examples.py
python scripts/build_m7_paper_assets.py
python scripts/build_m8_paper_assets.py
python scripts/build_m9_figure1.py
python scripts/build_m10_main_evidence.py
```

The builders read frozen outputs under `analysis/`, `audit/` and `reproduction/`
and write assets under `paper/`. They do not train models. These input records
and generated assets are maintained outside Git.

## Data sources and attribution

Scattering images come from [InverseBench, CaltechDATA](https://doi.org/10.22002/zg89b-mpv16).
The source dataset card specifies CC BY-NC-SA 4.0 for these images; the dataset
archive also carries an MIT metadata entry. Preserve the image-specific notice
when redistributing derived data. The physical convention follows InverseBench
commit `1db5932bc2995504491663d1e2c498e7317c5404`.

The real-measurement extension uses the [3D hyperspectral dataset of NMC grains
for broadband ptychographic tomography](https://doi.org/10.5281/zenodo.23189401),
released by its source authors under CC BY 4.0. The related article is Stolp et al.,
*Physical Review Letters* **134**, 257001 (2025),
[doi:10.1103/kbrf-gb9c](https://doi.org/10.1103/kbrf-gb9c). Native reconstruction
controls use [Pty-Chi](https://github.com/AdvancedPhotonSource/pty-chi), commit
`153ea64217b14fcc3f18e16445ba9f1c5bff4044`; install that optional dependency only
for its native reconstruction workflows.
