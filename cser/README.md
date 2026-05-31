# CSER — Conformal Submodular Expert Routing

Implementation of the AAAI plan (`docs/delivery/AAAI_UPGRADED_PLAN.md`).
This package is **separate from the existing C-QIN routing code** (`routing/`)
and drives the repo's 5 frozen expert models (`tasks/real_models.py`, mock
fallback `tasks/mock_models.py`).

> **New here? Start with [GETTING_STARTED.md](GETTING_STARTED.md)** — install,
> smoke test, and your first training/experiment run in a few minutes.

## Scope

The plan's flagship contribution rests on one empirical claim: **expert value is
submodular** (diminishing returns). Phase-1 builds the machinery to *test that
claim* and learn a context-dependent value function. Phase-2 adds the safety
gate, the integrated pipeline, and the experiment battery.

| Plan module | File | Status |
|-------------|------|--------|
| Oracle marginal-value labels (§4.2 training signal) | `value_oracle.py` | ✅ exact enumeration |
| Module 1 — Submodular Value Network (§4.2) | `svn.py` | ✅ + E5 ablation variants |
| SVN training (MSE + submodularity reg) | `train_svn.py` | ✅ |
| Module 2 — Conformal Safety Gate / Mondrian (§4.3) | `conformal.py` | ✅ split + Mondrian |
| Module 3 — Greedy Budgeted Selector (§4.4) | `greedy.py` | ✅ |
| Integrated CSER pipeline (§4.1, §4.5) | `pipeline.py` | ✅ |
| Selection baselines (B0/B1/B2/B4/oracle) | `baselines.py` | ✅ |
| E2 — Submodularity verification (§7) + γ fallback (§11) | `submodularity.py` | ✅ |
| E1/E3/E4/E5/E6 — main + ablations | `run_phase2.py` | ✅ |
| E7/E8/E9/E10 — scalability/robustness/contribution/oracle | `experiments_extra.py`, `run_phase3.py` | ✅ |
| Theorems 1/2/3 — proofs + empirical checks | `docs/delivery/CSER_THEOREMS.md`, `theory.py` | ✅ |

**Not yet done**: running with real model weights over a real video gallery
(weights are not in this repo) and drafting the paper prose. Everything runs
today on a self-contained synthetic frame gallery with mock experts.

**Full plan-to-code map**: see [CSER_IMPLEMENTATION.md](CSER_IMPLEMENTATION.md).
**Formal proofs**: see [../docs/delivery/CSER_THEOREMS.md](../docs/delivery/CSER_THEOREMS.md).

## Expert mapping

The plan's 5 frozen experts are the repo's real models (`tasks/real_models.py`,
mock fallback `tasks/mock_models.py`):

```
e0 = semantic   MobileCLIP    encode_text / encode_frames   mandatory base, cost 1.0
e1 = highlight  MomentDETR    score(frames) -> saliency       optional,      cost 2.0
e2 = face       SCRFD         detect(frames) -> (has, conf)   optional,      cost 2.0
e3 = face_id    ArcFace       embed(frames) -> vector         optional,      cost 3.0
e4 = scene      MobileNetV3   classify(frames) -> label       optional,      cost 1.5
```

`expert_features.py` runs all 5 models over each gallery video once and caches
per-video signals; selecting an optional expert adds its query-conditioned score
as a **soft** rerank, so the GT video is never eliminated and `f(S, q)` is well
defined for all 16 subsets (hard-filter safety is the Conformal Safety Gate's
job). With only 4 optional experts the lattice has 2⁴ = 16 elements, so oracle
marginal values are computed by **exact enumeration** — no Monte-Carlo sampling.

## Run

```bash
cd litevtr_multi_model_framework

# tests (self-contained, no external data)
python -m pytest cser/tests -q

# Phase-1 pipeline on the synthetic gallery (no external files)
python -m cser.run_phase1 --out-dir reports/cser_phase1

# Phase-2: full pipeline + baselines + experiments E1/E3/E4/E5/E6
python -m cser.run_phase2 --out-dir reports/cser_phase2

# Phase-3: experiments E7/E8/E9/E10 + empirical theorem verification
python -m cser.run_phase3 --out-dir reports/cser_phase3

# real numbers: real backbones over a real video gallery (all phases accept these)
python -m cser.run_phase2 --out-dir reports/cser_phase2_real \
    --videos /path/to/videos_dir --csv /path/to/queries.csv --real-models
```

The pipeline is **gallery-agnostic**: with no `--videos` it builds a synthetic
frame gallery and uses the mock experts (no external files); pass `--videos` +
`--real-models` to decode real videos and run the real backbones.

## Outputs (`--out-dir`)

**Phase 1** (`run_phase1`):
- `oracle_train.npz` / `oracle_test.npz` — exact value lattices + marginals
- `svn/svn.pt`, `svn/svn_config.json` — trained value network
- `submodularity_report.json` — **E2 verdict**: violation rate + weak-submod γ
- `greedy_vs_oracle.json` — SVN-greedy realised value vs oracle ceiling
- `phase1_summary.json` — roll-up

**Phase 2** (`run_phase2`):
- `e1_main_results.json` — CSER vs baselines (R@1/R@5/MRR/cost/coverage)
- `e3_conformal.json` — coverage vs α, split vs Mondrian set sizes (**E3 claim**)
- `e4_budget_curve.json` — R@1 vs avg experts across budgets 1..5
- `e5_svn_ablation.json` — SVN variants + submod-loss on/off
- `e6_safety_ablation.json` — conformal vs heuristic vs no-gate
- `phase2_summary.json` — roll-up + production gate config

## Reading the E2 verdict

`submodularity_report.json` decides the paper's framing (plan §11):

- `verdict = "submodular"` (violation < 5%) → keep the clean (1−1/e) story.
- `verdict = "weakly_submodular"` (5–10%) → use the (1−e^{−γ}) bound; `gamma_p10`
  is the conservative ratio to cite.
- `verdict = "non_submodular"` (> 10%) → the submodular framing needs rethinking
  before the paper relies on it. Surface this early.
