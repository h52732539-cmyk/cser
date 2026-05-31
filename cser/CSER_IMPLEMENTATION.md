# CSER Implementation — Complete Plan-to-Code Map

**Conformal Submodular Expert Routing** — full implementation of
`docs/delivery/AAAI_UPGRADED_PLAN.md`.

This document maps every section of the plan to concrete code, states what is
done vs. pending, explains the key design decisions, and gives the exact
commands to reproduce every experiment. It is the single entry point for anyone
picking up the project.

---

## 1. TL;DR status

| Plan item | Where | Status |
|-----------|-------|--------|
| Problem formulation (§3) | `experts.py`, `expert_features.py`, `retrieval.py` | ✅ |
| Module 1 — Submodular Value Network (§4.2) | `svn.py`, `train_svn.py` | ✅ |
| Module 2 — Conformal Safety Gate (§4.3) | `conformal.py` | ✅ |
| Module 3 — Greedy Budgeted Selector (§4.4) | `greedy.py` | ✅ |
| Integrated pipeline + combined guarantee (§4.1, §4.5) | `pipeline.py` | ✅ |
| Theorems 1/2/3 (§5) | `docs/delivery/CSER_THEOREMS.md` + `theory.py` | ✅ proofs + empirical checks |
| E1 main comparison (§7) | `run_phase2.py` | ✅ |
| E2 submodularity verification (§7) | `submodularity.py`, `run_phase1.py` | ✅ |
| E3 conformal coverage (§7) | `run_phase2.py` | ✅ |
| E4 budget curve (§7) | `run_phase2.py` | ✅ |
| E5 SVN ablation (§7) | `run_phase2.py` | ✅ |
| E6 safety ablation (§7) | `run_phase2.py` | ✅ |
| E7 scalability (§7) | `experiments_extra.py`, `run_phase3.py` | ✅ |
| E8 robustness (§7) | `experiments_extra.py`, `run_phase3.py` | ✅ |
| E9 expert contribution (§7) | `experiments_extra.py`, `run_phase3.py` | ✅ |
| E10 oracle comparison (§7) | `experiments_extra.py`, `run_phase3.py` | ✅ |

**All three modules, all ten experiments, and all three theorems are
implemented, wired to the repo's 5 real expert models** (`tasks/real_models.py`,
mock fallback `tasks/mock_models.py`). What remains is *running with real model
weights over a real video gallery* (weights are not in this repo) and drafting
the paper prose. Everything runs today on a self-contained synthetic frame
gallery with mock experts for logic validation.

---

## 2. The expert mapping

The plan describes 5 frozen *vision models*. These are **already implemented** in
the repo at `tasks/real_models.py` (real backbones) with deterministic mock
fallbacks in `tasks/mock_models.py`. CSER uses them directly:

```
e0 = semantic   MobileCLIP2-S0   encode_text / encode_frames   MANDATORY base, cost 1.0
e1 = highlight  MomentDETR       score(frames) -> saliency       optional,      cost 2.0
e2 = face       SCRFD            detect(frames) -> (has, conf)   optional,      cost 2.0
e3 = face_id    ArcFace          embed(frames) -> 512-D vector   optional,      cost 3.0
e4 = scene      MobileNetV3      classify(frames) -> label       optional,      cost 1.5
```

How it works:

- **`expert_features.py` runs all 5 models over every gallery video once** and
  caches per-video signals (mean CLIP embedding, max highlight saliency, max face
  confidence, mean ArcFace embedding, scene distribution). This mirrors the repo's
  offline-index philosophy: heavy model work happens once, queries are cheap.
- **Selecting an expert = adding its query-conditioned score as a soft rerank**
  on top of the semantic base (`retrieval.py`). The GT video is never filtered
  out, so f(S,q) is defined for all 16 subsets and the ranking always contains
  v*. Hard-filter safety is the Conformal Safety Gate's separate job.
- Experts have **overlapping, complementary** value (face + scene both fire on
  "a person at the beach") — the source of the submodularity the paper studies.
- **Cost is real**: budget is measured in model-call units (semantic 1.0 →
  face_id 3.0), per the plan's "model calls × frames" budget. Full set = 9.5.
- Only 4 optional experts ⇒ $2^4 = 16$ subsets ⇒ oracle marginal values by **exact
  enumeration**, no Monte-Carlo.

Run with real backbones via `--real-models` (needs weights); otherwise the mock
models run so the whole pipeline is testable with no external files. **To swap in
different models, only `experts.py` (roster/costs) and `expert_features.py`
(how a model's output becomes a signal) change** — SVN / conformal / greedy are
agnostic.

---

## 3. Module-by-module

### 3.1 Submodular Value Network (Module 1) — `svn.py`, `train_svn.py`

Predicts the marginal value $\hat v(e\mid S,q)$ of each optional expert given the
query feature and the already-selected set mask.

- **Architecture** (`variant="full"`): query encoder → expert embedding table →
  DeepSets set encoder → cross-attention of a [query⊕set] token over expert
  tokens → per-expert value head. ~300K params on the synthetic config.
- **Set conditioning** is the whole point: unlike the C-QIN MLP (which scores
  routes independently), the SVN's prediction for expert $e$ depends on which
  experts are already chosen — this is what lets it represent a context-dependent,
  diminishing-returns value function.
- **Training** (`train_svn.py`): expands each query into 16 (query, subset) rows;
  masked MSE on the true marginals + a submodularity-violation penalty
  $\mathcal{L}_{sub}$ (empty-set vs singleton-set marginals must not increase).
- **Ablation variants** (for E5): `full`, `no_cross_attn`, `no_set_conditioning`,
  plus a `lambda_sub=0` run (submod loss off).

### 3.2 Conformal Safety Gate (Module 2) — `conformal.py`

Produces $C(q)$ with $\mathbb{P}(v^\*\in C(q))\ge 1-\alpha$ (Theorem 1).

- Nonconformity score $s(q,v)=1-\widehat{\mathrm{sim}}(q,v)$ from
  `RetrievalEngine.semantic_norm`.
- `SplitConformal`: one global finite-sample-corrected threshold.
- `MondrianConformal`: per-difficulty-bin thresholds (bins by QPP margin) for
  tighter sets on easy queries while preserving per-bin coverage.
- `evaluate_coverage`: empirical coverage + average/median set size (E3).
- Replaces the plan's criticised "post-hoc Clopper-Pearson" with a
  distribution-free guarantee.

### 3.3 Greedy Budgeted Selector (Module 3) — `greedy.py`

Greedily adds the highest predicted-marginal feasible expert until the budget is
spent or the best remaining marginal drops below `stop_threshold`. Budget
compliance is guaranteed by construction (the invariant is maintained at every
step). This is the policy whose value Theorem 2 bounds.

### 3.4 Integrated pipeline — `pipeline.py`

`CSERPipeline.run` ties the three modules into one query-time call and returns a
`CSERResult` (rank, cost, experts used, conformal coverage indicator, set size).
Handles `gate=None` for the no-safety ablation.

---

## 4. Experiments — how to run & what they show

<!-- __APPEND_SECTION4__ -->

### 4.1 Three drivers, three phases

```bash
cd litevtr_multi_model_framework

# Phase 1: oracle labels -> train SVN -> verify submodularity (E2)
python -m cser.run_phase1 --out-dir reports/cser_phase1

# Phase 2: pipeline + baselines + E1, E3, E4, E5, E6
python -m cser.run_phase2 --out-dir reports/cser_phase2

# Phase 3: E7, E8, E9, E10 + empirical theorem verification
python -m cser.run_phase3 --out-dir reports/cser_phase3

# tests (all phases)
python -m pytest cser/tests -q
```

All three accept `--videos / --csv / --real-models` to swap the synthetic
mock-expert gallery for real backbones over real videos (see §5).

### 4.2 Experiment → artifact → paper claim

| Exp | Driver | Output file | Paper claim it supports |
|-----|--------|-------------|-------------------------|
| E1 | phase2 | `e1_main_results.json` | CSER beats random/cascade/UCB at fixed budget |
| E2 | phase1 | `submodularity_report.json` | value function is (weakly) submodular ⇒ greedy bound applies |
| E3 | phase2 | `e3_conformal.json` | empirical coverage ≥ 1−α for all α (Theorem 1) |
| E4 | phase2 | `e4_budget_curve.json` | CSER Pareto-dominates baselines across budgets |
| E5 | phase2 | `e5_svn_ablation.json` | set-conditioning + submod loss both help |
| E6 | phase2 | `e6_safety_ablation.json` | conformal gate vs heuristic/no-gate |
| E7 | phase3 | `e7_scalability.json` | latency + speedup vs gallery size |
| E8 | phase3 | `e8_robustness.json` | CSER degrades gracefully when expert signals are noisy |
| E9 | phase3 | `e9_expert_contribution.json` | which experts carry signal; SVN↔oracle correlation |
| E10 | phase3 | `e10_oracle_comparison.json` | CSER as % of per-query oracle ceiling |
| Thm 1/2/3 | phase3 | `theorem_verification.json` | the three bounds hold on held-out data |

---

## 5. Running with real expert models

The synthetic gallery uses **mock experts** and validates *logic*; paper numbers
need the real backbones run over real videos. Two things are required:

1. **Model weights.** SCRFD/ArcFace (InsightFace) and MobileNetV3 (torchvision)
   auto-download; MobileCLIP2-S0 and MomentDETR need checkpoints (paths in
   `tasks/real_models.py`).
2. **A video gallery + queries CSV** (`sentence`, `video_id`).

```bash
python -m cser.run_phase2 --out-dir reports/cser_phase2_real \
    --videos /path/to/videos_dir --csv /path/to/queries.csv \
    --real-models --epochs 300
```

`cser/data.py::load_video_dataset` decodes frames, runs the 5 experts via
`expert_features.extract_gallery_signals`, then the same pipeline applies. If a
real model can't construct, CSER warns and falls back to its mock for that run.

---

## 6. Key design decisions & rationale

1. **Soft-only expert selection.** Selecting an expert adds a soft rerank signal,
   never a hard filter. This decouples the *value* problem (Modules 1+3) from the
   *safety* problem (Module 2): $f(S,q)$ is always well defined, and the
   conformal gate is the sole owner of the "never drop $v^\*$" guarantee. Clean
   separation = clean combined theorem.

2. **Exact oracle, no sampling.** With $K=4$ optional experts the 16-subset
   lattice is enumerable, so SVN targets and the E2 submodularity measurement are
   exact. The plan's "sample random subsets for combinatorial coverage" is
   unnecessary at this $K$.

3. **Weak-submodular framing.** E2 on synthetic data gives ~9% violation and
   $\gamma_{p10}\approx 0.88$, so we cite the $(1-e^{-\gamma})$ bound rather than
   the fragile $(1-1/e)$. This is the §11 fallback, baked in from the start, and
   it is data-driven (the number comes from the lattice, not an assumption).

4. **Gallery-agnostic code.** Every driver runs with no external files
   (synthetic) or with `--cache` (real). No code path is hard-wired to a specific
   dataset.

5. **Reuse, don't reimplement.** The 5 experts are the repo's existing
   `tasks/real_models.py` (mock fallback `tasks/mock_models.py`); metrics come
   from `eval.metrics`. The `cser/` package adds only the new method — value
   network, conformal gate, greedy selector, experiments.

---

## 7. Honest limitations

- **Mock-expert numbers are for logic validation, not the paper.** With mocks,
  the CLIP text/frame embeddings are independent hashes, so semantic retrieval is
  near-random and the per-expert signals are heuristic stand-ins. Real numbers
  require `--real-models` over a real video gallery with real weights. Do not
  quote mock numbers as results.
- **Real weights are not in the repo.** MobileCLIP2-S0 and MomentDETR checkpoints
  are referenced by hardcoded paths in `tasks/real_models.py`; point them at your
  copies. SCRFD/ArcFace/MobileNetV3 auto-download.
- **Theorem 2's $\varepsilon$** is measured as the worst-case surrogate error over
  the full lattice; on a larger roster (more experts) this would need sampling.
- **The paper prose is not written** — this is the method/experiment/theory
  infrastructure, ready to produce tables/figures once real models run.

---

## 8. File index (`cser/`)

```
experts.py            expert roster (5 real models), subset utilities, cost model
expert_features.py    run the 5 models over the gallery -> per-video signals
retrieval.py          f(S,q) value function over expert signals
data.py               synthetic-frame + real-video dataset loaders
value_oracle.py       exact 16-subset enumeration -> value matrix + marginals
svn.py                Submodular Value Network (Module 1) + variants
train_svn.py          SVN training (masked MSE + submodularity penalty)
greedy.py             Greedy Budgeted Selector (Module 3)
conformal.py          Conformal Safety Gate (Module 2): split + Mondrian
pipeline.py           integrated CSER inference
baselines.py          B0 all / B1 random / B2 cascade / B4 UCB / oracle
submodularity.py      E2 verification (violation rate, gamma)
experiments_extra.py  E7 / E8 / E9 / E10
theory.py             empirical checks of Theorems 1/2/3
run_phase1.py         driver: oracle -> SVN -> E2
run_phase2.py         driver: pipeline + baselines + E1/E3/E4/E5/E6
run_phase3.py         driver: E7/E8/E9/E10 + theorem verification
tests/                pytest suites for all three phases
GETTING_STARTED.md    handoff guide (start here)
README.md             quick reference
CSER_IMPLEMENTATION.md  this document
```

Related: `docs/delivery/CSER_THEOREMS.md` (formal proofs),
`docs/delivery/AAAI_UPGRADED_PLAN.md` (the plan).

