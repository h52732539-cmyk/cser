# CSER — Getting Started

**Conformal Submodular Expert Routing** for budgeted video retrieval.
This guide takes you from a fresh checkout to running training and all
experiments in a few minutes. No prior context with the repo is needed.

> **One-line summary.** CSER learns a *set-conditioned value network* that picks
> the cheapest useful subset of retrieval "experts" per query (greedy under a
> compute budget), with a *conformal safety gate* that guarantees the correct
> video is never filtered out with probability ≥ 1−α.

---

## 0. What you received

The `cser/` package lives inside the larger `litevtr_multi_model_framework`
repo and **reuses that repo's 5 frozen expert models** (`tasks/real_models.py`,
with mock fallbacks in `tasks/mock_models.py`). Run everything *from the repo
root*. The package ships a synthetic frame-level gallery so you can run the
entire training + experiment pipeline with **no external data and no model
weights** (it uses the mock experts automatically).

The 5 experts (the plan's frozen multimodal models):

```
e0 = semantic   MobileCLIP    encode_text / encode_frames   MANDATORY base
e1 = highlight  MomentDETR    score(frames) -> saliency       optional
e2 = face       SCRFD         detect(frames) -> (has, conf)   optional
e3 = face_id    ArcFace       embed(frames) -> vector         optional
e4 = scene      MobileNetV3   classify(frames) -> label       optional
```

CSER learns which subset of optional experts to call per query (greedy under a
compute budget), with a conformal safety gate guaranteeing the correct video is
never filtered out with probability ≥ 1−α.

```
litevtr_multi_model_framework/          <- run commands from HERE
├── cser/                               <- the new method (what this guide is about)
│   ├── GETTING_STARTED.md              <- you are here
│   ├── README.md                       <- quick reference
│   ├── CSER_IMPLEMENTATION.md          <- full plan→code map + design rationale
│   ├── requirements-cser.txt           <- the only deps you need (numpy + torch)
│   ├── run_phase1.py / run_phase2.py / run_phase3.py
│   ├── tests/                          <- pytest suite (run this first!)
│   └── *.py                            <- the modules (see §6)
├── core/  eval/  metadata/             <- retrieval engine CSER builds on
├── data/cache/                         <- text embeddings live here
└── docs/delivery/
    ├── AAAI_UPGRADED_PLAN.md           <- the research plan CSER implements
    └── CSER_THEOREMS.md                <- formal proofs of the 3 guarantees
```

---

## 1. Install (2 minutes)

Python 3.8+ with `numpy` and `torch`. Nothing else is needed for CSER.

```bash
cd litevtr_multi_model_framework          # repo root
python -m venv .venv && source .venv/bin/activate   # optional but recommended
pip install -r cser/requirements-cser.txt
```

CPU is fine — the synthetic runs train in well under a minute. No GPU required.

---

## 2. Smoke test — confirm it works (1 minute)

Run the test suite first. If this is green, the whole pipeline is wired correctly.

```bash
python -m pytest cser/tests -q
```

Then run the Phase-1 pipeline on the built-in synthetic gallery:

```bash
python -m cser.run_phase1 --out-dir reports/cser_phase1
```

You should see the SVN train (val MSE dropping), then a submodularity verdict
like `verdict=weakly_submodular ... gamma_mean=0.97`. Artifacts land in
`reports/cser_phase1/`. If you got here, you're set.

<!-- __APPEND_GS_REST__ -->

---

## 3. The three phases (what to run and why)

Each phase is one self-contained driver. Run them in order; later phases retrain
the model internally so they don't depend on earlier outputs.

```bash
# Phase 1 — learn the value network + test the core scientific claim
python -m cser.run_phase1 --out-dir reports/cser_phase1
#   builds exact oracle value labels (all 16 expert subsets per query)
#   -> trains the Submodular Value Network (SVN)
#   -> E2: verifies submodularity (the assumption the whole method rests on)

# Phase 2 — full system vs baselines + safety
python -m cser.run_phase2 --out-dir reports/cser_phase2
#   E1 main comparison (CSER vs random / cascade / UCB-bandit / oracle)
#   E3 conformal coverage (does P(v* in C(q)) >= 1-alpha hold?)
#   E4 budget-vs-accuracy curve, E5 SVN ablation, E6 safety ablation

# Phase 3 — extended experiments + theorem verification
python -m cser.run_phase3 --out-dir reports/cser_phase3
#   E7 scalability, E8 robustness, E9 expert contribution, E10 oracle gap
#   + empirical check that Theorems 1/2/3 bounds hold on held-out data
```

Common flags (all phases): `--epochs`, `--budget` (compute budget B in expert
calls), `--alpha` (conformal level), `--seed`, `--metric {rr,recall@1,recall@5,recall@10}`.

---

## 4. How to train / experiment with your own settings

**Change the budget** (how many experts CSER may call per query):

```bash
python -m cser.run_phase2 --out-dir reports/b2 --budget 2.0
python -m cser.run_phase2 --out-dir reports/b5 --budget 5.0
```

**Change the safety level** (tighter coverage = larger conformal sets):

```bash
python -m cser.run_phase2 --out-dir reports/strict --alpha 0.01
```

**Train longer / bigger synthetic gallery:**

```bash
python -m cser.run_phase1 --out-dir reports/big --epochs 400 \
    --syn-videos 1000 --syn-queries 400
```

**Train just the SVN in your own script** (the programmatic entry points):

```python
from cser.data import build_synthetic_dataset      # or load_msrvtt_dataset
from cser.retrieval import RetrievalEngine
from cser.value_oracle import build_oracle_labels
from cser.train_svn import train_svn, SVNTrainConfig

ds = build_synthetic_dataset(n_videos=800, n_queries=300, seed=0)
eng = RetrievalEngine(ds.index, ds.meta_filter)
tr, cal, te = ds.split(seed=0)
emb = ds.query_embs[tr]
labels = build_oracle_labels(
    eng, emb, [ds.query_texts[i] for i in tr],
    [ds.gt_video_ids[i] for i in tr], [ds.intents[i] for i in tr])
model, history = train_svn(labels, SVNTrainConfig(epochs=300, variant="full"))
```

Then run inference with the integrated pipeline:

```python
from cser.pipeline import CSERPipeline
pipe = CSERPipeline(eng, model, conformal_gate=None, budget=3.0)
res = pipe.run(ds.query_embs[0], labels.query_feats[0],
               ds.gt_video_ids[0], ds.intents[0])
print(res.rank, res.cost, res.active_axes, res.gt_in_conformal_set)
```

---

## 5. Running with REAL expert models (for paper numbers)

The synthetic gallery uses **mock experts** and validates *logic*; real numbers
need the real backbones run over real videos. Two things are required:

1. **Model weights.** SCRFD / ArcFace (InsightFace) and MobileNetV3 (torchvision)
   auto-download. MobileCLIP2-S0 and MomentDETR need checkpoint files — see the
   hardcoded paths in `tasks/real_models.py` and point them at your copies.
2. **A video gallery + queries.** A directory of `.mp4` files (or a
   `manifest.json` of `[{"id","path"}]`) and a queries CSV with columns
   `sentence`, `video_id`.

Then add `--videos`, `--csv`, and `--real-models` to any phase:

```bash
python -m cser.run_phase2 --out-dir reports/real \
    --videos /path/to/videos_dir \
    --csv    /path/to/queries.csv \
    --real-models --epochs 300
```

If a real model fails to construct (missing weight/dep), CSER prints a warning
and **falls back to the mock** for that run — so the command never hard-crashes,
but check the log to confirm you actually got the real backbones.

> ⚠️ **Do not quote synthetic / mock-expert numbers in the paper.** The mock
> models are deterministic stand-ins. Real numbers require `--real-models` over a
> real video gallery with real weights.

---

## 6. Module map (where to look when changing things)

| File | Role | Touch this when… |
|------|------|------------------|
| `experts.py` | expert roster + costs + subset utilities | adding/removing experts, changing costs |
| `expert_features.py` | runs the 5 models over the gallery -> per-video signals | changing how a model's output becomes a signal |
| `retrieval.py` | the value function f(S, q) over expert signals | changing how a selected expert reranks |
| `data.py` | synthetic-frame + real-video loaders | adding a new dataset |
| `value_oracle.py` | exact enumeration → marginal-value labels | changing the training target |
| `svn.py` | Submodular Value Network (Module 1) | changing the model architecture |
| `train_svn.py` | training loop (MSE + submodularity penalty) | tuning the loss / optimiser |
| `conformal.py` | Conformal Safety Gate (Module 2) | changing the safety mechanism |
| `greedy.py` | Greedy Budgeted Selector (Module 3) | changing the selection policy |
| `pipeline.py` | integrated inference | end-to-end behaviour |
| `baselines.py` | comparison policies | adding a baseline |
| `submodularity.py` | E2 verification + γ | the submodularity analysis |
| `experiments_extra.py` | E7–E10 | the extended experiments |
| `theory.py` | empirical theorem checks | verifying bounds |

Deeper reading: **`CSER_IMPLEMENTATION.md`** (full plan→code map, design
decisions, limitations) and **`../docs/delivery/CSER_THEOREMS.md`** (the proofs).

---

## 7. Reading the key outputs

- **`reports/cser_phase1/submodularity_report.json`** — the go/no-go.
  `verdict ∈ {submodular, weakly_submodular, non_submodular}`; if
  `weakly_submodular`, cite the `(1−e^{−γ})` bound with `gamma_ratio_p10`.
- **`reports/cser_phase2/e1_main_results.json`** — CSER vs baselines
  (R@1, R@5, MRR, avg cost, avg experts, conformal coverage).
- **`reports/cser_phase2/e3_conformal.json`** — for each α, check
  `empirical_coverage ≥ target_coverage` (Theorem 1 on data).
- **`reports/cser_phase3/theorem_verification.json`** — `holds: true` for each
  theorem.

---

## 8. Troubleshooting

| Symptom | Fix |
|---------|-----|
| `ModuleNotFoundError: core` / `cser` | Run from the **repo root**, not inside `cser/`. The drivers add the root to `sys.path`; bare scripts won't. |
| `No module named torch` | `pip install -r cser/requirements-cser.txt` |
| Phase 3 feels slow | It builds a 1000-video oracle lattice; lower `--syn-videos 300` for a quick pass. |
| Want a faster smoke run | `python -m cser.run_phase1 --epochs 60 --syn-videos 60 --syn-queries 80` |

---

## 9. Suggested first session

```bash
cd litevtr_multi_model_framework
pip install -r cser/requirements-cser.txt
python -m pytest cser/tests -q                 # 1. confirm green
python -m cser.run_phase1 --out-dir reports/cser_phase1   # 2. train + submodularity
python -m cser.run_phase2 --out-dir reports/cser_phase2   # 3. baselines + safety
python -m cser.run_phase3 --out-dir reports/cser_phase3   # 4. extended + theory
# then read reports/*/...summary.json and CSER_IMPLEMENTATION.md
```

When you have the real `msrvtt_cache.npz`, re-run Phases 2–3 with `--cache ...`
to produce paper-ready numbers.

