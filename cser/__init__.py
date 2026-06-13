"""CSER — Conformal Submodular Expert Routing.

Implementation of AAAI_UPGRADED_PLAN.md (§4 method, §7 experiments, §10 roadmap).

Phase-1 (value + submodularity):
  * experts.py          — expert definitions + subset/mask utilities + cost model
  * expert_features.py  — run the 5 frozen models over a gallery -> per-video signals
  * retrieval.py        — f(S, q) value function over the expert signals
  * data.py             — gallery (synthetic frames now / real videos later) + queries
  * value_oracle.py     — enumerate all 2^K subsets -> exact marginal-value labels
  * svn.py              — Submodular Value Network (Module 1) + ablations
  * train_svn.py        — training loop (subset sampling, MSE + submodularity loss)
  * greedy.py           — Greedy Budgeted Selector (Module 3)
  * submodularity.py    — E2 verification (violation rate, weak-submodularity ratio)
  * run_phase1.py       — driver: data -> oracle -> train SVN -> verify submodularity

Phase-2 (safety + integrated pipeline + experiments):
  * conformal.py        — Conformal Safety Gate (Module 2): split + Mondrian
  * pipeline.py         — integrated CSER inference (SVN + GBS + CSG)
  * baselines.py        — selection baselines (all / random / cascade / UCB / oracle)
  * run_phase2.py       — experiments E1, E3, E4, E5, E6

Phase-3 (extended experiments + theory):
  * experiments_extra.py — E7 scalability, E8 robustness, E9 expert contribution, E10 oracle
  * theory.py            — empirical verification of Theorems 1/2/3 bounds
  * run_phase3.py        — experiments E7-E10 + theorem checks

See cser/CSER_IMPLEMENTATION.md for the full plan-to-code map and
docs/delivery/CSER_THEOREMS.md for the formal proofs.

The 5 "frozen experts" of the plan are the real models in ``tasks/real_models.py``
(mock fallbacks in ``tasks/mock_models.py``):

    e0 = semantic   MobileCLIP   (encode_text / encode_frames)  -- MANDATORY base
    e1 = highlight  MomentDETR   (score frames -> saliency)      -- optional
    e2 = face       SCRFD        (detect faces)                  -- optional
    e3 = face_id    ArcFace      (embed faces)                   -- optional
    e4 = scene      MobileNetV3  (classify scene)                -- optional

Selecting an expert adds its (query-conditioned) score signal as a soft rerank
on top of the semantic base, so f(S,q) is well defined for all 16 subsets. The
integrated pipeline applies a semantic top-k candidate prefilter and unions it
with the Conformal Safety Gate's protected set. With 4 optional experts the
subset lattice has only 2^4 = 16 elements, so oracle marginal values are
computed by exact enumeration.

Use real backbones with ``--real-models`` (needs weights); otherwise the mock
models run so the whole pipeline is testable with no external files.
"""

__version__ = "0.4.0-real-experts"
