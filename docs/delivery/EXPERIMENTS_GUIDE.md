# C-QIN Experiment Design — AAAI 2027 Submission Guide

> **Paper title**: Calibrated Query-Intent Planning for Black-Box Budgeted Video Retrieval
> **Core claim**: A 78K-param learned router achieves near-oracle retrieval while maintaining zero GT-filtered rate and robustness to noisy metadata, all without modifying any frozen expert model.

---

## Experiment Overview (10 experiments)

| # | Experiment | Table/Fig | Dataset | Purpose |
|---|---|---|---|---|
| E1 | Main results | Table 1 | MSR-VTT 1K (n=1000) | Core comparison |
| E2 | Component ablation | Table 2 | MSR-VTT 1K | Isolate each C-QIN component |
| E3 | Route bank ablation | Table 3 | MSR-VTT 1K | Justify 30-route design |
| E4 | Noise robustness | Table 4 + Fig 2 | MSR-VTT 1K | Graceful degradation vs cliff-edge |
| E5 | Cost-accuracy Pareto | Fig 3 | MSR-VTT 1K | Budget-aware tradeoff |
| E6 | Cross-dataset | Table 5 | QVHighlights + Charades-STA | Zero-shot transfer |
| E7 | Moment localization | Table 6 | QVHighlights | Downstream task benefit |
| E8 | Head analysis | Table 7 + Fig 4 | MSR-VTT 1K | What C-QIN learns |
| E9 | Calibration sensitivity | Fig 5 | MSR-VTT 1K | Threshold robustness |
| E10 | SOTA comparison | Table 8 | MSR-VTT 1K | Positioning in literature |

---

## E1: Main Results (Table 1)

### Setup
- **Dataset**: MSR-VTT 1K test, **full 1000 queries × 1000 videos**
- **Split**: 500 train / 100 calibration / 100 dev / 300 test (for router training); retrieval evaluated on all 1000 queries
- **Seeds**: 5 (42, 123, 456, 789, 1024)
- **Metadata**: Synthetic noisy (medium noise: time_shift=7d, geo_missing=30%, geo_wrong=10%)

### Metrics
R@1, R@5, R@10, MedR, MeanR_global (filtered→rank=1001), GT_filtered(%), Avg_cost

### Baselines

| ID | Method | Description |
|---|---|---|
| B0 | Semantic-only | MobileCLIP2 cosine + NNN/QAMP rerank, no metadata |
| B1 | Rule parser | Keyword-based intent → hard filter (industry practice) |
| B2 | Random router | Uniform random route from bank |
| B3 | QPP-only | Margin-based EASY/MEDIUM/HARD (no learning) |
| B5 | Always-hard-all | Hard filter on all detected axes (aggressive) |
| B6 | C-QIN uncalibrated | Both heads, no safety threshold |
| B7 | C-QIN calibrated | Both heads + calibrated dual-threshold |
| B8 | Cascade (no learning) | 3-stage escalation without C-QIN |
| B9 | C-QIN + soft fallback | Dual-threshold with soft degradation |
| B10 | C-QIN + budgeted cascade | Full system (main method) |
| B4 | Oracle route | Per-query best route (upper bound) |

### Statistical Testing
- Paired bootstrap (10000 resamples): B10 vs B1, B10 vs B8
- McNemar test for binary hit/miss
- 95% CI reported
- With n=1000 and expected Δ=6pp, power > 0.99 at α=0.05

### Expected Table

```
Method                  R@1↑   R@5↑   MeanR_g↓  GT_f↓   Cost
─────────────────────────────────────────────────────────────
B0 Semantic-only        31.7   52.9    37.1      0.0%    1.0
B1 Rule parser          40.3   61.4   177.7     16.4%    1.2
B5 Always-hard-all      40.3   61.4   177.7     16.4%    1.2
B8 Cascade              45.4   70.4    17.2      0.0%    1.1
B10 C-QIN+cascade       46.2   69.4    18.7      0.0%    1.9
B4 Oracle               56.8   77.1    13.4      0.0%    1.2
```

### Key claim to prove
B10 > B1 on R@1 (p<0.01) AND GT_filtered=0% vs 16.4% AND MeanR_global 18.7 vs 177.7

---

## E2: Component Ablation (Table 2)

### Setup
- **Dataset**: MSR-VTT 1K test, n=1000, 3 seeds
- **Protocol**: Additive ablation (start from semantic-only, add components one by one)

### Table Structure

```
Config                              R@1    ΔR@1   GT_f%   Cost
────────────────────────────────────────────────────────────────
Semantic-only (MobileCLIP2)         31.7    --     0.0%   1.0
+ Route bank (30 routes, random)    ~35    +3.3    ~8%   3.2
+ route_value_head (learned)        ~42    +7.0    ~5%   2.1
+ filter_safety_head                ~43    +1.0    0.0%  2.3
+ Calibration (dual-threshold)      ~44    +1.0    0.0%  2.2
+ Cascade early-exit                ~46    +2.0    0.0%  1.8
```

### Also: Leave-one-out from full system

```
Full C-QIN+cascade (B10)            46.2    base   0.0%  1.9
  w/o route_value_head              ~38    -8.2    0.0%  2.5
  w/o filter_safety_head            ~43    -3.2    ~7%   1.5
  w/o calibration                   ~39    -7.0    ~10%  1.0
  w/o keyword features              ~38    -8.0    0.0%  2.3
  w/o QPP features                  ~46    +0.0    0.0%  1.9
  w/o cascade                       ~44    -2.0    0.0%  2.2
  random route selection            ~36    -10.0   ~4%   1.6
```

---

## E3: Route Bank Ablation (Table 3)

### Setup
- **Dataset**: MSR-VTT 1K test, 3 seeds
- **Protocol**: Vary route bank composition

### Table Structure

```
Route bank config           Num_routes  R@1    GT_f%   Utilization_entropy
──────────────────────────────────────────────────────────────────────────
Semantic-only routes         5          ~38     0.0%   1.2
Hard-filter routes only     10          ~36    ~12%    2.1
Soft-filter routes only     10          ~40    ~3%     2.0
Semantic + Hard             15          ~42    ~4%     2.5
Semantic + Soft             15          ~43    ~1%     2.4
Full bank (H+S+Sem)        30          ~46     0.0%   3.1
Extended bank (+15 more)    45          ~46.3   0.0%   3.0
```

---

## E4: Noise Robustness (Table 4 + Figure 2)

### Setup
- **Dataset**: MSR-VTT 1K test, 300 queries × 3 seeds per noise level
- **Protocol**: 6 noise levels applied to synthetic metadata

### Noise Definitions

| Level | time_shift | geo_missing | geo_wrong | motion_flip | Description |
|---|---|---|---|---|---|
| clean | 0d | 0% | 0% | 0% | Perfect metadata |
| mild | 3d | 5% | 2% | 5% | Minor inaccuracies |
| medium | 7d | 30% | 10% | 15% | Realistic noise |
| heavy | 14d | 50% | 20% | 30% | Severe degradation |
| missing | 0d | 80% | 0% | 80% | Most metadata absent |
| conflict | 30d | 10% | 50% | 50% | Contradictory signals |

### Table Structure

```
Noise       B0(sem)  B1(rule)  B10(C-QIN)  B4(oracle)  B1_GTf  B10_GTf
─────────────────────────────────────────────────────────────────────────
clean        32.0%   57.7%     51.3%        64.7%      10.7%    0.0%
mild         32.0%   52.3%     49.3%        62.3%      12.3%    0.0%
medium       32.0%   41.7%     46.7%        58.0%      14.3%    0.0%
heavy        32.0%   32.7%     32.7%        48.0%      21.0%    0.0%
missing      32.0%   32.7%     32.7%        36.7%       4.7%    0.0%
conflict     32.0%   25.7%     32.3%        49.3%      49.3%    0.0%
```

### Figure 2: Line plot
- X-axis: noise level (ordered clean→conflict)
- Y-axis: R@1
- Lines: B0 (flat), B1 (crashes), B10 (graceful), B4 (oracle)
- Shaded region: B10 > B1 zone (medium onwards)

### Key claim
"Under conflicting metadata, rule parser drops 7pp BELOW semantic baseline (25.7% vs 32.0%) because it blindly trusts corrupted signals. C-QIN's calibrated safety head detects conflict and refuses hard filtering, maintaining baseline performance (32.3%)."

---

## E5: Cost-Accuracy Pareto (Figure 3)

### Setup
- **Dataset**: MSR-VTT 1K test, 3 seeds
- **Protocol**: Vary cascade budget threshold to trace curve

### Cost Definition
- MobileCLIP2 text encode = 0.1 (cheap, always called)
- MobileCLIP2 image encode = 1.0 (reference unit)
- MomentDETR = 2.0
- InsightFace SCRFD = 0.5
- ArcFace = 0.8
- MobileNetV3 = 0.3
- Avg_cost = sum of model calls per query / N_queries

### Points on Pareto

```
Method                    Cost    R@1     GT_f%
───────────────────────────────────────────────
Semantic-only             1.0     31.7%   0.0%
C-QIN (budget=low)        1.2     ~40%    0.0%
C-QIN (budget=medium)     1.8     ~44%    0.0%
C-QIN (budget=high)       2.5     ~46%    0.0%
Rule parser               3.5     40.3%   16.4%
All-models always         5.0     ~47%    0.0%
Oracle                    varies  56.8%   0.0%
```

### Key claim
"C-QIN at cost=1.8 already exceeds rule parser at cost=3.5, while maintaining zero GT-filtered. C-QIN Pareto-dominates rule parser at every budget level."

---

## E6: Cross-Dataset Generalization (Table 5)

### Setup
- **Datasets**: QVHighlights val (~500 queries), Charades-STA test (~1000 queries)
- **Protocol**: Train C-QIN on MSR-VTT only. Zero-shot transfer to other datasets.
- **Metadata**: Regenerated synthetic per dataset (same noise model)
- **Seeds**: 3

### Table Structure

```
Dataset        Method          R@1    R@5    R@10   GT_f%
─────────────────────────────────────────────────────────
QVHighlights   Semantic-only   ~25    ~48    ~60    0.0%
QVHighlights   Rule parser     ~32    ~55    ~65    ~18%
QVHighlights   C-QIN (0-shot)  ~36    ~58    ~68    0.0%
Charades-STA   Semantic-only   ~18    ~40    ~52    0.0%
Charades-STA   Rule parser     ~24    ~46    ~58    ~20%
Charades-STA   C-QIN (0-shot)  ~28    ~50    ~62    0.0%
```

### Key claim
"C-QIN's routing generalizes zero-shot because it learns query-intent patterns (temporal/spatial/action), not dataset-specific video features."

### Prerequisites
- QVHighlights CLIP features downloaded and cached
- Charades-STA VGG/I3D features downloaded and cached
- Evaluation scripts adapted (existing `experiments/run_qvh_eval.py`)

---

## E7: Moment Localization (Table 6)

### Setup
- **Dataset**: QVHighlights val split (standard moment retrieval benchmark)
- **Protocol**: C-QIN retrieves top-K videos → MomentDETR localizes moments
- **Metrics**: R1@IoU=0.5, R1@IoU=0.7, mAP@0.5, mAP@0.75

### Table Structure

```
Retrieval method          R1@0.5  R1@0.7  mAP@0.5  mAP@0.75
─────────────────────────────────────────────────────────────
Oracle video (GT given)    ~55     ~35     ~50      ~30
Semantic-only → DETR       ~30     ~18     ~25      ~14
Rule parser → DETR         ~35     ~22     ~30      ~17
C-QIN → DETR               ~40     ~25     ~35      ~20
```

### Key claim
"Better retrieval routing directly improves downstream moment localization. C-QIN selects videos where MomentDETR can succeed."

### Implementation
- Use existing `experiments/run_qvh_eval.py`
- MomentDETR already integrated at `tasks/real_models.py:MomentDETRHighlightModel`
- SegmentAggregator at `core/segment_aggregator.py` for [t_start, t_end] output

---

## E8: C-QIN Head Analysis (Table 7 + Figure 4)

### Setup
- **Dataset**: MSR-VTT 1K test

### Sub-experiments

**(a) Route utilization histogram**
- Which routes are selected most often?
- Expected: ~5 routes handle 80% of queries (long-tail)

**(b) Safety head calibration quality**
- Reliability diagram: predicted safety score vs actual GT_filtered rate
- Expected Calibration Error (ECE): uncalibrated ~0.15, calibrated ~0.03

**(c) Head conflict analysis**

```
Scenario                    Fraction    R@1 impact
────────────────────────────────────────────────────
Both agree (safe route)      ~70%       +0 (normal)
Both agree (semantic only)   ~15%       +0 (normal)
Conflict (value→hard, safety→veto)  ~15%   -2pp but prevents GT_filtered
```

**(d) Feature importance** (via permutation)

```
Feature group       ΔR@1 when zeroed
────────────────────────────────────
CLIP text emb (512D)    -3.5pp
QPP statistics (6D)     -0.0pp
Keyword indicators (5D) -8.0pp
Meta availability (4D)  -1.2pp
Budget vector (4D)      -0.5pp
```

---

## E9: Calibration Sensitivity (Figure 5)

### Setup
- **Dataset**: MSR-VTT 1K test, 3 seeds
- **Protocol**: Sweep tau_hard ∈ {0.1, 0.2, ..., 0.9}, tau_soft_ratio ∈ {0.3, 0.4, ..., 0.8}

### Visualization
- Heatmap: x=tau_hard, y=soft_ratio, color=R@1
- Overlay: contour where GT_filtered=0%
- Show broad "safe zone" (not knife-edge)

### Key claim
"Calibration is robust: a wide region of threshold space (tau_hard ∈ [0.6, 0.95], soft_ratio ∈ [0.4, 0.7]) achieves both high R@1 and zero GT-filtered."

---

## E10: SOTA Comparison (Table 8)

### Setup
- **Dataset**: MSR-VTT 1K test (standard benchmark)
- **Framing**: C-QIN competes on cost-accuracy-safety tradeoff, not raw R@1

### Table Structure

```
Method              Params(train)  Models  R@1   Cost  GT_f%  Setting
──────────────────────────────────────────────────────────────────────
CLIP4Clip-B/32      150M           1       44.5  1.0   0%     Fine-tuned
X-CLIP-B/32         150M           1       46.1  1.0   0%     Fine-tuned
TS2-Net             150M           1       47.0  1.0   0%     Fine-tuned
InternVideo2-1B     1B             1       55.9  1.0   0%     Fine-tuned
MobileCLIP2-S0      30M            1       29.4  1.0   0%     Zero-shot
CNPR (ours, prev)   0              1       39.2  1.0   0%     Training-free
Rule parser         0              5       40.3  3.5   16.4%  Rule-based
C-QIN (ours)        78K            5       46.2  1.9   0%     Learned router
```

### Key narrative
"C-QIN with 78K trainable parameters achieves R@1 competitive with 150M fine-tuned models, while using 5 frozen lightweight models at half the cost of naive multi-model deployment, with zero safety violations."

---

## Statistical Protocol (all experiments)

1. **Seeds**: 5 for main results (E1), 3 for ablations (E2-E9)
2. **Tests**: Paired bootstrap (10000 resamples) + McNemar
3. **Reporting**: mean ± std, 95% CI, p-value
4. **Multiple comparisons**: Bonferroni when >3 methods
5. **Effect size**: Cohen's d for key comparisons
6. **Power**: n=1000 with d=0.15 → power>0.99 at α=0.05
7. **GT_filtered CI**: Binomial Clopper-Pearson upper bound (report "0/N observed, CI upper < X%")

---

## Honest Disclosure (must appear in paper)

1. Metadata is **synthetic** (sampled from plausible distributions, not extracted from real video EXIF/GPS)
2. The 5 frozen models are **lightweight edge models**, not SOTA large models
3. C-QIN is trained on MSR-VTT; cross-dataset results are **zero-shot transfer of the router only**
4. The "budgeted" setting assumes **heterogeneous model costs**, realistic for edge but differs from standard benchmarks
5. GT_filtered measures **false elimination** — a safety metric specific to the budgeted routing setting
6. Power/energy numbers are **estimates**, not measured on real hardware

---

## Timeline (2 weeks)

### Week 1
| Day | Task | Output |
|---|---|---|
| 1-2 | Scale to n=1000, run E1 (5 seeds) + E2 (3 seeds) | Table 1, Table 2 |
| 3 | Run E3 (route bank ablation) + E5 (Pareto) | Table 3, Figure 3 |
| 4 | Run E4 (noise sweep, 6 levels × 3 seeds) | Table 4, Figure 2 |
| 5-7 | Build QVHighlights cache, adapt pipeline | Cache ready |

### Week 2
| Day | Task | Output |
|---|---|---|
| 1-2 | Run E6 (cross-dataset) + E7 (moment loc) | Table 5, Table 6 |
| 3 | Run E8 (head analysis) + E9 (calibration heatmap) | Table 7, Fig 4-5 |
| 4 | Compile E10 (SOTA numbers from papers) | Table 8 |
| 5-7 | Statistical tests, figure generation, buffer | All final |

---

## Prerequisites Checklist

- [ ] QVHighlights CLIP features downloaded (~5GB)
- [ ] Charades-STA features downloaded (~3GB)
- [ ] MSR-VTT full 1K evaluation scaled (currently n=300 → n=1000)
- [ ] 5-seed infrastructure in `run_final_eval.py`
- [ ] MomentDETR evaluation metrics (R1@IoU) implemented
- [ ] Pareto curve generation script
- [ ] Calibration heatmap generation script
- [ ] SOTA numbers collected from published papers

---

## Success Criteria (go/no-go for AAAI submission)

### Must achieve (hard requirements)
- [ ] B10 > B1 on R@1 with p < 0.01 (n=1000)
- [ ] B10 GT_filtered = 0% across all noise levels
- [ ] B10 MeanR_global < B1 MeanR_global by > 5x
- [ ] Cross-dataset (E6): C-QIN > rule parser on at least 1 dataset
- [ ] Ablation (E2): route_value_head contributes > 5pp

### Should achieve (soft requirements)
- [ ] B10 > B8 (cascade) by > 1pp
- [ ] Noise sweep crossover at medium noise
- [ ] Moment localization (E7): C-QIN > semantic-only by > 5pp R1@0.5
- [ ] Calibration heatmap shows broad safe zone
- [ ] Oracle gap closed > 50%

### If not achieved → pivot to MLSys/ACM MM
- C-QIN improvement < 3pp over rule parser
- GT_filtered > 0% in any noise condition
- Cross-dataset shows no transfer
- Calibration is brittle (narrow safe zone)
