# CQIN Failure Root Cause Analysis

**Job**: `9801800`  
**Scope**: CQIN only  
**Report root**: `reports/full/9801800/cqin/`

## 1. Verdict

The CQIN branch completed, but the reported CQIN metrics are not a valid estimate
of the learned router's deployable performance. There are three confirmed
problems:

1. The main evaluation and Pareto sweep construct different CQIN features at
   training time and test time.
2. The B8 and B10 cascades use the ground-truth rank at inference time to decide
   whether to escalate and which result to return.
3. A separate head-analysis path uses the correct CQIN test features and still
   shows a genuine value-head collapse: all 570 test queries select the semantic
   fallback route.

The first two problems invalidate the current headline metrics. The third shows
that fixing the evaluation scripts alone will not be sufficient.

## 2. Confirmed Root Causes

### P0: Train/Test Feature Construction Mismatch

The training helper in `scripts/run_final_eval.py:161-171` builds CQIN features
from:

- the real query text;
- the real query embedding;
- the top-20 semantic scores;
- the parsed intent;
- metadata availability.

The CQIN test wrappers do not reuse that helper:

- `scripts/run_final_eval.py:433-443` passes an empty query and 20 zero scores to
  B7, B9, and B10.
- `routing/baselines.py:139-153` does the same for B6 and additionally replaces
  metadata availability with four zeros.
- `scripts/run_pareto_sweep.py:281-286` repeats the empty-query and zero-score
  pattern.
- `scripts/run_final_eval.py:612-614` repeats it in the noise sweep.

This creates a deterministic train/test distribution shift. The six QPP
dimensions are informative during training and constant during evaluation. The
raw query text is also unavailable to the lexical indicator logic at test time.

The main-table QPP ablation is therefore invalid by construction:
`scripts/run_final_eval.py:489-493` zeros QPP dimensions that were already
generated from zero scores.

### P0: Cascade Evaluation Uses Oracle Information

The B8 baseline in `routing/baselines.py:175-192` uses `RouteResult.rank` to:

- stop after stage 1 if the GT is already rank 1;
- decide whether stage 2 improved over stage 1;
- return the best GT rank across stages.

The B10 planner in `routing/calibrated_planner_v2.py:278-311` has the same
problem. It escalates when the GT rank is poor and returns the minimum GT rank
across stages.

At online inference time the GT video and its rank are unknown. These cascades
are oracle-assisted evaluation procedures, not deployable retrieval policies.
Their reported improvements must not be used as CQIN gains.

### P0: Value Head Collapses Even With Correct Test Features

`scripts/run_head_analysis.py:236-268` builds both training and test features
with the real query text and semantic top-20 scores. It does not contain the
main-evaluation feature bug.

Its saved result in
`reports/full/9801800/cqin/head_analysis/head_analysis.json` reports:

| Diagnostic | Result |
|---|---:|
| Test queries | 570 |
| Selected route | `R00_semantic_only_top500` |
| Queries selecting R00 | 570 |
| Non-R00 selections | 0 |
| Feature-importance change for every feature group | `0.00pp` |
| Route/safety conflicts | 0 |

This is a real learning failure: the route-value head does not use query-level
features to produce meaningful route variation.

The zero conflict rate is not evidence that the two heads cooperate well. It is
a consequence of always selecting the no-filter semantic fallback.

A read-only reproduction with the job `9801800` cache, seed `42`, and the same
`350 / 80 / 570` split confirms the separation between the two failures:

| Diagnostic | Correct test features | Broken main-eval features |
|---|---:|---:|
| R00 value-head argmax | `570 / 570` | `570 / 570` |
| Mean safety: time | `0.985` | `0.840` |
| Mean safety: geo | `0.930` | `0.725` |
| Mean safety: motion | `0.934` | `0.741` |
| Mean safety: device | `0.991` | `0.859` |

The value head is already collapsed before the test-feature bug is applied. The
bug is still a P0 issue because it substantially changes the safety head inputs
and therefore contaminates calibration and fallback behavior.

## 3. Why The Value Head Is Prone To Collapse

The following implementation choices are consistent with the observed
collapse:

1. `routing/train_qin.py:134-138` trains 30 route logits with unweighted
   cross-entropy plus Huber utility regression. There is no treatment for
   oracle-route imbalance or redundant routes.
2. `routing/route_schema.py:126-135` defines R00 as the always-available,
   low-cost semantic fallback. It is the easiest safe local optimum.
3. `routing/train_qin.py:137-138` uses unweighted BCE for survival labels. The
   full-run head analysis shows safety outputs concentrated near 1.0.
4. `routing/qin_model.py:68-130` defaults the budget feature to `low`. The
   experiment scripts do not pass query-specific budget tiers, so the four
   budget dimensions are constant during training and test.
5. The full-run head experiment has only 350 training queries for a 531-D input
   and 30-way route decision.

A read-only reconstruction with the job `9801800` cache, seed `42`, and the same
350-query training split directly confirms the imbalance:

| Item | Current-cache result |
|---|---:|
| R00 oracle-route share | `169 / 350 = 48.3%` |
| Oracle R@1 ceiling on the training labels | `56.6%` |
| Survival mean: time / geo / motion / device | `98.9% / 92.6% / 92.9% / 100%` |
| Unique budget feature vectors | only `[1, 0, 0, 0]` (`low`) |

Two older saved label archives show the same pattern:

| Archive | R00 oracle share | Survival mean: time / geo / motion / device |
|---|---:|---|
| `reports/aaai_main/route_bank_train.npz` | `341 / 700 = 48.7%` | `97.7% / 92.0% / 90.6% / 100%` |
| `reports/aaai_v2/route_bank_train.npz` | `295 / 600 = 49.2%` | `97.8% / 91.8% / 90.3% / 100%` |

The exact contribution of class imbalance, redundant routes, and the joint loss
must be measured after fixing the feature path. The collapse itself is already
confirmed by the full-run head analysis.

## 4. Additional Protocol Problems

### Synthetic Metadata Is Injected From Ground Truth

`scripts/run_final_eval.py:82-146` and `scripts/run_pareto_sweep.py:120-188`
generate random metadata for each gallery video. They then enrich each query
intent with metadata copied from that query's GT video with 50% probability.

This is acceptable as a synthetic stress harness, but it is not a real MSR-VTT
metadata protocol. Metadata-filter gains cannot be presented as benchmark gains
without a separate, clearly labeled protocol.

### Budget Routes Do Not Execute Distinct Expensive Operations

The route schema exposes `allow_image_model_calls` and
`allow_dense_refinement`, but `routing/route_executor.py:90-190` does not execute
different expert calls or dense refinement for those flags. It only reports a
tier proxy from `COST_TABLE`.

The Pareto sweep also defines a second cost scale in
`scripts/run_pareto_sweep.py:56-77`. As a result, the current Pareto curve is not
evidence for real compute-quality tradeoffs.

### Calibration Heatmap Is Not Interpretable

All 54 calibration settings produce the same result. The sweep evaluates the
oracle-assisted B10 cascade in `scripts/run_calibration_sweep.py:121-140`, so it
does not isolate the calibration threshold effect.

## 5. Artifact Usability

| Artifact | Usable now? | Reason |
|---|---|---|
| `final_eval/main_results.csv` learned CQIN rows | No | Test feature mismatch |
| `final_eval/main_results.csv` B8/B10 rows | No | Oracle-assisted cascade |
| `pareto_sweep/pareto_sweep.csv` | No | Test feature mismatch and proxy-only budget |
| `calibration_sweep/calibration_heatmap.csv` | No | Oracle-assisted cascade masks threshold behavior |
| `route_bank_ablation/route_bank_ablation.json` | Exploratory only | Correct CQIN features, but oracle-assisted cascade evaluation |
| `training_sensitivity/training_sensitivity.json` | Exploratory only | Correct CQIN features, but oracle-assisted cascade evaluation |
| `head_analysis/head_analysis.json` | Yes, as a failure diagnostic | Correct feature path directly demonstrates value-head collapse |

## 6. Repair Order

1. Create one CQIN inference-feature builder and use it in training, calibration,
   main evaluation, Pareto, noise sweep, and ablations. Add a parity test that
   compares train and inference features for the same query.
2. Remove GT-rank access from B8 and B10. Escalation must use observable signals
   such as QPP uncertainty, planner confidence, latency budget, or candidate
   statistics.
3. Rerun `run_head_analysis.py` first. Do not rerun the full CQIN suite until the
   raw planner selects more than one route and at least one feature group has
   non-zero importance.
4. Rebalance the route objective: report oracle-route histograms, try class
   weights or a ranking loss, remove redundant routes, and start from the
   `10_soft` bank before expanding to 30 routes.
5. Make budget conditioning real: pass requested budget into feature extraction,
   use one cost definition, and either implement the expensive route operations
   or rename the proxy tiers.
6. Separate the synthetic metadata stress test from the final MSR-VTT protocol.
   Do not copy GT metadata into final-test query intents.

The next execution should be a CQIN-only repair run. Re-running CSER is not
necessary until this CQIN branch is corrected.
