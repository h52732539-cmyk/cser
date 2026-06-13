# CQIN / CSER MSR-VTT Full Run Result Analysis

**Job**: `9801800`  
**Run window**: 2026-05-31 18:00:25 +08:00 to 2026-06-02 05:15:03 +08:00  
**Report root**: `reports/full/9801800/`  
**Reference plan**: `docs/delivery/AAAI_UPGRADED_PLAN.md`

## 1. Executive Verdict

The full Slurm job completed successfully and the run used the real MSR-VTT
videos plus all five real expert adapter classes. It is a valid diagnostic run.

The results do **not** yet reach the expected AAAI-upgraded outcome. The current
artifacts support a narrower statement:

> Real-model CSER has a useful conformal-set coverage diagnostic and the measured
> reciprocal-rank value lattice is approximately submodular, but the learned
> router, operational safety integration, cost model, latency evidence, and
> baseline protocol are not yet sufficient for the planned paper claims.

The strongest blockers are:

1. CSER `R@1=28.0%`, below the planned `38.8%`, and tied with UCB rather than
   dominating the baselines.
2. The safety gate measures whether the GT is in `C(q)` but does not actually
   filter candidates. `GT_filtered_rate` is fixed to `0.0` in evaluation.
3. The value function has a `17.425%` monotonicity-violation rate. The clean
   monotone-submodular greedy theorem assumption is therefore not satisfied.
4. Theorem 2 passes only vacuously: `epsilon=1.4989` gives a negative lower bound
   (`RHS=-5.701`).
5. The SVN router is weak: its oracle-marginal correlation is `0.175`, and its
   mean reciprocal-rank value is below semantic-only retrieval.
6. The latency experiment reports `speedup < 1`, covers only galleries up to 1K,
   and times cached reranking rather than real end-to-end expert execution.

## 2. Run Integrity

### Passed

- `reports/setup/cqin_cser_full_latest.status` records:
  `state=complete`, `detail=all_experiments_complete`, `exit_code=0`.
- Static preflight marked all five experts ready:
  MobileCLIP2-S0, MomentDETR, SCRFD, ArcFace, and MobileNetV3.
- Runtime preflight recorded `all_real=true`.
- Each CSER phase logged `[cser] using REAL expert models`.
- The prepared cache contains 1000 gallery videos and 1000 queries.
- The full run produced CQIN final evaluation plus CSER phases 1, 2, and 3.

### Runtime Caveat

InsightFace requested `CUDAExecutionProvider`, but ONNX Runtime reported that
only `AzureExecutionProvider` and `CPUExecutionProvider` were available. The
InsightFace components therefore ran on CPU. This does not invalidate the
retrieval metrics, but it prevents using this run as GPU end-to-end latency
evidence.

The stderr file also contains repeated ONNX Runtime thread-affinity warnings.
They did not stop the job, but should be cleaned up before the next timing run.

## 3. Dataset Protocol Caveat

`scripts/prepare_msrvtt_real_1k.py` deliberately selects one caption per video.
The resulting real cache has:

| Item | Value |
|---|---:|
| Gallery videos | 1000 |
| Queries | 1000 |
| Captions per video | 1 |
| Cached MobileCLIP frames per video | 6 |

CSER then splits these 1000 test-gallery queries into `600 train / 150 cal / 250
test` for router training, conformal calibration, and final evaluation.

This is appropriate for a diagnostic router-development run, but it is not yet
a final paper protocol. The final protocol should explicitly separate router
training/calibration data from the official final test set and should state
whether evaluation uses one caption or the full-caption retrieval setup.

## 4. CSER Scorecard Against the Plan

| Experiment | Planned claim | Actual result | Verdict |
|---|---|---|---|
| E1 main result | CSER `R@1=38.8%`, `R@5=61.0%`, `0%` GT elimination, `2.1` experts | CSER `R@1=28.0%`, `R@5=57.6%`, `0%` reported GT elimination, `2.064` experts, cost `3.184` | Fail |
| E2 submodularity | Violation rate `<5%` | Submodularity violation `4.3%`, gamma mean `0.975`, gamma p10 `0.968`; monotonicity violation `17.425%` | Partial |
| E3 conformal coverage | Coverage `>=1-alpha`; Mondrian sets tighter than split | Coverage passes for all alpha values; at `alpha=0.05`, split `96.8%`, Mondrian `98.0%`; Mondrian average set is larger (`239.1` vs `215.9`) | Partial |
| E4 budget curve | CSER Pareto-dominates baselines | CSER does not dominate: at budget `3.0`, CSER `R@1=28.8%` vs fixed/all-feasible `30.4%`; curve omits random and UCB | Fail |
| E5 SVN ablation | Set conditioning and cross-attention improve retrieval | Full SVN ties no-set-conditioning on `R@1=28.0%`; no-set-conditioning has better `R@5` and MRR at lower cost | Fail |
| E6 safety ablation | Gate variants trade retrieval against filtering safety | All four variants have identical retrieval metrics and `GT_filtered_rate=0.0` | Invalid as evidence |
| E7 scalability | 1K/10K/50K galleries and `15x-35x` speedup | Only 250/500/1000 galleries; speedup `0.169x-0.247x` | Fail |
| E8 robustness | Graceful degradation and safety under noisy signals | CSER `R@1` drops from `32.7%` to `29.3%` and remains above cascade; reported GT filtering is always `0.0` by construction | Partial |
| E9 expert contribution | Interpretable expert value patterns and accurate SVN routing | SVN-oracle correlation `0.175`; optional-expert mean oracle marginals are non-positive; `face_id` is always zero | Fail |
| E10 oracle comparison | CSER reaches `89.8%` of oracle | CSER reaches `86.0%` of oracle RR value, below semantic-only `90.1%`; true-value greedy reaches `100%` | Fail |

## 5. Main CSER Results

From `reports/full/9801800/cser/phase2/e1_main_results.json`:

| Method | R@1 | R@5 | MRR | Avg cost | Avg experts |
|---|---:|---:|---:|---:|---:|
| B0 all-experts label at budget 5 | 26.4% | 50.0% | 0.372 | 5.000 | 3.000 |
| B1 random | 26.8% | 56.4% | 0.399 | 3.480 | 2.296 |
| B2 fixed cascade | 26.4% | 50.0% | 0.372 | 5.000 | 3.000 |
| B4 UCB bandit | 28.0% | 55.6% | 0.404 | 4.348 | 2.528 |
| B-oracle | 35.2% | 61.2% | 0.474 | 1.730 | 1.368 |
| B6 CSER | 28.0% | 57.6% | 0.408 | 3.184 | 2.064 |

CSER improves over the fixed cascade at the selected budget and improves UCB
`R@5` by `2.0pp`, but it does not improve UCB `R@1`. It also remains `7.2pp`
below the oracle ceiling.

The `B0_all_experts` label is misleading at budget `5.0`: the five-expert full
cost is `9.5`, so the baseline selects the most expensive feasible subset rather
than all five experts. The actual five-expert row appears in the E4 sweep at
budget `9.5`, where `R@1=25.2%`. This confirms that optional experts can hurt
retrieval, which is useful evidence for selection but conflicts with the clean
monotone-value theorem.

## 6. Theory Assessment

### Theorem 1: Conformal Coverage

The conformal membership diagnostic works on the held-out 250-query split:

| Alpha | Target | Split coverage | Mondrian coverage | Split avg set | Mondrian avg set |
|---:|---:|---:|---:|---:|---:|
| 0.01 | 99% | 100.0% | 100.0% | 977.6 | 1000.0 |
| 0.05 | 95% | 96.8% | 98.0% | 215.9 | 239.1 |
| 0.10 | 90% | 92.8% | 94.8% | 76.8 | 104.3 |
| 0.20 | 80% | 83.6% | 84.4% | 34.7 | 37.4 |

The coverage claim is supported. The planned adaptive-efficiency claim is not:
Mondrian sets are consistently larger than split-conformal sets.

### Theorem 2: Greedy Approximation

The reported data are:

| Item | Value |
|---|---:|
| Submodularity violation | 4.3% |
| Monotonicity violation | 17.425% |
| Gamma p10 | 0.968 |
| Approximation factor | 0.620 |
| Worst-case surrogate error epsilon | 1.4989 |
| Realised LHS | 0.408 |
| Bound RHS | -5.701 |

The `bound_holds=true` flag is not a useful empirical result because any
non-negative retrieval value beats a negative lower bound. More importantly, the
clean monotone-submodular assumption is not satisfied by the measured lattice.

### Theorem 3: Combined Guarantee

The current `all_three_hold=true` flag should not be quoted as a paper result:

- Theorem 1 is a valid conformal-set membership check.
- Theorem 2 is vacuous at the measured epsilon and its monotonicity assumption is
  not met.
- The safety gate is not applied to an actual candidate-filtering operation.

## 7. Implementation Boundaries That Affect Claims

### Safety Is Not Operationally Integrated

`cser/pipeline.py` computes `gt_in_conformal_set` and `conformal_set_size`, but
then ranks the complete gallery. It always returns `gt_filtered=False`.
`cser/run_phase2.py` also writes `GT_filtered_rate=0.0` unconditionally.

As a result:

- E6 variants are identical by construction.
- The current run validates conformal coverage, not safe candidate filtering.
- The planned filtering-efficiency story is not yet implemented.

### Cost Is a Proxy, Not Measured Online Expert Compute

All five expensive experts are executed over the gallery up front and their
signals are cached. Query-time expert selection chooses which cached score
vectors to blend into reranking. The reported cost is a hand-configured
expert-unit proxy.

This is a coherent offline-index setting, but it differs from the plan's
per-query "which experts to call" framing. The paper must either:

1. Reframe CSER as budgeted query-time access to precomputed expert indexes, or
2. Implement and measure genuinely incremental expert execution.

### Some Optional Experts Cannot Contribute Under the Current Queries

`face_id` requires a reference face embedding in the query prior. CSV queries
only provide text, so `face_emb` remains `None`. This explains its exact-zero
oracle marginal contribution.

The scene expert is also weak in this run: it has positive marginal value for
only `1.6%` of queries. The current ImageNet-to-scene mapping and lexical scene
cues need review before using scene routing as a central paper example.

## 8. CQIN Results From the Same Job

The original CQIN branch completed, but its headline numbers are not
interpretable as learned-router performance:

1. The main CQIN evaluation and Pareto sweep use empty query text and zero QPP
   scores at test time even though training uses real query text and semantic
   top-score statistics.
2. The B8 and B10 cascades use the GT rank to decide whether to escalate and
   which stage result to return. That oracle information is unavailable during
   online inference.
3. The separate head-analysis path uses correct test features and still shows a
   genuine collapse: all 570 queries choose `R00_semantic_only_top500`, with
   `0.00pp` permutation importance for every feature group.

The earlier CQIN table should therefore not be quoted. The focused evidence,
affected artifacts, and repair order are documented in
`docs/delivery/CQIN_FAILURE_ROOT_CAUSE_ANALYSIS_9801800.md`.

## 9. Recommended Next Actions

### Priority 0: Correct the Claim Surface

1. Do not quote `38.8% R@1`, `25% cost`, `15x-35x speedup`, or
   `all_three_hold=true` from this run.
2. Mark the current outputs as real-model diagnostic results, not final paper
   tables.

### Priority 1: Implement the Missing Safety Path

1. Add an actual candidate filter `F(q)` to the retrieval pipeline.
2. Enforce that no item in `C(q)` can be removed.
3. Measure actual candidate reduction, actual GT elimination, retrieval metrics,
   and latency for Mondrian, split, heuristic, and no-gate variants.

### Priority 2: Resolve the Objective and Theorem Mismatch

1. Decide whether to enforce monotonicity in aggregation or adopt a
   non-monotone-submodular formulation.
2. Reduce surrogate error enough to produce a non-vacuous lower bound.
3. Train and report the objective used for the headline table. The current SVN
   uses reciprocal rank while the planned headline is Recall@1.

### Priority 3: Repair Expert Utility

1. Remove `face_id` from the current text-only experiment or add a benchmark
   slice with reference-face queries.
2. Audit scene-label mapping and scene lexical cues.
3. Tune score aggregation so optional experts help when selected and do not
   systematically reduce semantic-only retrieval.
4. Use the true-greedy `100%` oracle result as the debugging target: the lattice
   contains useful selections, but the learned surrogate is not identifying
   them reliably.

### Priority 4: Rebuild the Evaluation Protocol

1. Train and calibrate the router outside the final MSR-VTT test query set.
2. State and validate the caption protocol explicitly.
3. Add multi-seed CSER results and significance intervals.
4. Add the missing RL/PPO and original CQIN baselines to CSER E1.
5. Compare against the actual all-five-expert baseline at cost `9.5`.
6. Evaluate E7 on 1K/10K/50K galleries with end-to-end timing.
7. Fix ONNX Runtime GPU provider availability before reporting GPU latency.

## 10. Bottom Line

The run succeeded operationally and exposed the right next engineering work. It
did not validate the planned AAAI-ready claim set. The best paper path is to
retain the real-model data pipeline, conformal coverage analysis, and exact
oracle lattice, then repair safety integration, evaluation protocol, and router
quality before rerunning the final experiment battery.
