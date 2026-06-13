# CSER Repair Summary After Job 9801800

**Source diagnostic**: `docs/delivery/EXPERIMENT_RESULT_ANALYSIS_9801800.md`
**Repair scope**: CSER correctness and reporting paths only

## 1. Implemented Repairs

### Operational candidate filtering

`CSERPipeline` now applies a semantic top-k prefilter and unions the result with
the safety gate prediction set:

```text
final_candidates(q) = semantic_top_k(q) union C(q)
```

This guarantees that no member of `C(q)` is removed while allowing actual
candidate reduction. Phase-2 now reports:

- `GT_filtered_rate`
- `avg_candidates_after_filter`
- `candidate_reduction_rate`
- `hard_filter_activation_rate`

E6 is no longer a fixed-output coverage-only comparison.

### Robustness and theorem reporting

- E8 now evaluates CSER through the operational pipeline and reports actual
  candidate-filter statistics.
- E7 artifacts explicitly label latency as `cached_score_rerank_only`.
- Theorem 2 reports whether its lower bound is non-vacuous.
- Theorem 3 no longer reports `all_three_hold=true` when the greedy bound is
  negative or the measured monotonicity assumption fails.

### Experiment integrity

- Explicit `--real-models` runs fail closed if a backbone cannot initialize.
  They no longer silently fall back to mock adapters.
- MRR now counts filtered or missing GT results as reciprocal rank zero.
- Phase-2 artifacts label cost as `offline_index_expert_unit_proxy`.

## 2. Safety-Only Cache Validation

The existing `data/msrvtt_real_1k` MobileCLIP cache was used for a read-only
top-100 safety-path calculation with the original seed-42 `600 / 150 / 250`
split:

| Gate | Protected-set coverage | GT filtered | Avg candidates | Reduction |
|---|---:|---:|---:|---:|
| Mondrian conformal | 97.2% | 2.4% | 231.9 / 1000 | 76.8% |
| Split conformal | 96.4% | 3.2% | 190.0 / 1000 | 81.0% |
| Heuristic threshold | 98.8% | 1.2% | 285.5 / 1000 | 71.4% |
| No gate | N/A | 6.4% | 100.0 / 1000 | 90.0% |

This validates the repaired safety semantics. It is not a replacement for a new
five-expert Phase-2 run: the cache-only check uses the prepared MobileCLIP cache
and does not recompute optional-expert reranking.

## 3. Remaining Research Issues

The repair does not make job `9801800` paper-ready. A new experiment iteration
is still required for:

1. Weak SVN routing: low oracle-marginal correlation and semantic-only beating
   learned SVN-greedy value.
2. Objective mismatch: the measured value lattice violates monotonicity, so the
   clean monotone-submodular theorem assumptions do not hold.
3. Expert utility: `face_id` has no text-only query reference embedding and the
   scene expert contributes weakly.
4. Latency evidence: current timing covers cached score reranking, not
   end-to-end incremental expert execution.
5. Protocol quality: router training and calibration must be separated from the
   official final test protocol, and caption handling must be stated explicitly.

## 4. Verification

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /hpc2hdd/home/yyan047/miniconda3/envs/cser/bin/python \
  -m pytest cser/tests tests/eval/test_metrics.py -q
```

Result: `54 passed`.
