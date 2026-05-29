# LiteVTR Multi-Model Framework — Benchmark Report

## Table 1: Overall Multi-Model Benchmark Summary

| Strategy        |   Avg Wall (ms) |   Avg Frames |   Avg Accuracy |   face_det |   face_emb |   highlight |   retrieval |   scene |   face_det_bnd_mae |   face_det_seg_iou |   highlight_bnd_mae |   highlight_seg_iou |   retrieval_seg_iou | Speedup   | Frames-   | Δface_det   | Δface_emb   | Δhighlight   | Δretrieval   | Δscene   |
|-----------------|-----------------|--------------|----------------|------------|------------|-------------|-------------|---------|--------------------|--------------------|---------------------|---------------------|---------------------|-----------|-----------|-------------|-------------|--------------|--------------|----------|
| A_independent   |            4720 |          280 |          1     |          1 |      1     |       1     |       1     |       1 |                0   |              1     |               0     |               1     |               1     | 1.00x     | 0.0%      | +0.00pp     | +0.00pp     | +0.00pp      | +0.00pp      | +0.00pp  |
| B_union_fps     |            2176 |           86 |          0.999 |          1 |      1     |       1     |       1     |       1 |                0.3 |              0.992 |               0     |               1     |               1     | 2.17x     | 69.3%     | +0.00pp     | +0.00pp     | +0.00pp      | +0.00pp      | +0.00pp  |
| C_framework     |             867 |           70 |          0.731 |          1 |      0.518 |       0.594 |       0.867 |       1 |                0.6 |              0.988 |              19.25  |               0.447 |               0.431 | 5.45x     | 75.1%     | +0.00pp     | -48.24pp    | -40.57pp     | -13.33pp     | +0.00pp  |
| C1_no_prefilter |             476 |           78 |          0.716 |          1 |      0.427 |       0.483 |       0.933 |       1 |                0   |              1     |              13.502 |               0.309 |               0.579 | 9.91x     | 72.3%     | +0.00pp     | -57.29pp    | -51.68pp     | -6.67pp      | +0.00pp  |
| C2_no_two_stage |             494 |           35 |          0.599 |          1 |      0     |       0.469 |       0.867 |       1 |                0.6 |              0.988 |              23.543 |               0.29  |               0.179 | 9.56x     | 87.5%     | +0.00pp     | -100.00pp   | -53.11pp     | -13.33pp     | +0.00pp  |

## Table 2: Per-Video Breakdown (C_framework)

| video    | duration   |   prefilter (ms) |   stage1_decode (ms) |   stage1_compute (ms) |   stage2_decode (ms) |   stage2_compute (ms) |   total (ms) |   frames |   S1 frames |   S2 frames |   intervals |
|----------|------------|------------------|----------------------|-----------------------|----------------------|-----------------------|--------------|----------|-------------|-------------|-------------|
| synth_00 | 51.0s      |              189 |                  192 |                  1478 |                   59 |                     7 |         1926 |       72 |          40 |          32 |           3 |
| synth_01 | 37.6s      |              110 |                   90 |                    12 |                   56 |                     7 |          276 |       68 |          29 |          39 |           3 |
| synth_02 | 54.3s      |              141 |                  117 |                    14 |                   82 |                     7 |          363 |       88 |          45 |          43 |           4 |
| synth_03 | 47.9s      |              141 |                  101 |                     7 |                   46 |                     8 |          305 |       68 |          40 |          28 |           3 |
| synth_04 | 23.8s      |               87 |                   75 |                  1222 |                   66 |                    14 |         1465 |       53 |          21 |          32 |           2 |

## Table 3: Efficiency-Accuracy Tradeoff

| Strategy        |   Avg Wall (ms) |   Avg Accuracy |   Eff-Acc Score |
|-----------------|-----------------|----------------|-----------------|
| A_independent   |            4720 |          1     |           0.212 |
| B_union_fps     |            2176 |          0.999 |           0.459 |
| C_framework     |             867 |          0.731 |           0.843 |
| C1_no_prefilter |             476 |          0.716 |           1.504 |
| C2_no_two_stage |             494 |          0.599 |           1.213 |

---

**Legend.** Speedup = Wall_A / Wall_X. Δmetric = metric_X − metric_A (in pp). Eff-Acc = Avg Accuracy / (Avg Wall / 1000s).
