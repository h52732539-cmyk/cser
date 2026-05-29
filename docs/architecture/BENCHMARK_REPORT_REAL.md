# LiteVTR Multi-Model Framework — Benchmark Report

## Table 1: Overall Multi-Model Benchmark Summary

| Strategy        |   Avg Wall (ms) |   Avg Frames |   Avg Accuracy |   face_det |   face_emb |   highlight |   retrieval |   scene |   face_det_bnd_mae |   face_det_seg_iou |   highlight_bnd_mae |   highlight_seg_iou |   retrieval_seg_iou | Speedup   | Frames-   | Δface_det   | Δface_emb   | Δhighlight   | Δretrieval   | Δscene   |
|-----------------|-----------------|--------------|----------------|------------|------------|-------------|-------------|---------|--------------------|--------------------|---------------------|---------------------|---------------------|-----------|-----------|-------------|-------------|--------------|--------------|----------|
| A_independent   |            9104 |          280 |          1     |          1 |      1     |       1     |       1     |       1 |                0   |              1     |               0     |               1     |               1     | 1.00x     | 0.0%      | +0.00pp     | +0.00pp     | +0.00pp      | +0.00pp      | +0.00pp  |
| B_union_fps     |           15351 |           86 |          0.999 |          1 |      1     |       1     |       1     |       1 |                0.3 |              0.992 |               0     |               1     |               1     | 0.59x     | 69.3%     | +0.00pp     | +0.00pp     | +0.00pp      | +0.00pp      | +0.00pp  |
| C_framework     |            7084 |           70 |          0.703 |          1 |      0.518 |       0.594 |       0.667 |       1 |                0.6 |              0.988 |              19.25  |               0.447 |               0.414 | 1.29x     | 75.1%     | +0.00pp     | -48.24pp    | -40.57pp     | -33.33pp     | +0.00pp  |
| C1_no_prefilter |            6498 |           78 |          0.687 |          1 |      0.427 |       0.483 |       0.733 |       1 |                0   |              1     |              13.502 |               0.309 |               0.544 | 1.40x     | 72.3%     | +0.00pp     | -57.29pp    | -51.68pp     | -26.67pp     | +0.00pp  |
| C2_no_two_stage |            3650 |           35 |          0.599 |          1 |      0     |       0.469 |       0.867 |       1 |                0.6 |              0.988 |              23.543 |               0.29  |               0.179 | 2.49x     | 87.5%     | +0.00pp     | -100.00pp   | -53.11pp     | -13.33pp     | +0.00pp  |

## Table 2: Per-Video Breakdown (C_framework)

| video    | duration   |   prefilter (ms) |   stage1_decode (ms) |   stage1_compute (ms) |   stage2_decode (ms) |   stage2_compute (ms) |   total (ms) |   frames |   S1 frames |   S2 frames |   intervals |
|----------|------------|------------------|----------------------|-----------------------|----------------------|-----------------------|--------------|----------|-------------|-------------|-------------|
| synth_00 | 51.0s      |              118 |                   79 |                  3373 |                   39 |                  2939 |         6551 |       72 |          40 |          32 |           3 |
| synth_01 | 37.6s      |              130 |                  112 |                  3745 |                   44 |                  4537 |         8568 |       68 |          29 |          39 |           3 |
| synth_02 | 54.3s      |              164 |                  178 |                  4021 |                   53 |                  3279 |         7696 |       88 |          45 |          43 |           4 |
| synth_03 | 47.9s      |              260 |                  126 |                  3906 |                   42 |                  2203 |         6543 |       68 |          40 |          28 |           3 |
| synth_04 | 23.8s      |              162 |                  129 |                  3566 |                   36 |                  2168 |         6062 |       53 |          21 |          32 |           2 |

## Table 3: Efficiency-Accuracy Tradeoff

| Strategy        |   Avg Wall (ms) |   Avg Accuracy |   Eff-Acc Score |
|-----------------|-----------------|----------------|-----------------|
| A_independent   |            9104 |          1     |           0.11  |
| B_union_fps     |           15351 |          0.999 |           0.065 |
| C_framework     |            7084 |          0.703 |           0.099 |
| C1_no_prefilter |            6498 |          0.687 |           0.106 |
| C2_no_two_stage |            3650 |          0.599 |           0.164 |

---

**Legend.** Speedup = Wall_A / Wall_X. Δmetric = metric_X − metric_A (in pp). Eff-Acc = Avg Accuracy / (Avg Wall / 1000s).
