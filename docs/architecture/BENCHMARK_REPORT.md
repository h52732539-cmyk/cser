# LiteVTR Multi-Model Framework — Benchmark Report

## Table 1: Overall Multi-Model Benchmark Summary

| Strategy        |   Avg Wall (ms) |   Avg Frames |   Avg Accuracy |   face_det |   face_emb |   highlight |   retrieval |   scene |   face_det_bnd_mae |   face_det_seg_iou |   highlight_bnd_mae |   highlight_seg_iou |   retrieval_seg_iou | Speedup   | Frames-   | Δface_det   | Δface_emb   | Δhighlight   | Δretrieval   | Δscene   |
|-----------------|-----------------|--------------|----------------|------------|------------|-------------|-------------|---------|--------------------|--------------------|---------------------|---------------------|---------------------|-----------|-----------|-------------|-------------|--------------|--------------|----------|
| A_independent   |             446 |          280 |          1     |          1 |      1     |       1     |       1     |   1     |                0   |              1     |               0     |               1     |               1     | 1.00x     | 0.0%      | +0.00pp     | +0.00pp     | +0.00pp      | +0.00pp      | +0.00pp  |
| B_union_fps     |             145 |           86 |          0.971 |          1 |      1     |       1     |       1     |   0.775 |                0.3 |              0.992 |               0     |               1     |               1     | 3.07x     | 69.3%     | +0.00pp     | +0.00pp     | +0.00pp      | +0.00pp      | -22.48pp |
| C_framework     |             176 |          106 |          0.843 |          1 |      0.962 |       0.948 |       0.933 |   0.825 |                0.6 |              0.988 |               4.179 |               0.566 |               0.519 | 2.53x     | 62.1%     | +0.00pp     | -3.85pp     | -5.17pp      | -6.67pp      | -17.49pp |
| C1_no_prefilter |             128 |          126 |          0.906 |          1 |      0.962 |       0.985 |       1     |   1     |                0   |              1     |               2.792 |               0.711 |               0.59  | 3.48x     | 55.0%     | +0.00pp     | -3.85pp     | -1.54pp      | +0.00pp      | +0.00pp  |
| C2_no_two_stage |             113 |           35 |          0.583 |          1 |      0     |       0.618 |       0.533 |   0.825 |                0.6 |              0.988 |              18.204 |               0.42  |               0.279 | 3.96x     | 87.5%     | +0.00pp     | -100.00pp   | -38.19pp     | -46.67pp     | -17.49pp |

## Table 2: Per-Video Breakdown (C_framework)

| video    | duration   |   prefilter (ms) |   stage1_decode (ms) |   stage1_compute (ms) |   stage2_decode (ms) |   stage2_compute (ms) |   total (ms) |   frames |   S1 frames |   S2 frames |   intervals |
|----------|------------|------------------|----------------------|-----------------------|----------------------|-----------------------|--------------|----------|-------------|-------------|-------------|
| synth_00 | 51.0s      |               73 |                   54 |                     3 |                   57 |                     5 |          192 |      118 |          40 |          78 |           2 |
| synth_01 | 37.6s      |               54 |                   45 |                     2 |                   46 |                     4 |          153 |       89 |          29 |          60 |           2 |
| synth_02 | 54.3s      |               78 |                   65 |                     3 |                   71 |                     6 |          224 |      139 |          45 |          94 |           1 |
| synth_03 | 47.9s      |               71 |                   59 |                     3 |                   66 |                     7 |          205 |      122 |          40 |          82 |           1 |
| synth_04 | 23.8s      |               39 |                   30 |                     2 |                   32 |                     3 |          107 |       63 |          21 |          42 |           1 |

## Table 3: Efficiency-Accuracy Tradeoff

| Strategy        |   Avg Wall (ms) |   Avg Accuracy |   Eff-Acc Score |
|-----------------|-----------------|----------------|-----------------|
| A_independent   |             446 |          1     |           2.242 |
| B_union_fps     |             145 |          0.971 |           6.688 |
| C_framework     |             176 |          0.843 |           4.784 |
| C1_no_prefilter |             128 |          0.906 |           7.063 |
| C2_no_two_stage |             113 |          0.583 |           5.18  |

---

**Legend.** Speedup = Wall_A / Wall_X. Δmetric = metric_X − metric_A (in pp). Eff-Acc = Avg Accuracy / (Avg Wall / 1000s).
