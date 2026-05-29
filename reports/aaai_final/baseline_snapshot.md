# Final Eval Baseline Snapshot

```json
{
  "seeds": [
    42,
    123,
    456
  ],
  "epochs": 200,
  "noise_default": "medium",
  "delta": 0.1,
  "soft_ratio": 0.6,
  "split": "50/10/10/30 (train/cal/dev/test)",
  "route_bank": "configs/route_bank_30.yaml",
  "WORST_RANK": 1001,
  "cache": "E:\\Work\\HKUST(2025)\\video_query\\video_retrieval_code_no_dataset\\data\\cache\\msrvtt_cache.npz",
  "csv": "E:\\Work\\HKUST(2025)\\video_query\\video_retrieval_code_no_dataset\\data\\msrvtt_test_1k.csv",
  "timestamp": "2026-05-10 20:08"
}
```

## Key Numbers
- B0 (semantic): R@1 = 31.7%
- B10 (C-QIN cascade): R@1 = 46.2%
- B4 (oracle): R@1 = 56.8%
- Oracle gap closed: 58.0%
- Seeds: [42, 123, 456]
- Test n: 300
