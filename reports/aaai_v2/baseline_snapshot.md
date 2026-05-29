# Baseline Snapshot

```json
{
  "date": "2026-05-09 23:38",
  "seed": 42,
  "split": {
    "train": 600,
    "cal": 150,
    "test": 250
  },
  "noise_config": {
    "time_shift_std": 7.0,
    "geo_missing": 0.3,
    "geo_wrong": 0.1
  },
  "calibration_delta": 0.1,
  "soft_ratio": 0.6,
  "route_bank_size": 30,
  "epochs": 200,
  "model_params": 78562,
  "cache_path": "E:\\Work\\HKUST(2025)\\video_query\\video_retrieval_code_no_dataset\\data\\cache\\msrvtt_cache.npz",
  "csv_path": "E:\\Work\\HKUST(2025)\\video_query\\video_retrieval_code_no_dataset\\data\\msrvtt_test_1k.csv"
}
```

## Results

- B0_semantic_only: R@1=31.6% MeanR=36.6 GT_filt=0.0%
- B1_rule_parser: R@1=39.2% MeanR=14.8 GT_filt=14.0%
- B2_qpp_only: R@1=32.8% MeanR=34.2 GT_filt=0.0%
- B4_oracle: R@1=56.0% MeanR=14.6 GT_filt=0.0%
- B5_always_hard_all: R@1=39.2% MeanR=14.8 GT_filt=14.0%
- B7_cqin_calibrated_v1: R@1=45.6% MeanR=24.5 GT_filt=0.0%
- B8_cascade: R@1=44.0% MeanR=18.6 GT_filt=0.0%
- B9_cqin_soft_fallback: R@1=45.6% MeanR=24.5 GT_filt=0.0%
- B10_cqin_budgeted_cascade: R@1=46.0% MeanR=18.4 GT_filt=0.0%

## Delta Sweep

```json
{
  "delta=0.00": {
    "time": {
      "tau": 1.0,
      "enabled": false,
      "ucb": 1.0
    },
    "geo": {
      "tau": 1.0,
      "enabled": false,
      "ucb": 1.0
    },
    "motion": {
      "tau": 1.0,
      "enabled": false,
      "ucb": 1.0
    },
    "device": {
      "tau": 1.0,
      "enabled": false,
      "ucb": 1.0
    }
  },
  "delta=0.01": {
    "time": {
      "tau": 1.0,
      "enabled": false,
      "ucb": 1.0
    },
    "geo": {
      "tau": 1.0,
      "enabled": false,
      "ucb": 1.0
    },
    "motion": {
      "tau": 1.0,
      "enabled": false,
      "ucb": 1.0
    },
    "device": {
      "tau": 1.0,
      "enabled": false,
      "ucb": 1.0
    }
  },
  "delta=0.03": {
    "time": {
      "tau": 1.0,
      "enabled": false,
      "ucb": 1.0
    },
    "geo": {
      "tau": 1.0,
      "enabled": false,
      "ucb": 1.0
    },
    "motion": {
      "tau": 1.0,
      "enabled": false,
      "ucb": 1.0
    },
    "device": {
      "tau": 0.985518217086792,
      "enabled": true,
      "ucb": 0.01977343816450844
    }
  },
  "delta=0.05": {
    "time": {
      "tau": 0.970183253288269,
      "enabled": true,
      "ucb": 0.04566359632764001
    },
    "geo": {
      "tau": 0.9399389028549194,
      "enabled": true,
      "ucb": 0.04437517261818429
    },
    "motion": {
      "tau": 1.0,
      "enabled": false,
      "ucb": 1.0
    },
    "device": {
      "tau": 0.985518217086792,
      "enabled": true,
      "ucb": 0.01977343816450844
    }
  },
  "delta=0.10": {
    "time": {
      "tau": 0.9067564606666565,
      "enabled": true,
      "ucb": 0.050877067978523885
    },
    "geo": {
      "tau": 0.9054273962974548,
      "enabled": true,
      "ucb": 0.09402694403881356
    },
    "motion": {
      "tau": 0.8378589153289795,
      "enabled": true,
      "ucb": 0.095183420660454
    },
    "device": {
      "tau": 0.985518217086792,
      "enabled": true,
      "ucb": 0.01977343816450844
    }
  },
  "delta=0.15": {
    "time": {
      "tau": 0.9067564606666565,
      "enabled": true,
      "ucb": 0.050877067978523885
    },
    "geo": {
      "tau": 0.833267092704773,
      "enabled": true,
      "ucb": 0.13425700799112406
    },
    "motion": {
      "tau": 0.7583419680595398,
      "enabled": true,
      "ucb": 0.11045753184423883
    },
    "device": {
      "tau": 0.985518217086792,
      "enabled": true,
      "ucb": 0.01977343816450844
    }
  }
}
```
