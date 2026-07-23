# Validation — `basic_bayes`
_2026-07-22T10:51:41.952154+00:00_ · git `cea2ae4f` (dirty) · verdict **PASS**

| check | status |
|---|---|
| 0. verify (model-specific) | ✅ PASS |
| 2. fit convergence / spread | ✅ PASS |
| 3. AIC/BIC well-formed | ✅ PASS |
| 4. CV validity | ✅ PASS |
| 5. shape reproduction | ✅ PASS |

## Per-subject fits (12/12 present)
| subj | NLL | AIC | k | start_spread | conv | hit_max | s |
|---|---|---|---|---|---|---|---|
| 1 | 38428.1 | 76874.1 | 9 | 508.3 | False | True | 1028 |
| 2 | 36834.3 | 73686.6 | 9 | 95.4 | True | False | 360 |
| 3 | 41990.7 | 83999.4 | 9 | 381.4 | False | True | 1024 |
| 4 | 23186.5 | 46391.0 | 9 | 93.1 | True | False | 959 |
| 5 | 30981.5 | 61981.1 | 9 | 852.8 | False | True | 1080 |
| 6 | 37610.2 | 75238.3 | 9 | 461.4 | False | True | 996 |
| 7 | 27137.1 | 54292.3 | 9 | 154.9 | True | False | 429 |
| 8 | 31001.9 | 62021.8 | 9 | 74.0 | True | False | 448 |
| 9 | 42312.9 | 84643.7 | 9 | 153.9 | True | False | 521 |
| 10 | 30987.7 | 61993.3 | 9 | 191.1 | True | False | 461 |
| 11 | 31472.9 | 62963.8 | 9 | 314.7 | False | True | 477 |
| 12 | 32804.9 | 65627.8 | 9 | 198.4 | True | False | 474 |