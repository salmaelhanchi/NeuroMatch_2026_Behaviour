# Validation — `switch`
_2026-07-22T10:51:38.394277+00:00_ · git `cea2ae4f` (dirty) · verdict **FAIL**

| check | status |
|---|---|
| 0. verify (model-specific) | ✅ PASS |
| 2. fit convergence / spread | ✅ PASS |
| 3. AIC/BIC well-formed | ✅ PASS |
| 4. CV validity | ❌ FAIL |
| 5. shape reproduction | ✅ PASS |

## Per-subject fits (12/12 present)
| subj | NLL | AIC | k | start_spread | conv | hit_max | s |
|---|---|---|---|---|---|---|---|
| 1 | 38529.1 | 77076.2 | 9 | 340.6 | False | True | 1169 |
| 2 | 36254.2 | 72526.5 | 9 | 11.3 | True | False | 340 |
| 3 | 41738.7 | 83495.5 | 9 | 330.9 | False | True | 1229 |
| 4 | 23076.0 | 46169.9 | 9 | 51.1 | False | True | 710 |
| 5 | 30981.5 | 61980.9 | 9 | 676.8 | False | True | 699 |
| 6 | 37585.5 | 75188.9 | 9 | 438.1 | False | True | 800 |
| 7 | 27092.2 | 54202.4 | 9 | 131.1 | False | True | 282 |
| 8 | 30950.2 | 61918.3 | 9 | 19.8 | False | True | 326 |
| 9 | 42237.9 | 84493.7 | 9 | 193.0 | False | True | 416 |
| 10 | 30889.5 | 61797.1 | 9 | 235.7 | False | True | 329 |
| 11 | 31297.2 | 62612.3 | 9 | 104.7 | True | False | 336 |
| 12 | 32580.9 | 65179.7 | 9 | 93.6 | False | True | 334 |