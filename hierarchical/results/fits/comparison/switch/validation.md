# Validation — `switch`
_2026-07-22T20:08:34.737300+00:00_ · git `c2e094d6` (dirty) · verdict **FAIL**

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
| 1 | 38526.8 | 77071.5 | 9 | 327.7 | False | True | 628 |
| 2 | 36254.2 | 72526.4 | 9 | 4.4 | False | True | 609 |
| 3 | 41728.8 | 83475.5 | 9 | 330.1 | False | True | 672 |
| 4 | 23074.9 | 46167.7 | 9 | 23.5 | False | True | 447 |
| 5 | 30864.8 | 61747.7 | 9 | 514.0 | False | True | 494 |
| 6 | 37569.4 | 75156.7 | 9 | 252.3 | False | True | 591 |
| 7 | 27088.8 | 54195.6 | 9 | 109.0 | False | True | 508 |
| 8 | 30947.4 | 61912.8 | 9 | 15.0 | False | True | 516 |
| 9 | 42232.3 | 84482.7 | 9 | 136.1 | False | True | 623 |
| 10 | 30887.2 | 61792.4 | 9 | 210.8 | False | True | 504 |
| 11 | 31296.8 | 62611.7 | 9 | 38.8 | False | True | 520 |
| 12 | 32580.8 | 65179.7 | 9 | 86.4 | False | True | 518 |