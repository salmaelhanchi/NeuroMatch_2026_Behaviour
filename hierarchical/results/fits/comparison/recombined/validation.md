# Validation — `recombined`
_2026-07-22T10:51:50.631564+00:00_ · git `cea2ae4f` (dirty) · verdict **PASS**

| check | status |
|---|---|
| 0. verify (model-specific) | ✅ PASS |
| 2. fit convergence / spread | ✅ PASS |
| 3. AIC/BIC well-formed | ✅ PASS |
| 4. CV validity | ⏭️ SKIP |
| 5. shape reproduction | ✅ PASS |

> ⚠️ **start_spread is 0 for every subject** — these fits look single-start (stale or a mis-wired fit path). Refit through `multistart()`.

## Per-subject fits (12/12 present)
| subj | NLL | AIC | k | start_spread | conv | hit_max | s |
|---|---|---|---|---|---|---|---|
| 1 | 39534.8 | 79083.6 | 7 | 0.0 | True | False | 5703 |
| 2 | 36810.0 | 73633.9 | 7 | 0.0 | True | False | 2054 |
| 3 | 43054.8 | 86123.6 | 7 | 0.0 | True | False | 10858 |
| 4 | 23429.7 | 46873.3 | 7 | 0.0 | True | False | 1426 |
| 5 | 31912.8 | 63839.6 | 7 | 0.0 | True | False | 5627 |
| 6 | 38210.9 | 76435.8 | 7 | 0.0 | True | False | 9001 |
| 7 | 27317.9 | 54649.9 | 7 | 0.0 | False | True | 2360 |
| 8 | 31083.5 | 62181.1 | 7 | 0.0 | True | False | 1590 |
| 9 | 42749.0 | 85512.1 | 7 | 0.0 | True | False | 3007 |
| 10 | 31183.2 | 62380.5 | 7 | 0.0 | False | True | 2626 |
| 11 | 31676.7 | 63367.4 | 7 | 0.0 | True | False | 2173 |
| 12 | 33042.6 | 66099.2 | 7 | 0.0 | True | False | 1435 |