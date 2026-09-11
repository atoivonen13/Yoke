# KN LodeRunner — study log

Dense late-time 9-band forecasting. Metric: **late-time RMSE (mag)**, phase
2 d < t ≤ 10 d, unless noted. Eval via `eval_dense_latetime_9band.py`.

**Why this file exists:** `studyIDX` is a launch-time template placeholder
(`<studyIDX>` in `training_input.tmpl`), filled on the cluster and written only to
rundir / CSV filenames — it is **never committed**. So a commit cannot be mapped
to a study number from git alone. This log is the manual bridge: study number →
commit → config → result. Add a row when a run is launched; fill RMSE when it
returns.

Noise floor: run-to-run seed noise ≈ 0.05–0.07 mag; 1000-object bootstrap CI
≈ ±0.068. Treat differences below ~0.07 mag as noise.

| Study | Commit | RMSE | Key config | Notes |
|------:|--------|-----:|------------|-------|
| 065 | _TBD_ | _TBD_ | _TBD_ | Prior baseline (late Aug). Commit not recoverable from git — please confirm. |
| 075 | `e6883db` | 1.8373 | bypass MLP, waist 32, point head, W=2.0/M=12, phase F=10, hidden 128 | Pre-quantile champion. "revert changes back to run 75". |
| 076 | `e6883db`* | 1.9372 | 075 + phase min-period 0.3 d | Regressed; min-period reverted to 1.0 d. |
| 077 | `260c524`* | 2.1137 | 075 + anchor-pinned subsample, W=2/M=12 | Regressed. (Also hit ModuleNotFoundError — new file not copied to cluster.) |
| 078 | `260c524`* | 2.1411 | subsample at wide window W=5/M=24 | Regressed further. Subsample abandoned; reverted to legacy trailing selection (option A). |
| 079 | `a1166b7` | 1.9054 | bypass MLP, waist 32, **point** head (N_QUANTILES=1) | Same commit as 080, point head. |
| 080 | `a1166b7` | **1.8318** | bypass MLP, waist 32, **quantile** head (N_QUANTILES=3, levels 0.1/0.5/0.9) | **Current champion.** Median-pinball collapsed blue-band biases (u −0.80→−0.095, g −0.62→−0.035). Calibrated bands (coverage 0.777). |
| 081 | `9ef46e8`* | 2.0208 | backbone ON (frozen), waist **8** (forced), quantile head | Regressed vs 080. Confound: waist narrowed 32→8 (non-bypass path needs backbone_channels=8). Biases returned (u −1.03, g −1.04) → underfitting. |
| 082 | _this commit_ | _running_ | backbone ON, **decoder tail unfrozen** (PatchExpand[-1] + up_connect[-1] + linear4unpatch) at 0.1× head LR, quantile head, waist 8 | Capacity test. Tail is downstream of the 8-channel input bottleneck; expected to stay near ~2.0 if that bottleneck dominates. |

\* Commit is the nearest committed state matching that study's config; the study
number itself is not in the commit (see note above). Where a study reused an
earlier commit's code with only a config toggle, the same hash is listed.

## Baseline map (quick reference)

- **075 champion (pre-quantile):** `e6883db` — bypass MLP, waist 32, point head.
- **080 champion (current):** `a1166b7` — same as 075 + quantile head.
- **081/082 backbone experiments:** `9ef46e8` and later — backbone un-bypassed.
