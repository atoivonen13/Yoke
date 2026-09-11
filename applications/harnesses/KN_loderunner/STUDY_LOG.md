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
| 080 | `a1166b7` | **1.8582** (1000) / 1.8318 (100) | bypass MLP, waist 32, **quantile** head (N_QUANTILES=3, levels 0.1/0.5/0.9) | **Champion.** Median-pinball collapsed blue-band biases vs the point head. Calibrated bands (coverage 0.777). 100-obj number was optimistically biased (+0.026 vs 1000). |
| 081 | `9ef46e8`* | 2.0208 (100) | backbone ON (frozen), waist **8** (forced), quantile head | Regressed vs 080. Confound: waist narrowed 32→8 (non-bypass path needs backbone_channels=8). Biases returned (u −1.03, g −1.04) → underfitting. |
| 082 | _prev commit_ | 1.8492 (100) | backbone ON, **decoder tail unfrozen** (PatchExpand[-1] + up_connect[-1] + linear4unpatch) at 0.1× head LR, quantile head, waist 8 | Capacity test. Unfreezing the tail recovered **0.17 mag vs 081** → backbone path was capacity-limited, not dead weight. But only **ties** 080 at higher cost + waist-8 bottleneck. No reason to adopt. |
| 083 | _prev commit_ | 1.8900 (1000) / 1.8576 (100) | **bypass MLP, waist 64** (BYPASS_CHANNELS=64), quantile head, W=2.0/M=12 | Waist-width capacity test. **Ties** 080 at 1000 obj (Δ0.032, within noise), if anything slightly worse → waist saturates by 32; 64 is wasted capacity. Confirms the **aleatoric floor**. **080 (waist 32) stays champion** — and is the efficient one. |
| 084 | _this commit_ | _TBD_ | **backbone ON**, waist 8, **frozen encoder + fine-tuned bottleneck+decoder** @ 0.1× LR (BACKBONE_FINETUNE_SCOPE="decoder", BACKBONE_TAIL_LR_MULT=0.1), quantile head | **Goal shift: shared backbone for multiphysics reuse.** Not chasing single-task RMSE — testing whether a frozen shared encoder + a small per-task decoder is a strong reusable config. 082 only tuned the thin tail (bottleneck + up_stage1's 2 blocks stayed frozen); 084 fine-tunes the whole decoder. If it beats the 080/082 tie → adopt freeze-encoder/tune-decoder as the multiphysics template. If it ties → decoder capacity isn't the limiter; recipe validated as *sufficient*, reuse for efficiency. |

**Eval-size note:** 100-object evals run ~0.03 mag optimistic vs 1000-object
(080: 1.8318 → 1.8582). Quote the **1000-object** number when comparing studies;
the 100-obj bootstrap CI half-width alone is ±0.068.

## Conclusion (as of study 083)

Model is at the **aleatoric floor** (~1.86 late-time RMSE at 1000 objects),
confirmed 3 ways: capacity (waist 32→64 flat), backbone (082 tie), and data
(no train/test gap).
**Champion: study 080** — bypass MLP, waist 32, quantile head. Remaining signal is
NOT capacity: it's (a) per-band systematic bias (ztfg/u still −0.8 to −0.9) and
(b) context CONTENT (photometric errors, per-band recency), not model size.

\* Commit is the nearest committed state matching that study's config; the study
number itself is not in the commit (see note above). Where a study reused an
earlier commit's code with only a config toggle, the same hash is listed.

## Baseline map (quick reference)

- **075 champion (pre-quantile):** `e6883db` — bypass MLP, waist 32, point head.
- **080 champion (current):** `a1166b7` — same as 075 + quantile head.
- **081/082 backbone experiments:** `9ef46e8` and later — backbone un-bypassed.
