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
| 084 | _prev commit_ | 1.9350 (100) | **backbone ON**, waist 8, **frozen encoder + fine-tuned bottleneck+decoder** @ 0.1× LR (scope="decoder", mult=0.1), quantile head, **BATCH_SIZE=2** (eff. 8) | **Regressed** ~0.10 vs 080 (fully trained 40 ep, best.pth). Every band under-fades (larger −biases than 080); worst RMSE g/r 2.48. But CONFOUNDED: batch dropped 5→2 to fit the full decoder's activations in VRAM, while 080/082/083 ran batch 5 (eff. 20) — so scope is tangled with a noisier/smaller-LR regime. 085 controls for it. |
| 085 | _this commit_ | 1.8939 (100) | **backbone ON**, waist 8, **tail unfrozen** (scope="tail", mult=0.1), quantile head, **BATCH_SIZE=2** (eff. 8) | **Batch-2 twin of 082 — the clean control for 084.** Two clean 1-variable reads: (1) SCOPE at fixed batch — 085 (tail) 1.8939 **beats** 084 (decoder) 1.9350 by 0.041 → unfreezing the whole decoder genuinely hurts, not just the batch. (2) BATCH at fixed scope — 082 (tail, batch 5) 1.8492 beats 085 (tail, batch 2) 1.8939 by 0.045 → the batch cut is real & material and stacks. Also: 085 biases far better-centered than 084 (ztfr −0.20 vs −0.63, u −0.19 vs −0.57, r −0.14 vs −0.76) → tail scope generalizes; full decoder drifts bright (under-fades). |
| 086a | _this commit_ | 2.9360 (100) | **SPATIAL RENDER** + backbone **frozen**, waist 8, quantile head | **Interface fix, not a config toggle.** Root cause of the 080-tie: 081–085 tiled the conditioner's [B,8] vector into a spatially-**constant** image and global-pooled the output, forcing the Swin backbone to act as a frozen per-channel MLP (the bypass MLP's exact job → ties by construction). 086 renders the light curve as a 2D field (rows=time, cols=band, bilinear tent splat/event) and **gathers** the prediction at the target's (row(Dt), band-col) instead of global-pooling — finally exercising 2D windowed attention on structured input. Conditioner/output_head dropped for a small `read_head`; output contract [B,Q,9] unchanged (rollout/eval need no change). 086a = frozen backbone (only read_head trains). **Catastrophic (+1.10 vs 080).** Persistence collapse: the gather reads rows 224–1119 (forecast region), which is ZERO by construction — only context rows 0–224 get splatted (~0.44% of the field non-zero). A frozen encoder cannot manufacture signal in the empty readout region. Root failure of the whole render line. |
| 087 | _this commit_ | 2.1899 (100) | **bypass MLP control** (SPATIAL_RENDER=False, BYPASS_BACKBONE=True), waist per config, quantile head | Bypass sanity control run alongside the render experiments (NOT the 080 champion config — differs in other flags). Beats render 086a by ~0.75 but still well above 080, i.e. render is far worse than plain bypass. Confirms the render interface is the problem, not the task. |
| 088 | _this commit_ | 2.4136 (100) | SPATIAL RENDER + **tail unfrozen** @ 0.1× LR (scope="tail") + **TREND_DECAY_ANCHOR**, waist 8, quantile head | Render + partial backbone unfreeze + trend anchor, the best attempt to rescue the render line. Recovered +0.52 vs 086a (2.94→2.41) — tail unfreeze lets the backbone push *some* signal into the forecast rows — but still loses to bypass 087 (2.19) and is ~0.55 above 080. Under-fades rather than collapses. **Render line abandoned after this:** even with unfreeze + anchor, gathering from a structurally-empty region can't beat the bypass MLP. |
| 089 | `cb2bba9` | 2.0153 (100) | **DENSE-CONTEXT probe** (PROBE_DENSE_CONTEXT=True), bypass MLP, **waist 8** (BYPASS_CHANNELS=None), trend anchor, quantile head | Ceiling probe for distillation: draw model context from the DENSE set (all in-window detections, truncated at 2 d — dense-within-window, not leakage), scored region unchanged so RMSE is comparable. **Go/no-go = does privileged dense context beat sparse 080?** CONFOUNDED as a 080 comparison: waist was 8, not 32 (None pins waist to backbone_channels=8). Lands right on 081 (waist 8, sparse) 2.0208 → at fixed waist 8, dense context is ~neutral (2.02→2.02); the gap vs 080 is the waist, not the context. 091 removes the confound. |
| 090 | `cb2bba9` | 1.9616 (100) | dense-context probe, bypass MLP, **waist 8**, **TREND_DECAY_ANCHOR=False** (flat-hold), quantile head | 089 with the anchor flipped off. NOT anchor-matched to 080 (080 uses trend), so it introduces a variable rather than removing one — my initial framing was backwards. Useful only as: (089 vs 090) at fixed dense+waist-8, flat-hold beats trend by ~0.05; and confirms dense context loses to sparse 080 under both anchors. Still waist-8 confounded vs 080. |
| 091 | `cb2bba9` | 2.0555 (100) | **DENSE-CONTEXT probe, CLEAN** — bypass MLP, **waist 32**, **trend anchor** (both matched to 080), quantile head | **The true single-variable test: context source is the ONLY difference from 080 (1.8318 @100).** Dense context **regresses +0.22** — and is the *worst* of the dense runs, worse than the waist-8 dense runs (089 2.0153, 090 1.9616). So at full capacity dense context actively HURTS, and hurts more, not less. Deepest blue under-fade yet (ztfg −1.06, u −0.75, i −0.71, r −0.63). **Definitive no-go for distillation:** a teacher with privileged dense context forecasts *worse* than the sparse student — nothing to transfer. Late-time info genuinely isn't in the 2 d window regardless of sampling density. |

**Eval-size note:** 100-object evals run ~0.03 mag optimistic vs 1000-object
(080: 1.8318 → 1.8582). Quote the **1000-object** number when comparing studies;
the 100-obj bootstrap CI half-width alone is ±0.068.

## Conclusion (as of study 091)

Model is at the **aleatoric floor** (~1.86 late-time RMSE at 1000 objects),
confirmed **6 ways**: capacity (waist 32→64 flat, 083), backbone (082 tie), data
(no train/test gap), spatial render (086–088 all worse), and now
**context density** (091: dense context regresses +0.22 vs sparse 080, and is
worst at full waist). The last is the most direct: enriching the *input itself*
doesn't help, so the late-time 2→10 d signal is genuinely not present in the 2 d
context window regardless of sampling density.

**Two lines closed:**
- **Spatial render (086–088): dead.** The gather reads a forecast region that is
  zero by construction (~0.44% of the field non-zero); a mostly-frozen encoder
  can't manufacture signal there. Best rescue (088, tail-unfreeze + trend anchor)
  still lost to plain bypass. Abandoned.
- **Distillation: ruled out empirically (091).** A dense-context teacher forecasts
  *worse* than the sparse student — no privileged signal to transfer.

**Champion: study 080** — bypass MLP, waist 32, quantile head, trend anchor
(1.8582 @1000). Remaining signal is NOT capacity, architecture, or context
density: it's (a) per-band systematic bias (ztfg/u still −0.8 to −0.9) and
(b) context CONTENT (photometric errors, per-band recency), not anything we can
reach by feeding more or rendering differently.

\* Commit is the nearest committed state matching that study's config; the study
number itself is not in the commit (see note above). Where a study reused an
earlier commit's code with only a config toggle, the same hash is listed.

## Baseline map (quick reference)

- **075 champion (pre-quantile):** `e6883db` — bypass MLP, waist 32, point head.
- **080 champion (current):** `a1166b7` — same as 075 + quantile head.
- **081/082 backbone experiments:** `9ef46e8` and later — backbone un-bypassed.
