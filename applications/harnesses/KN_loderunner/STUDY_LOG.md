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
| 089 | `cb2bba9` | 2.0153 (100) | **DENSE-CONTEXT probe** (PROBE_DENSE_CONTEXT=True), bypass MLP, **waist 8** (BYPASS_CHANNELS=None), trend anchor, quantile head, **BATCH_SIZE=2** | Ceiling probe for distillation: draw model context from the DENSE set (all in-window detections, truncated at 2 d — dense-within-window, not leakage), scored region unchanged so RMSE is comparable. **Go/no-go = does privileged dense context help?** Two confounds vs the 080 champion (see ⚠️ banner below): (1) waist 8 not 32; (2) **batch 2 not 5**. Compare only within the batch-2 family (090/091/092). |
| 090 | `cb2bba9` | 1.9616 (100) | dense-context probe, bypass MLP, **waist 8**, **TREND_DECAY_ANCHOR=False** (flat-hold), quantile head, **BATCH_SIZE=2** | 089 with the anchor flipped off. Within-regime read: (089 vs 090) at fixed dense+waist-8+batch-2, flat-hold beats trend by ~0.05. NOT comparable to 080 (batch + waist + anchor all differ). |
| 091 | `cb2bba9` | 2.0555 (100) | **DENSE-CONTEXT probe** — bypass MLP, **waist 32**, **trend anchor**, quantile head, **BATCH_SIZE=2** | Intended as the clean single-variable test vs 080, but **it is NOT** — 080 ran batch 5, this ran batch 2 (see ⚠️). ~~"Definitive no-go, dense regresses +0.22 vs 080"~~ **RETRACTED:** that compared batch-2 dense against batch-5 sparse; the offset was mostly the batch cut, present in every batch-2 run. The valid comparison is same-regime: **091 (dense, batch 2) 2.0555 vs 092 (sparse, batch 2) 2.1289 → dense slightly BEATS sparse by ~0.07** (within the ±0.068 @100 noise). Dense context is NOT harmful; distillation is NOT ruled out. Needs a batch-5 rerun (093+) to settle. |
| 092 | `19b6fd7` | 2.1289 (100) | sparse-context reproduction control, bypass MLP, **waist 32**, trend anchor, quantile head, **BATCH_SIZE=2** | Meant to reproduce 080 (~1.83). **Failed to — landed 2.1289 (+0.30).** Diagnosis: code path is byte-identical to 080 (dataset/epoch/eval/optimizer all verified unchanged); the drift is the **launch regime** — the active CSV row runs BATCH_SIZE=2, but 080 ran batch 5 (eff. 20). STUDY_LOG already records the batch 5→2 penalty as ~0.045 (082→085); here the full batch-2 + schedule effect is ~0.30. So 092 is "080's architecture, batch-2 regime," not a reproduction. **This run is what exposed the batch confound in 089–091.** Reproduction retried at batch 5 as 093. |
| 093 | `19b6fd7` | **1.8792 (1000)** / 1.9121 (100) | sparse-context reproduction of 080, bypass MLP, **waist 32**, trend anchor, quantile head, **BATCH_SIZE=5** | **✅ CLEAN REPRODUCTION of 080** (1.8582 @1000). Δ = **+0.021 @1000**, within noise. Confirms (1) no harness drift — current code + moved data + optimizer-builder refactor all reproduce the champion; (2) the batch confound was the whole 092 story (batch 2→5 recovered +0.22 @100); (3) the @100 residual vs 080 was eval-size optimism (1.9121@100 → 1.8792@1000). **Re-anchors the batch-5 reference regime.** One difference: bias signature flipped — several bands positive (u +0.14, g +0.12, ztfi +0.14) vs 080's persistent negative under-fade. Same RMSE, different minimum; the blue under-fade is largely gone here. **093 is now the sparse batch-5 baseline for the distillation go/no-go** (needs a dense batch-5 twin to compare, evaluated with --probe_dense_context). |

> ⚠️ **BATCH CONFOUND (studies 089–092).** All four ran at **BATCH_SIZE=2**
> (the reduced regime introduced for 084/085 to fit VRAM), while champion **080
> ran batch 5** (eff. 20). The batch cut alone costs ~0.045 mag (082→085), and the
> full batch-2 + shortened schedule costs ~0.30 here (092 vs 080). **Therefore any
> comparison of an 089–092 number to 080's 1.8318 is invalid** — the gap is mostly
> batch, not the variable under test. Within the batch-2 family the comparisons are
> clean: at matched regime **dense context ≈ or slightly beats sparse** (091 2.0555
> vs 092 2.1289), so the earlier "dense context hurts → distillation dead" call was
> an artifact and is **retracted**. **Resolved by 093:** the batch-5 sparse
> reproduction landed 1.8792 @1000 (Δ +0.021 vs 080) — clean, no drift. Batch was
> the whole confound. The distillation go/no-go now needs a **dense batch-5 twin**
> compared to 093 (1.8792), not 080.

**Eval-size note:** 100-object evals run ~0.03 mag optimistic vs 1000-object
(080: 1.8318 → 1.8582). Quote the **1000-object** number when comparing studies;
the 100-obj bootstrap CI half-width alone is ±0.068.

## Conclusion (as of study 092)

Model is at the **aleatoric floor** (~1.86 late-time RMSE at 1000 objects),
confirmed **4 ways**: capacity (waist 32→64 flat, 083), backbone (082 tie), data
(no train/test gap), and spatial render (086–088 all worse).

**One line closed, one still open:**
- **Spatial render (086–088): dead.** The gather reads a forecast region that is
  zero by construction (~0.44% of the field non-zero); a mostly-frozen encoder
  can't manufacture signal there. Best rescue (088, tail-unfreeze + trend anchor)
  still lost to plain bypass. Abandoned.
- **Distillation: STILL OPEN.** The 089–091 "no-go" was **retracted** — it rested
  on comparing batch-2 dense runs against batch-5 sparse 080 (see ⚠️ banner). **093
  (sparse, batch 5) reproduced 080 cleanly** (1.8792 @1000, Δ +0.021), confirming
  the batch confound was the whole story and re-anchoring the batch-5 regime.
  Whether a dense-context teacher genuinely beats the sparse student is still
  **undetermined**; the actual go/no-go is a **dense batch-5 twin** compared to
  093 (1.8792), evaluated with `--probe_dense_context`.

**Champion: study 080** — bypass MLP, waist 32, quantile head, trend anchor
(1.8582 @1000), reproduced by **093** (1.8792 @1000) in the current code. Its
remaining error is likely (a) per-band systematic bias and (b) context CONTENT
(photometric errors, per-band recency). **Batch-5 is the reference regime** (084–092
were all batch-2 and are not directly comparable to it); 093 re-anchors it and is
the sparse baseline all future dense/distillation runs must be compared against.

\* Commit is the nearest committed state matching that study's config; the study
number itself is not in the commit (see note above). Where a study reused an
earlier commit's code with only a config toggle, the same hash is listed.

## Baseline map (quick reference)

- **075 champion (pre-quantile):** `e6883db` — bypass MLP, waist 32, point head.
- **080 champion (current):** `a1166b7` — same as 075 + quantile head.
- **081/082 backbone experiments:** `9ef46e8` and later — backbone un-bypassed.
