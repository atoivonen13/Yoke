# KN LodeRunner — study log

Dense late-time 9-band forecasting. Metric: **late-time RMSE (mag)**, phase
2 d < t ≤ 10 d, unless noted (studies 095–100 use 2 d < t ≤ 7 d; see 🔻 banner).
Eval via `eval_dense_latetime_9band.py`. **Current champion: study 104, 1.4287
@1000 on 2→10** (backbone ON + tail unfrozen, waist 8, quantile head, delta OFF,
anchor OFF, rollout OFF).

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
| 094 | `19b6fd7` | **1.8733 (1000)** | **DENSE-CONTEXT twin of 093** (PROBE_DENSE_CONTEXT=True), bypass MLP, **waist 32**, trend anchor, quantile head, **BATCH_SIZE=5** | **THE CLEAN DISTILLATION GO/NO-GO.** Identical to 093 in every knob (waist 32, trend anchor, batch 5); context source is the SOLE difference. Result: **Δ = −0.006 @1000 vs 093** (1.8733 vs 1.8792) — dense context is statistically **indistinguishable** from sparse, far inside the noise floor. **DISTILLATION = NO-GO, clean call.** Privileged dense in-window context buys nothing, so a dense-context teacher is no stronger than the sparse student → nothing to transfer. Confirms the late-time 2→10 d signal is genuinely absent from the 2 d window regardless of sampling density — an information limit, not capacity or modeling. This is the correctly-controlled version of the retracted 089–091 claim. |
| 095 | _TBD_ | 1.9168 (1000, 998 obj / 12485 pts) | **SHORTER HORIZON: scored 2→7 d** (TARGET_HORIZON_DAYS 8→5, eval `--late_time_max_days 7`), scheduled sampling **ON** (n_rollout_steps=12), sparse context, bypass MLP, **waist 32**, trend anchor, quantile head, **BATCH_SIZE=5** | ⚠️ **NOT comparable to ≤094** (those scored 2→**10**; this scores 2→**7**, an easier near-region population → different RMSE by construction, see 🔻 banner). Sole purpose was to establish the shorter horizon + serve as the scheduled-sampling-**ON** baseline for the 096 twin. Clean same-region comparison is **095 vs 096 only**. |
| 096 | _TBD_ | **1.7896 (1000, 998 obj / 12485 pts)** / 1.7104 (100, 1218 pts) | **scheduled sampling OFF** (n_rollout_steps 12→**1**), scored 2→7 d, sparse context, bypass MLP, **waist 32**, trend anchor, quantile head, **BATCH_SIZE=5** | **Rollout-off twin of 095 — clean single-variable test.** Everything matched to 095 (region, horizon, context, waist, anchor, batch); only scheduled sampling differs. Result: **Δ = −0.127 @1000 vs 095** (1.7896 vs 1.9168), well outside the ±0.02 @1000 noise → **scheduled sampling was HURTING.** 12-step rollout optimized for an autoregressive task never run at eval (direct single-pass) — paid its cost, never collected its benefit. Cleanest bias signature of the campaign (ztfi −0.08, i +0.04, z +0.21, y −0.06 near-centered; residual offenders are the known blue-band limit: u −1.29, g −0.41, ztfg −0.50, r −0.52). @100→@1000 gap (1.7104→1.7896, +0.079) confirms the eval-size optimism model. **Confirmed a genuine win by the 093-on-2→7 baseline below:** 096 beats the champion 1.8677 by −0.078 on the identical region. Caveat: 096 is a horizon-5 *specialist* (can't forecast 7→10 d); this is a win within the 7-day deployment target, not strict domination of the horizon-8 generalist. |
| 093′ | `19b6fd7` | 1.8677 (1000, 998 obj / 12485 pts) | **093 checkpoint re-scored on 2→7 d** (eval `--late_time_max_days 7`, no retrain), scheduled sampling ON (as-trained), sparse, waist 32, trend anchor, batch 5 | **The owed same-region baseline** — makes 095/096 comparable to the champion. Not a new training run; identical weights to 093, only the scored region shrank 2→10 → 2→7. **Δ vs 093-on-2→10 = −0.012** (1.8677 vs 1.8792): dropping the 7→10 d points barely helps → the 2→7 region is **NOT meaningfully easier**, so 096's low number is not a region artifact. **Δ vs 096 = +0.078** (096 wins): scheduled-sampling-off beats the champion on the matched region — a second, independent confirmation of the 095↔096 twin result. Bias all-positive here (ztfi +0.25, i +0.27, z +0.18, y +0.28, g +0.27) — different minimum than 096's blue-negative signature; same RMSE-class, different failure mode. |
| 097 | _TBD_ | 1.6119 (100, 1218 pts) | **TREND ANCHOR OFF** (rollout-off twin of 096 with anchor removed), scored 2→7 d, sparse, bypass MLP, **waist 32**, quantile head, PREDICT_DELTA=True, **BATCH_SIZE=5** | Anchor-off A/B vs 096 (@100: 1.6119 vs 1.7104 = **−0.099**, ~1.5σ at ±0.068). **Surprise: the analytic trend anchor was a CONSTRAINT, not a crutch** — the head fits late-time fade *better* without being pinned to a per-band local-slope extrapolation. Refutes the "anchor is doing the forecasting" hypothesis. Blue bands went *more* negative without the anchor pinning them (u −1.39, g −1.06) — win redistributes error but nets better. ⚠️ @100 only; not yet @1000. |
| 098 | _TBD_ | 1.7024 (100) @ep40; **2.3400 (100) @ep3** | **POINT HEAD + HUBER** (N_QUANTILES=1, LOSS_TYPE=huber δ=0.1), anchor off, delta on, 2→7 d, sparse, waist 32, batch 5 | **Diagnostic run — reverts the quantile head, expected to regress, and did** (~0.09 worse than 097's quantile head @100; blue biases returned u −1.77, g −1.34, as the 080 finding predicts). **KEY RESULT — resolves the "flat loss" mystery:** the ep3→ep40 eval descended **2.34 → 1.70 mag (−0.64!)**, so training was never dead. The flat training-loss curve was a plotting artifact of three stacked effects: (1) loss is in **normalized z-score units**, eval in mag; (2) **log y-axis** compresses the change; (3) **Huber δ=0.1 caps** the large-residual tail so the objective has a small numerical floor and "starts low." Fix: loss plot default flipped to **linear**. |
| 099 | _TBD_ | 1.6136 (100) | **PREDICT_DELTA OFF** (absolute head) + point/Huber (099 = 098 with delta off), anchor off, 2→7 d, sparse, waist 32, batch 5 | Delta-off A/B under the Huber head: 098→099 = 1.7024→1.6136 = **−0.089**. Mirrors the quantile-head delta-off effect (097→100, −0.088) → **delta-off is a real, head-independent win.** Absolute head fades better because the delta prior pinned forecasts to the bright near-peak last obs (structural under-fade). ⚠️ @100 only. |
| 100 | _TBD_ | **1.4738 (1000, 998 obj / 12485 pts)** / 1.5240 (100) | **NEW CHAMPION (2→7). = 096 with PREDICT_DELTA OFF.** Quantile head, anchor off, rollout off, absolute head, 2→7 d, sparse, bypass MLP, **waist 32**, **BATCH_SIZE=5** | **Biggest single jump in the log.** Δ vs 096 = **−0.316 @1000** (1.4738 vs 1.7896) — delta-off is the SOLE change from 096, so the full −0.316 is attributable to removing the delta persistence prior. Stacks with the quantile>Huber effect (−0.09) and rollout-off (−0.127). **The "aleatoric floor" (081–094) was substantially self-inflicted by the delta anchor**, which pinned forecasts to the bright near-peak last obs → chronic under-fade. Removing it let every band fade properly: cleanest @1000 bias signature ever (ztfg −0.05, y −0.07, r −0.25, z −0.20; even u lifted 2.91→2.49, bias −1.26→−0.94). @100→@1000 went the *right* way (1.5240→1.4738) — atypical, but the win holds massively either way. ⚠️ Still 2→7; NOT comparable to the original 2→10 champion (1.86) — 101 tests that. |
| 101 | _TBD_ | **1.4640 (1000, 17589 pts, 2→10)** / 1.6222 (100) | **★ UNCONDITIONAL CHAMPION (superseded by 104).** HORIZON 8 (2→10) carrying the 100 winners — TARGET_HORIZON_DAYS 5→8, **rollout off, delta off, anchor off, quantile head**, sparse, waist 32, batch 5 | **Beats the original 080 champion (1.8582 @1000, same 2→10 region) by −0.394 mag.** The delta-off + rollout-off wins are NOT artifacts of the easier 2→7 region — they carry to the full horizon. Like 100, @100→@1000 went the *right* way (1.6222→1.4640): the model generalizes to the full set better than the subset. Biases confirm the mechanism at the hard horizon: **u collapsed −1.0/−1.3 → −0.67**, i −0.07, z −0.07, r −0.16. Residual offenders ztfg −0.75 / g −0.64 = the genuine blue-band info limit in the 7→10 d tail. Eval used `--late_time_max_days 10.0` (default restored); plot default `--fixed_forecast_max_days 8`. |
| 102 | _TBD_ | **1.4562 (1000, 17589 pts, 2→10)** | **DENSE-CONTEXT twin of 101** (PROBE_DENSE_CONTEXT=True), else identical to 101 (waist 32, delta/anchor/rollout off, quantile, batch 5) | **Distillation re-test against the NEW floor.** After the floor moved ~0.4 mag (delta/rollout fixes), re-ran the go/no-go: does privileged dense in-window context help now? **Δ = −0.008 @1000 vs 101** (1.4562 vs 1.4640) — again statistically indistinguishable, deep inside the ±0.068 noise. **DISTILLATION STILL NO-GO**, now confirmed at the post-delta floor (was 094: Δ −0.006 at the old floor). The late-time signal is genuinely absent from the 2 d window regardless of sampling density — an information limit that survived the ~0.4 mag improvement. |
| 103 | _TBD_ | **1.4978 (1000, 17589 pts, 2→10)** | **WAIST-8 BYPASS CONTROL.** 101 with BYPASS_CHANNELS 32→**None** (waist 32→8), backbone still OFF, sparse, quantile, delta/anchor/rollout off, batch 5 | **Isolates the pure waist-narrowing cost** so the backbone-on run (104, forced to waist 8) has a fair matched-waist partner. **Δ = +0.034 @1000 vs 101** (1.4978 vs 1.4640) — narrowing the waist 32→8 costs a small but real 0.034 mag (just outside noise). This is the capacity penalty any backbone-on run must first recover before it can claim a net win. Biases mostly negative (u −0.56, g −0.44, r −0.35, z −0.21) — mild under-fade from the tighter waist. **103, not 101, is the correct comparison for 104.** |
| 105 | _TBD_ | _pending_ | **SPATIAL RENDER + INTERPOLATED CONTEXT** (RENDER_INTERPOLATE=True) — draw each band's context as continuous piecewise-linear segments between detections instead of sparse tent dots; backbone ON + tail unfrozen (as 104-ish render mode), waist 8, quantile, delta/anchor/rollout off, batch 5 | **Revives the render line with a denser encoding of the SAME context info** ("use more of the pixels"). Fills the ~107 blank rows between sparse detections in the 2 d context window (verified: a 3-detection band goes 13→224 nonzero rows; singleton bands fall back to the tent splat so no coverage is lost). ⚠️ **Fills only the CONTEXT region; the forecast region stays 0** — this tests whether continuous context helps the backbone *propagate* into the empty readout, NOT the readout gap itself (forecast-row prefill is the deferred next step). Baseline to beat: render was ~2.2–2.9 in 086–088 but that predated the delta/rollout/backbone fixes; the real target is the 104 champion (1.4287). |
| 104 | _TBD_ | **1.4287 (1000, 17589 pts, 2→10)** / 1.4876 (100) | **★ NEW UNCONDITIONAL CHAMPION.** = 103 + **BACKBONE ON, tail unfrozen** (BYPASS_BACKBONE=False, BACKBONE_TAIL_LR_MULT=0.1, scope="tail"), waist 8 (forced), sparse, quantile, delta/anchor/rollout off, batch 5 | **First time in the whole campaign the backbone HELPED.** Two clean reads: (1) **vs the matched control 103** (waist 8, backbone off): Δ = **−0.069 @1000** (1.4287 vs 1.4978) — at identical waist the unfrozen backbone tail adds real signal the bypass MLP cannot. (2) **vs old champion 101** (waist 32, backbone off): Δ = **−0.035** even while paying the +0.034 waist-8 penalty → the backbone contribution (~0.07) more than covers the capacity it gave up. **Overturns the 082/085/086 verdict** that backbone-on merely ties the MLP: that held under *frozen* constant-image mode; the *unfrozen tail* changes the story. Atypically, @1000 (1.4287) came in BELOW @100 (1.4876) — the @100 was pessimistic here. Biases: u −0.86 (still the dominant offender, RMSE 2.40), g −0.50, z −0.42; reds/y tight (r −0.15, i −0.18, y +0.20). Residual blue-band under-fade is where **redshift conditioning** should bite next. |

> ⚠️ **BATCH CONFOUND (studies 089–092).** All four ran at **BATCH_SIZE=2**
> (the reduced regime introduced for 084/085 to fit VRAM), while champion **080
> ran batch 5** (eff. 20). The batch cut alone costs ~0.045 mag (082→085), and the
> full batch-2 + shortened schedule costs ~0.30 here (092 vs 080). **Therefore any
> comparison of an 089–092 number to 080's 1.8318 is invalid** — the gap is mostly
> batch, not the variable under test. Within the batch-2 family the comparisons are
> clean: at matched regime **dense context ≈ or slightly beats sparse** (091 2.0555
> vs 092 2.1289), so the earlier "dense context hurts → distillation dead" call was
> an artifact and is **retracted**. **Resolved by 093+094:** the batch-5 sparse
> reproduction landed 1.8792 (Δ +0.021 vs 080) and its dense twin 094 landed 1.8733
> (Δ −0.006 vs 093) — batch was the whole confound, and at matched regime dense
> context ≈ sparse. Distillation is a clean **no-go** (see 094).

> 🔻 **HORIZON-REGION CHANGE (studies 095+).** Starting at 095 the scored region
> is **2 < phase ≤ 7 d** (was 2 < phase ≤ 10 d for ≤094). The shorter horizon
> scores an **easier, nearer** population, so its RMSE is **lower by construction**
> and is **NOT comparable** to the 080/093 champion (1.88 on 2→10) or any ≤094 row.
> Valid same-region comparisons among these runs are only **095 ↔ 096 ↔ 093′**
> (all 2→7). **Owed baseline — now RESOLVED (093′):** re-scoring the 093 checkpoint
> on 2→7 (no retrain) gave **1.8677**, only −0.012 vs its own 2→10 number → the
> 2→7 region is **not meaningfully easier**, so 096's 1.7896 is a **real** win
> (−0.078 vs the champion on the matched region), not a region artifact. Reference
> frames: eval
> `--late_time_max_days` is **phase from trigger**; train `TARGET_HORIZON_DAYS` is
> **lead from anchor**; phase = lead + CONTEXT_WINDOW_DAYS (2 d), so 7 d phase ↔
> lead 5.

**Eval-size note:** 100-object evals run ~0.03 mag optimistic vs 1000-object
(080: 1.8318 → 1.8582). Quote the **1000-object** number when comparing studies;
the 100-obj bootstrap CI half-width alone is ±0.068.

## Update (studies 095–096): scheduled sampling was hurting

**First lever to move the floor since 080.** Turning scheduled sampling OFF
(n_rollout_steps 12→1) improved late-time RMSE by **−0.078 mag** against the
champion on the matched 2→7 region (096 1.7896 vs 093′ 1.8677), confirmed two
independent ways: the direct 095↔096 twin (−0.127, rollout the only variable) and
096-vs-093′ (−0.078, same region). Root cause: 12-step rollout optimized for an
autoregressive inference path we never run — eval is **DIRECT single-pass**, so the
rollout training was a pure train/eval task mismatch, paying exposure-bias cost with
no matching benefit. The 2→7 region itself is not meaningfully easier (093′ vs 093:
−0.012), so the gain is real, not a region artifact.

**Caveat — specialist vs generalist.** 096 trains to horizon 5 (phase 7), so it is a
7-day *specialist* and cannot forecast 7→10 d; it wins **within the 7-day deployment
target**, not by dominating the horizon-8 generalist. **Next:** retrain rollout-off
at horizon 8 to claim the general champion (strong prior it lands below 093's 1.8792
on 2→10). Note TARGET_HORIZON_DAYS uniform-Δt multi-lead sampling is a SEPARATE
mechanism from rollout and is preserved at n_rollout_steps=1.

## Update (studies 097–100): the delta anchor was the floor — "aleatoric floor" LARGELY OVERTURNED

**The ~1.88 "aleatoric floor" was substantially self-inflicted.** Three stacked
levers, each an independent single-variable A/B, dropped late-time RMSE from the
096 champion (1.7896 @1000, 2→7) to **1.4738 @1000 (study 100)** — a **−0.316 mag**
improvement, the largest jump in the log:

1. **Rollout / scheduled sampling OFF** (095→096, −0.127): train/eval task match.
2. **Quantile head > point+Huber** (~−0.09, confirmed both delta on 096>098 and off
   099>100): the pinball 0.5 term controls blue-band bias; Huber δ=0.1 is an
   objective/metric mismatch (median-like objective, L2 metric).
3. **PREDICT_DELTA OFF** (096→100, −0.316 as the sole change; mirrored head-
   independently by 098→099 −0.089): **the biggest one.** The delta head pinned each
   forecast to the *bright near-peak last observation* in the 2 d window, structurally
   biasing against fading → the chronic under-fade behind every negative bias in
   081–094. The absolute head fades freely; study 100's @1000 biases are the most
   centered ever (ztfg −0.05, y −0.07, even u lifted −1.26→−0.94).

**What this means for the old conclusion below:** the "floor confirmed 5 ways"
narrative was measuring a floor created by the delta parameterization + rollout, not
a true aleatoric limit. Capacity/backbone/render/distillation are still closed (those
findings stand), but the model was **not** at the information limit — it was
mis-parameterized. **Distillation (094 no-go) should arguably be re-tested** now that
the sparse baseline moved ~0.3 mag.

**Flat-loss mystery resolved (098):** training was never dead — the ep3→ep40 eval
descended 2.34→1.70 mag. The flat *training-loss* curve was a plotting artifact:
normalized z-score units + log y-axis + Huber-capped tail. Loss plot default is now
**linear**.

**Caveat:** all of 095–100 are scored **2→7** and are NOT comparable to the original
2→10 champion (1.86). **Study 101** carries the 100 winners to horizon 8 (2→10) to
claim the unconditional general champion — pending.

## ★ CURRENT CHAMPION (study 104): 1.4287 @1000 on 2→10

**Config:** **backbone ON, tail unfrozen** (BYPASS_BACKBONE = False,
BACKBONE_TAIL_LR_MULT = 0.1, BACKBONE_FINETUNE_SCOPE = "tail"), **waist 8** (forced
by backbone-on: BYPASS_CHANNELS = None), **quantile head (3, levels 0.1/0.5/0.9)**,
**PREDICT_DELTA = False** (absolute head), **TREND_DECAY_ANCHOR = False**,
**n_rollout_steps = 1** (scheduled sampling off), TARGET_HORIZON_DAYS = 8, sparse
context, BATCH_SIZE = 5. Eval on 2→10 with `--late_time_max_days 10.0`.

**Result: 1.4287 @1000.** Two clean reads (all @1000, all 2→10):
- **vs matched-waist control 103** (waist 8, backbone OFF): **−0.069** — at
  identical waist the unfrozen backbone tail adds real signal the bypass MLP can't.
- **vs prior champion 101** (waist 32, backbone OFF): **−0.035**, even while paying
  the +0.034 waist-8 capacity penalty (isolated by 103).

**This overturns the 082/085/086 verdict** that turning the backbone on merely ties
the bypass MLP. That held under *frozen constant-image* mode (which forces the
backbone into the MLP's exact job); the **unfrozen tail** breaks the tie. First win
from the backbone in the whole campaign.

**Predecessor context:** the ~1.88 "aleatoric floor" (studies 081–094) was NOT an
information limit — the delta persistence prior + scheduled-sampling mismatch
created it. Studies 096–101 removed both (rollout-off −0.127, quantile>Huber ~−0.09,
delta-off ~−0.32) to reach 1.4640; 104 then adds the backbone lever on top.

**Still open / next:**
- **Blue-band tail** (u −0.86 RMSE 2.40, g −0.50 @2→10) is the dominant remaining
  error — the PRIMARY next lever is the **redshift conditioning-scalar idea**
  (see memory), now that the backbone lever is banked.
- **Distillation is settled NO-GO** at the new floor too (102: Δ −0.008 vs 101).
- **Re-test capacity** (waist/backbone-decoder scope, 084) and **spatial render**
  (088) against the new floor — lower priority than redshift.

## Conclusion (as of study 094 — SUPERSEDED by studies 095–101; see above)

> ⚠️ The framing below is **retained for history but is WRONG.** The "aleatoric
> floor confirmed 5 ways" was measuring a floor created by the delta
> parameterization + rollout, not a true information limit. Study 101 (1.4640)
> broke it by −0.39 mag. The capacity/backbone/render/data findings still stand as
> individual results, but the *conclusion that ~1.88 was irreducible* does not.

Model is at the **aleatoric floor** (~1.88 late-time RMSE at 1000 objects),
confirmed **5 ways**: capacity (waist 32→64 flat, 083), backbone (082 tie), data
(no train/test gap), spatial render (086–088 all worse), and **context density**
(094: dense context = sparse to within −0.006, clean batch-5 test).

**Both lines closed:**
- **Spatial render (086–088): dead.** The gather reads a forecast region that is
  zero by construction (~0.44% of the field non-zero); a mostly-frozen encoder
  can't manufacture signal there. Best rescue (088, tail-unfreeze + trend anchor)
  still lost to plain bypass. Abandoned.
- **Distillation: NO-GO (clean call, study 094).** The 089–091 "no-go" was
  **retracted** as a batch artifact, then re-tested properly: 093 (sparse, batch 5)
  reproduced 080 at 1.8792, and its dense-context twin 094 landed 1.8733 — **Δ
  −0.006, indistinguishable.** Privileged dense context buys nothing, so a
  dense-context teacher is no stronger than the sparse student. The late-time
  signal is genuinely not in the 2 d window regardless of sampling density — an
  information limit. No pipeline to build.

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
- **080 champion (superseded):** `a1166b7` — 075 + quantile head; 1.8582 @1000.
- **101 champion (superseded):** bypass MLP, waist 32, quantile head, delta OFF,
  anchor OFF, rollout OFF; **1.4640 @1000 on 2→10.**
- **104 champion (current):** backbone ON + tail unfrozen (mult 0.1, scope="tail"),
  waist 8, quantile head, delta OFF, anchor OFF, rollout OFF; **1.4287 @1000 on
  2→10.** First backbone-on win; beats matched-waist control 103 (1.4978) by −0.069.
- **081/082 backbone experiments:** `9ef46e8` and later — backbone un-bypassed.
