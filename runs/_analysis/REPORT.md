# cellfluxv2 — Stage 1 / Stage 2 wandb readout (2026-05-13)

Three wandb runs pulled from the cloud:

| Tag | run | name | steps | bs | lr | source_noise | balance_cond | scales (t, drug) |
|---|---|---|---|---|---|---|---|---|
| BASELINE | `cellfluxv2-stage1/hyeekygn` | lyric-lake-5 | 2000 | 512 | 2e-4 | 0.0 | **false** | n/a |
| BALANCED | `cellfluxv2-stage1/xhl7cib2`  | valiant-resonance-6 | 1000 | 128 | 2e-4 | 0.0 | **true** | 1.0 / 1.0 |
| STAGE2   | `cellfluxv2-stage2/95qevp4r`  | autumn-cosmos-1 | 2000 | 128 | 1e-4 | 0.1 | true (loaded ckpt) | 1.0 / 1.0 |

All on the RTX PRO 6000, same commit `5581968`, same metadata + latent + fingerprint sources.

Plots live in `plots/` next to this file.

---

## Useful charts (and what they actually say)

### 1. `02_mse_gap.png` — the headline diagnostic
`MSE(shuffled cond) − MSE(real cond)`. Positive ⇒ the model uses the real drug; near zero ⇒ collapse.

| Run | step 0 | end |
|---|---|---|
| BASELINE | 0.0003 | 0.0038 (slowly creeps) |
| BALANCED | 0.0044 | 0.0009 (**regresses**) |
| STAGE2   | 0.0085 | 0.014–0.022 (peaks step 1400) |

**Reading**: Fix-1's magnitude rebalance does NOT by itself make the model use the drug — in stage 1 alone the gap actually *decays* over training. Stage 2 (source-noise + warm start) is where the gap finally widens. Drug conditioning becomes load-bearing only once the trivial `v ≈ x1 − x0` identity is broken.

### 2. `03_drug_swap_cos.png` — `cos(v_real_drug, v_swapped_drug)`
1.0 = drug irrelevant, lower = drug shifts the velocity field.

| Run | step 0 | end |
|---|---|---|
| BASELINE | 0.9999 | 0.9986 (drug essentially ignored) |
| BALANCED | 0.9805 → **0.999** | model *forgets* drug during stage 1 |
| STAGE2   | 0.998 → ~0.995 (stable) | small but persistent drug signal |

Same story as the MSE gap: balanced stage 1 *unlearns* drug; stage 2 holds onto ~0.5 pp of cosine separation. Still nowhere near the memo's "drug-swap cos median ≤ ~0.95" target.

### 3. `04_cond_rms.png` — the magnitude fix is working
- Left panel: post-norm `condition_embedding_rms` is pinned at 1.0 for BALANCED/STAGE2 (RMSNorm doing its job), vs **0.05** for baseline (the 20× starvation predicted by the memo).
- Right panel: raw drug-RMS (solid) and raw time-RMS (dashed) now track each other ~1:1 in BALANCED/STAGE2 (~0.20–0.34). Pre-Fix-1 ratio was ~1:5 per the memo.

**Reading**: Fix-1 fixed the magnitude imbalance cleanly. The remaining collapse is *not* magnitude — the model is choosing to ignore the (now comparably-loud) drug signal.

### 4. `01_loss.png` — loss is misleading here
Stage 2 has the lowest absolute loss because lr=1e-4 + warm init + source noise removes the early loud transient. Don't read "stage 2 is best" off this — it's not training-from-scratch comparable to the others. The BALANCED loss plateau also looks high but that's a 4× smaller batch.

### 5. `05_v_pred_vs_target.png` — flow-matching fit is healthy
All three converge `v_pred_rms` to 1.3–1.8 against `v_target_rms ≈ 2.65`. Under-prediction is normal mid-training; nothing alarming.

### 6. `06_grad_norm.png` — optimization is stable
All runs stay 1–5, well below clip=1.0 → no explosions, no dead grads.

### `00_dashboard.png`
Six-panel overview combining the above. Good single-page artifact for sharing.

---

## What this tells us → what to do next

**Solid:**
- Fix-1's magnitude rebalance works exactly as designed (cond RMS 0.05 → 1.0, raw drug↔time RMS now commensurate).
- Stage 2 with source_noise=0.1 is where the drug conditioning **first becomes load-bearing** (mse_gap 10× wider than stage 1, drug-swap cos drops by ~0.5 pp).
- No optimization pathology anywhere (grad_norm, v_pred_rms, t_mean all healthy).

**Not solid:**
- Stage 1 alone, even with Fix-1, **regresses** on both diagnostics over 1k steps. Running this 20k will not save it.
- Stage 2 smoke is the right *direction* but the magnitudes are still small: drug-swap cos ≥ 0.994 means real and shuffled drugs give nearly the same velocity. The memo's stage-1 success criterion (`Δloss ≥ +0.02`, `Δgate ≥ ~30%`) is not met by these proxies.

**Confounds before any next big run:**
- BASELINE used bs=512, the other two bs=128. Don't read absolute losses across them.
- Memo prescribes `cond = 0.5·(RMSNorm(t) + RMSNorm(drug))` → both scales = 0.5. This run uses scale=1 for both, i.e. **2× the prescribed magnitude**. That alone could be why the gate-effect is below criterion.
- Drug-swap cos at step 0 of BALANCED is already 0.98 — that's after the very first weight update, so it likely reflects FP init noise more than real conditioning. The meaningful comparison is step 100 onward.

**Recommended next experiments, ranked:**

1. **One-variable smoke at the memo's prescribed scale** — same config as BALANCED, but `time_scale=0.5, condition_scale=0.5`. 1–2k steps. If `mse_gap` no longer decays during stage 1, Fix-1 is actually validated. (Cheapest experiment, highest information.)

2. **Apples-to-apples baseline rerun** — pre-Fix-1 config but with bs=128/1k steps to match BALANCED, so loss numbers and gap trajectories are directly comparable.

3. **Image-space conditioning lift** — pull the STAGE2 final.pt, generate samples with real drug vs shuffled drug vs null on a 32-compound test set, report image cosine. The MSE/swap-cos diagnostics are proxies; the question we actually care about is "does the generated image look like the drug we asked for?".

4. **Only after #1 and #3 are green:** longer stage-2 (10–20k) from the best balanced ckpt. Don't burn the long stage-2 budget on a checkpoint whose lift hasn't been image-verified.

**Hard no:**
- Do NOT start a long pre-Fix-1 stage 1 — magnitude bug confirmed in baseline (cond RMS 0.05).
- Do NOT promote BALANCED final.pt to a 20k stage-2 on the basis of the smoke alone. The smoke says "promising direction", not "fix validated".
