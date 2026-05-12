# cellfluxv2_repro

A clean **CellFluxV2-style reproduction baseline** trained on **RxRx3 MAE
latents of shape `(169, 8)`**.

> This is **not** an exact reproduction of the CellFluxV2 paper's VAE-latent
> setup. The latent representation is provided by the project's own MAE
> encoder over rxrx3-core (frozen, precomputed to GCS), so the velocity
> model, normalization, and pairing logic are matched to that latent
> geometry rather than to the paper's VAE.

## What is reproduced

- Distribution-to-distribution flow matching in latent space.
- Two-stage training:
  - **Stage 1**: Gaussian noise → treated latent.
  - **Stage 2**: same-batch control latent → treated latent.
- Noisy interpolant: `x_t = (1-t)x0 + t·x1 + sin²(πt)·ε`, with
  velocity target `v = x1 - x0 + π·sin(2πt)·ε`.
- Classifier-free guidance via condition dropout, with the paper-style
  blend `v_cfg = α·v_cond + (1-α)·v_uncond` (`α = 1.0` means no guidance).
- Chemical condition: **Morgan/ECFP fingerprints** (radius=2, n_bits=1024
  by default) — **not** MoLFormer, not drug token sequences.
- Simple DiT velocity model conditioned via adaLN-Zero on
  `time_emb + cond_emb`.

## What is **not** in scope (v0)

- MMDiT, MoLFormer, drug token sequences, contrastive losses.
- FID / MoA evaluation. The first eval gate is latent-space diagnostics
  (drug-swap cosine, condition vs. shuffled-condition MSE gap, embedding
  RMS, gradient norms).

## Status

Build is gated by smoke tests. Current progress: **step 1 — fingerprints**.
See the plan in conversation for the full sequenced build order.
