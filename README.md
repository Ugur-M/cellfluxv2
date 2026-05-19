# cellfluxv2_repo

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

Build is gated by tests and smoke scripts. Each step adds one slice and
keeps the suite green before the next one lands.

- **Step 1** — Morgan fingerprints (precompute, dedupe report,
  ``use_chirality`` metadata).
- **Step 2** — metadata split + strict same-(experiment, plate) pair
  index, no fallback pairing.
- **Step 3** — per-channel latent norm stats + plate cache with
  ``address_to_rows`` and finite validation.
- **Step 4** — ``CellFluxDataset`` (stage 1: Gaussian noise → treated
  latent; stage 2: same-plate control → treated latent). Strict on
  missing addresses; no ``skip_on_missing`` fallback.
- **Step 5** — pure-torch rectified + noisy flow paths
  (``x_t = (1-t)x0 + t·x1 + sin²(πt)·σ·ε``), separate source-noise
  augmentation.
- **Step 6** — ``DiTVelocity`` velocity model: 169-token DiT with
  adaLN-Zero, time and condition embeddings.
- **Step 7** — Euler sampler + classifier-free-guidance utilities
  (``v_cfg = α·v_cond + (1-α)·v_uncond``).
- **Step 8** — single-call ``train_step`` + latent diagnostics suite
  (drug-swap velocity cosine, real-vs-shuffled MSE gap, embedding RMS).
- **Step 9 / 10** — Stage 1 training-readiness plumbing and runner:
  epoch-wise dataset resampling, missing-address split filter,
  checkpoint save/load, JSONL logging, seed util, ``configs/stage1.yaml``,
  ``cellfluxv2.train.stage1`` entry-point, plus a tiny end-to-end smoke
  on real rxrx3 latents.

### Running the suite

```
pytest                                  # full unit-test suite
python scripts/smoke_synthetic_overfit.py  # CPU overfit gate (no GCS)
python scripts/smoke_stage1_tiny.py         # 5-step Stage 1 on real data
```

### Stage 1 GPU smoke (RTX PRO 6000)

```
python -m cellfluxv2.train.stage1 \
  --config configs/stage1.yaml \
  --device cuda \
  --batch-size 128 \
  --num-workers 4 \
  --max-steps 2000 \
  --output-dir runs/stage1_gpu_smoke
```

The relative data paths in ``configs/stage1.yaml`` resolve from the
studio root (where ``2DGen/`` and ``cellfluxv2_repro/`` are siblings),
so run that command from ``/teamspace/studios/this_studio/`` with
``cellfluxv2_repro/src`` on ``PYTHONPATH`` (or in editable-install
mode). ``runs/`` is gitignored; checkpoints, ``train.jsonl``, and
``config.json`` will land in ``runs/stage1_gpu_smoke/``.

W&B is wired through the optional ``wandb:`` block in
``configs/stage1.yaml`` (project ``cellfluxv2-stage1`` by default). The
JSONL sink is always authoritative; wandb is a parallel sink that
degrades to a no-op on import / login / network failures. Toggle from
the CLI:

```
--no-wandb                  # force-disable for this run
--wandb-project NAME
--wandb-run-name NAME
--wandb-mode {online,offline,disabled}
```
