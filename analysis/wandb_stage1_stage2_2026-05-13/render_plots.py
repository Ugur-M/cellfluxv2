"""Render the useful curves for the three runs into PNGs."""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PLOTS_DIR = Path("plots")
PLOTS_DIR.mkdir(exist_ok=True)


def load(path):
    rows = json.loads(Path(path).read_text())
    cols = {}
    for r in rows:
        for k in set(list(cols.keys()) + list(r.keys())):
            v = r.get(k, None)
            cols.setdefault(k, []).append(np.nan if v is None else v)
    return {k: np.array(v, dtype=float) for k, v in cols.items()}


def aligned(d, key):
    x = d["_step"]
    y = d[key]
    mask = ~np.isnan(y) & ~np.isnan(x)
    return x[mask], y[mask]


baseline = load("stage1_baseline_cloud.json")
cond = load("stage1_cond_balance_cloud.json")
stage2 = load("stage2_smoke_cloud.json")

RUNS = [
    ("Stage1 BASELINE (pre Fix-1, 2k steps, bs=512)", baseline, "tab:red"),
    ("Stage1 BALANCED (Fix-1: RMSNorm cond, 1k steps, bs=128)", cond, "tab:blue"),
    ("Stage2 SMOKE (from balanced ckpt, 2k steps, bs=128, source_noise=0.1)", stage2, "tab:green"),
]


def smooth(y, w=21):
    if len(y) < w:
        return y
    pad = w // 2
    padded = np.concatenate([np.full(pad, y[0]), y, np.full(pad, y[-1])])
    k = np.ones(w) / w
    return np.convolve(padded, k, mode="valid")


def fig_loss():
    fig, ax = plt.subplots(figsize=(10, 5))
    for label, d, c in RUNS:
        x, y = aligned(d, "loss")
        ax.plot(x, y, color=c, alpha=0.25, lw=0.6)
        ax.plot(x, smooth(y), color=c, lw=2, label=label)
    ax.set_xlabel("step")
    ax.set_ylabel("flow-matching loss")
    ax.set_title("Loss (raw + 21-step moving avg)")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "01_loss.png", dpi=130)
    plt.close(fig)


def fig_mse_gap():
    """The headline diagnostic: shuffled_cond - real_cond MSE.
    A positive gap means the model uses the real drug; collapse pins it near zero."""
    fig, ax = plt.subplots(figsize=(10, 5))
    for label, d, c in RUNS:
        if "diagnostics/mse_gap_shuffled_minus_real" not in d:
            continue
        x, y = aligned(d, "diagnostics/mse_gap_shuffled_minus_real")
        ax.plot(x, y, color=c, alpha=0.3, lw=0.6)
        ax.plot(x, smooth(y, 5), color=c, lw=2, label=label)
    ax.axhline(0.0, color="k", lw=0.5, ls="--")
    ax.set_xlabel("step")
    ax.set_ylabel("MSE(shuffled cond) − MSE(real cond)")
    ax.set_title("Drug-conditioning gap: >0 means real drug helps (closer = collapse)")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "02_mse_gap.png", dpi=130)
    plt.close(fig)


def fig_drug_swap_cos():
    """Cosine similarity between v-prediction with real vs. swapped drug.
    Near 1.0 = model ignores drug. <0.95 = drug starting to matter."""
    fig, ax = plt.subplots(figsize=(10, 5))
    for label, d, c in RUNS:
        if "diagnostics/drug_swap_v_cos" not in d:
            continue
        x, y = aligned(d, "diagnostics/drug_swap_v_cos")
        ax.plot(x, y, color=c, alpha=0.3, lw=0.6)
        ax.plot(x, smooth(y, 5), color=c, lw=2, label=label)
    ax.axhline(1.0, color="k", lw=0.5, ls="--")
    ax.set_xlabel("step")
    ax.set_ylabel("cos( v_real_drug , v_swapped_drug )")
    ax.set_title("Drug-swap cosine: 1.0 = drug irrelevant, lower = drug shifts the velocity field")
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "03_drug_swap_cos.png", dpi=130)
    plt.close(fig)


def fig_cond_rms():
    """Conditioning embedding RMS — pre vs post RMSNorm.
    Baseline has only the raw 'condition_embedding_rms' which is ~0.05 (the bug)."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=False)

    ax = axes[0]
    for label, d, c in RUNS:
        if "diagnostics/condition_embedding_rms" not in d:
            continue
        x, y = aligned(d, "diagnostics/condition_embedding_rms")
        ax.plot(x, y, color=c, lw=1.5, label=label)
    ax.set_xlabel("step")
    ax.set_ylabel("condition_embedding_rms (post-Fix-1: used/RMSNormed)")
    ax.set_title("Condition embedding RMS (raw for baseline, post-norm for Fix-1 runs)")
    ax.legend(fontsize=8, loc="center right")
    ax.grid(alpha=0.3)

    ax = axes[1]
    # baseline raw vs the explicit raw on cond/stage2
    for label, d, c in RUNS:
        if "diagnostics/condition_embedding_raw_rms" in d:
            x, y = aligned(d, "diagnostics/condition_embedding_raw_rms")
            ax.plot(x, y, color=c, ls="-", lw=1.5, label=f"{label[:25]} drug_raw")
        if "diagnostics/time_embedding_raw_rms" in d:
            x, y = aligned(d, "diagnostics/time_embedding_raw_rms")
            ax.plot(x, y, color=c, ls="--", lw=1.5, label=f"{label[:25]} time_raw")
    ax.set_xlabel("step")
    ax.set_ylabel("raw RMS before RMSNorm")
    ax.set_title("Raw drug-RMS (solid) vs raw time-RMS (dashed) — should be commensurate")
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "04_cond_rms.png", dpi=130)
    plt.close(fig)


def fig_v_pred_rms():
    fig, ax = plt.subplots(figsize=(10, 5))
    for label, d, c in RUNS:
        x_p, y_p = aligned(d, "v_pred_rms")
        x_t, y_t = aligned(d, "v_target_rms")
        ax.plot(x_p, smooth(y_p), color=c, lw=2, label=f"{label[:30]} | v_pred_rms")
        ax.plot(x_t, smooth(y_t), color=c, ls=":", lw=1, label=f"{label[:30]} | v_target_rms")
    ax.set_xlabel("step")
    ax.set_ylabel("RMS")
    ax.set_title("v-prediction RMS vs v-target RMS — gap = under-prediction of motion")
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "05_v_pred_vs_target.png", dpi=130)
    plt.close(fig)


def fig_grad():
    fig, ax = plt.subplots(figsize=(10, 5))
    for label, d, c in RUNS:
        x, y = aligned(d, "grad_norm")
        ax.plot(x, y, color=c, alpha=0.25, lw=0.5)
        ax.plot(x, smooth(y), color=c, lw=2, label=label)
    ax.set_xlabel("step")
    ax.set_ylabel("grad_norm")
    ax.set_title("Gradient norm (post-clip target=1.0)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "06_grad_norm.png", dpi=130)
    plt.close(fig)


def fig_combined_dashboard():
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    panels = [
        ("loss", "Loss", True),
        ("diagnostics/mse_gap_shuffled_minus_real", "MSE gap (shuffled − real)", False),
        ("diagnostics/drug_swap_v_cos", "Drug-swap v cosine", False),
        ("diagnostics/condition_embedding_rms", "cond_embedding_rms (post-norm)", False),
        ("v_pred_rms", "v_pred_rms", True),
        ("grad_norm", "grad_norm", True),
    ]
    for ax, (key, title, fill) in zip(axes.flat, panels):
        for label, d, c in RUNS:
            if key not in d:
                continue
            x, y = aligned(d, key)
            if len(x) == 0:
                continue
            if fill:
                ax.plot(x, y, color=c, alpha=0.2, lw=0.5)
            ax.plot(x, smooth(y, 11 if fill else 5), color=c, lw=1.6, label=label.split(" (")[0])
        ax.set_title(title)
        ax.grid(alpha=0.3)
    axes.flat[0].legend(fontsize=7, loc="upper right")
    fig.suptitle("Stage1 baseline vs Stage1 balanced vs Stage2 smoke — dashboard", fontsize=12)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "00_dashboard.png", dpi=130)
    plt.close(fig)


fig_loss()
fig_mse_gap()
fig_drug_swap_cos()
fig_cond_rms()
fig_v_pred_rms()
fig_grad()
fig_combined_dashboard()
print("Wrote plots to", PLOTS_DIR.resolve())
