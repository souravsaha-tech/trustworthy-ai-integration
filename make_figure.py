"""Figure 3: E1 (infrastructure failure) and E2 (model failure) time series, seed 1."""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
S = dict(np.load("results/series_seed1.npz"))

def bin5(x):
    return x.reshape(-1, 5).mean(axis=1)
BLUE, ORANGE, INK = "#1f4e79", "#c55a11", "#262626"
styles = {"naive": dict(color=INK, ls=(0, (1.5, 1.5)), lw=1.6, label="No time budget (2 s client timeout)"),
          "budget": dict(color=ORANGE, ls=(0, (5, 2)), lw=1.6, label="Time budget + fallback"),
          "budget_cb": dict(color=BLUE, ls="-", lw=2.0, label="Time budget + fallback + circuit breaker")}
fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), gridspec_kw=dict(wspace=0.32))
t = np.arange(0, 600, 5) + 2.5

def shade(ax):
    ax.axvspan(200, 320, color="#e6e6e6", lw=0); ax.axvspan(320, 380, color="#c8c8c8", lw=0)

ax = axes[0]; shade(ax)
for m, st in styles.items():
    ax.plot(t, 100 * bin5(S[f"e1_{m}_failed_share"]), **st)
ax.set_ylim(-3, 103); ax.set_xlabel("Time (s)"); ax.set_ylabel("Payments failed (%)")
ax.set_title("(a) E1: Failed payments", loc="left", fontsize=9.5, weight="bold")
ax.text(260, 88, "model\nslow", ha="center", fontsize=7.5); ax.text(350, 88, "model\ndown", ha="center", fontsize=7.5)
ax.text(8, 6, "Time-budget designs: 0%", ha="left", fontsize=7.5, color=BLUE)

ax = axes[1]; shade(ax)
for m, st in styles.items():
    ax.plot(t, 100 * bin5(S[f"e1_{m}_model_share"]), **st)
ax.set_ylim(-3, 103); ax.set_xlabel("Time (s)"); ax.set_ylabel("Decisions scored by the model (%)")
ax.set_title("(b) E1: Decisions scored by the model", loc="left", fontsize=9.5, weight="bold")

ax = axes[2]
days = np.arange(len(S["e2_true_recall"]))
ax.axvspan(25, 34, color="#e6e6e6", lw=0)
tr = S["e2_true_recall"]; k = 3
smooth = np.convolve(tr, np.ones(k) / k, mode="same"); smooth[:1] = tr[:1]; smooth[-1:] = tr[-1:]
ax.plot(days, 100 * smooth, color="#7f7f7f", lw=1.3, label="True fraud interception (3-day avg.)")
ax.plot(days, 100 * S["e2_recall_seen"], color=BLUE, lw=2.0, label="Outcome monitor (labels arrive late)")
ax.axhline(100 * (S["e2_base_recall"] - 0.10), color=BLUE, lw=0.8, ls=(0, (4, 2)))
ax.set_ylim(0, 100); ax.set_xlabel("Day"); ax.set_ylabel("Fraud intercepted (%)")
ax.set_title("(c) E2: Silent fraud drift", loc="left", fontsize=9.5, weight="bold")
ax2 = ax.twinx(); ax2.plot(days, S["e2_max_psi"], color=ORANGE, lw=1.4, ls=(0, (5, 2)), label="Input-drift PSI (alarm at 0.2)")
ax2.set_ylim(0, 0.25); ax2.axhline(0.2, color=ORANGE, lw=0.8); ax2.set_ylabel("Max PSI across inputs", color=ORANGE)
ax2.tick_params(axis="y", colors=ORANGE)
ax2.text(0, 0.203, "PSI alarm threshold", ha="left", va="bottom", fontsize=7, color=ORANGE)
ax.text(0, 100 * (S["e2_base_recall"] - 0.10) - 1.5, "outcome alarm threshold", ha="left", va="top", fontsize=7, color=BLUE)
h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, fontsize=7, loc="lower left", frameon=True, framealpha=0.95)
for a in axes:
    a.spines["top"].set_visible(False)
axes[0].spines["right"].set_visible(False); axes[1].spines["right"].set_visible(False)
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=3, fontsize=8, frameon=False, bbox_to_anchor=(0.37, 1.04))
for ext in ("png", "pdf"):
    fig.savefig(f"results/Figure3_evaluation.{ext}", dpi=300, bbox_inches="tight", pad_inches=0.05)
