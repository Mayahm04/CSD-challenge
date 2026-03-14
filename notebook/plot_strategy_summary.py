"""
Beautiful summary plot of both strategies (Stage-1 RF vs Stage-2 Ensemble).
Run from CSD-challenge/ directory.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path

OUT = Path("model_artifact/plots")
OUT.mkdir(exist_ok=True)

# ── palette ───────────────────────────────────────────────────────────────────
BG      = "#0F1117"
CARD    = "#1A1D27"
CARD2   = "#22263A"
ACCENT1 = "#4C9BE8"   # blue  — Stage 1
ACCENT2 = "#2A9D8F"   # teal  — Stage 2 / ens_ev
GOLD    = "#F4C430"
RED     = "#E76F51"
GRAY    = "#8892A4"
WHITE   = "#E8ECF4"

# ── data ─────────────────────────────────────────────────────────────────────
strategies   = ["RF only\n(Stage-1)", "LGBM EV\n(Stage-2)", "XGB EV\n(Stage-2)", "Ensemble EV\n(Stage-2) ★"]
strat_keys   = ["rf_prob", "lgbm_ev", "xgb_ev", "ens_ev"]
fold1_profit = [1.946, 3.573, 3.712, 3.914]
fold2_profit = [1.829, 0.854, 1.155, 1.104]
total_profit = [3.775, 4.427, 4.867, 5.019]
bar_colors   = [GRAY, ACCENT1, "#F4A261", ACCENT2]

# Stage-1 RF metrics per fold
s1_metrics = {
    "AUC":  [0.834, 0.866],
    "F1":   [0.431, 0.414],
    "Prec": [0.341, 0.334],
    "Rec":  [0.585, 0.547],
}

# @K metrics for ens_ev
ens_topk = {"Fold1": {"Prec": 0.531, "Rec": 0.135, "F1": 0.215},
             "Fold2": {"Prec": 0.475, "Rec": 0.139, "F1": 0.215}}

# ── figure ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 12), facecolor=BG)
gs  = gridspec.GridSpec(3, 4, figure=fig, hspace=0.55, wspace=0.38,
                        left=0.05, right=0.97, top=0.88, bottom=0.07)

def dark_ax(ax):
    ax.set_facecolor(CARD)
    for spine in ax.spines.values():
        spine.set_edgecolor("#2E3347")
    ax.tick_params(colors=GRAY, labelsize=9)
    ax.xaxis.label.set_color(GRAY)
    ax.yaxis.label.set_color(GRAY)
    ax.title.set_color(WHITE)
    ax.grid(axis="y", color="#2E3347", linewidth=0.6, alpha=0.8)
    ax.set_axisbelow(True)
    return ax

# ── title ─────────────────────────────────────────────────────────────────────
fig.text(0.5, 0.945, "2-Stage Pipeline — Strategy & Performance Summary",
         ha="center", va="center", fontsize=19, fontweight="bold", color=WHITE)
fig.text(0.5, 0.915, "Stage 1: Random Forest (Rachid, config_id=7)   ·   "
                     "Stage 2: Ensemble LGBM + XGB on all data (Maya)",
         ha="center", va="center", fontsize=11, color=GRAY)

# ══════════════════════════════════════════════════════════════════════════════
# [Row 0, col 0-1]  Total profit stacked bar — all 4 strategies
# ══════════════════════════════════════════════════════════════════════════════
ax1 = dark_ax(fig.add_subplot(gs[0, :2]))
x = np.arange(4)
w = 0.5
b1 = ax1.bar(x, fold1_profit, w, color=bar_colors, alpha=0.95, label="Fold1 — Val 2022")
b2 = ax1.bar(x, fold2_profit, w, bottom=fold1_profit,
             color=bar_colors, alpha=0.45, hatch="///", label="Fold2 — Val 2023")

for i, (tot, col) in enumerate(zip(total_profit, bar_colors)):
    ax1.text(i, tot + 0.06, f"{tot:.2f}M€",
             ha="center", va="bottom", fontsize=10, fontweight="bold",
             color=GOLD if i == 3 else WHITE)

ax1.set_xticks(x)
ax1.set_xticklabels(strategies, fontsize=9, color=WHITE)
ax1.set_ylabel("Profit (M€)", color=GRAY)
ax1.set_ylim(0, 6.2)
ax1.set_title("Total Realized Profit by Strategy", fontsize=12, fontweight="bold", pad=8)
leg = ax1.legend(fontsize=9, loc="upper left",
                 facecolor=CARD2, edgecolor="#2E3347", labelcolor=WHITE)

# highlight best bar label
ax1.get_xticklabels()[3].set_color(ACCENT2)
ax1.get_xticklabels()[3].set_fontweight("bold")

# delta annotation
ax1.annotate(f"+{total_profit[3]-total_profit[0]:.2f}M€\nvs RF only",
             xy=(3, total_profit[3]), xytext=(2.35, 5.5),
             arrowprops=dict(arrowstyle="->", color=GOLD, lw=1.5),
             fontsize=9, color=GOLD, fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.3", facecolor=CARD2, edgecolor=GOLD, linewidth=1))

# ══════════════════════════════════════════════════════════════════════════════
# [Row 0, col 2-3]  Fold1 vs Fold2 generalization scatter
# ══════════════════════════════════════════════════════════════════════════════
ax2 = dark_ax(fig.add_subplot(gs[0, 2:]))
labels_short = ["RF only", "LGBM EV", "XGB EV", "Ens EV ★"]
for i, (f1, f2, col, lbl) in enumerate(zip(fold1_profit, fold2_profit, bar_colors, labels_short)):
    ax2.scatter(f1, f2, s=220, color=col, zorder=5, edgecolors=WHITE, linewidths=0.5)
    ax2.annotate(lbl, (f1, f2), textcoords="offset points",
                 xytext=(8, 5), fontsize=9.5, color=col, fontweight="bold")

mn, mx = 0.5, 4.5
ax2.plot([mn, mx], [mn, mx], color=GRAY, linestyle="--", alpha=0.4, linewidth=1, label="Fold1 = Fold2")
ax2.set_xlabel("Fold1 Profit (M€) — Val 2022", fontsize=10)
ax2.set_ylabel("Fold2 Profit (M€) — Val 2023", fontsize=10)
ax2.set_title("Fold1 vs Fold2 — Generalization", fontsize=12, fontweight="bold", pad=8)
ax2.set_xlim(1.2, 4.5); ax2.set_ylim(0.5, 2.3)
ax2.text(3.0, 0.62, "← closer to diagonal = better generalization",
         fontsize=8, color=GRAY, style="italic")

# ══════════════════════════════════════════════════════════════════════════════
# [Row 1, col 0-1]  Stage-1 RF classification metrics per fold
# ══════════════════════════════════════════════════════════════════════════════
ax3 = dark_ax(fig.add_subplot(gs[1, :2]))
metric_names = list(s1_metrics.keys())
fold_cols = [ACCENT1, "#F4A261"]
fold_labels = ["Fold1 — Val 2022", "Fold2 — Val 2023"]
xs = np.arange(len(metric_names))
w3 = 0.3
for i, (fold_vals, fc, fl) in enumerate(zip(
        [list(v[0] for v in s1_metrics.values()),
         list(v[1] for v in s1_metrics.values())],
        fold_cols, fold_labels)):
    bars = ax3.bar(xs + (i-0.5)*w3, fold_vals, w3, color=fc,
                   alpha=0.9, label=fl, edgecolor=BG, linewidth=0.5)
    for bar, v in zip(bars, fold_vals):
        ax3.text(bar.get_x() + bar.get_width()/2, v + 0.012,
                 f"{v:.3f}", ha="center", va="bottom", fontsize=8.5,
                 color=WHITE, fontweight="bold")

ax3.set_xticks(xs)
ax3.set_xticklabels(metric_names, fontsize=11, color=WHITE)
ax3.set_ylim(0, 1.05)
ax3.set_ylabel("Score", color=GRAY)
ax3.set_title("Stage-1 RF — Classification Metrics (threshold=0.5)", fontsize=12, fontweight="bold", pad=8)
ax3.axhline(0.5, color=GRAY, linestyle=":", alpha=0.4, linewidth=0.8)
leg3 = ax3.legend(fontsize=9, facecolor=CARD2, edgecolor="#2E3347", labelcolor=WHITE)

# ══════════════════════════════════════════════════════════════════════════════
# [Row 1, col 2]  Stage-2 regression R²
# ══════════════════════════════════════════════════════════════════════════════
ax4 = dark_ax(fig.add_subplot(gs[1, 2]))
r2_data = {"LGBM": [-0.113, 0.066], "XGB": [-0.087, 0.033]}
xs4 = np.arange(2)
for i, (model, vals) in enumerate(r2_data.items()):
    col = ACCENT1 if model == "LGBM" else "#F4A261"
    bars = ax4.bar(xs4 + (i-0.5)*0.3, vals, 0.28, color=col, alpha=0.9,
                   label=model, edgecolor=BG, linewidth=0.5)
    for bar, v in zip(bars, vals):
        ypos = v + 0.004 if v >= 0 else v - 0.012
        ax4.text(bar.get_x() + bar.get_width()/2, ypos,
                 f"{v:.3f}", ha="center", va="bottom" if v >= 0 else "top",
                 fontsize=8.5, color=WHITE, fontweight="bold")
ax4.axhline(0, color=RED, linestyle="--", alpha=0.7, linewidth=1.2)
ax4.text(1.38, 0.004, "0 = naïve baseline", fontsize=7.5, color=RED, style="italic")
ax4.set_xticks(xs4); ax4.set_xticklabels(["Fold1\n(2022)", "Fold2\n(2023)"], color=WHITE)
ax4.set_title("Stage-2 R²\n(regression quality)", fontsize=11, fontweight="bold", pad=8)
ax4.set_ylim(-0.18, 0.13)
leg4 = ax4.legend(fontsize=8.5, facecolor=CARD2, edgecolor="#2E3347", labelcolor=WHITE)

# ══════════════════════════════════════════════════════════════════════════════
# [Row 1, col 3]  Precision@K / Recall@K / F1@K — Ensemble only
# ══════════════════════════════════════════════════════════════════════════════
ax5 = dark_ax(fig.add_subplot(gs[1, 3]))
topk_m = ["Prec@K", "Rec@K", "F1@K"]
topk_cols = ["#E9C46A", ACCENT2, RED]
f1_vals = [ens_topk["Fold1"]["Prec"], ens_topk["Fold1"]["Rec"], ens_topk["Fold1"]["F1"]]
f2_vals = [ens_topk["Fold2"]["Prec"], ens_topk["Fold2"]["Rec"], ens_topk["Fold2"]["F1"]]
xs5 = np.arange(3)
for i, (vals, fl) in enumerate([(f1_vals, "Fold1"), (f2_vals, "Fold2")]):
    fc = ACCENT1 if fl == "Fold1" else "#F4A261"
    bars = ax5.bar(xs5 + (i-0.5)*0.3, vals, 0.28, color=fc, alpha=0.9,
                   label=fl, edgecolor=BG, linewidth=0.5)
    for bar, v in zip(bars, vals):
        ax5.text(bar.get_x() + bar.get_width()/2, v + 0.005,
                 f"{v:.3f}", ha="center", va="bottom",
                 fontsize=8.5, color=WHITE, fontweight="bold")
ax5.set_xticks(xs5); ax5.set_xticklabels(topk_m, color=WHITE)
ax5.set_title("Ensemble EV — @K Metrics\n(quality of top-K selection)", fontsize=11, fontweight="bold", pad=8)
ax5.set_ylim(0, 0.75)
leg5 = ax5.legend(fontsize=8.5, facecolor=CARD2, edgecolor="#2E3347", labelcolor=WHITE)

# ══════════════════════════════════════════════════════════════════════════════
# [Row 2]  Pipeline diagram + KPI cards
# ══════════════════════════════════════════════════════════════════════════════
ax6 = fig.add_subplot(gs[2, :])
ax6.set_facecolor(BG)
ax6.axis("off")

def kpi_card(ax, x, y, w, h, title, value, subtitle, color, fig):
    ax_inset = fig.add_axes([x, y, w, h], facecolor=CARD2)
    ax_inset.axis("off")
    for spine in ax_inset.spines.values():
        spine.set_edgecolor(color)
        spine.set_linewidth(2)
    ax_inset.text(0.5, 0.78, title, ha="center", va="center",
                  fontsize=9, color=GRAY, transform=ax_inset.transAxes)
    ax_inset.text(0.5, 0.42, value, ha="center", va="center",
                  fontsize=17, fontweight="bold", color=color, transform=ax_inset.transAxes)
    ax_inset.text(0.5, 0.12, subtitle, ha="center", va="center",
                  fontsize=8, color=GRAY, transform=ax_inset.transAxes)

# KPI cards — bottom row
kpi_card(fig, 0.05,  0.03, 0.12, 0.12, "Stage-1 Best AUC",    "0.866",     "Fold2 — Val 2023",           ACCENT1, fig)
kpi_card(fig, 0.20,  0.03, 0.12, 0.12, "Stage-1 Best F1",     "0.431",     "Fold1 — Val 2022",           ACCENT1, fig)
kpi_card(fig, 0.35,  0.03, 0.12, 0.12, "Best Strategy",       "Ens EV",    "LGBM + XGB ensemble",        ACCENT2, fig)
kpi_card(fig, 0.50,  0.03, 0.12, 0.12, "Total Profit",        "5.02M€",    "Fold1 + Fold2",              GOLD,    fig)
kpi_card(fig, 0.65,  0.03, 0.12, 0.12, "Optimal K",           "100 / 86",  "Fold1 / Fold2",              ACCENT2, fig)
kpi_card(fig, 0.80,  0.03, 0.12, 0.12, "Gain vs RF only",     "+1.24M€",   "+33% over baseline",         GOLD,    fig)

# pipeline arrow
fig.text(0.5, 0.175,
         "PIPELINE:   RF OOF prob  ──→  × max(0, (LGBM_pred + XGB_pred) / 2)  ──→  top-K per month",
         ha="center", va="center", fontsize=11, color=GRAY,
         bbox=dict(boxstyle="round,pad=0.5", facecolor=CARD2, edgecolor="#2E3347", linewidth=1.5))

# ── save ──────────────────────────────────────────────────────────────────────
plt.savefig(OUT / "strategy_summary.png", dpi=160, bbox_inches="tight", facecolor=BG)
plt.close()
print(f"✓ Saved → {OUT}/strategy_summary.png")
