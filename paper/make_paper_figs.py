"""Standalone figure generation for the IEEE-CS paper (English labels).

Produces two publication-quality figures under paper/figuras/:
  - fig_method_en.png   : Cold-Start LOPO evaluation protocol (context + horizons + windows)
  - fig_results_en.png  : median rRMSE (7-day roll-out, operational window) per configuration

Run:  ./.venv/Scripts/python.exe paper/make_paper_figs.py
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = os.path.join(os.path.dirname(__file__), "figuras")
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "savefig.dpi": 200,
                     "savefig.bbox": "tight"})


def _save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path)
    plt.close(fig)
    print("[paper]", path)


def fig_method():
    """Cold-Start LOPO protocol: synthetic context + horizons + evaluation windows."""
    fig, ax = plt.subplots(figsize=(9.2, 3.9))
    ax.set_xlim(0, 24); ax.set_ylim(0, 4.4); ax.axis("off")

    y0 = 3.05
    ax.add_patch(plt.Rectangle((1, y0), 7, 0.9, fc="#FFF2CC", ec="#BF8F00"))
    ax.text(4.5, y0 + 0.45, "Synthetic context (168 h)\n"
            r"$y = \mathrm{capacity}\times PR_{\mathrm{regional}}\times \mathrm{solar\ profile}$",
            ha="center", va="center", fontsize=8.5)
    ax.add_patch(plt.Rectangle((8, y0), 2.2, 0.9, fc="#C6E0B4", ec="#538135"))
    ax.text(9.1, y0 + 0.45, "day1\n(24 h)", ha="center", va="center", fontsize=8.5)
    ax.add_patch(plt.Rectangle((8, 1.85), 14.6, 1.1, fc="#DEEBF7", ec="#1F4E79"))
    ax.text(15.3, 2.4, "rollout7d: 7 autoregressive days\n"
            "(feeds back the predicted median, never the ground truth)",
            ha="center", va="center", fontsize=8)
    ax.annotate("", xy=(9.1, y0), xytext=(9.1, 2.95),
                arrowprops=dict(arrowstyle="->", lw=1.1))

    ax.annotate("", xy=(22.8, 1.6), xytext=(1, 1.6),
                arrowprops=dict(arrowstyle="->", lw=1.2))
    ax.text(22.6, 1.33, "time (canonical hourly grid)", fontsize=8, ha="right")

    ax.text(1, 1.05, "Evaluation window start (sensitivity analysis):",
            fontsize=8.5, fontweight="bold")
    ax.text(1, 0.68, "Raw: first recorded hour (includes commissioning ramp)"
            "     Operational: first sustained production\n"
            "Seasonal: first sustained window starting in Summer / Autumn / "
            "Winter / Spring (later history of the same plant)",
            fontsize=8, va="top")
    fig.suptitle("Zero-shot Cold-Start evaluation protocol (LOPO)", fontsize=11)
    _save(fig, "fig_method_en.png")


def fig_results():
    """Median rRMSE (7-day roll-out, operational window) per configuration."""
    # (label, scope, rRMSE roll7d %) sorted best -> worst
    rows = [
        ("XGBoost", "Local", 62.2),
        ("XGBoost", "Global", 95.4),
        ("N-HiTS", "Local", 104.0),
        ("LSTM", "Global", 118.2),
        ("Informer", "Global", 129.3),
        ("TFT", "Global", 133.1),
        ("LSTM", "Local", 146.9),
        ("N-HiTS", "Global", 150.4),
    ]
    labels = [f"{m} ({s})" for m, s, _ in rows]
    vals = [v for _, _, v in rows]
    colors = ["#2CA02C" if s == "Local" else "#1F77B4" for _, s, _ in rows]

    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    ypos = range(len(rows))
    ax.barh(list(ypos), vals, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_yticks(list(ypos)); ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Median rRMSE — 7-day roll-out, operational window (%)", fontsize=9)
    for i, v in enumerate(vals):
        ax.text(v + 1.5, i, f"{v:.0f}", va="center", fontsize=8.5)
    ax.axvline(62.2, ls="--", lw=0.9, color="#2CA02C", alpha=0.6)
    ax.set_xlim(0, 168)
    handles = [plt.Rectangle((0, 0), 1, 1, fc="#2CA02C", ec="black"),
               plt.Rectangle((0, 0), 1, 1, fc="#1F77B4", ec="black")]
    ax.legend(handles, ["Strictly local", "Global (Cross-Site)"],
              loc="upper right", fontsize=8.5, framealpha=0.95)
    ax.grid(axis="x", ls=":", alpha=0.4)
    fig.tight_layout()
    _save(fig, "fig_results_en.png")


if __name__ == "__main__":
    fig_method()
    fig_results()
    print("done")
