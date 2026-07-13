"""Standalone figure generation for the IEEE-CS paper (English labels).

Produces publication-quality figures under paper/figuras/:
  - fig_method_en.png    : Cold-Start LOPO evaluation protocol (context + horizons + windows)
  - fig_results_en.png   : median rRMSE (7-day roll-out, operational window) per configuration
  - fig_medallion_en.png : Medallion data-lineage of the ETL pipeline (Bronze/Silver/Gold)

Run:  ./.venv/Scripts/python.exe paper/make_paper_figs.py
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = os.path.join(os.path.dirname(__file__), "figuras")
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "savefig.dpi": 200,
                     "savefig.bbox": "tight"})


def _save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path)
    plt.close(fig)
    print("[paper]", path)


def _arrow(ax, p1, p2, color="#555555"):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=14,
                                 color=color, lw=1.5, shrinkA=2, shrinkB=2))


def fig_medallion():
    """Medallion data-lineage of the ETL pipeline (source systems -> Bronze/Silver/Gold)."""
    src_fc, src_ec = "#9DC3E6", "#2E75B6"
    brz_fc, brz_ec = "#C9A227", "#7F6000"
    slv_fc, slv_ec = "#E7E6E6", "#808080"
    gld_fc, gld_ec = "#FFD966", "#BF9000"

    def tbl(cx, cy, w, h, text, fc, ec, tc="#1A1A1A", fs=8.5):
        ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                                    boxstyle="round,pad=0.015", fc=fc, ec=ec, lw=1.3))
        ax.text(cx, cy, text, ha="center", va="center", fontsize=fs,
                family="monospace", color=tc)

    def zone(x0, x1, y0, y1, title, subtitle, ec):
        ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                                    boxstyle="round,pad=0.05", fc="none", ec=ec,
                                    lw=1.7, linestyle=(0, (6, 4))))
        ax.text((x0 + x1) / 2, y1 - 0.6, title, ha="center", fontsize=13, fontweight="bold")
        ax.text((x0 + x1) / 2, y1 - 1.15, subtitle, ha="center", fontsize=8.5,
                color="#555555", family="monospace")

    fig, ax = plt.subplots(figsize=(13, 6.4))
    ax.set_xlim(0, 26); ax.set_ylim(0, 14); ax.axis("off")

    for cy, txt in [(10.4, "Grid operator\n(generation)"),
                    (7.0, "Plant master\n(geo, capacity)"),
                    (3.6, "CR2\n(meteorology)")]:
        ax.add_patch(FancyBboxPatch((0.4, cy - 1.0), 3.0, 2.0, boxstyle="round,pad=0.02",
                                    fc=src_fc, ec=src_ec, lw=1.3))
        ax.text(1.9, cy, txt, ha="center", va="center", fontsize=8.8)

    zone(4.4, 9.8, 1.5, 13.0, "Bronze", "schema = bronze", brz_ec)
    zone(10.7, 16.5, 1.5, 13.0, "Silver", "schema = silver", slv_ec)
    zone(17.4, 25.6, 1.5, 13.0, "Gold", "schema = gold", gld_ec)

    tbl(7.1, 9.4, 3.6, 1.3, "maestro_\ngeneracion", brz_fc, brz_ec, tc="#241D00")
    tbl(7.1, 4.6, 3.6, 1.3, "maestro_\nexogenas", brz_fc, brz_ec, tc="#241D00")
    tbl(13.6, 8.2, 3.7, 1.3, "silver_\nunified", slv_fc, slv_ec)
    tbl(13.6, 4.6, 3.7, 1.2, "silver_dl", slv_fc, slv_ec)
    gx, gy = 21.5, 8.2
    for dx, dy in ((0.7, -0.7), (0.35, -0.35)):
        tbl(gx + dx, gy + dy, 4.8, 1.3, "", gld_fc, gld_ec)
    tbl(gx, gy, 4.8, 1.3, "{macrozona}/{estacion}/\n<id>_test", gld_fc, gld_ec, fs=8.0)

    _arrow(ax, (3.4, 10.4), (5.3, 9.7))
    _arrow(ax, (3.4, 7.0), (5.3, 9.1))
    _arrow(ax, (3.4, 3.6), (5.3, 4.6))
    _arrow(ax, (8.9, 9.4), (11.75, 8.5))
    _arrow(ax, (8.9, 4.6), (11.75, 7.9))
    ax.text(10.25, 6.55, "join", ha="center", fontsize=7.5, style="italic", color="#777777")
    _arrow(ax, (13.6, 7.55), (13.6, 5.2))
    _arrow(ax, (15.45, 8.2), (19.1, 8.2))
    ax.text(17.3, 8.55, "LOPO\nsplit", ha="center", fontsize=7.5, style="italic", color="#777777")

    ax.text(1.9, 12.1, "raw CSV · 2014–2024", ha="center", fontsize=8,
            style="italic", color="#333333")
    ax.text(13.6, 9.55, "94 plants · ≈ 4.37 M rows", ha="center", fontsize=7.8, color="#333333")
    ax.text(13.6, 3.75, "float32 · label-encoded", ha="center", fontsize=7.8, color="#333333")
    ax.text(21.8, 9.7, "94 LOPO partitions", ha="center", fontsize=7.8, color="#333333")
    ax.text(15.0, 0.55, "Generation y is never imputed · exogenous imputation flagged "
            "· zero data leakage enforced by automated tests",
            ha="center", fontsize=8.3, style="italic", color="#555555")

    fig.suptitle("Medallion ETL data-lineage: source systems → Bronze / Silver / Gold",
                 fontsize=13)
    _save(fig, "fig_medallion_en.png")


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
    fig_medallion()
    print("done")
