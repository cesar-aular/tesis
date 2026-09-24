"""Análisis para la versión final de conferencia (paper #228, CWPR/JCC 2026).

Responde a los comentarios de los revisores SIN reentrenar nada: todo se deriva de
artefactos ya en disco.

Entradas
--------
- results/total/visuals/all_metrics_summary.csv : una fila por (Model, Horizon, Window, Planta)
- data/silver/silver_unified.parquet            : capacidad instalada por planta (MW)
- results/total/<model>/<macrozona>/<estacion>/<planta>/<horizon_dir>/preds.parquet

Salidas
-------
- docs/tablas/tab_main94.tex        : Table I recomputada sobre 94 plantas (+ NRMSE, IQR)
- docs/tablas/tab_seasonal8.tex     : Table II con las 8 configuraciones (añade LSTM local)
- docs/tablas/tab_stats.tex         : Friedman + Wilcoxon-Holm + win counts
- docs/figuras/fig_timeseries_en.png: observado vs. predicho, 24 h y 7 d, banda P5-P95

Uso:  ./.venv/Scripts/python.exe docs/make_review_analysis.py
"""
from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

STRATEGY = "total"
RES = ROOT / "results" / STRATEGY
SUMMARY = RES / "visuals" / "all_metrics_summary.csv"
SILVER = ROOT / "data" / "silver" / "silver_unified.parquet"
OUT_TAB = Path(__file__).resolve().parent / "tablas"
OUT_FIG = Path(__file__).resolve().parent / "figuras"
OUT_TAB.mkdir(exist_ok=True)
OUT_FIG.mkdir(exist_ok=True)

plt.rcParams.update({"font.family": "DejaVu Sans", "savefig.dpi": 200,
                     "savefig.bbox": "tight"})

# Etiquetas de presentación (inglés, como el paper)
PRETTY = {
    "XGB_LOCAL": ("XGBoost", "Local"),
    "XGB_GLOBAL": ("XGBoost", "Global"),
    "NHITS_LOCAL": ("N-HiTS", "Local"),
    "NHITS": ("N-HiTS", "Global"),
    "LSTM_LOCAL": ("LSTM", "Local"),
    "LSTM": ("LSTM", "Global"),
    "TFT": ("TFT", "Global"),
    "INFORMER": ("Informer", "Global"),
}
MODELS = list(PRETTY)
ROLLOUT = "7-Day Rollout"
DAY1 = "Day 1"


# ----------------------------------------------------------------------------
# Carga
# ----------------------------------------------------------------------------
def load_summary() -> pd.DataFrame:
    if not SUMMARY.exists():
        sys.exit(f"[ERROR] No existe {SUMMARY}. Se requiere la corrida 'total'.")
    df = pd.read_csv(SUMMARY)
    print(f"[carga] {SUMMARY.name}: {len(df):,} filas, "
          f"{df['Planta'].nunique()} plantas, {df['Model'].nunique()} modelos")
    return df


def load_capacity() -> pd.DataFrame:
    cap = (pd.read_parquet(SILVER, columns=["unique_id", "potencia_neta_mw", "macrozona"])
           .drop_duplicates("unique_id")
           .rename(columns={"unique_id": "Planta", "potencia_neta_mw": "capacidad_mw"}))
    print(f"[carga] capacidad: {len(cap)} plantas "
          f"(mediana {cap['capacidad_mw'].median():.2f} MW)")
    return cap


# ----------------------------------------------------------------------------
# 1.1 Vista "cold-start window" sobre las 94 plantas
# ----------------------------------------------------------------------------
def build_window94(df: pd.DataFrame) -> pd.DataFrame:
    """Operational donde existe (rampa distinguible), Raw en el resto.

    `eval_window.eval_window_variants` solo emite '_operational' si el inicio de
    producción sostenida está a MÁS de 24 h del primer registro. Para las plantas
    sin esa variante la rampa es <=24 h, de modo que la ventana Raw ya arranca en
    producción sostenida y es la ventana cold-start limpia de esa planta.
    """
    op = df[df["Window"] == "Operational"]
    raw = df[df["Window"] == "Raw"]
    plants_op = set(op["Planta"].unique())
    plants_raw = set(raw["Planta"].unique())

    merged = pd.concat([op, raw[~raw["Planta"].isin(plants_op)]], ignore_index=True)
    merged["WindowSource"] = np.where(merged["Planta"].isin(plants_op),
                                      "Operational", "Raw")
    n_op, n_raw = len(plants_op), len(plants_raw - plants_op)
    print(f"[ventana94] Operational={n_op} plantas | Raw(sin rampa)={n_raw} "
          f"| total={merged['Planta'].nunique()}")
    return merged


def audit_missing_operational(df: pd.DataFrame) -> None:
    """Clasifica POR QUÉ 61 plantas no tienen ventana Operational.

    Casos posibles en eval_window_variants:
      (a) start_op <= 24            -> sin rampa; Raw ya es la ventana limpia
      (b) start_op is None          -> nunca hay producción sostenida
      (c) start_op + 336 > len(y)   -> la ventana operacional no cabe en la serie
    Solo (a) justifica sustituir por Raw.
    """
    from src.ml.utils.eval_window import (WINDOW_HOURS, eval_window_variants,
                                          find_operational_start)
    from src.ml.utils.grid import make_hourly_grid

    op_plants = set(df[df["Window"] == "Operational"]["Planta"].unique())
    all_plants = sorted(df["Planta"].unique())
    missing = [p for p in all_plants if p not in op_plants]
    print(f"\n[auditoría] {len(missing)} plantas sin ventana Operational; clasificando...")

    silver = pd.read_parquet(SILVER, columns=["unique_id", "ds", "y", "potencia_neta_mw"])
    cases = {"a_sin_rampa": [], "b_sin_produccion": [], "c_no_cabe": []}
    for p in missing:
        s = silver[silver["unique_id"] == p][["unique_id", "ds", "y", "potencia_neta_mw"]]
        if s.empty:
            cases["b_sin_produccion"].append(p)
            continue
        cap = float(s["potencia_neta_mw"].iloc[0])
        grid = make_hourly_grid(s.rename(columns={"unique_id": "unique_id"}), p)
        start_op = find_operational_start(grid["y"].reset_index(drop=True), cap)
        if start_op is None:
            cases["b_sin_produccion"].append(p)
        elif start_op <= 24:
            cases["a_sin_rampa"].append(p)
        elif start_op + WINDOW_HOURS > len(grid):
            cases["c_no_cabe"].append(p)
        else:
            cases.setdefault("d_inesperado", []).append(p)

    for k, v in cases.items():
        print(f"           {k:18} {len(v):3d}" + (f"  ej: {v[:3]}" if v else ""))
    return cases


# ----------------------------------------------------------------------------
# 1.2 NRMSE normalizado por capacidad
# ----------------------------------------------------------------------------
def add_nrmse(df: pd.DataFrame, cap: pd.DataFrame) -> pd.DataFrame:
    out = df.merge(cap[["Planta", "capacidad_mw"]], on="Planta", how="left")
    missing = out["capacidad_mw"].isna().sum()
    if missing:
        print(f"[NRMSE] AVISO: {missing} filas sin capacidad")
    out["NRMSE"] = out["RMSE"] / out["capacidad_mw"] * 100.0
    return out


# ----------------------------------------------------------------------------
# 1.3 Tests estadísticos
# ----------------------------------------------------------------------------
def paired_matrix(df: pd.DataFrame, metric: str, horizon: str) -> pd.DataFrame:
    """Matriz Planta x Model completa (solo plantas con los 8 modelos)."""
    sub = df[df["Horizon"] == horizon]
    m = sub.pivot_table(index="Planta", columns="Model", values=metric, aggfunc="first")
    m = m.reindex(columns=MODELS).dropna()
    return m


def holm(pvals: list[float]) -> list[float]:
    """Corrección Holm-Bonferroni; devuelve p ajustados en el orden de entrada."""
    n = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(n, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        val = (n - rank) * pvals[idx]
        running = max(running, val)
        adj[idx] = min(1.0, running)
    return adj.tolist()


def run_stats(mat: pd.DataFrame, metric: str, horizon: str) -> dict:
    n_plants = len(mat)
    print(f"\n[tests] métrica={metric} horizonte={horizon} "
          f"diseño completo {n_plants}x{mat.shape[1]}")

    chi2, p_fried = friedmanchisquare(*[mat[m].to_numpy() for m in mat.columns])
    print(f"        Friedman: chi2={chi2:.2f}  p={p_fried:.3e}")

    # ranking medio (1 = mejor)
    ranks = mat.rank(axis=1, method="average")
    mean_rank = ranks.mean().sort_values()
    print("        ranking medio:",
          ", ".join(f"{m}={r:.2f}" for m, r in mean_rank.items()))

    pairs, raw_p, stats_rows = [], [], []
    for a, b in combinations(mat.columns, 2):
        d = mat[a].to_numpy() - mat[b].to_numpy()
        if np.allclose(d, 0):
            stat, p = np.nan, 1.0
        else:
            stat, p = wilcoxon(mat[a], mat[b], zero_method="wilcox")
        pairs.append((a, b))
        raw_p.append(float(p))
        wins_a = int((mat[a] < mat[b]).sum())
        stats_rows.append({"a": a, "b": b, "median_diff": float(np.median(d)),
                           "wins_a": wins_a, "wins_b": int((mat[b] < mat[a]).sum()),
                           "p_raw": float(p)})
    p_adj = holm(raw_p)
    for row, pa in zip(stats_rows, p_adj):
        row["p_holm"] = pa

    # win counts globales: nº de plantas donde cada modelo es el mejor
    winners = mat.idxmin(axis=1).value_counts().reindex(MODELS).fillna(0).astype(int)
    print("        win counts (mejor por planta):",
          ", ".join(f"{m}={c}" for m, c in winners.items() if c))

    return {"n": n_plants, "chi2": float(chi2), "p_friedman": float(p_fried),
            "mean_rank": mean_rank, "pairs": stats_rows, "winners": winners}


KEY_CONTRASTS = [
    ("XGB_LOCAL", "XGB_GLOBAL"),   # titular: mejor local vs mejor global
    ("LSTM", "LSTM_LOCAL"),        # matiz a igualdad de arquitectura
    ("NHITS", "NHITS_LOCAL"),      # contra-ejemplo del matiz
    ("XGB_GLOBAL", "LSTM"),        # mejor global vs mejor red global
]


def fmt_p_simple(p: float) -> str:
    if p < 1e-4:
        return "$<10^{-4}$"
    if p < 0.001:
        return "$<0.001$"
    return f"${p:.3f}$"


# ----------------------------------------------------------------------------
# Emisión de tablas LaTeX
# ----------------------------------------------------------------------------
def write_tab_main94(df94: pd.DataFrame) -> None:
    """Table I sobre 94 plantas: rRMSE (continuidad) + NRMSE (interpretable) + IQR."""
    rows = []
    for m in MODELS:
        s_roll = df94[(df94["Model"] == m) & (df94["Horizon"] == ROLLOUT)]
        s_day1 = df94[(df94["Model"] == m) & (df94["Horizon"] == DAY1)]
        rows.append({
            "Model": PRETTY[m][0], "Scope": PRETTY[m][1],
            "rRMSE_d1": s_day1["rRMSE"].median(),
            "rRMSE_r7": s_roll["rRMSE"].median(),
            "NRMSE_d1": s_day1["NRMSE"].median(),
            "NRMSE_r7": s_roll["NRMSE"].median(),
            "NRMSE_q25": s_roll["NRMSE"].quantile(.25),
            "NRMSE_q75": s_roll["NRMSE"].quantile(.75),
            "Cov": s_roll["Coverage_90_diurna"].median(),
            "Pin": s_roll["Pinball_P95"].median(),
            "_sort": s_roll["NRMSE"].median(),
        })
    tab = pd.DataFrame(rows).sort_values("_sort")
    best = {c: tab[c].min() for c in ["rRMSE_d1", "rRMSE_r7", "NRMSE_d1", "NRMSE_r7", "Pin"]}

    def b(v, col, fmt="{:.1f}"):
        s = fmt.format(v)
        return r"\textbf{" + s + "}" if np.isclose(v, best[col]) else s

    lines = [r"\begin{tabular}{llccccccc}", r"\toprule",
             r" &  & \multicolumn{2}{c}{rRMSE (\%)} & \multicolumn{2}{c}{NRMSE (\% cap.)} & & & \\",
             r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}",
             r"Model & Scope & day1 & roll7d & day1 & roll7d & IQR$_{\text{NRMSE}}$ & Cov$_{90}$ & Pinball \\",
             r"\midrule"]
    for _, r in tab.iterrows():
        lines.append(
            f"{r['Model']} & {r['Scope']} & "
            f"{b(r['rRMSE_d1'],'rRMSE_d1')} & {b(r['rRMSE_r7'],'rRMSE_r7')} & "
            f"{b(r['NRMSE_d1'],'NRMSE_d1')} & {b(r['NRMSE_r7'],'NRMSE_r7')} & "
            f"[{r['NRMSE_q25']:.1f}, {r['NRMSE_q75']:.1f}] & "
            f"{r['Cov']:.2f} & {b(r['Pin'],'Pin','{:.3f}')} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT_TAB / "tab_main94.tex").write_text("\n".join(lines), encoding="utf-8")
    print(f"[tabla] tab_main94.tex ({len(tab)} filas)")


def write_tab_seasonal(df: pd.DataFrame, cap: pd.DataFrame) -> None:
    """Table II con las 8 configuraciones (el paper omitía LSTM local)."""
    seasons = ["Verano", "Otoño", "Invierno", "Primavera"]
    sub = df[(df["Horizon"] == ROLLOUT) & (df["Window"].isin(seasons))]
    piv = sub.pivot_table(index="Model", columns="Window", values="rRMSE",
                          aggfunc="median").reindex(index=MODELS, columns=seasons)
    npl = sub.groupby(["Model", "Window"])["Planta"].nunique()

    lines = [r"\begin{tabular}{llcccc}", r"\toprule",
             r"Model & Scope & Summer & Autumn & Winter & Spring \\", r"\midrule"]
    for m in MODELS:
        v = piv.loc[m]
        lines.append(f"{PRETTY[m][0]} & {PRETTY[m][1]} & "
                     + " & ".join(f"{v[s]:.1f}" for s in seasons) + r" \\")
    lines += [r"\midrule",
              r"\textit{plants} & & "
              + " & ".join(str(int(npl.xs(s, level='Window').max())) for s in seasons)
              + r" \\", r"\bottomrule", r"\end{tabular}"]
    (OUT_TAB / "tab_seasonal8.tex").write_text("\n".join(lines), encoding="utf-8")
    missing = [m for m in MODELS if piv.loc[m].isna().all()]
    print(f"[tabla] tab_seasonal8.tex (8 configs; sin datos: {missing or 'ninguna'})")


def write_tab_stats(res: dict, metric_label: str) -> None:
    # Etiqueta compacta: si comparten familia -> "XGB: local vs. global"
    short = {"XGBoost": "XGB", "N-HiTS": "N-HiTS", "LSTM": "LSTM",
             "TFT": "TFT", "Informer": "Informer"}

    def label(a: str, b: str) -> str:
        fa, sa = PRETTY[a]
        fb, sb = PRETTY[b]
        fa, fb = short[fa], short[fb]
        if fa == fb:
            return f"{fa}: {sa.lower()} vs.\\ {sb.lower()}"
        if sa == sb:
            return f"{sa}: {fa} vs.\\ {fb}"
        return f"{fa} {sa.lower()} vs.\\ {fb} {sb.lower()}"

    lines = [r"\footnotesize", r"\begin{tabular}{llrrr}", r"\toprule",
             r"Contrast & Better & $\Delta$ & Wins & $p$ \\",
             r"\midrule"]
    by_pair = {(r["a"], r["b"]): r for r in res["pairs"]}
    for a, b in KEY_CONTRASTS:
        r = by_pair.get((a, b)) or by_pair.get((b, a))
        if r is None:
            continue
        flip = (r["a"], r["b"]) != (a, b)
        diff = -r["median_diff"] if flip else r["median_diff"]
        wa = r["wins_b"] if flip else r["wins_a"]
        wb = r["wins_a"] if flip else r["wins_b"]
        better_full = PRETTY[a] if diff < 0 else PRETTY[b]
        sig = r["p_holm"] < 0.05
        better = (better_full[1].lower() if PRETTY[a][0] == PRETTY[b][0]
                  else f"{short[better_full[0]]} {better_full[1].lower()}")
        lines.append(f"{label(a, b)} & {better if sig else '---'} & {abs(diff):.1f} & "
                     f"{wa}/{wb} & {fmt_p_simple(r['p_holm'])} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT_TAB / "tab_stats.tex").write_text("\n".join(lines), encoding="utf-8")
    print(f"[tabla] tab_stats.tex ({len(KEY_CONTRASTS)} contrastes clave)")

    # resumen de texto para redactar el paper
    summary = [f"Friedman chi2={res['chi2']:.2f}, p={res['p_friedman']:.3e}, n={res['n']} plantas",
               "ranking medio: " + ", ".join(f"{m}={r:.2f}" for m, r in res["mean_rank"].items()),
               "win counts: " + ", ".join(f"{m}={c}" for m, c in res["winners"].items()),
               f"metrica={metric_label}", "", "TODOS los contrastes (Holm):"]
    for r in sorted(res["pairs"], key=lambda x: x["p_holm"]):
        summary.append(f"  {r['a']:12} vs {r['b']:12} dmed={r['median_diff']:+7.2f} "
                       f"wins={r['wins_a']:2}/{r['wins_b']:2} p_holm={r['p_holm']:.3e}")
    (OUT_TAB / "stats_summary.txt").write_text("\n".join(summary), encoding="utf-8")


# ----------------------------------------------------------------------------
# 1.4 Figura de series temporales (R1.4)
# ----------------------------------------------------------------------------
def fig_results_nrmse(df94: pd.DataFrame) -> None:
    """Ranking por NRMSE (roll-out) sobre las 94 plantas: reemplaza la figura
    antigua, que mostraba rRMSE sobre el subconjunto de 33 y ya no concuerda
    con la Table I."""
    med = (df94[df94["Horizon"] == ROLLOUT]
           .groupby("Model")["NRMSE"].median().reindex(MODELS).sort_values())
    labels = [f"{PRETTY[m][0]} ({PRETTY[m][1]})" for m in med.index]
    colors = ["#2CA02C" if PRETTY[m][1] == "Local" else "#1F77B4" for m in med.index]

    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    ypos = range(len(med))
    ax.barh(list(ypos), med.to_numpy(), color=colors, edgecolor="black", linewidth=0.5)
    ax.set_yticks(list(ypos)); ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Median NRMSE — 7-day roll-out, 94 plants (% of installed capacity)",
                  fontsize=9)
    for i, v in enumerate(med.to_numpy()):
        ax.text(v + 0.4, i, f"{v:.1f}", va="center", fontsize=8.5)
    ax.axvline(med.min(), ls="--", lw=0.9, color="#2CA02C", alpha=0.6)
    ax.set_xlim(0, med.max() * 1.18)
    handles = [plt.Rectangle((0, 0), 1, 1, fc="#2CA02C", ec="black"),
               plt.Rectangle((0, 0), 1, 1, fc="#1F77B4", ec="black")]
    ax.legend(handles, ["Strictly local", "Global (Cross-Site)"],
              loc="lower right", fontsize=8.5, framealpha=0.95)
    ax.grid(axis="x", ls=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(OUT_FIG / "fig_results_en.png")
    plt.close(fig)
    print(f"[figura] fig_results_en.png regenerada (NRMSE, 94 plantas)")


SEASON_DIR = {"Verano": "_verano", "Otoño": "_otono", "Invierno": "_invierno",
              "Primavera": "_primavera", "Operational": "_operational", "Raw": ""}


def preds_path(model: str, macro: str, est: str, planta: str,
               horizon: str, window: str) -> Path:
    hdir = ("day1" if horizon == DAY1 else "rollout7d") + SEASON_DIR[window]
    return RES / model.lower() / macro / est / planta / hdir / "preds.parquet"


def fig_timeseries(df: pd.DataFrame) -> None:
    """Observado vs. predicho (P50) con banda P5-P95, para 24 h y 7 d.

    Dos columnas (day1 / rollout7d) x dos filas (un modelo global y uno local)
    sobre la MISMA planta y ventana, para que la comparación sea directa.
    """
    # elegir una planta con las 4 combinaciones disponibles y generación no trivial
    cand = df[(df["Window"] == "Operational") & (df["Horizon"] == ROLLOUT)
              & df["rRMSE"].notna()]
    pick = None
    for planta in cand["Planta"].unique():
        rows = df[(df["Planta"] == planta) & (df["Window"] == "Operational")]
        if rows.empty:
            continue
        macro = rows["Macrozona"].iloc[0]
        est = rows["Estacion_Conexion"].iloc[0]
        paths = {(mo, ho): preds_path(mo, macro, est, planta, ho, "Operational")
                 for mo in ("XGB_GLOBAL", "XGB_LOCAL") for ho in (DAY1, ROLLOUT)}
        if all(p.exists() for p in paths.values()):
            pick = (planta, macro, est, paths)
            break
    if pick is None:
        print("[figura] AVISO: no se halló planta con las 4 combinaciones; se omite.")
        return
    planta, macro, est, paths = pick
    print(f"[figura] serie temporal: planta={planta} ({macro}, {est})")

    fig, axes = plt.subplots(2, 2, figsize=(11, 5.4), sharey="row")
    titles = {DAY1: "24-hour horizon (day1)", ROLLOUT: "7-day horizon (rollout7d)"}
    for i, mo in enumerate(("XGB_GLOBAL", "XGB_LOCAL")):
        for j, ho in enumerate((DAY1, ROLLOUT)):
            ax = axes[i, j]
            d = pd.read_parquet(paths[(mo, ho)])
            ax.plot(d["ds"], d["y"], color="black", lw=1.3, label="Observed")
            ax.plot(d["ds"], d["y_pred"], color="#1F77B4", lw=1.3, label="P50 forecast")
            ax.fill_between(d["ds"], d["y_pred_lo_90"], d["y_pred_hi_90"],
                            color="#1F77B4", alpha=0.22, label="P5–P95 interval")
            lbl = f"{PRETTY[mo][0]} ({PRETTY[mo][1]})"
            if i == 0:
                ax.set_title(titles[ho], fontsize=10)
            if j == 0:
                ax.set_ylabel(f"{lbl}\nGeneration (MWh)", fontsize=9)
            ax.tick_params(axis="x", labelsize=7, rotation=30)
            ax.tick_params(axis="y", labelsize=8)
            ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=8, loc="upper left", framealpha=0.9)
    fig.suptitle("Observed vs. predicted generation with 90% prediction intervals "
                 f"(plant: {planta.replace('_', ' ')})", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT_FIG / "fig_timeseries_en.png")
    plt.close(fig)
    print(f"[figura] {OUT_FIG / 'fig_timeseries_en.png'}")


# ----------------------------------------------------------------------------
def main() -> None:
    df = load_summary()
    cap = load_capacity()
    df = add_nrmse(df, cap)

    df94 = build_window94(df)
    audit_missing_operational(df)

    # Sanity check: reproducir las medianas actuales de Table I (subconjunto 33)
    op33 = df[(df["Window"] == "Operational") & (df["Horizon"] == ROLLOUT)]
    chk = op33.groupby("Model")["rRMSE"].median().round(1)
    print("\n[sanity] rRMSE mediano roll-out, ventana Operational (n=33, debe "
          "coincidir con la Table I actual):")
    for m in MODELS:
        print(f"         {m:12} {chk.get(m, float('nan'))}")

    write_tab_main94(df94)
    write_tab_seasonal(df, cap)

    # Tests sobre la vista de 94 plantas, métrica NRMSE (bloque completo)
    mat = paired_matrix(df94, "NRMSE", ROLLOUT)
    res = run_stats(mat, "NRMSE", ROLLOUT)
    write_tab_stats(res, "NRMSE (% of installed capacity), 7-day roll-out")

    fig_results_nrmse(df94)
    fig_timeseries(df)
    print("\n[listo] tablas en docs/tablas/, figuras en docs/figuras/")


if __name__ == "__main__":
    main()
