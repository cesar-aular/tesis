"""Generador de artefactos para el documento de tesis (latex-tesis/).

Produce, a partir de los resultados consolidados de una estrategia
(results/{strategy}/visuals/all_metrics_summary.csv) y del dataset Silver:

- latex-tesis/figuras/*.png : diagramas metodológicos y figuras de resultados.
- latex-tesis/tablas/*.tex  : fragmentos tabulares (\\input) con métricas reales.

Es un artefacto de REPORTE (no forma parte del pipeline de datos productivo):
se ejecuta bajo demanda cuando hay resultados nuevos que volcar al documento.

Uso:  ./.venv/Scripts/python.exe -m src.ml.report_tesis --strategy half
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import pandas as pd

FIG_DIR = Path("latex-tesis/figuras")
TAB_DIR = Path("latex-tesis/tablas")

MODEL_ORDER = ["XGB_LOCAL", "LSTM_LOCAL", "NHITS_LOCAL",
               "XGB_GLOBAL", "LSTM", "NHITS", "TFT", "INFORMER"]
MODEL_LABEL = {
    "XGB_LOCAL": "XGBoost local", "LSTM_LOCAL": "LSTM local",
    "NHITS_LOCAL": "N-HiTS local", "XGB_GLOBAL": "XGBoost global",
    "LSTM": "LSTM global", "NHITS": "N-HiTS global",
    "TFT": "TFT global", "INFORMER": "Informer global",
}
LOCAL_MODELS = {"XGB_LOCAL", "LSTM_LOCAL", "NHITS_LOCAL"}
SEASONS = ["Verano", "Otoño", "Invierno", "Primavera"]

C_LOCAL = "#FF7F0E"   # naranja: paradigma local
C_GLOBAL = "#1F77B4"  # azul: paradigma global


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def _load_summary(strategy: str) -> pd.DataFrame:
    path = Path(f"results/{strategy}/visuals/all_metrics_summary.csv")
    df = pd.read_csv(path)
    df = df[df.Model.isin(MODEL_ORDER)]
    return df


def _fmt(x, dec=1):
    return "--" if pd.isna(x) else f"{x:.{dec}f}"


def _write_tex(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    print(f"[Tesis] {path}")


def _save_fig(fig, name: str):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / name
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[Tesis] {out}")


def _box(ax, xy, w, h, text, fc="#E8F0FE", ec="#1F4E79", fontsize=9, bold=False):
    ax.add_patch(FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.02",
                                fc=fc, ec=ec, lw=1.4))
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center",
            fontsize=fontsize, fontweight="bold" if bold else "normal", wrap=True)


def _arrow(ax, p1, p2, color="#555555"):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=16,
                                 color=color, lw=1.6, shrinkA=2, shrinkB=2))


# --------------------------------------------------------------------------
# Diagramas metodológicos (Capítulo IV)
# --------------------------------------------------------------------------
def fig_solucion():
    """Concepto de la solución: transferencia Cross-Site hacia planta nueva."""
    fig, ax = plt.subplots(figsize=(10, 5.4))
    ax.set_xlim(0, 10); ax.set_ylim(0, 5.4); ax.axis("off")

    for i, y in enumerate((4.3, 3.4, 2.5)):
        _box(ax, (0.3, y), 2.1, 0.7, f"Planta consolidada {i+1}\n(años de historia)",
             fc="#DEEBF7")
    ax.text(1.35, 2.25, "⋮  (N−1 plantas)", ha="center", fontsize=10)

    _box(ax, (3.4, 2.9), 2.6, 1.6,
         "MODELO GLOBAL\nCross-Site\n(TFT / Informer /\nN-HiTS / LSTM)",
         fc="#C6E0B4", bold=True)
    _box(ax, (3.4, 0.6), 2.6, 1.1,
         "Exógenas conocidas:\nclima + geografía\n(sin generación del target)",
         fc="#FFF2CC")
    _box(ax, (7.0, 2.9), 2.6, 1.6,
         "PLANTA NUEVA\n(Cold-Start:\nsin historial propio)", fc="#FCE4D6", bold=True)
    _box(ax, (7.0, 0.6), 2.6, 1.1,
         "Pronóstico day-1 y\nroll-out 7 días\nP5 / P50 / P95", fc="#E2EFDA")

    for y in (4.65, 3.75, 2.85):
        _arrow(ax, (2.4, y), (3.4, min(max(y, 3.1), 4.3)))
    _arrow(ax, (4.7, 2.9), (4.7, 1.7))
    _arrow(ax, (6.0, 3.7), (7.0, 3.7), color="#C00000")
    ax.text(6.5, 3.95, "transferencia\nZero-Shot", ha="center", fontsize=8.5,
            color="#C00000", fontweight="bold")
    _arrow(ax, (8.3, 2.9), (8.3, 1.7))
    fig.suptitle("Transferencia Cross-Site: pre-entrenamiento multi-planta e "
                 "inferencia sobre una planta sin historia", fontsize=11)
    _save_fig(fig, "fig_solucion.png")


def fig_etl_flow():
    """Flujograma del ETL Medallion con reglas de calidad."""
    fig, ax = plt.subplots(figsize=(11.5, 5.0))
    ax.set_xlim(0, 11.5); ax.set_ylim(0, 5.0); ax.axis("off")

    stages = [
        ("LANDING", "CSV crudos\nCEN (generación)\nCR2 (meteorología)\nMaestro instalaciones", "#F2F2F2"),
        ("BRONZE", "Parseo paralelo\nRecorte 2014–2024\nDescarte de −9999\nAgregación multi-unidad\nColapso seguro DST", "#FBE5D6"),
        ("SILVER", "Macrozonas geográficas\nFeatures cíclicas sin/cos\nImputación exógenas\n(ffill/bfill + dummy)\ny: JAMÁS imputada", "#DEEBF7"),
        ("GOLD / DL", "Particiones LOPO\npor planta (validación)\nsilver_dl numérico\nfloat32", "#E2EFDA"),
    ]
    box_w, box_h, y0 = 2.5, 2.4, 2.3
    x = 0.25
    centers = []
    for title, body, color in stages:
        # Caja vacía; título arriba y cuerpo debajo en y DISTINTAS (evita el
        # choque del patrón anterior, que centraba ambos en el mismo punto)
        ax.add_patch(FancyBboxPatch((x, y0), box_w, box_h, boxstyle="round,pad=0.02",
                                    fc=color, ec="#1F4E79", lw=1.4))
        cx = x + box_w / 2
        ax.text(cx, y0 + box_h - 0.33, title, ha="center", va="center",
                fontsize=11, fontweight="bold")
        ax.text(cx, y0 + box_h / 2 - 0.42, body, ha="center", va="center", fontsize=8)
        centers.append(cx)
        if x > 0.5:
            _arrow(ax, (x - 0.28, y0 + box_h / 2), (x, y0 + box_h / 2))
        x += box_w + 0.35

    ax.add_patch(FancyBboxPatch((2.0, 0.35), 7.5, 1.2, boxstyle="round,pad=0.02",
                                fc="#FFF2CC", ec="#BF8F00", lw=1.4))
    ax.text(5.75, 0.95, "Reglas anti-leakage: sin PR = y/capacidad · prior regional "
            "calculado solo con plantas de entrenamiento\n(dentro de cada split LOPO) · "
            "pytest bloquea la ejecución si una regla se rompe (gate TDD)",
            ha="center", va="center", fontsize=8.5)
    _arrow(ax, (centers[2], y0), (5.75, 1.55), color="#C00000")
    fig.suptitle("Pipeline Medallion de preparación de datos y control de calidad",
                 fontsize=12)
    _save_fig(fig, "fig_etl_flow.png")


def fig_lopo():
    """Esquema de rotación Leave-One-Plant-Out."""
    n = 8
    folds = [0, 3, 7]
    fig, axes = plt.subplots(1, len(folds), figsize=(10.5, 2.9))
    for k, (ax, held) in enumerate(zip(axes, folds)):
        for i in range(n):
            color = "#FF7F0E" if i == held else "#1F77B4"
            ax.barh(i, 1, color=color, edgecolor="white", height=0.82)
        ax.set_yticks(range(n))
        ax.set_yticklabels([f"Planta {i+1}" for i in range(n)], fontsize=7.5)
        ax.set_xticks([])
        ax.invert_yaxis()
        title = f"Fold {held+1}: se oculta la planta {held+1}"
        ax.set_title(title, fontsize=9)
        if k == 1:
            ax.set_xlabel("azul = entrenamiento (N−1)  ·  naranja = test (Cold-Start)",
                          fontsize=9)
    fig.suptitle("Validación Leave-One-Plant-Out: la planta objetivo es invisible "
                 "durante el entrenamiento global", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    _save_fig(fig, "fig_lopo.png")


def fig_eval_protocolo():
    """Protocolo de evaluación: contexto sintético + horizontes + ventanas."""
    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.set_xlim(0, 24); ax.set_ylim(0, 4.6); ax.axis("off")

    y0 = 3.3
    ax.add_patch(plt.Rectangle((1, y0), 7, 0.9, fc="#FFF2CC", ec="#BF8F00"))
    ax.text(4.5, y0 + 0.45, "Contexto sintético (168 h)\n"
            r"$y = capacidad \times PR_{regional} \times perfil\ solar$",
            ha="center", va="center", fontsize=9)
    ax.add_patch(plt.Rectangle((8, y0), 2.2, 0.9, fc="#C6E0B4", ec="#538135"))
    ax.text(9.1, y0 + 0.45, "day1\n(24 h)", ha="center", va="center", fontsize=9)
    ax.add_patch(plt.Rectangle((8, 2.2), 14.6, 0.9, fc="#DEEBF7", ec="#1F4E79"))
    ax.text(15.3, 2.65, "rollout7d: 7 días autorregresivos "
            "(realimenta la mediana predicha, nunca la realidad)",
            ha="center", va="center", fontsize=9)
    _arrow(ax, (9.1, y0), (9.1, 3.1))

    ax.annotate("", xy=(22.8, 1.75), xytext=(1, 1.75),
                arrowprops=dict(arrowstyle="->", lw=1.2))
    ax.text(22.6, 1.5, "tiempo (grilla horaria continua)", fontsize=8.5, ha="right")

    ax.text(1, 1.15, "Inicio de la ventana (análisis de sensibilidad):",
            fontsize=9, fontweight="bold")
    ax.text(1, 0.75, "• Raw: primera hora registrada (incluye rampa de puesta en marcha)"
            "     • Operacional: primera producción sostenida\n"
            "• Estacionales: primera ventana sostenida que inicia en Verano / Otoño / "
            "Invierno / Primavera (historia posterior de la misma planta)",
            fontsize=8.5, va="top")
    fig.suptitle("Protocolo de evaluación Cold-Start: contexto sintético, horizontes "
                 "y ventanas de sensibilidad", fontsize=11)
    _save_fig(fig, "fig_eval_protocolo.png")


def fig_medallion_impl():
    """Pipeline implementado con volúmenes reales (Capítulo V)."""
    fig, ax = plt.subplots(figsize=(11, 3.6))
    ax.set_xlim(0, 11); ax.set_ylim(0, 3.6); ax.axis("off")
    stages = [
        ("LANDING", "139 archivos CEN\n+ estaciones CR2\n2014–2024", "#F2F2F2"),
        ("BRONZE", "CSV por planta\ny por estación\nmeteorológica", "#FBE5D6"),
        ("SILVER", "silver_unified\n94 plantas\n≈ 4,37 M filas", "#DEEBF7"),
        ("GOLD / DL", "94 particiones LOPO\nsilver_dl float32\n(solo numérico)", "#E2EFDA"),
        ("ML / REPORTE", "8 modelos ×\nventanas × horizontes\nmétricas + gráficas", "#FCE4D6"),
    ]
    x = 0.3
    for title, body, color in stages:
        _box(ax, (x, 1.2), 1.85, 1.6, "", fc=color)
        ax.text(x + 0.92, 2.45, title, ha="center", fontsize=9.5, fontweight="bold")
        ax.text(x + 0.92, 1.85, body, ha="center", va="center", fontsize=8)
        if x > 0.5:
            _arrow(ax, (x - 0.35, 2.0), (x, 2.0))
        x += 2.2
    ax.text(5.5, 0.55, "Orquestador único (src/orchestrator.py) · gate TDD (pytest) "
            "previo a cada corrida · idempotencia con marcadores de completitud",
            ha="center", fontsize=8.5, style="italic")
    fig.suptitle("Pipeline Medallion implementado (volúmenes reales tras depuración)",
                 fontsize=11)
    _save_fig(fig, "fig_medallion_impl.png")


def fig_mapa_muestra():
    """Distribución de la muestra por macrozona (norte -> sur). Cap. I."""
    silver = Path("data/silver/silver_unified.parquet")
    if not silver.exists():
        print("[Tesis] silver_unified no disponible; fig_mapa omitida.")
        return
    df = pd.read_parquet(silver, columns=["unique_id", "macrozona", "potencia_neta_mw"])
    plantas = df.drop_duplicates("unique_id")
    orden = ["Norte Grande", "Norte Chico", "Zona Central", "Zona Sur"]
    g = plantas.groupby("macrozona", observed=True)
    counts = g.size().reindex(orden)
    caps = g["potencia_neta_mw"].sum().reindex(orden)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    ys = np.arange(len(orden))
    ax1.barh(ys, counts, color="#1F77B4")
    ax1.set_yticks(ys); ax1.set_yticklabels(orden)
    ax1.invert_yaxis()
    ax1.set_xlabel("Número de plantas")
    ax1.set_title("Plantas por macrozona")
    for y, v in zip(ys, counts):
        ax1.text(v + 0.5, y, str(int(v)), va="center", fontsize=9)
    ax2.barh(ys, caps, color="#FF7F0E")
    ax2.set_xlabel("Capacidad instalada (MW)")
    ax2.set_title("Capacidad por macrozona")
    for y, v in zip(ys, caps):
        ax2.text(v + 8, y, f"{v:,.0f}", va="center", fontsize=9)
    for ax in (ax1, ax2):
        ax.grid(axis="x", alpha=0.3)
    fig.suptitle("Muestra de plantas fotovoltaicas agrupadas por macrozona "
                 "(ordenadas de norte a sur)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    _save_fig(fig, "fig_mapa_muestra.png")


def fig_tft_arch():
    """Esquema de la arquitectura TFT (Cap. II, marco teórico)."""
    fig, ax = plt.subplots(figsize=(9.5, 6.2))
    ax.set_xlim(0, 9.5); ax.set_ylim(0, 6.2); ax.axis("off")

    _box(ax, (0.4, 0.4), 2.5, 0.8, "Covariables estáticas\n(macrozona, capacidad)",
         fc="#FCE4D6")
    _box(ax, (3.5, 0.4), 2.5, 0.8, "Pasado observado\n(y, clima histórico)",
         fc="#DEEBF7")
    _box(ax, (6.6, 0.4), 2.5, 0.8, "Futuro conocido\n(calendario, pronóstico)",
         fc="#E2EFDA")

    _box(ax, (2.0, 1.7), 5.5, 0.65,
         "Redes de Selección de Variables (VSN) + codificadores estáticos (GRN)",
         fc="#FFF2CC")
    _box(ax, (2.0, 2.8), 2.6, 0.65, "Codificador LSTM\n(secuencia pasada)", fc="#DEEBF7")
    _box(ax, (4.9, 2.8), 2.6, 0.65, "Decodificador LSTM\n(horizonte futuro)", fc="#E2EFDA")
    _box(ax, (2.0, 3.9), 5.5, 0.65,
         "Atención multi-cabezal interpretable (dependencias de largo plazo)",
         fc="#D9D2E9")
    _box(ax, (2.0, 5.0), 5.5, 0.65,
         "Compuertas GRN + salida cuantílica  →  P5 / P50 / P95", fc="#C6E0B4")

    for x in (1.65, 4.75, 7.85):
        _arrow(ax, (x, 1.2), (min(max(x, 2.6), 6.9), 1.7))
    _arrow(ax, (3.3, 2.35), (3.3, 2.8)); _arrow(ax, (6.2, 2.35), (6.2, 2.8))
    _arrow(ax, (3.3, 3.45), (3.3, 3.9)); _arrow(ax, (6.2, 3.45), (6.2, 3.9))
    _arrow(ax, (4.75, 4.55), (4.75, 5.0))
    fig.suptitle("Arquitectura del Temporal Fusion Transformer (Lim et al., 2021)",
                 fontsize=11)
    _save_fig(fig, "fig_tft_arch.png")


def fig_cross_site():
    """Local aislado vs. Cross-Site (Cap. II, marco teórico)."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax in axes:
        ax.set_xlim(0, 5.5); ax.set_ylim(0, 4.4); ax.axis("off")

    ax = axes[0]
    ax.set_title("(a) Entrenamiento local aislado", fontsize=10.5)
    for i, y in enumerate((3.3, 2.0, 0.7)):
        _box(ax, (0.3, y), 1.9, 0.8, f"Planta {i+1}\n(su propia historia)", fc="#DEEBF7")
        _box(ax, (3.3, y), 1.9, 0.8, f"Modelo local {i+1}", fc="#FCE4D6")
        _arrow(ax, (2.2, y + 0.4), (3.3, y + 0.4))
    ax.text(2.75, 0.15, "una planta nueva NO tiene historia → sin modelo",
            ha="center", fontsize=8.5, color="#C00000")

    ax = axes[1]
    ax.set_title("(b) Cross-Site Learning (global)", fontsize=10.5)
    for i, y in enumerate((3.3, 2.0, 0.7)):
        _box(ax, (0.3, y), 1.9, 0.8, f"Planta {i+1}\n(su propia historia)", fc="#DEEBF7")
    _box(ax, (2.9, 1.8), 1.6, 1.2, "MODELO\nGLOBAL\núnico", fc="#C6E0B4", bold=True)
    for y in (3.7, 2.4, 1.1):
        _arrow(ax, (2.2, y), (2.9, min(max(y, 2.1), 2.7)))
    _box(ax, (4.0, 3.3), 1.4, 0.8, "Planta\nNUEVA", fc="#FCE4D6", bold=True)
    _arrow(ax, (4.1, 2.9), (4.6, 3.3), color="#C00000")
    ax.text(4.9, 2.75, "Zero-Shot", fontsize=8.5, color="#C00000", ha="center")
    fig.suptitle("Paradigma local aislado versus transferencia de conocimiento "
                 "inter-plantas", fontsize=11)
    _save_fig(fig, "fig_cross_site.png")


# --------------------------------------------------------------------------
# Figuras de resultados (Capítulo V)
# --------------------------------------------------------------------------
def fig_resultados_global(df: pd.DataFrame):
    """Barras: rRMSE mediano por modelo y horizonte (ventana Operacional)."""
    op = df[df.Window == "Operational"]
    med = (op.pivot_table(index="Model", columns="Horizon", values="rRMSE",
                          aggfunc="median").reindex(MODEL_ORDER))
    fig, ax = plt.subplots(figsize=(10, 4.6))
    xs = np.arange(len(med))
    w = 0.38
    ax.bar(xs - w / 2, med["Day 1"], w, label="Day 1 (24 h)", color="#4C9BE8")
    ax.bar(xs + w / 2, med["7-Day Rollout"], w, label="Roll-out 7 días", color="#2B5D8C")
    for i, m in enumerate(med.index):
        if m in LOCAL_MODELS:
            ax.axvspan(i - 0.5, i + 0.5, color=C_LOCAL, alpha=0.07)
    ax.set_xticks(xs)
    ax.set_xticklabels([MODEL_LABEL[m].replace(" ", "\n") for m in med.index],
                       fontsize=8.5)
    ax.set_ylabel("rRMSE mediano (%)")
    ax.set_title("Error relativo por arquitectura y horizonte — ventana Operacional "
                 f"(n = {op[op.Horizon == '7-Day Rollout'].Planta.nunique()} plantas)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.text(0.5, -0.22, "fondo naranja = paradigma local (requiere historia propia de la planta)",
            transform=ax.transAxes, ha="center", va="top", fontsize=8.5, color=C_LOCAL)
    _save_fig(fig, "fig_resultados_global.png")


def fig_estacional(df: pd.DataFrame):
    """Sensibilidad estacional (rollout7d): 4 modelos representativos."""
    keep = ["XGB_LOCAL", "XGB_GLOBAL", "LSTM", "TFT"]
    sea = df[(df.Horizon == "7-Day Rollout") & df.Window.isin(SEASONS)
             & df.Model.isin(keep)]
    med = sea.pivot_table(index="Window", columns="Model", values="rRMSE",
                          aggfunc="median").reindex(SEASONS)[keep]
    fig, ax = plt.subplots(figsize=(9.5, 4.4))
    xs = np.arange(len(SEASONS))
    w = 0.2
    colors = {"XGB_LOCAL": C_LOCAL, "XGB_GLOBAL": "#2B5D8C",
              "LSTM": "#4C9BE8", "TFT": "#7FB3D5"}
    for j, m in enumerate(keep):
        ax.bar(xs + (j - 1.5) * w, med[m], w, label=MODEL_LABEL[m], color=colors[m])
    ax.set_xticks(xs); ax.set_xticklabels(SEASONS)
    ax.set_ylabel("rRMSE mediano (%)")
    ax.set_title("Sensibilidad estacional del Cold-Start (roll-out 7 días)")
    ax.legend(fontsize=8.5)
    ax.grid(axis="y", alpha=0.3)
    _save_fig(fig, "fig_estacional.png")


def fig_ejemplo_rollout(df: pd.DataFrame, strategy: str):
    """Roll-out real vs. predicho (con bandas) para una planta representativa."""
    cand = df[(df.Model == "LSTM") & (df.Horizon == "7-Day Rollout")
              & (df.Window == "Operational")].sort_values("rRMSE")
    if cand.empty:
        print("[Tesis] Sin candidatos para figura de ejemplo. Omitida.")
        return
    # Preferir un ejemplo donde el forzamiento fisico nocturno opero (pred=0 de
    # noche); si la radiacion interpolada quedo >=5 en huecos, el piso nocturno
    # del modelo queda visible y confunde al lector.
    row = preds = None
    for _, c in cand.head(8).iterrows():
        pq = Path(f"results/{strategy}/lstm/{c['Macrozona']}/{c['Estacion_Conexion']}/"
                  f"{c['Planta']}/rollout7d_operational/preds.parquet")
        if not pq.exists():
            continue
        p = pd.read_parquet(pq).sort_values("ds")
        if (p["y_pred"] == 0).mean() >= 0.2:
            row, preds = c, p
            break
        if row is None:
            row, preds = c, p  # fallback: el de menor rRMSE disponible
    if preds is None:
        print("[Tesis] Sin preds.parquet disponibles. Figura de ejemplo omitida.")
        return
    planta, macro = row["Planta"], row["Macrozona"]
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.plot(preds["ds"], preds["y"], color="black", lw=1.4, label="Generación real")
    ax.plot(preds["ds"], preds["y_pred"], color="#1F77B4", lw=1.3, ls="--",
            label="LSTM global (mediana)")
    if "y_pred_lo_90" in preds.columns:
        ax.fill_between(preds["ds"], preds["y_pred_lo_90"], preds["y_pred_hi_90"],
                        color="#1F77B4", alpha=0.18, label="Intervalo 90 % (P5–P95)")
    ax.set_ylabel("Generación (MWh)")
    ax.set_title(f"Roll-out Cold-Start de 7 días — planta {planta} "
                 f"({macro}, ventana operacional) · rRMSE = {row['rRMSE']:.1f} %")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    _save_fig(fig, "fig_ejemplo_rollout.png")


# --------------------------------------------------------------------------
# Tablas .tex
# --------------------------------------------------------------------------
def tab_dataset():
    silver = Path("data/silver/silver_unified.parquet")
    if not silver.exists():
        print("[Tesis] silver_unified no disponible; tabla dataset omitida.")
        return
    df = pd.read_parquet(silver, columns=["unique_id", "ds", "macrozona",
                                          "potencia_neta_mw", "y"])
    g = df.groupby("macrozona", observed=True)
    caps = df.drop_duplicates("unique_id").groupby("macrozona", observed=True)["potencia_neta_mw"]
    rows = []
    for mz in sorted(df["macrozona"].dropna().unique()):
        sub = g.get_group(mz)
        rows.append((mz, sub["unique_id"].nunique(), caps.sum()[mz],
                     len(sub) / 1e6, str(sub["ds"].min().date()),
                     str(sub["ds"].max().date())))
    total = ("\\textbf{Total}", df["unique_id"].nunique(),
             df.drop_duplicates("unique_id")["potencia_neta_mw"].sum(),
             len(df) / 1e6, str(df["ds"].min().date()), str(df["ds"].max().date()))
    lines = [r"\begin{tabular}{lrrrcc}", r"\toprule",
             r"Macrozona & Plantas & Capacidad (MW) & Obs. (millones) & Desde & Hasta \\",
             r"\midrule"]
    for mz, n, cap, obs, d0, d1 in rows:
        lines.append(f"{mz} & {n} & {cap:,.0f} & {obs:.2f} & {d0} & {d1} \\\\")
    lines += [r"\midrule",
              f"{total[0]} & {total[1]} & {total[2]:,.0f} & {total[3]:.2f} & {total[4]} & {total[5]} \\\\",
              r"\bottomrule", r"\end{tabular}"]
    _write_tex(TAB_DIR / "tab_dataset.tex", "\n".join(lines))


def tab_hparams(strategy: str):
    rows = []
    # Decisión de diseño: precisión completa en todas las arquitecturas
    fp32 = {"lstm": "fp32", "nhits": "fp32", "tft": "fp32", "informer": "fp32"}
    hidden_name = {"lstm": "encoder\\_hidden\\_size", "nhits": "mlp\\_units",
                   "tft": "hidden\\_size", "informer": "hidden\\_size"}
    for m in ("lstm", "nhits", "tft", "informer"):
        p = Path(f"models/{strategy}/{m}/best_params.json")
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        rows.append((MODEL_LABEL[m.upper()].replace(" global", ""),
                     d.get("input_size", 168), hidden_name[m],
                     d.get("hidden_size", 64),
                     f"{d.get('learning_rate', 1e-3):.2e}", fp32[m]))
    lines = [r"\begin{tabular}{lclccc}", r"\toprule",
             r"Arquitectura & Ventana entrada (h) & Parám. de dimensión & Valor & "
             r"Tasa aprendizaje & Precisión \\", r"\midrule"]
    for r in rows:
        lines.append(" & ".join(str(v) for v in r) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    _write_tex(TAB_DIR / "tab_hparams.tex", "\n".join(lines))


def tab_resultados(df: pd.DataFrame):
    """Tabla principal: medianas por modelo x horizonte, Raw y Operacional."""
    def block(window):
        sub = df[df.Window == window]
        piv = sub.pivot_table(index="Model", columns="Horizon",
                              values=["rRMSE", "sMAPE", "RMSE"], aggfunc="median")
        n = sub[sub.Horizon == "7-Day Rollout"].groupby("Model").Planta.nunique()
        return piv.reindex(MODEL_ORDER), n

    for window, fname in (("Raw", "tab_resultados_raw.tex"),
                          ("Operational", "tab_resultados_operacional.tex")):
        piv, n = block(window)
        lines = [r"\begin{tabular}{lrrrrrrr}", r"\toprule",
                 r" & \multicolumn{3}{c}{Day 1 (24 h)} & "
                 r"\multicolumn{3}{c}{Roll-out 7 días} & \\",
                 r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
                 r"Modelo & RMSE & sMAPE & rRMSE & RMSE & sMAPE & rRMSE & $n$ \\",
                 r" & (MWh) & (\%) & (\%) & (MWh) & (\%) & (\%) & \\",
                 r"\midrule"]
        for m in MODEL_ORDER:
            if m not in piv.index or pd.isna(piv.loc[m].get(("rRMSE", "Day 1"), np.nan)):
                continue
            r = piv.loc[m]
            lines.append(
                f"{MODEL_LABEL[m]} & {_fmt(r[('RMSE', 'Day 1')], 2)} & "
                f"{_fmt(r[('sMAPE', 'Day 1')])} & {_fmt(r[('rRMSE', 'Day 1')])} & "
                f"{_fmt(r[('RMSE', '7-Day Rollout')], 2)} & "
                f"{_fmt(r[('sMAPE', '7-Day Rollout')])} & "
                f"{_fmt(r[('rRMSE', '7-Day Rollout')])} & {n.get(m, 0)} \\\\")
        lines += [r"\bottomrule", r"\end{tabular}"]
        _write_tex(TAB_DIR / fname, "\n".join(lines))


def tab_estacional(df: pd.DataFrame):
    """Eje estacional principal: zero-shot por estación con todas las plantas.

    Una tabla por horizonte (day1 y rollout7d) + fila de n (plantas evaluadas
    por estación, rango entre modelos)."""
    for horizon, fname in (("Day 1", "tab_estacional_day1.tex"),
                           ("7-Day Rollout", "tab_estacional.tex")):
        sea = df[(df.Horizon == horizon) & df.Window.isin(SEASONS)]
        if sea.empty:
            continue
        piv = sea.pivot_table(index="Model", columns="Window", values="rRMSE",
                              aggfunc="median").reindex(MODEL_ORDER)
        piv = piv[[s for s in SEASONS if s in piv.columns]]
        ns = sea.groupby(["Model", "Window"]).Planta.nunique().unstack()
        lines = [r"\begin{tabular}{lrrrr}", r"\toprule",
                 r"Modelo & Verano & Otoño & Invierno & Primavera \\", r"\midrule"]
        for m in MODEL_ORDER:
            if m not in piv.index or piv.loc[m].isna().all():
                continue
            r = piv.loc[m]
            vals = " & ".join(_fmt(r.get(s)) for s in SEASONS)
            lines.append(f"{MODEL_LABEL[m]} & {vals} \\\\")
        n_cells = []
        for s in SEASONS:
            if s in ns.columns:
                mn, mx = int(ns[s].min()), int(ns[s].max())
                n_cells.append(str(mn) if mn == mx else f"{mn}--{mx}")
            else:
                n_cells.append("--")
        lines += [r"\midrule",
                  r"\textit{n plantas} & " + " & ".join(n_cells) + r" \\",
                  r"\bottomrule", r"\end{tabular}"]
        _write_tex(TAB_DIR / fname, "\n".join(lines))


def tab_probabilistico(df: pd.DataFrame):
    cols = ["Coverage_90", "Coverage_90_diurna", "Pinball_P05", "Pinball_P95"]
    have = [c for c in cols if c in df.columns]
    dl = df[(df.Horizon == "7-Day Rollout") & (df.Window == "Operational")
            & df.Coverage_90.notna()]
    med = dl.groupby("Model")[have].median()
    med = med.reindex([m for m in MODEL_ORDER if m in med.index])
    diurna = "Coverage_90_diurna" in have
    header = (r"Modelo & Cobertura 90\,\% & Cobertura diurna & Pinball P05 & Pinball P95 \\"
              if diurna else r"Modelo & Cobertura 90\,\% & Pinball P05 & Pinball P95 \\")
    sub = (r" & (con noche) & (horas $y>0$) & (MWh) & (MWh) \\"
           if diurna else r" & (ideal $\approx 0{,}90$) & (MWh) & (MWh) \\")
    colspec = "lcccc" if diurna else "lccc"
    lines = [r"\begin{tabular}{" + colspec + "}", r"\toprule", header, sub, r"\midrule"]
    for m, r in med.iterrows():
        if diurna:
            lines.append(f"{MODEL_LABEL[m]} & {_fmt(r['Coverage_90'], 2)} & "
                         f"{_fmt(r['Coverage_90_diurna'], 2)} & "
                         f"{_fmt(r['Pinball_P05'], 3)} & {_fmt(r['Pinball_P95'], 3)} \\\\")
        else:
            lines.append(f"{MODEL_LABEL[m]} & {_fmt(r['Coverage_90'], 2)} & "
                         f"{_fmt(r['Pinball_P05'], 3)} & {_fmt(r['Pinball_P95'], 3)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    _write_tex(TAB_DIR / "tab_probabilistico.tex", "\n".join(lines))


def _count_params(strategy: str) -> dict:
    """Número de parámetros por modelo.

    - DL: parámetros entrenables de torch, reconstruyendo la arquitectura con
      los hiperparámetros óptimos cacheados (sin entrenar). Los locales
      comparten arquitectura con su global (reusan sus hiperparámetros).
    - XGB: nodos totales del ensamble (splits + hojas de todos los árboles y
      cuantiles) — la medida de complejidad análoga en modelos de árboles.
    """
    import joblib
    from src.ml.utils.dl_models import MODEL_SPECS, _build

    counts = {}
    for m, spec in MODEL_SPECS.items():
        p = Path(f"models/{strategy}/{m}/best_params.json")
        params = json.loads(p.read_text()) if p.exists() else {}
        try:
            obj = _build(spec, hidden=params.get("hidden_size", 64),
                         lr=1e-3, max_steps=1, batch_size=1,
                         windows_batch_size=1,
                         input_size=params.get("input_size", 168))
            n = int(sum(t.numel() for t in obj.parameters()))
            counts[m.upper()] = n
            counts[f"{m.upper()}_LOCAL"] = n  # misma arquitectura/hparams
        except Exception as e:
            print(f"[Tesis] No se pudo contar parámetros de {m}: {e}")

    def _xgb_nodes(path: Path):
        model = joblib.load(path)
        return len(model.get_booster().trees_to_dataframe())

    gpath = Path("models/xgb_global/xgb_global_model.joblib")
    if gpath.exists():
        counts["XGB_GLOBAL"] = _xgb_nodes(gpath)
    locales = sorted(Path("models/xgb_local").glob("xgb_local_*.joblib"))
    if locales:
        sample = [_xgb_nodes(p) for p in locales[:10]]  # muestra representativa
        counts["XGB_LOCAL"] = int(np.median(sample))
    return counts


def tab_params(df: pd.DataFrame, strategy: str):
    """Complejidad vs desempeño: #parámetros y rRMSE mediano por modelo."""
    counts = _count_params(strategy)
    if not counts:
        return
    roll = df[(df.Horizon == "7-Day Rollout") & (df.Window == "Raw")]
    med = roll.groupby("Model")["rRMSE"].median()

    lines = [r"\begin{tabular}{lrrc}", r"\toprule",
             r"Modelo & Parámetros & rRMSE roll-out (\%) & Tipo de parámetro \\",
             r"\midrule"]
    tipo = {"XGB": "nodos de árbol", "DL": "pesos entrenables"}
    for m in MODEL_ORDER:
        if m not in counts:
            continue
        t = tipo["XGB"] if m.startswith("XGB") else tipo["DL"]
        lines.append(f"{MODEL_LABEL[m]} & {counts[m]:,} & "
                     f"{_fmt(med.get(m))} & {t} \\\\".replace(",", "."))
    lines += [r"\bottomrule", r"\end{tabular}"]
    _write_tex(TAB_DIR / "tab_params.tex", "\n".join(lines))

    # Scatter complejidad (log) vs error
    fig, ax = plt.subplots(figsize=(9, 5))
    for m in MODEL_ORDER:
        if m not in counts or m not in med.index or pd.isna(med[m]):
            continue
        color = C_LOCAL if m in LOCAL_MODELS else C_GLOBAL
        ax.scatter(counts[m], med[m], s=90, color=color, zorder=3)
        ax.annotate(MODEL_LABEL[m], (counts[m], med[m]),
                    textcoords="offset points", xytext=(8, 6), fontsize=8.5)
    ax.set_xscale("log")
    ax.set_xlabel("Número de parámetros (escala log)")
    ax.set_ylabel("rRMSE mediano (%) — roll-out 7 días, ventana Raw")
    ax.set_title("Complejidad del modelo vs. error de pronóstico Cold-Start")
    ax.grid(alpha=0.3)
    ax.scatter([], [], color=C_GLOBAL, label="paradigma global")
    ax.scatter([], [], color=C_LOCAL, label="paradigma local")
    ax.legend()
    _save_fig(fig, "fig_complejidad.png")


def tab_anexo_plantas(df: pd.DataFrame):
    """Anexo: rRMSE rollout7d por planta (ventana Raw) para modelos clave."""
    keep = ["XGB_LOCAL", "XGB_GLOBAL", "LSTM", "TFT"]
    sub = df[(df.Horizon == "7-Day Rollout") & (df.Window == "Raw")
             & df.Model.isin(keep)]
    piv = sub.pivot_table(index=["Macrozona", "Planta"], columns="Model",
                          values="rRMSE", aggfunc="median")[keep]
    lines = [r"{\small", r"\begin{longtable}{llrrrr}", r"\toprule",
             r"Macrozona & Planta & XGB local & XGB global & LSTM & TFT \\",
             r"\midrule", r"\endhead"]
    for (mz, pl), r in piv.sort_index().iterrows():
        pl_tex = str(pl).replace("_", r"\_")
        vals = " & ".join(_fmt(r[m]) for m in keep)
        lines.append(f"{mz} & {pl_tex} & {vals} \\\\")
    lines += [r"\bottomrule", r"\end{longtable}", r"}"]
    _write_tex(TAB_DIR / "tab_anexo_plantas.tex", "\n".join(lines))


def tab_anexo_imputacion():
    """Horas-planta imputadas por variable exógena y macrozona (log de Silver)."""
    log = Path("data/silver/imputation_log.csv")
    if not log.exists():
        print("[Tesis] imputation_log.csv no disponible; tabla omitida.")
        return
    df = pd.read_csv(log)
    piv = (df.groupby(["variable_imputada", "macrozona"]).size()
             .unstack(fill_value=0))
    piv["Total"] = piv.sum(axis=1)
    zonas = list(piv.columns)
    lines = [r"\begin{tabular}{l" + "r" * len(zonas) + "}", r"\toprule",
             "Variable imputada & " + " & ".join(zonas) + r" \\",
             r"\midrule"]
    for var, r in piv.iterrows():
        vals = " & ".join(f"{int(v):,}" for v in r)
        lines.append(f"{str(var).replace('_', chr(92) + '_')} & {vals} \\\\")
    tot = piv.sum()
    lines += [r"\midrule",
              r"\textbf{Total} & " + " & ".join(f"{int(v):,}" for v in tot) + r" \\",
              r"\bottomrule", r"\end{tabular}"]
    _write_tex(TAB_DIR / "tab_anexo_imputacion.tex", "\n".join(lines))


# --------------------------------------------------------------------------
def main(strategy: str = "half"):
    df = _load_summary(strategy)
    print(f"[Tesis] {len(df)} filas de métricas ({strategy}).")

    # Diagramas conceptuales y metodológicos (caps. I, II y IV)
    fig_mapa_muestra()
    fig_tft_arch()
    fig_cross_site()
    fig_solucion()
    fig_etl_flow()
    fig_lopo()
    fig_eval_protocolo()
    fig_medallion_impl()

    # Resultados (cap. V)
    fig_resultados_global(df)
    fig_estacional(df)
    fig_ejemplo_rollout(df, strategy)

    tab_dataset()
    tab_hparams(strategy)
    tab_resultados(df)
    tab_estacional(df)
    tab_probabilistico(df)
    tab_params(df, strategy)
    tab_anexo_plantas(df)
    tab_anexo_imputacion()
    print("[Tesis] Artefactos generados en latex-tesis/{figuras,tablas}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", default="half", choices=["toy", "half", "total"])
    args = parser.parse_args()
    main(args.strategy)
