"""Figuras propias de la CHARLA (defensa/conferencia), en español.

Genera dos figuras que NO existen en la tesis, sin tocar ninguna figura de la
tesis (`fig_resultados_global.png` la usa chp5.tex y está congelada):

1. fig_mapa_plantas.png       - ubicación real de las plantas en Chile
2. fig_resultados_nrmse_es.png - ranking por NRMSE sobre las 94 plantas

--- Mapa ---

Las coordenadas viven en `data/landing/estaciones_solar.json`, que es NDJSON:
718 Feature GeoJSON, uno por línea (json.load() del archivo completo FALLA).
Geometría Point con coordinates = [lon, lat, alt].

Esas coordenadas NO sobreviven al ETL (silver solo conserva `macrozona`), así que
aquí se hace el join por nombre normalizado contra los 94 `unique_id` evaluados.

Validación: la latitud casada debe concordar con la macrozona ya asignada en
silver, usando los mismos cortes que `bronze_to_silver.get_macrozona_by_latitud`.

Salida: latex-tesis/figuras/fig_mapa_plantas.png

Uso:  ./.venv/Scripts/python.exe latex-tesis/make_mapa_plantas.py
"""
from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
GEO = ROOT / "data" / "landing" / "estaciones_solar.json"
SILVER = ROOT / "data" / "silver" / "silver_unified.parquet"
OUT = Path(__file__).resolve().parent / "figuras" / "fig_mapa_plantas.png"

plt.rcParams.update({"font.family": "DejaVu Sans", "savefig.dpi": 200,
                     "savefig.bbox": "tight"})

# Cortes de latitud de bronze_to_silver.get_macrozona_by_latitud
LAT_CUTS = [(-26.0, "Norte Grande"), (-32.1, "Norte Chico"),
            (-36.2, "Zona Central"), (-44.0, "Zona Sur")]
ORDER = ["Norte Grande", "Norte Chico", "Zona Central", "Zona Sur"]
COLORS = {"Norte Grande": "#D95F02", "Norte Chico": "#E7B416",
          "Zona Central": "#1F77B4", "Zona Sur": "#2CA02C"}

# Tokens de segmento/genéricos que no identifican a la planta
STOP = {"pfv", "pmgd", "pmg", "psf", "fv", "pv", "solar", "parque", "planta",
        "fotovoltaica", "fotovoltaico", "central", "el", "la", "los", "las",
        "de", "del", "y"}


BOUNDARY = Path(__file__).resolve().parent / "chile_boundary.geojson"
REGIONS = Path(__file__).resolve().parent / "chile_regions.geojson"


def draw_chile(ax) -> None:
    """Dibuja Chile continental: relleno del país + fronteras regionales.

    Los GeoJSON (Natural Earth, dominio público) viven en el repo, así que la
    figura es reproducible sin red y sin geopandas/cartopy.
    """
    if BOUNDARY.exists():
        geo = json.loads(BOUNDARY.read_text(encoding="utf-8"))
        polys = geo["geometry"]["coordinates"]
        for poly in polys:
            for i, ring in enumerate(poly):
                xs = [c[0] for c in ring]
                ys = [c[1] for c in ring]
                if i == 0:                      # anillo exterior: tierra
                    ax.fill(xs, ys, facecolor="#F4F4F1", edgecolor="#6E6E6E",
                            linewidth=0.9, zorder=1)
                else:                           # huecos (lagos)
                    ax.fill(xs, ys, facecolor="white", edgecolor="#6E6E6E",
                            linewidth=0.4, zorder=1)
        print(f"[mapa] contorno de Chile ({len(polys)} polígonos)")
    else:
        print(f"[aviso] falta {BOUNDARY.name}; sin mapa de fondo")

    if REGIONS.exists():
        reg = json.loads(REGIONS.read_text(encoding="utf-8"))["regions"]
        for r in reg:
            for poly in r["coordinates"]:
                for ring in poly:              # bordes de región (líneas finas)
                    ax.plot([c[0] for c in ring], [c[1] for c in ring],
                            color="#A9A9A9", lw=0.45, zorder=2, solid_joinstyle="round")
        print(f"[mapa] fronteras de {len(reg)} regiones")
    else:
        print(f"[aviso] falta {REGIONS.name}; sin fronteras regionales")


def macrozona_by_lat(lat: float) -> str:
    for cut, name in LAT_CUTS:
        if lat > cut:
            return name
    return "Zona Austral"


def norm_tokens(s: str) -> frozenset[str]:
    """Conjunto de tokens significativos, sin acentos ni puntuación."""
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    s = re.sub(r"^\s*\w*\d+\w*_", " ", s)        # prefijo '239_' / 'EXT_001_'
    s = re.sub(r"[^A-Za-z0-9]+", " ", s).lower()
    return frozenset(t for t in s.split() if t and t not in STOP)


def load_geo() -> list[dict]:
    feats = []
    with GEO.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            f = json.loads(line)
            lon, lat = f["geometry"]["coordinates"][:2]
            feats.append({"name": f["properties"]["Name"],
                          "tokens": norm_tokens(f["properties"]["Name"]),
                          "lon": float(lon), "lat": float(lat)})
    print(f"[geo] {len(feats)} features leídos de estaciones_solar.json (NDJSON)")
    return feats


def main() -> None:
    plants = (pd.read_parquet(SILVER, columns=["unique_id", "macrozona",
                                               "potencia_neta_mw"])
              .drop_duplicates("unique_id"))
    print(f"[silver] {len(plants)} plantas evaluadas")
    feats = load_geo()

    rows, ambiguous, unmatched = [], [], []
    for _, p in plants.iterrows():
        tk = norm_tokens(p["unique_id"])
        hits = [f for f in feats if f["tokens"] == tk]
        if not hits:  # relajar: subconjunto no vacío
            hits = [f for f in feats
                    if tk and (tk <= f["tokens"] or f["tokens"] <= tk)]
        if len(hits) == 1:
            f = hits[0]
            rows.append({"unique_id": p["unique_id"], "lon": f["lon"],
                         "lat": f["lat"], "macrozona": p["macrozona"],
                         "mw": p["potencia_neta_mw"]})
        elif len(hits) > 1:
            # desempate: el que concuerde con la macrozona de silver
            ok = [f for f in hits if macrozona_by_lat(f["lat"]) == p["macrozona"]]
            if len(ok) >= 1:
                f = ok[0]
                rows.append({"unique_id": p["unique_id"], "lon": f["lon"],
                             "lat": f["lat"], "macrozona": p["macrozona"],
                             "mw": p["potencia_neta_mw"]})
            else:
                ambiguous.append(p["unique_id"])
        else:
            unmatched.append(p["unique_id"])

    df = pd.DataFrame(rows)
    print(f"[join] casadas={len(df)}/{len(plants)} | ambiguas={len(ambiguous)} "
          f"| sin match={len(unmatched)}")
    if unmatched:
        print(f"       sin match: {unmatched}")

    # Validación: latitud vs. macrozona declarada en silver
    df["macro_lat"] = df["lat"].apply(macrozona_by_lat)
    bad = df[df["macro_lat"] != df["macrozona"]]
    print(f"[validación] coincide macrozona(lat) con silver en "
          f"{len(df) - len(bad)}/{len(df)} plantas")
    if len(bad):
        print(f"             discrepancias: {bad['unique_id'].tolist()[:5]}")

    # ---------------- figura ----------------
    fig, ax = plt.subplots(figsize=(4.3, 8.4))

    # Contorno real de Chile continental (Natural Earth 50m, dominio público)
    draw_chile(ax)

    for mz in ORDER:
        sub = df[df["macrozona"] == mz]
        if sub.empty:
            continue
        ax.scatter(sub["lon"], sub["lat"],
                   s=16 + 5.0 * sub["mw"] ** 0.62,
                   c=COLORS[mz], edgecolors="black", linewidths=0.45,
                   alpha=0.9, label=f"{mz} ({len(sub)})", zorder=4)

    # referencia de latitud: cortes de macrozona
    for cut, _ in LAT_CUTS[:-1]:
        ax.axhline(cut, color="#888888", lw=0.7, ls=":", zorder=2)
    ax.set_xlabel("Longitud (°)", fontsize=9)
    ax.set_ylabel("Latitud (°)", fontsize=9)
    ax.set_xlim(-76.5, -65.5)
    ax.set_ylim(-39.5, -16.8)
    ax.tick_params(labelsize=8)
    # escala geográfica: 1° de latitud es más largo que 1° de longitud
    ax.set_aspect(1.0 / np.cos(np.radians(28.0)))
    ax.legend(fontsize=7.5, loc="lower left", framealpha=0.95,
              title="Macrozona (n)", title_fontsize=8)
    ax.set_title(f"Ubicación de las plantas evaluadas\n"
                 f"({len(df)} de {len(plants)} georreferenciadas · "
                 f"área ∝ capacidad instalada)", fontsize=10)
    fig.tight_layout()
    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT)
    plt.close(fig)
    print(f"[figura] {OUT}")


def fig_resultados_nrmse() -> None:
    """Ranking por NRMSE (roll-out, 94 plantas) en español, para la charla.

    Figura NUEVA: no reemplaza `fig_resultados_global.png`, que pertenece a la
    tesis ya cerrada.
    """
    summary = ROOT / "results" / "total" / "visuals" / "all_metrics_summary.csv"
    if not summary.exists():
        print("[aviso] no existe all_metrics_summary.csv; se omite la figura NRMSE")
        return
    df = pd.read_csv(summary)
    cap = (pd.read_parquet(SILVER, columns=["unique_id", "potencia_neta_mw"])
           .drop_duplicates("unique_id")
           .rename(columns={"unique_id": "Planta", "potencia_neta_mw": "cap"}))
    df = df.merge(cap, on="Planta", how="left")
    df["NRMSE"] = df["RMSE"] / df["cap"] * 100.0

    # ventana cold-start: Operational donde exista, Raw en el resto
    op = df[df["Window"] == "Operational"]
    raw = df[df["Window"] == "Raw"]
    plants_op = set(op["Planta"])
    d = pd.concat([op, raw[~raw["Planta"].isin(plants_op)]], ignore_index=True)
    d = d[d["Horizon"] == "7-Day Rollout"]

    pretty = {"XGB_LOCAL": ("XGBoost", "local"), "XGB_GLOBAL": ("XGBoost", "global"),
              "NHITS_LOCAL": ("N-HiTS", "local"), "NHITS": ("N-HiTS", "global"),
              "LSTM_LOCAL": ("LSTM", "local"), "LSTM": ("LSTM", "global"),
              "TFT": ("TFT", "global"), "INFORMER": ("Informer", "global")}
    med = d.groupby("Model")["NRMSE"].median().reindex(pretty).sort_values()

    labels = [f"{pretty[m][0]} ({pretty[m][1]})" for m in med.index]
    colors = ["#2CA02C" if pretty[m][1] == "local" else "#1F77B4" for m in med.index]

    fig, ax = plt.subplots(figsize=(7.2, 3.5))
    y = range(len(med))
    ax.barh(list(y), med.to_numpy(), color=colors, edgecolor="black", linewidth=0.5)
    ax.set_yticks(list(y)); ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("NRMSE mediano — roll-out 7 días, 94 plantas (% de la capacidad)",
                  fontsize=9)
    for i, v in enumerate(med.to_numpy()):
        ax.text(v + 0.4, i, f"{v:.1f}", va="center", fontsize=8.5)
    ax.axvline(med.min(), ls="--", lw=0.9, color="#2CA02C", alpha=0.6)
    ax.set_xlim(0, med.max() * 1.2)
    handles = [plt.Rectangle((0, 0), 1, 1, fc="#2CA02C", ec="black"),
               plt.Rectangle((0, 0), 1, 1, fc="#1F77B4", ec="black")]
    ax.legend(handles, ["Estrictamente local", "Global (Cross-Site)"],
              loc="lower right", fontsize=8.5, framealpha=0.95)
    ax.grid(axis="x", ls=":", alpha=0.4)
    fig.tight_layout()
    out = OUT.parent / "fig_resultados_nrmse_es.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"[figura] {out}")


if __name__ == "__main__":
    main()
    fig_resultados_nrmse()
