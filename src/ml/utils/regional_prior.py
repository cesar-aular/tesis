"""Prior regional de eficiencia (Performance Ratio) LIBRE DE LEAKAGE.

Lección crítica del proyecto: la columna `PR = y / potencia_neta_mw` calculada
fila a fila en Silver era el target disfrazado (corr(PR, y) = 1.0) e invalidaba
los benchmarks XGBoost (rRMSE falso de ~0.8% vs ~56% real).

El reemplazo correcto es un prior AGREGADO por macrozona + estación del año,
calculado EXCLUSIVAMENTE con las plantas de entrenamiento dentro de cada split
LOPO. La planta objetivo recibe el prior de su región, jamás un valor derivado
de su propio y.
"""
import pandas as pd

DEFAULT_PR = 0.25  # factor de planta solar típico si no hay información regional


def compute_regional_pr(train_df: pd.DataFrame) -> pd.DataFrame:
    """Calcula la tabla de priors [macrozona, estacion_año, pr_regional].

    Eficiencia DIURNA por planta y estación: mean(y | y > 0) / potencia_neta_mw,
    luego promedio ENTRE plantas de la macrozona (evita que plantas grandes
    dominen). `train_df` DEBE ser el set N-1 del split LOPO (sin la objetivo).

    CALIBRACIÓN (auditoría 2026-07-04): la media sobre TODAS las horas (noches
    en cero incluidas) da un factor de planta (~0.2-0.3) que, usado como
    amplitud PICO del perfil sintético normalizado, deprimía el contexto
    Cold-Start ~3-4x respecto de una planta típica. La media sobre horas
    productivas es la amplitud correcta para un perfil con pico 1.0.
    """
    productive = train_df[train_df['y'] > 0]
    per_plant = (productive
                 .groupby(['unique_id', 'macrozona', 'estacion_año'], observed=True)
                 .agg(mean_y=('y', 'mean'), cap=('potencia_neta_mw', 'first'))
                 .reset_index())
    per_plant = per_plant[per_plant['cap'] > 0]
    per_plant['eff'] = per_plant['mean_y'] / per_plant['cap']

    table = (per_plant
             .groupby(['macrozona', 'estacion_año'], observed=True)['eff']
             .mean()
             .reset_index(name='pr_regional'))
    return table


def lookup_regional_pr(table: pd.DataFrame, macrozona: str, estacion: str) -> float:
    """Devuelve el prior para una macrozona+estación, con fallbacks seguros.

    Fallbacks: media de la macrozona -> media global de la tabla -> DEFAULT_PR.
    El resultado se acota a (0, 1] (un PR fuera de ese rango no es físico).
    """
    if table is None or table.empty:
        return DEFAULT_PR

    exact = table[(table['macrozona'] == macrozona) & (table['estacion_año'] == estacion)]
    if not exact.empty:
        value = float(exact['pr_regional'].iloc[0])
    else:
        mz = table[table['macrozona'] == macrozona]
        value = float(mz['pr_regional'].mean()) if not mz.empty else float(table['pr_regional'].mean())

    if not (value > 0):
        return DEFAULT_PR
    return min(value, 1.0)


def merge_regional_pr(df: pd.DataFrame, table: pd.DataFrame) -> pd.DataFrame:
    """Agrega el prior como feature `pr_regional` (por macrozona+estación).

    Usa un mapeo (no merge) para PRESERVAR los dtypes originales del frame:
    un merge sobre columnas categóricas puede degradarlas a object y
    desalinear los códigos de categoría entre train y test en XGBoost.
    Seguro para train y test: la tabla proviene solo de plantas de entrenamiento.
    """
    out = df.copy()
    if table is None or table.empty:
        out['pr_regional'] = DEFAULT_PR
        return out

    global_mean = float(table['pr_regional'].mean())
    fallback = global_mean if global_mean > 0 else DEFAULT_PR
    mapping = {(str(mz), str(es)): float(pr)
               for mz, es, pr in table[['macrozona', 'estacion_año', 'pr_regional']].itertuples(index=False)}

    keys = pd.Series(list(zip(out['macrozona'].astype(str), out['estacion_año'].astype(str))),
                     index=out.index)
    out['pr_regional'] = keys.map(mapping).fillna(fallback).clip(upper=1.0).astype('float32')
    return out
