import argparse
import gc
from pathlib import Path

from src.ml.utils.quiet import silence_noise
silence_noise()  # antes de que NeuralForecast/Lightning configuren sus loggers

import pandas as pd

from src.ml.utils.data_loader import load_lopo_split
from src.ml.utils.idempotency import check_plant_model_done
from src.ml.utils.regional_prior import compute_regional_pr, lookup_regional_pr, merge_regional_pr
# Modelos
from src.ml.xgb_local import train as xgb_local_train, test as xgb_local_test
from src.ml.xgb_global import train as xgb_global_train, test as xgb_global_test
from src.ml.lstm import train as lstm_train, test as lstm_test, tune as lstm_tune
from src.ml.nhits import train as nhits_train, test as nhits_test, tune as nhits_tune
from src.ml.tft import train as tft_train, test as tft_test, tune as tft_tune
from src.ml.informer import train as informer_train, test as informer_test, tune as informer_tune
from src.ml.utils.dl_local import run_local_dl
from src.ml.visualize import generate_global_metrics_report

XGB_MODELS = ("xgb_local", "xgb_global")
DL_MODELS = {
    "lstm": (lstm_tune, lstm_train, lstm_test),
    "nhits": (nhits_tune, nhits_train, nhits_test),
    "tft": (tft_tune, tft_train, tft_test),
    "informer": (informer_tune, informer_train, informer_test),
}
# Baselines DL ESTRICTAMENTE locales (anteproyecto: LSTM local; NHITS agregado
# por simetria). Entrenan solo con la historia de la planta objetivo.
LOCAL_DL_MODELS = ("lstm", "nhits")


def _load_silver_unified(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    # ANTI-LEAKAGE defensivo: PR = y/capacidad es el target disfrazado.
    # Silver ya no la genera, pero un parquet antiguo podria contenerla.
    if 'PR' in df.columns:
        print("[WARN] Columna PR (leakage) detectada en silver_unified: descartada. "
              "Regenera Silver con el ETL actualizado.")
        df = df.drop(columns=['PR'])
    # Reducir memoria: floats a 32 bits, strings a category
    float_cols = df.select_dtypes(include=['float64']).columns
    df[float_cols] = df[float_cols].astype('float32')
    for col in df.select_dtypes(include=['object', 'string']).columns:
        if col not in ('unique_id', 'ds'):
            df[col] = df[col].astype('category')
    return df


def _cast_xgb_categories(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.select_dtypes(include=['object', 'string']).columns:
        if col not in ('unique_id', 'ds'):
            df[col] = df[col].astype('category')
    return df


def run_ml_pipeline(strategy: str = "toy", force: bool = False):
    print("\n--- Iniciando Pipeline ML (LOPO & Cold-Start) ---")

    silver_unified_path = Path("data/silver/silver_unified.parquet")
    silver_dl_path = Path("data/silver/silver_dl.parquet")

    if not silver_unified_path.exists() or not silver_dl_path.exists():
        print("Error: Faltan datasets en data/silver/. Ejecuta el ETL y prepare_dl_dataset.py primero.")
        return

    results_dir = Path(f"results/{strategy}")
    results_dir.mkdir(parents=True, exist_ok=True)

    # Cargas UNICAS (antes se releia el parquet de 147MB por modelo y por planta)
    print("[INFO] Cargando silver_unified (una sola vez)...")
    silver_xgb = _load_silver_unified(silver_unified_path)
    print("[INFO] Cargando silver_dl (una sola vez)...")
    silver_dl = pd.read_parquet(silver_dl_path)

    plantas = silver_dl["unique_id"].unique().tolist()
    print(f"[INFO] Total Plantas Disponibles: {len(plantas)}")

    if strategy == "toy":
        plantas_to_test = plantas[:2]
        print(f"[Estrategia TOY] Reduciendo a {len(plantas_to_test)} plantas.")
    elif strategy == "half":
        plantas_to_test = plantas[:len(plantas) // 2]
        print(f"[Estrategia HALF] Reduciendo a {len(plantas_to_test)} plantas.")
    else:
        plantas_to_test = plantas
        print(f"[Estrategia TOTAL] Usando todas las plantas.")

    for i, planta in enumerate(plantas_to_test, 1):
        print(f"\n[{i}/{len(plantas_to_test)}] Evaluando LOPO sobre planta: {planta}")

        # Idempotencia REAL: salta modelos ya completados para esta planta
        pending_xgb = [m for m in XGB_MODELS
                       if force or not check_plant_model_done(results_dir, m, planta)]
        pending_dl = [m for m in DL_MODELS
                      if force or not check_plant_model_done(results_dir, m, planta)]
        pending_local_dl = [m for m in LOCAL_DL_MODELS
                            if force or not check_plant_model_done(results_dir, f"{m}_local", planta)]
        if not pending_xgb and not pending_dl and not pending_local_dl:
            print(f"[SKIP] Todos los modelos ya evaluados para {planta} (idempotencia).")
            continue

        # ---- Split LOPO + prior regional (leakage-free) sobre datos tabulares ----
        train_xgb, test_xgb = load_lopo_split(silver_xgb, planta)
        macrozona = str(test_xgb['macrozona'].iloc[0])
        estacion = str(test_xgb['estacion_año'].iloc[0])

        # Prior de eficiencia calculado SOLO con plantas de entrenamiento (N-1)
        pr_table = compute_regional_pr(train_xgb)
        regional_pr = lookup_regional_pr(pr_table, macrozona, estacion)
        # Cada ventana estacional se siembra con el PR de SU estación (la ventana
        # _invierno usa el PR invernal de la macrozona, no el de puesta en marcha)
        pr_by_window = {"": regional_pr, "_operational": regional_pr}
        for sfx, season in (("_verano", "Verano"), ("_otono", "Otoño"),
                            ("_invierno", "Invierno"), ("_primavera", "Primavera")):
            pr_by_window[sfx] = lookup_regional_pr(pr_table, macrozona, season)
        print(f"[INFO] {planta}: macrozona={macrozona}, estacion={estacion}, "
              f"PR regional (train-only)={regional_pr:.3f}")

        if pending_xgb:
            # Feature pr_regional unida por macrozona+estacion (tabla train-only)
            train_x = _cast_xgb_categories(merge_regional_pr(train_xgb, pr_table))
            test_x = _cast_xgb_categories(merge_regional_pr(test_xgb, pr_table))

            if "xgb_local" in pending_xgb:
                # Baseline local: usa SOLO la historia de la planta objetivo
                xgb_local_train.run_train(test_x, planta, strategy)
                xgb_local_test.run_test(test_x, results_dir, macrozona, estacion, planta)

            if "xgb_global" in pending_xgb:
                train_g = train_x
                if strategy == "toy":
                    # Smoke test: ultimo anio, primeras 10 plantas
                    sub_plantas = train_g['unique_id'].unique().tolist()[:10]
                    min_date = train_g['ds'].max() - pd.DateOffset(years=1)
                    train_g = train_g[(train_g['unique_id'].isin(sub_plantas)) &
                                      (train_g['ds'] >= min_date)]
                xgb_global_train.run_train(train_g, strategy)
                xgb_global_test.run_test(test_x, results_dir, macrozona, estacion, planta)
                del train_g

            del train_x, test_x
            gc.collect()

        del train_xgb, test_xgb
        gc.collect()

        # ---- Modelos Deep Learning (NeuralForecast) ----
        if pending_dl or pending_local_dl:
            train_dl, test_dl = load_lopo_split(silver_dl, planta)

            for model_name in pending_dl:
                tune_mod, train_mod, test_mod = DL_MODELS[model_name]
                print(f"[INFO] {model_name.upper()} para {planta}...")
                try:
                    tune_mod.run_tune(train_dl, strategy)
                    train_mod.run_train(train_dl, strategy)
                    test_mod.run_test(test_dl, silver_dl, results_dir, macrozona,
                                      estacion, planta, strategy, regional_pr=pr_by_window)
                except Exception as e:
                    # Un modelo fallido no debe matar una corrida de horas.
                    # Sin marker de completitud -> la idempotencia lo reintenta.
                    # (El retry fp32 anti-NaN se eliminó: todo entrena en fp32.)
                    print(f"[ERROR] {model_name.upper()} fallo para {planta}: {e}. Continuando.")
                gc.collect()
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # ---- Baselines DL estrictamente locales (historia propia) ----
            for model_name in pending_local_dl:
                print(f"[INFO] {model_name.upper()}_LOCAL para {planta}...")
                try:
                    run_local_dl(model_name, test_dl, results_dir, macrozona,
                                 estacion, planta, strategy)
                except Exception as e:
                    print(f"[ERROR] {model_name.upper()}_LOCAL fallo para {planta}: {e}. Continuando.")
                gc.collect()
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            del train_dl, test_dl
            gc.collect()

    print("\n--- Pipeline ML Completado ---")
    generate_global_metrics_report(strategy)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", type=str, default="toy", choices=["toy", "half", "total"])
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    run_ml_pipeline(strategy=args.strategy, force=args.force)
