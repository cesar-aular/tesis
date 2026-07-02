import os
import argparse
from pathlib import Path
import pandas as pd

from src.ml.utils.data_loader import load_lopo_split
from src.ml.utils.idempotency import check_run_completed
# Local imports for models
from src.ml.xgb_local import train as xgb_local_train, test as xgb_local_test
from src.ml.xgb_global import train as xgb_global_train, test as xgb_global_test
from src.ml.lstm import train as lstm_train, test as lstm_test, tune as lstm_tune
from src.ml.nhits import train as nhits_train, test as nhits_test, tune as nhits_tune
from src.ml.tft import train as tft_train, test as tft_test, tune as tft_tune
from src.ml.visualize import generate_global_metrics_report

def run_ml_pipeline(strategy: str = "toy", force: bool = False):
    print("\n--- Iniciando Pipeline ML (LOPO & Cold-Start) ---")
    
    silver_unified_path = Path("data/silver/silver_unified.parquet")
    silver_dl_path = Path("data/silver/silver_dl.parquet")
    
    if not silver_unified_path.exists() or not silver_dl_path.exists():
        print("Error: No se encontraron los datasets en data/silver/. Ejecuta ETL y prepare_dl_dataset.py primero.")
        return
        
    results_dir = Path(f"results/{strategy}")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Get plant list from a lightweight load
    df_temp = pd.read_parquet(silver_dl_path, columns=["unique_id"])
    plantas = df_temp["unique_id"].unique().tolist()
    del df_temp
    import gc
    gc.collect()
    
    print(f"[INFO] Total Plantas Disponibles: {len(plantas)}")
    
    if strategy == "toy":
        plantas_to_test = plantas[:2]
        print(f"[Estrategia TOY] Reduciendo a {len(plantas_to_test)} plantas.")
    elif strategy == "half":
        plantas_to_test = plantas[:len(plantas)//2]
        print(f"[Estrategia HALF] Reduciendo a {len(plantas_to_test)} plantas.")
    else:
        plantas_to_test = plantas
        print(f"[Estrategia TOTAL] Usando todas las plantas.")
        
    for i, planta in enumerate(plantas_to_test, 1):
        print(f"\n[{i}/{len(plantas_to_test)}] Evaluando LOPO sobre planta: {planta}")
        
        macrozona = "Norte"
        estacion = "Verano"
        
        # XGBoost Local (Solo heuristica, no usa train global)
        if not check_run_completed(results_dir / "xgb_local" / macrozona / estacion / planta, "xgb_local", planta, estacion):
            print("[INFO] Cargando silver_xgb...")
            silver_xgb = pd.read_parquet(silver_unified_path)
            cat_cols = [c for c in silver_xgb.select_dtypes(include=['object', 'string']).columns if c not in ['unique_id', 'ds']]
            for col in cat_cols: silver_xgb[col] = silver_xgb[col].astype('category')
            train_xgb, test_xgb = load_lopo_split(silver_xgb, planta)
            macrozona = str(test_xgb['macrozona'].iloc[0])
            estacion = str(test_xgb['estacion_a\u00f1o'].iloc[0])
            
            xgb_local_train.run_train(silver_xgb, planta, strategy)
            xgb_local_test.run_test(test_xgb, results_dir, macrozona, estacion, planta)
            del train_xgb; del test_xgb; del silver_xgb; gc.collect()
            
        # XGBoost Global
        if not check_run_completed(results_dir / "xgb_global" / macrozona / estacion / planta, "xgb_global", planta, estacion):
            print("[INFO] Cargando silver_xgb...")
            silver_xgb = pd.read_parquet(silver_unified_path)
            cat_cols = [c for c in silver_xgb.select_dtypes(include=['object', 'string']).columns if c not in ['unique_id', 'ds']]
            for col in cat_cols: silver_xgb[col] = silver_xgb[col].astype('category')
            train_xgb, test_xgb = load_lopo_split(silver_xgb, planta)
            macrozona = str(test_xgb['macrozona'].iloc[0])
            estacion = str(test_xgb['estacion_a\u00f1o'].iloc[0])
            
            xgb_global_train.run_train(train_xgb, strategy)
            xgb_global_test.run_test(test_xgb, results_dir, macrozona, estacion, planta)
            del train_xgb; del test_xgb; del silver_xgb; gc.collect()
            
        # LSTM
        if not check_run_completed(results_dir / "lstm" / macrozona / estacion / planta, "lstm", planta, estacion):
            print("[INFO] Cargando silver_dl para LSTM...")
            silver_dl = pd.read_parquet(silver_dl_path)
            train_dl, test_dl = load_lopo_split(silver_dl, planta)
            
            lstm_tune.run_tune(train_dl, strategy)
            lstm_train.run_train(train_dl, strategy)
            lstm_test.run_test(test_dl, silver_dl, results_dir, macrozona, estacion, planta, strategy)
            del train_dl; del test_dl; del silver_dl; gc.collect()
            import torch
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            
        # NHITS
        if not check_run_completed(results_dir / "nhits" / macrozona / estacion / planta, "nhits", planta, estacion):
            print("[INFO] Cargando silver_dl para NHITS...")
            silver_dl = pd.read_parquet(silver_dl_path)
            train_dl, test_dl = load_lopo_split(silver_dl, planta)
            
            nhits_tune.run_tune(train_dl, strategy)
            nhits_train.run_train(train_dl, strategy)
            nhits_test.run_test(test_dl, silver_dl, results_dir, macrozona, estacion, planta, strategy)
            del train_dl; del test_dl; del silver_dl; gc.collect()
            import torch
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            
        # TFT
        if not check_run_completed(results_dir / "tft" / macrozona / estacion / planta, "tft", planta, estacion):
            print("[INFO] Cargando silver_dl para TFT...")
            silver_dl = pd.read_parquet(silver_dl_path)
            train_dl, test_dl = load_lopo_split(silver_dl, planta)
            
            tft_tune.run_tune(train_dl, strategy)
            tft_train.run_train(train_dl, strategy)
            tft_test.run_test(test_dl, silver_dl, results_dir, macrozona, estacion, planta, strategy)
            del train_dl; del test_dl; del silver_dl; gc.collect()
            import torch
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            
    print("\n--- Pipeline ML Completado ---")
    generate_global_metrics_report(strategy)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", type=str, default="toy", choices=["toy", "half", "total"])
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    
    run_ml_pipeline(strategy=args.strategy, force=args.force)
