import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pathlib import Path
import glob
import json

def plot_forecast_rollout(df_real: pd.DataFrame, df_pred: pd.DataFrame, output_path: Path, model_name: str, planta: str):
    """
    Dibuja el pronostico (y_pred) vs la realidad (y) con bandas de incertidumbre cuantilica si existen.
    """
    plt.figure(figsize=(14, 6))
    
    # Asegurar orden
    df_real = df_real.sort_values('ds')
    df_pred = df_pred.sort_values('ds')
    
    plt.plot(df_real['ds'], df_real['y'], label='Real (y)', color='black', alpha=0.7)
    plt.plot(df_pred['ds'], df_pred['y_pred'], label=f'{model_name} (Median/Point)', color='blue', linestyle='--')
    
    # Dibujar bandas probabilísticas si existen (DL QuantileLoss)
    if 'y_pred_lo_90' in df_pred.columns and 'y_pred_hi_90' in df_pred.columns:
        plt.fill_between(
            df_pred['ds'],
            df_pred['y_pred_lo_90'],
            df_pred['y_pred_hi_90'],
            color='blue',
            alpha=0.2,
            label='90% Confidence Interval'
        )
        
    plt.title(f"Forecasting Roll-Out - {planta} [{model_name}]")
    plt.xlabel("Fecha")
    plt.ylabel("Generacion MW")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()

def generate_unified_forecast_plot(planta: str, macrozona: str, estacion: str, horizon: str, strategy: str = "toy"):
    """
    Genera un grafico de lineas comparando todos los modelos para una planta y un horizonte (day1 o rollout7d).
    """
    results_dir = Path(f"results/{strategy}")
    if not results_dir.exists():
        return
        
    plt.figure(figsize=(16, 7))
    df_real = None
    
    colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown']
    c_idx = 0
    
    # iterar sobre los modelos
    for model_dir in results_dir.iterdir():
        if not model_dir.is_dir() or model_dir.name == "visuals": continue
        model_name = model_dir.name
        
        # Buscar el preds.parquet
        # Ruta esperada: results/{strategy}/{model}/{macrozona}/{estacion}/{planta}/{horizon}/preds.parquet
        preds_path = model_dir / macrozona / estacion / planta / horizon / "preds.parquet"
        
        if not preds_path.exists():
            continue
            
        df_pred = pd.read_parquet(preds_path)
        
        if df_real is None:
            df_real = df_pred[['ds', 'y']].drop_duplicates().sort_values('ds')
            plt.plot(df_real['ds'], df_real['y'], label='Real (y)', color='black', alpha=0.8, linewidth=2)
            
        if 'y_pred' in df_pred.columns:
            plt.plot(df_pred['ds'], df_pred['y_pred'], label=f'{model_name}', color=colors[c_idx % len(colors)], linestyle='--')
            
            if 'y_pred_lo_90' in df_pred.columns and 'y_pred_hi_90' in df_pred.columns:
                plt.fill_between(
                    df_pred['ds'],
                    df_pred['y_pred_lo_90'],
                    df_pred['y_pred_hi_90'],
                    color=colors[c_idx % len(colors)],
                    alpha=0.15,
                    label=f'{model_name} (90% CI)'
                )
                
            c_idx += 1
            
    if df_real is not None:
        plt.title(f"Prediccion Unificada - {planta} ({macrozona}, {estacion}) [{horizon}]")
        plt.xlabel("Fecha")
        plt.ylabel("Generacion MW")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        out_dir = results_dir / "visuals" / macrozona / estacion / planta
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"unified_forecast_{horizon}.png"
        
        plt.savefig(out_path, dpi=150)
    plt.close()

def generate_global_metrics_report(strategy: str = "toy"):
    print(f"[Reporte] Consolidando metricas de Day 1 y Rollout 7D para estrategia '{strategy}'...")
    results_dir = Path(f"results/{strategy}")
    if not results_dir.exists():
        return
        
    all_metrics = []
    plantas_evaluadas = set() # (planta, macrozona, estacion)
    
    # Buscar todos los JSON de completitud
    for json_path in results_dir.rglob("*_done.json"):
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            model_name = data.get("model", "unknown")
            planta = data.get("planta", "")
            estacion = data.get("estacion", "")
            metrics = data.get("metrics", {})
            
            # Inferir macrozona y horizonte
            # Ruta tipica: results/strategy/model/macrozona/estacion/planta/horizon/file.json
            parts = json_path.relative_to(results_dir).parts
            if len(parts) >= 5:
                macrozona = parts[1]
            else:
                macrozona = "unknown"
                
            if planta and macrozona != "unknown":
                plantas_evaluadas.add((planta, macrozona, estacion))
                
            if "day1" in model_name:
                base_model = model_name.replace("_day1", "").upper()
                horizon = "Day 1"
            elif "rollout7d" in model_name:
                base_model = model_name.replace("_rollout7d", "").upper()
                horizon = "7-Day Rollout"
            else:
                base_model = model_name.split("_")[0].upper()
                horizon = "unknown"
                if "local" in model_name:
                    base_model = "XGB LOCAL"
                elif "global" in model_name:
                    base_model = "XGB GLOBAL"
                    
            row = {
                "Model": base_model,
                "Horizon": horizon,
                "Macrozona": macrozona,
                "Estacion": estacion,
                "Planta": planta,
                "RMSE": metrics.get("RMSE", 0),
                "MAE": metrics.get("MAE", 0),
                "rMAE": metrics.get("rMAE", 0),
            }
            all_metrics.append(row)
        except Exception as e:
            print(f"Error procesando {json_path}: {e}")
            
    # Graficos unificados de prediccion por planta
    for (p, m, e) in plantas_evaluadas:
        generate_unified_forecast_plot(p, m, e, "day1", strategy)
        generate_unified_forecast_plot(p, m, e, "rollout7d", strategy)
        
    if not all_metrics:
        print("[Reporte] No se encontraron metricas para unificar.")
        return
        
    df_metrics = pd.DataFrame(all_metrics)
    
    # Save global summary
    visuals_dir = results_dir / "visuals"
    visuals_dir.mkdir(parents=True, exist_ok=True)
    df_metrics.to_csv(visuals_dir / "all_metrics_summary.csv", index=False)
    
    # Grouped RMSE barplots por Macrozona y Estacion
    for macrozona in df_metrics["Macrozona"].unique():
        for estacion in df_metrics["Estacion"].unique():
            df_subset = df_metrics[(df_metrics["Macrozona"] == macrozona) & (df_metrics["Estacion"] == estacion)]
            if df_subset.empty: continue
            
            plt.figure(figsize=(12, 6))
            sns.barplot(data=df_subset, x="Model", y="RMSE", hue="Horizon", ci=None)
            plt.title(f"Comparacion de RMSE - {macrozona} ({estacion}) [Todos los modelos]")
            plt.ylabel("RMSE")
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            
            out_dir = visuals_dir / macrozona / estacion
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / "unified_rmse_comparison.png"
            plt.savefig(out_path, dpi=150)
            plt.close()
            
            # Export sub-csv
            df_subset.to_csv(out_dir / "metrics_summary.csv", index=False)
            
    print(f"[Reporte] Graficas unificadas de RMSE y CSVs guardadas por zona y estacion en {visuals_dir}")
