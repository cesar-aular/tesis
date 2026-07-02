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
                
            # Ventana de sensibilidad: '_operational' = desde produccion sostenida;
            # '_verano'/'_otono'/'_invierno'/'_primavera' = ventana en esa estacion
            season_windows = {"_verano": "Verano", "_otono": "Otoño",
                              "_invierno": "Invierno", "_primavera": "Primavera"}
            window = "Raw"
            if model_name.endswith("_operational"):
                window = "Operational"
                model_name = model_name[:-len("_operational")]
            else:
                for sfx, nombre in season_windows.items():
                    if model_name.endswith(sfx):
                        window = nombre
                        model_name = model_name[:-len(sfx)]
                        break

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
                # Window = estacion donde EMPIEZA la ventana de evaluacion
                # (Raw/Operational = momento real de conexion; Verano..Primavera =
                # sensibilidad estacional simulando el Cold-Start en esa estacion)
                "Window": window,
                "Macrozona": macrozona,
                # Estacion_Conexion = estacion del PRIMER registro de la planta
                # (cuando ocurrio su Cold-Start real). NO es la estacion evaluada:
                # una planta conectada en Verano tambien se evalua con ventanas
                # de Invierno tomadas de su historia posterior.
                "Estacion_Conexion": estacion,
                "Planta": planta,
                "RMSE": metrics.get("RMSE", 0),
                "MAE": metrics.get("MAE", 0),
                "sMAPE": metrics.get("sMAPE", 0),
                "rRMSE": metrics.get("rRMSE", 0),
                # Metricas probabilisticas (solo modelos DL con MQLoss level=[90])
                "Coverage_90": metrics.get("Coverage_90"),
                "Pinball_P05": metrics.get("Pinball_P05"),
                "Pinball_P95": metrics.get("Pinball_P95"),
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
    
    # Sensibilidad estacional GLOBAL: rRMSE por modelo x ventana (todas las
    # plantas; cada planta aporta ventanas de sus 4 estaciones si su historia
    # las contiene). Esta es la vista correcta para comparar estaciones —
    # el cruce con la estacion de conexion solo particiona la muestra.
    df_roll = df_metrics[df_metrics["Horizon"] == "7-Day Rollout"]
    if not df_roll.empty:
        plt.figure(figsize=(14, 6))
        orden = ["Raw", "Operational", "Verano", "Otoño", "Invierno", "Primavera"]
        orden = [w for w in orden if w in df_roll["Window"].unique()]
        sns.barplot(data=df_roll, x="Window", y="rRMSE", hue="Model",
                    order=orden, errorbar=None, estimator="median")
        plt.title("Sensibilidad de la ventana Cold-Start (rRMSE mediano, rollout 7d, todas las plantas)")
        plt.ylabel("rRMSE mediano (%)")
        plt.xlabel("Ventana de evaluacion")
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(visuals_dir / "window_sensitivity_global.png", dpi=150)
        plt.close()

    # Grouped RMSE barplots por Macrozona, Estacion de conexion y Ventana
    for macrozona in df_metrics["Macrozona"].unique():
        for estacion in df_metrics["Estacion_Conexion"].unique():
            df_subset = df_metrics[(df_metrics["Macrozona"] == macrozona) & (df_metrics["Estacion_Conexion"] == estacion)]
            if df_subset.empty: continue

            out_dir = visuals_dir / macrozona / estacion
            out_dir.mkdir(parents=True, exist_ok=True)

            for window in df_subset["Window"].unique():
                df_w = df_subset[df_subset["Window"] == window]
                if df_w.empty: continue
                plt.figure(figsize=(12, 6))
                sns.barplot(data=df_w, x="Model", y="RMSE", hue="Horizon", errorbar=None)
                plt.title(f"Comparacion de RMSE - {macrozona} (conexion: {estacion}) [Ventana evaluada: {window}]")
                plt.ylabel("RMSE")
                plt.grid(axis='y', alpha=0.3)
                plt.tight_layout()
                plt.savefig(out_dir / f"unified_rmse_comparison_{window.lower()}.png", dpi=150)
                plt.close()

            # Export sub-csv (todas las ventanas, columna Window)
            df_subset.to_csv(out_dir / "metrics_summary.csv", index=False)
            
    print(f"[Reporte] Graficas unificadas de RMSE y CSVs guardadas por zona y estacion en {visuals_dir}")
