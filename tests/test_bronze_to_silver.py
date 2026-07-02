import os
import pandas as pd
import numpy as np
import pytest
from pathlib import Path

# Suponiendo que la función principal será process_silver
from src.etl.bronze_to_silver import process_silver

def test_bronze_to_silver_pipeline(tmp_path):
    """
    Prueba unitaria para la creación de la Capa Silver (Unified).
    """
    bronze_dir = tmp_path / "bronze"
    silver_dir = tmp_path / "silver"
    
    bronze_dir.mkdir()
    silver_dir.mkdir()
    
    # 1. Crear directorios y maestros simulados en Bronze
    (bronze_dir / "generacion").mkdir(parents=True, exist_ok=True)
    (bronze_dir / "exogenas" / "radiacion").mkdir(parents=True, exist_ok=True)
    
    # Maestro Generacion
    maestro_gen = pd.DataFrame({
        "nombre": ["Planta 1", "Planta 2"],
        "region": ["Antofagasta", "Metropolitana"],
        "potencia_neta_mw": [10.0, 20.0]
    })
    maestro_gen.to_parquet(bronze_dir / "maestro_generacion.parquet", index=False)
    
    # Maestro Exogenas
    maestro_exo = pd.DataFrame({
        "nombre": ["Estacion Norte", "Estacion Centro"],
        "latitud": [-22.0, -33.5]
    })
    maestro_exo.to_parquet(bronze_dir / "maestro_exogenas.parquet", index=False)
    
    # Datos de Generacion (Norte Grande y Zona Central)
    pd.DataFrame({
        "unique_id": ["planta_1", "planta_1"],
        "ds": ["2020-01-01 12:00:00", "2020-01-01 13:00:00"],
        "generacion_mwh": [8.0, 9.0]
    }).to_csv(bronze_dir / "generacion" / "planta_1.csv", index=False)
    
    pd.DataFrame({
        "unique_id": ["planta_2", "planta_2"],
        "ds": ["2020-01-01 12:00:00", "2020-01-01 13:00:00"],
        "generacion_mwh": [15.0, 16.0]
    }).to_csv(bronze_dir / "generacion" / "planta_2.csv", index=False)
    
    # Datos de Exogenas
    pd.DataFrame({
        "unique_id": ["estacion_norte", "estacion_norte"],
        "ds": ["2020-01-01 12:00:00", "2020-01-01 13:00:00"],
        "radiacion": [1000, 1100]
    }).to_csv(bronze_dir / "exogenas" / "radiacion" / "estacion_norte.csv", index=False)
    
    pd.DataFrame({
        "unique_id": ["estacion_centro", "estacion_centro"],
        "ds": ["2020-01-01 12:00:00", "2020-01-01 13:00:00"],
        "radiacion": [800, 850]
    }).to_csv(bronze_dir / "exogenas" / "radiacion" / "estacion_centro.csv", index=False)
    
    # 2. Ejecutar la logica de Silver
    process_silver(bronze_dir, silver_dir)
    
    # 3. Aserciones
    unified_parquet = silver_dir / "silver_unified.parquet"
    unified_csv = silver_dir / "silver_unified.csv"
    
    assert unified_parquet.exists(), "No se creó el dataset unificado en parquet"
    assert unified_csv.exists(), "No se creó el dataset unificado en csv"
    
    df_silver = pd.read_parquet(unified_parquet)
    
    # Validaciones estructurales para ML (Panel Data)
    expected_cols = [
        "unique_id", "ds", "y", "macrozona", "potencia_neta_mw", "PR",
        "radiacion", "sin_hour", "cos_hour", "sin_day_year", "cos_day_year",
        "sin_day_month", "cos_day_month", "sin_month", "cos_month", "estacion_año"
    ]
    for col in expected_cols:
        assert col in df_silver.columns, f"Falta la columna esperada: {col}"
        
    # Verificar Macrozonas asignadas correctamente
    norte_df = df_silver[df_silver["unique_id"] == "planta_1"]
    assert norte_df.iloc[0]["macrozona"] == "Norte Grande"
    assert norte_df.iloc[0]["radiacion"] == 1000.0  # El promedio de la macrozona debe cruzar correctamente
    
    centro_df = df_silver[df_silver["unique_id"] == "planta_2"]
    assert centro_df.iloc[0]["macrozona"] == "Zona Central"
    assert centro_df.iloc[0]["radiacion"] == 800.0
    
    # Verificar Performance Ratio (PR)
    # PR = Generacion / (Potencia * horas) -> Generacion: 8, Potencia: 10, horas: 1 -> PR = 0.8
    assert np.isclose(norte_df.iloc[0]["PR"], 0.8)
