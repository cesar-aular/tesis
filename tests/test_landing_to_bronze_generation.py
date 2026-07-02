import os
import pandas as pd
import pytest
from pathlib import Path

from src.etl.landing_to_bronze_generation import process_generation

def test_landing_to_bronze_generation_pipeline(tmp_path):
    """
    Verifica que el ETL de Generación cumpla con:
    1. Filtrar plantas estrictamente por Tecnología == 'Solar fotovoltaica'.
    2. Eliminar plantas sin 'Nombre' o sin 'Región'.
    3. Exportar el maestro a Parquet en bronze.
    4. Desglosar las series temporales por planta en formato CSV.
    5. Generar gráficos exploratorios en la carpeta visuales.
    """
    landing_dir = tmp_path / "landing"
    bronze_dir = tmp_path / "bronze"
    visuales_dir = tmp_path / "visuales"
    
    landing_dir.mkdir()
    (landing_dir / "generacion_raw").mkdir()
    
    # 1. Crear maestro dummy (A es válida, B no es solar, C no tiene nombre, D no tiene región)
    maestro_data = pd.DataFrame({
        "Nombre": ["Planta Solar A", "Planta Viento B", None, "Planta Solar D"],
        "Tecnología": ["Solar fotovoltaica", "Eólica", "Solar fotovoltaica", "Solar fotovoltaica"],
        "Región": ["Metropolitana", "Coquimbo", "Atacama", None],
        "Potencia Neta [MW]": [10.5, 20.0, 5.0, 15.0]
    })
    maestro_data.to_excel(landing_dir / "datos-centrales-generacion.xlsx", index=False)
    
    # 2. Crear CSV raw dummy simulando 2014
    import numpy as np
    dates = pd.date_range("2014-01-01", periods=800, freq="h")
    raw_csv = pd.DataFrame({
        "Date": dates.strftime("%d/%m/%Y"),
        "Hour": dates.hour + 1,
        "planta": ["Planta Solar A"] * 800,
        "generacion_mwh": (np.random.rand(800) * 100).astype(str)
    })
    # Add a couple of rows for Planta Viento B
    raw_csv.loc[0, "planta"] = "Planta Viento B"
    raw_csv.loc[1, "planta"] = "Planta Viento B"
    raw_csv.to_csv(landing_dir / "generacion_raw" / "201401GxSEN.csv", sep=';', index=False)
    
    # 3. Ejecutar el ETL
    process_generation(landing_dir, bronze_dir, visuales_dir)
    
    # --- ASERCIONES (Fase Verde) ---
    
    # A. Maestro guardado en Parquet y contiene SOLO a 'Planta Solar A'
    maestro_path = bronze_dir / "maestro_generacion.parquet"
    assert maestro_path.exists(), "El archivo maestro_generacion.parquet no fue creado"
    
    df_maestro_out = pd.read_parquet(maestro_path)
    assert len(df_maestro_out) == 1
    assert df_maestro_out.iloc[0]["nombre"] == "Planta Solar A"
    assert "potencia_neta_mw" in df_maestro_out.columns, "Falta la columna de capacidad instalada"
    assert float(df_maestro_out.iloc[0]["potencia_neta_mw"]) == 10.5
    
    # B. Series temporales desglosadas en CSV
    planta_a_csv = bronze_dir / "generacion" / "planta_solar_a.csv"
    assert planta_a_csv.exists(), "El CSV individual para la planta no fue creado"
    
    # C. Generación de Visuales Exploratorios
    visuales_generacion_dir = visuales_dir / "generacion-raw"
    assert visuales_generacion_dir.exists(), "La carpeta de visuales no fue creada"
    assert len(list(visuales_generacion_dir.glob("*.png"))) > 0, "No se generaron gráficos exploratorios"
