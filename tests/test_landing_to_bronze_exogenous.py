import os
import pandas as pd
import pytest
from pathlib import Path

from src.etl.landing_to_bronze_exogenous import process_exogenous

def test_landing_to_bronze_exogenous_pipeline(tmp_path):
    landing_dir = tmp_path / "landing"
    bronze_dir = tmp_path / "bronze"
    visuales_dir = tmp_path / "visuales"
    
    landing_dir.mkdir()
    (landing_dir / "variables_externas").mkdir()
    var_dir = landing_dir / "variables_externas" / "radiacion-global"
    var_dir.mkdir()
    
    # 1. Crear maestro dummy de estaciones
    # Estación C no tiene latitud, Estación D no tiene nombre
    maestro_data = pd.DataFrame({
        "código": ["200001", "200002", "200003", "200004"],
        "nombre": ["Estacion A", "Estacion B", "Estacion C", None],
        "latitud": [-20.0, -21.0, None, -22.0],
        "longitud": [-70.0, -71.0, -72.0, -73.0],
        "altura": [100, 200, 300, 400]
    })
    maestro_data.to_excel(landing_dir / "datos-centrales-exogenas.xlsx", index=False)
    
    # 2. Crear CSV raw dummy
    raw_csv = pd.DataFrame({
        "fecha": ["2014-01-01", "2014-01-02", "2014-01-01"],
        "estacion": ["Estacion A", "Estacion A", "Estacion C"],
        "valor": [100.5, 110.2, 50.0]
    })
    raw_csv.to_csv(var_dir / "201401_radiacion.csv", sep=',', index=False)
    
    # 3. Ejecutar ETL
    process_exogenous(landing_dir, bronze_dir, visuales_dir)
    
    # --- ASERCIONES --- 
    maestro_path = bronze_dir / "maestro_exogenas.parquet"
    assert maestro_path.exists(), "El maestro no fue creado"
    
    df_maestro_out = pd.read_parquet(maestro_path)
    assert len(df_maestro_out) == 2, "Deben quedar solo las estaciones válidas (A y B)"
    assert set(df_maestro_out["nombre"].tolist()) == {"Estacion A", "Estacion B"}
    
    # 4. Validar existencia de los CSV separados (ahora dentro de subcarpetas por variable)
    assert (bronze_dir / "exogenas" / "radiacion-global" / "estacion_a.csv").exists(), "No se creó el CSV individual para la estacion A en su subcarpeta"
    assert not (bronze_dir / "exogenas" / "radiacion-global" / "estacion_c.csv").exists(), "No se debió crear CSV para estacion C (no validada)"
    
    vis_dir = visuales_dir / "exogenas-raw"
    assert vis_dir.exists()
    assert len(list(vis_dir.glob("*.png"))) > 0, "No se generaron gráficos exploratorios en las subcarpetas"
