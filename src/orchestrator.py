import argparse
import sys
import pytest
from pathlib import Path

# Add project root to PYTHONPATH
sys.path.append(str(Path(__file__).parent.parent))

from src.etl.landing_to_bronze_generation import process_generation
from src.etl.landing_to_bronze_exogenous import process_exogenous
from src.etl.bronze_to_silver import process_silver

def run_tests():
    print("\n--- Ejecutando Pruebas de Integridad (TDD) ---")
    retcode = pytest.main(["tests/"])
    if retcode != 0:
        print("\nERROR CRÍTICO: Las pruebas fallaron. Abortando el pipeline para asegurar la integridad de los datos.")
        sys.exit(1)
    print("Todas las pruebas pasaron satisfactoriamente.\n")

def main():
    parser = argparse.ArgumentParser(description="Orquestador Central del Proyecto Tesis (Pipeline Continuo)")
    parser.add_argument("--force", action="store_true", help="Fuerza la re-ejecución ignorando idempotencia.")
    parser.add_argument("--clean", action="store_true", help="Borra las carpetas de salida en bronze y visuales antes de ejecutar.")
    parser.add_argument("--skip-tests", action="store_true", help="Omite la ejecución de las pruebas.")
    parser.add_argument("--strategy", type=str, default="toy", choices=["toy", "half", "total"], help="Estrategia de ejecución ML.")
    args = parser.parse_args()
    
    # 1. Pipeline de Integridad
    if not args.skip_tests:
        run_tests()
    else:
        print("WARNING: Saltando pruebas de integridad (--skip-tests).")
        
    # Rutas base
    project_root = Path(__file__).parent.parent
    landing_dir = project_root / "data" / "landing"
    bronze_dir = project_root / "data" / "bronze"
    silver_dir = project_root / "data" / "silver"
    visuales_dir = project_root / "visuales"
    
    # 1.5 Limpieza de directorios (Clean)
    if args.clean:
        import shutil
        print("\n[CLEAN] Borrando directorios de salida (bronze, silver y visuales)...")
        if bronze_dir.exists():
            shutil.rmtree(bronze_dir, ignore_errors=True)
        if silver_dir.exists():
            shutil.rmtree(silver_dir, ignore_errors=True)
        if visuales_dir.exists():
            shutil.rmtree(visuales_dir, ignore_errors=True)
        # Re-crear bases
        bronze_dir.mkdir(parents=True, exist_ok=True)
        silver_dir.mkdir(parents=True, exist_ok=True)
        visuales_dir.mkdir(parents=True, exist_ok=True)
        # --clean implica --force para las funciones subyacentes
        args.force = True
    
    # --- ML Pipeline (Idempotency and Trigger) ---
    gold_dir = project_root / "data" / "gold"
    skip_etl = False
    
    if gold_dir.exists() and any(gold_dir.iterdir()) and not args.force:
        print("\n[INFO] Capa Gold detectada y no se paso --force/--clean.")
        print("[INFO] Saltando el pipeline ETL completo (Idempotencia) -> Pasando directo a ML.")
        skip_etl = True
    
    if not skip_etl:
        print("--- Iniciando Pipeline ETL ---")
        
        print("\n--- Ejecutando Capa Bronze (Generacion) ---")
        try:
            from src.etl.landing_to_bronze_generation import process_generation
            process_generation("data/landing", "data/bronze", "visuales", force=args.force)
        except Exception as e:
            print(f"Error en Capa Bronze (Generacion): {e}")
            sys.exit(1)
            
        print("\n--- Ejecutando Capa Bronze (Exogenas) ---")
        try:
            from src.etl.landing_to_bronze_exogenous import process_exogenous
            process_exogenous("data/landing", "data/bronze", "visuales", force=args.force)
        except Exception as e:
            print(f"Error en Capa Bronze (Exogenas): {e}")
            sys.exit(1)
            
        print("\n--- Ejecutando Capa Silver ---")
        try:
            from src.etl.bronze_to_silver import process_silver
            process_silver("data/bronze", "data/silver")
        except Exception as e:
            print(f"Error en Capa Silver: {e}")
            sys.exit(1)
            
        print("\n--- Ejecutando Capa Gold (Train/Test Split) ---")
        try:
            from src.etl.silver_to_gold_split import process_gold_split
            process_gold_split("data/silver", "data/gold")
        except Exception as e:
            print(f"Error en Capa Gold: {e}")
            sys.exit(1)
            
        print("\n[OK] Pipeline ETL Completo Ejecutado Exitosamente!")

    # --- Iniciar ML ---
    print("\n--- Iniciando Pipeline ML (LOPO & Cold-Start) ---")
    try:
        from src.ml.orchestrator_ml import run_ml_pipeline
        # We can pass specific args if we want, like the strategy
        # For now, default to toy if not specified via orchestrator, but let's add the argument
        strategy = getattr(args, 'strategy', 'toy')
        run_ml_pipeline(strategy=strategy)
    except ImportError:
        print("Módulo src.ml.orchestrator_ml aún no está creado.")
    except Exception as e:
        print(f"Error en Pipeline ML: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
