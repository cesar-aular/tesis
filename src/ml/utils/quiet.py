"""Configuración de arranque de librerías ML (orden de imports + verbosidad).

POLÍTICA DE WARNINGS (decisión de César, 2026-07-02): los warnings se
RESUELVEN en su origen (pasando los parámetros que la librería pide o
corrigiendo el uso), no se ocultan. Este módulo NO contiene
warnings.filterwarnings — si aparece un warning nuevo, es accionable:
arreglarlo donde se genera.

Lo que sí hace:
1. Orden de carga anti-segfault (Windows DLL hell): pyarrow debe cargar sus
   DLLs antes que torch/lightning, o el import perezoso de pyarrow.dataset
   dentro de read_parquet revienta el proceso (exit 139).
2. Niveles de LOGGING (verbosidad de INFO/banners, no warnings): Lightning
   anuncia GPU/seeds/ranks por cada fit (~50x por corrida LOPO) y Optuna
   loguea cada trial; se elevan a WARNING/ERROR — sus warnings reales siguen
   visibles.
"""
import logging


def silence_noise():
    # ORDEN CRITICO (Windows DLL hell): pyarrow DEBE cargar sus DLLs antes que
    # torch/lightning. Si lightning carga primero, el import perezoso de
    # pyarrow.dataset (dentro de read_parquet) hace segfault (exit 139).
    try:
        import pyarrow.dataset   # noqa: F401
        import pyarrow.parquet   # noqa: F401
    except ImportError:
        pass

    # Importar lightning ANTES de configurar: fija su nivel INFO en el import
    # y pisaria cualquier configuracion hecha antes.
    try:
        import pytorch_lightning  # noqa: F401
        import lightning_fabric   # noqa: F401
    except ImportError:
        pass

    # --- Lightning / Fabric: banners INFO fuera; warnings reales siguen ---
    for name in ("pytorch_lightning", "lightning.pytorch", "lightning_fabric",
                 "lightning_fabric.utilities.seed",
                 "pytorch_lightning.utilities.rank_zero",
                 "pytorch_lightning.accelerators.cuda", "lightning"):
        logging.getLogger(name).setLevel(logging.WARNING)

    # --- Optuna: sin INFO por trial (el resumen del TUNE lo imprime dl_models) ---
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        pass
