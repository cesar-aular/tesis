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

import pandas as pd

# Copy-on-Write (pandas 2.x): df.copy() y los slices se vuelven perezosos —
# solo se materializa la columna que efectivamente se muta. Mitigación OOM de
# RAM clave con la base completa (~4.3M filas): el camino DL hacía 2-3 copias
# profundas de ~350MB por fold que CoW convierte en vistas.
# (La mitigación de VRAM vive en la geometría de muestreo de dl_models:
# batch_size de series y step_size de ventanas — expandable_segments de torch
# no está soportado en Windows.)
pd.options.mode.copy_on_write = True


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
