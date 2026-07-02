"""Silenciador central de ruido de librerías.

No cambia NINGÚN comportamiento de los modelos — solo la verbosidad:
- PyTorch Lightning: banners (GPU available, Seed set, LOCAL_RANK), warnings de
  scheduler/dataloader y resúmenes de módulos.
- Optuna: logs INFO por trial (el resumen del TUNE ya lo imprime dl_models).
- pandas/sklearn: FutureWarnings repetitivos que no podemos accionar aquí.

Llamar silence_noise() al inicio de cualquier entrypoint de ML.
"""
import logging
import os
import warnings


def silence_noise():
    # Importar primero: lightning fija su nivel INFO EN el import y pisaria
    # cualquier configuracion hecha antes.
    try:
        import pytorch_lightning  # noqa: F401
        import lightning_fabric   # noqa: F401
    except ImportError:
        pass

    # --- Lightning / Fabric: solo errores (mata banners y PossibleUserWarning) ---
    for name in ("pytorch_lightning", "lightning.pytorch", "lightning_fabric",
                 "lightning_fabric.utilities.seed",
                 "pytorch_lightning.utilities.rank_zero",
                 "pytorch_lightning.accelerators.cuda", "lightning"):
        logging.getLogger(name).setLevel(logging.ERROR)

    # --- Optuna: sin INFO por trial ---
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        pass

    # --- Warnings de librerías que no podemos accionar sin pelear con NF ---
    warnings.filterwarnings("ignore", category=UserWarning, module=r"pytorch_lightning.*")
    warnings.filterwarnings("ignore", category=UserWarning, module=r"lightning.*")
    warnings.filterwarnings("ignore", category=UserWarning, module=r"torch.*")
    warnings.filterwarnings("ignore", category=FutureWarning, module=r"neuralforecast.*")
    warnings.filterwarnings("ignore", category=FutureWarning, module=r"pandas.*")
    warnings.filterwarnings("ignore", category=FutureWarning, module=r"sklearn.*")
    # lr_scheduler.step() before optimizer.step(): interno de Lightning+AMP
    warnings.filterwarnings("ignore", message=r".*lr_scheduler\.step.*")
    # val_check_steps > max_steps en trials cortos: esperado
    warnings.filterwarnings("ignore", message=r".*val_check_steps is greater than max_steps.*")

    # Que los DataLoader workers hereden el silencio (Windows spawn)
    os.environ.setdefault("PYTHONWARNINGS", "ignore::UserWarning,ignore::FutureWarning")
