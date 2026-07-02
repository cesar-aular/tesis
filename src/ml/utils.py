from pathlib import Path

def check_idempotency(output_file: Path | str, force: bool = False) -> bool:
    """
    Verifica si un archivo de salida ya existe y, basado en el flag 'force',
    determina si la etapa debe ejecutarse.
    
    Retorna:
    - True: Si la etapa debe ser omitida (el archivo existe y no se forzó).
    - False: Si la etapa debe ejecutarse (el archivo no existe o se forzó).
    """
    output_file = Path(output_file)
    if output_file.exists() and not force:
        return True # Ya existe, omitir ejecución
    return False # Ejecutar
