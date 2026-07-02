# KI: Fuzzy Matching para Nombres de Plantas Solares (SEN vs Mapa)

**Contexto:**
En los registros crudos del SEN (2014-2024), el `Codigo Central` a menudo difiere ligeramente del `Nombre Proyecto` oficial del Mapa del Ministerio (ej. "SOLAR EL AGUILA I" vs "EL AGUILA I"). Un cruce estricto (`==`) descarta plantas válidas.

**Solución Técnica Estandarizada:**
Para cruzar la metadata del mapa con los CSVs de generación, SE DEBE usar la librería estándar `difflib.get_close_matches` junto con un saneamiento previo de strings.

**Implementación Aprobada (No reinventar):**
```python
import re
from unidecode import unidecode
from difflib import get_close_matches

def sanitize(name):
    # Eliminar puntuación, espacios extra y caracteres no alfanuméricos
    return re.sub(r'[^\w\s-]', '', str(name).strip().upper())

def match_plants(nombre_mapa, lista_nombres_sen, cutoff=0.7):
    sanitized_mapa = sanitize(unidecode(nombre_mapa))
    sanitized_sen = [sanitize(unidecode(p)) for p in lista_nombres_sen]
    
    # 1. Intentar coincidencia de subcadena (muy eficiente)
    for idx, p in enumerate(sanitized_sen):
        if sanitized_mapa in p or p in sanitized_mapa:
            return lista_nombres_sen[idx]
            
    # 2. Fallback a Fuzzy Matching
    matches = get_close_matches(sanitized_mapa, sanitized_sen, n=1, cutoff=cutoff)
    if matches:
        index_match = sanitized_sen.index(matches[0])
        return lista_nombres_sen[index_match]
        
    return None
```
