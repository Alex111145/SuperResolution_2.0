import sys
import os
from pathlib import Path

def setup_paths():
 
    UTILS_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT = UTILS_DIR.parent
    MODELS_DIR = PROJECT_ROOT / "models"
    

    paths_to_add = [
        MODELS_DIR / "BasicSR"
    ]
    
    print(f"Configurazione percorsi Python (Root: {PROJECT_ROOT})...")
    
    for p in paths_to_add:
        if p.exists():
            str_p = str(p)
            if str_p not in sys.path:
                sys.path.insert(0, str_p)
                print(f"Aggiunto al path: {p.name}")
        else:
            
            print(f"ATTENZIONE: Percorso necessario non trovato: {p}")

setup_paths()

def import_external_archs():
    print("Importazione Moduli Esterni...")
    
    RRDBNet = None
    
   
    try:
        from basicsr.archs.rrdbnet_arch import RRDBNet
        print("BasicSR (RRDBNet) importato correttamente.")
    except ImportError as e:
        print(f"Errore import BasicSR: {e}")

    return RRDBNet
