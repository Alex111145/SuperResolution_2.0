import sys
import os
from pathlib import Path

def setup_paths():
    # --- MODIFICA PATH: Setup Nuova Struttura ---
    # Il file è in 'utils/', quindi:
    UTILS_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT = UTILS_DIR.parent
    MODELS_DIR = PROJECT_ROOT / "models"
    
    # Le librerie esterne devono essere dentro 'models/BasicSR', 'models/HAT'
    paths_to_add = [
        MODELS_DIR / "BasicSR",
        MODELS_DIR / "HAT"
    ]
    
    print(f"Configurazione percorsi Python (Root: {PROJECT_ROOT})...")
    
    for p in paths_to_add:
        if p.exists():
            str_p = str(p)
            if str_p not in sys.path:
                sys.path.insert(0, str_p)
                print(f"Aggiunto al path: {p.name}")
        else:
            print(f"ATTENZIONE: Percorso non trovato: {p}")

setup_paths()

def import_external_archs():
    """Tenta di importare le architetture e stampa errori specifici se fallisce."""
    print("Importazione Moduli Esterni...")
    
    RRDBNet = None
    HAT = None
    
    try:
        from basicsr.archs.rrdbnet_arch import RRDBNet
        print("BasicSR (RRDBNet) importato correttamente.")
    except ImportError as e:
        print(f"Errore import BasicSR: {e}")

    try:
        from hat.archs.hat_arch import HAT
        print("HAT importato correttamente.")
    except ImportError as e:
        try:
            from archs.hat_arch import HAT
            print("HAT importato (path alternativo).")
        except ImportError as e2:
            print(f"Errore import HAT: {e}")

    return RRDBNet, HAT