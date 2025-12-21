import sys
import os
from pathlib import Path

def setup_paths():
    # Il file si trova in 'utils/', quindi risaliamo alla root del progetto
    UTILS_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT = UTILS_DIR.parent
    MODELS_DIR = PROJECT_ROOT / "models"
    
    # Rimosso HAT poiché il progetto utilizza SwinIR come architettura principale
    # BasicSR viene mantenuto in quanto fornisce classi base e utility necessarie
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
            # Segnala la mancanza di BasicSR, essenziale per alcune utility di training
            print(f"ATTENZIONE: Percorso necessario non trovato: {p}")

setup_paths()

def import_external_archs():
    """Tenta di importare solo le architetture effettivamente necessarie."""
    print("Importazione Moduli Esterni...")
    
    RRDBNet = None
    
    # Importiamo RRDBNet da BasicSR (usato spesso come baseline o blocco ausiliario)
    try:
        from basicsr.archs.rrdbnet_arch import RRDBNet
        print("BasicSR (RRDBNet) importato correttamente.")
    except ImportError as e:
        print(f"Errore import BasicSR: {e}")

    # HAT è stato rimosso dalla logica di importazione
    return RRDBNet
