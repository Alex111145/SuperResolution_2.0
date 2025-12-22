import sys
import torch
import numpy as np
import json
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from typing import List, Optional

# Configurazione percorsi
CURRENT_SCRIPT = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_SCRIPT.parent
OUTPUT_ROOT = PROJECT_ROOT / "outputs"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import dal nuovo progetto HAT
from models.hat_arch import HAT
from dataset.astronomical_dataset import AstronomicalDataset
from utils.metrics import TrainMetrics

def save_as_tiff16(tensor, path):
    """Salva un tensore come file TIFF a 16-bit."""
    arr = tensor.squeeze().float().cpu().numpy()
    arr = np.clip(arr, 0, 1)
    # Riconverte nel range 0-65535 per lo standard astronomico
    arr_u16 = (arr * 65535).astype(np.uint16)
    Image.fromarray(arr_u16, mode='I;16').save(path)

def get_available_targets(output_root: Path) -> List[str]:
    """Elenca le cartelle di output disponibili."""
    if not output_root.is_dir(): return []
    return sorted([p.name for p in output_root.iterdir() if p.is_dir()])

def run_inference(target_folder: str):
    """Esegue l'inferenza usando il modello HAT."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Utilizzo dispositivo: {device}")

    OUTPUT_DIR = OUTPUT_ROOT / target_folder / "inference_results_tiff"
    CHECKPOINT_DIR = OUTPUT_ROOT / target_folder / "checkpoints"
    
    # Ricerca del miglior checkpoint disponibile
    checkpoints = list(CHECKPOINT_DIR.glob("best_gan_model.pth"))
    if not checkpoints:
        checkpoints = list(CHECKPOINT_DIR.glob("latest_checkpoint.pth"))
    
    if not checkpoints:
        print("Nessun checkpoint trovato.")
        return
    
    CHECKPOINT_PATH = checkpoints[0]
    print(f"Caricamento modello: {CHECKPOINT_PATH}")

    # Inizializzazione HAT Engine (configurazione x4 come in train_hat.py)
    model = HAT(
        img_size=128, 
        in_chans=1,        # 1 canale per dati astronomici
        embed_dim=180, 
        depths=(6, 6, 6, 6, 6, 6), 
        num_heads=(6, 6, 6, 6, 6, 6),
        window_size=7, 
        upscale=4, 
        upsampler='pixelshuffle'
    ).to(device)

    # Caricamento dei pesi
    try:
        state_dict = torch.load(CHECKPOINT_PATH, map_location=device)
        # Estrazione pesi se salvati in un dizionario checkpoint complesso
        if 'net_g' in state_dict: state_dict = state_dict['net_g']
            
        # Rimozione eventuale prefisso DDP 'module.'
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k.replace("module.", "") 
            new_state_dict[name] = v
        
        model.load_state_dict(new_state_dict, strict=True)
        print("Pesi HAT caricati con successo.")
    except Exception as e:
        print(f"Errore caricamento pesi: {e}")
        return

    model.eval()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("\nModello HAT pronto per l'inferenza.")
    print("Nota: Assicurati di caricare le immagini tramite AstronomicalDataset o pre-processarle a 16-bit.")

if __name__ == "__main__":
    targets = get_available_targets(OUTPUT_ROOT)
    if targets:
        print("Cartelle di addestramento trovate:", targets)
        sel = input("Inserisci il nome della cartella target: ")
        run_inference(sel)
    else:
        print(f"Nessun output trovato in {OUTPUT_ROOT}")
