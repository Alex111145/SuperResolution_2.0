import sys
import torch
import numpy as np
import json
from pathlib import Path
from torch.utils.data import DataLoader
from PIL import Image
from tqdm import tqdm
from typing import List

# Configurazione percorsi
CURRENT_SCRIPT = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_SCRIPT.parent
OUTPUT_ROOT = PROJECT_ROOT / "outputs"
ROOT_DATA_DIR = PROJECT_ROOT / "data"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import dai moduli del progetto
from models.architecture import SwinIR
from dataset.astronomical_dataset import AstronomicalDataset

torch.backends.cudnn.benchmark = True

def save_as_tiff16(tensor, path):
    """Salva un tensor come immagine TIFF a 16-bit."""
    arr = tensor.squeeze().float().cpu().numpy()
    arr = np.clip(arr, 0, 1)
    arr_u16 = (arr * 65535).astype(np.uint16)
    Image.fromarray(arr_u16, mode='I;16').save(path)

def get_available_targets(output_root: Path) -> List[str]:
    """Recupera le cartelle di output disponibili."""
    if not output_root.is_dir(): return []
    return sorted([p.name for p in output_root.iterdir() if p.is_dir()])

def run_test(target_model_folder: str):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Utilizzo dispositivo: {device}")

    # Percorsi checkpoint e risultati
    OUTPUT_DIR = OUTPUT_ROOT / target_model_folder / "test_results_tiff"
    CHECKPOINT_DIR = OUTPUT_ROOT / target_model_folder / "checkpoints"
    
    # Ricerca automatica del miglior checkpoint disponibile
    checkpoints = list(CHECKPOINT_DIR.glob("best_gan_model.pth"))
    if not checkpoints:
        checkpoints = list(CHECKPOINT_DIR.glob("latest_checkpoint.pth"))
    
    if not checkpoints:
        print(f"Nessun checkpoint trovato in {CHECKPOINT_DIR}")
        return
    
    CHECKPOINT_PATH = checkpoints[0]
    print(f"Caricamento pesi: {CHECKPOINT_PATH}")

    # Inizializzazione modello SwinIR
    model = SwinIR(upscale=4, in_chans=1, img_size=128, window_size=8,
                   img_range=1.0, depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                   mlp_ratio=2, upsampler='pixelshuffle', resi_connection='1conv').to(device)

    # Caricamento dello stato del modello
    try:
        state_dict = torch.load(CHECKPOINT_PATH, map_location=device)
        if 'net_g' in state_dict: state_dict = state_dict['net_g']
        elif 'model_state_dict' in state_dict: state_dict = state_dict['model_state_dict']
            
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k.replace("module.", "") 
            new_state_dict[name] = v
        
        model.load_state_dict(new_state_dict, strict=True)
        print("Pesi caricati correttamente.")
    except Exception as e:
        print(f"Errore caricamento pesi: {e}")
        return

    model.eval()

    # --- LOGICA CARICAMENTO JSON E DATASET ---
    # Estrae il nome del target (es. "M1") dalla cartella (es. "M1_DDP_SwinIR")
    target_name = target_model_folder.split('_')[0]
    
    # Costruisce il percorso verso il file JSON di validazione
    json_path = ROOT_DATA_DIR / target_name / "8_dataset_split" / "splits_json" / "val.json"
    
    if not json_path.exists():
        print(f"Errore: File JSON non trovato in {json_path}")
        print("Controlla che la cartella 'data' contenga i file generati durante il training.")
        return

    print(f"Caricamento dataset da: {json_path.name}")
    # Inizializza il dataset senza augmentation per l'inferenza
    test_ds = AstronomicalDataset(str(json_path), base_path=PROJECT_ROOT, augment=False)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=2)

    # Creazione cartella di output
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Loop di inferenza
    print(f"Inizio inferenza su {len(test_ds)} immagini...")
    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader, desc="Esecuzione")):
            lr_img = batch['lr'].to(device)
            
            # Esecuzione modello SwinIR
            sr_img = model(lr_img)
            
            # Clamp dei valori tra 0 e 1 per sicurezza
            sr_img = sr_img.clamp(0, 1)
            
            # Salvataggio risultato in TIFF 16-bit
            save_path = OUTPUT_DIR / f"sr_result_{i:04d}.tif"
            save_as_tiff16(sr_img, save_path)

    print(f"\nInferenza completata correttamente!")
    print(f"Risultati salvati in: {OUTPUT_DIR}")

if __name__ == "__main__":
    targets = get_available_targets(OUTPUT_ROOT)
    if targets:
        print("\nCartelle di output trovate:")
        for idx, t in enumerate(targets):
            print(f" [{idx}] {t}")
        
        sel_idx = input("\nInserisci il numero o il nome della cartella: ")
        
        # Gestione input sia come indice che come stringa
        if sel_idx.isdigit() and int(sel_idx) < len(targets):
            sel_folder = targets[int(sel_idx)]
        else:
            sel_folder = sel_idx
            
        run_test(sel_folder)
    else:
        print(f"Nessuna cartella trovata in {OUTPUT_ROOT}. Esegui prima il training.")
