import sys
import torch
import numpy as np
import json
import os
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from typing import List
from torch.utils.data import DataLoader

# Configurazione percorsi
CURRENT_SCRIPT = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_SCRIPT.parent
OUTPUT_ROOT = PROJECT_ROOT / "outputs"
DATA_ROOT = PROJECT_ROOT / "data"

# Aggiunge la root del progetto al path se manca
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# --- FIX IMPORT PRIORITARIO (Necessario per HAT/BasicSR) ---
BASICSR_PATH = PROJECT_ROOT / "models" / "BasicSR"
if BASICSR_PATH.exists():
    sys.path.insert(0, str(BASICSR_PATH))
else:
    print(f" [WARN] Cartella BasicSR non trovata in: {BASICSR_PATH}")

# Import dal progetto
try:
    from models.hat_arch import HAT
    from dataset.astronomical_dataset import AstronomicalDataset
except ImportError as e:
    print(f"Errore Import Modelli: {e}")
    sys.exit(1)

def save_as_tiff16(tensor, path):
    """Salva un tensore come file TIFF a 16-bit (Scientifico)."""
    arr = tensor.squeeze().float().cpu().numpy()
    arr = np.clip(arr, 0, 1)
    # Riconverte nel range 0-65535 per lo standard astronomico
    arr_u16 = (arr * 65535).astype(np.uint16)
    Image.fromarray(arr_u16, mode='I;16').save(path)

def tensor_to_img_8bit(tensor):
    """Converte tensore in array numpy uint8 per visualizzazione."""
    arr = tensor.squeeze().float().cpu().numpy()
    arr = np.clip(arr, 0, 1)
    return (arr * 255).astype(np.uint8)

def save_comparison(lr_tensor, sr_tensor, hr_tensor, filename):
    """
    Salva un'immagine composta (Tris): 
    LR (Upscaled Nearest) | SR (Output) | HR (Target)
    """
    lr_img = tensor_to_img_8bit(lr_tensor)
    sr_img = tensor_to_img_8bit(sr_tensor)
    hr_img = tensor_to_img_8bit(hr_tensor)
    
    # Ridimensiona LR alle dimensioni di SR per confronto visivo 
    # (Usa Nearest Neighbor per mostrare chiaramente i pixel originali)
    h, w = sr_img.shape
    lr_pil = Image.fromarray(lr_img).resize((w, h), resample=Image.NEAREST)
    lr_resized = np.array(lr_pil)
    
    # Concatenazione orizzontale
    combined = np.hstack((lr_resized, sr_img, hr_img))
    Image.fromarray(combined).save(filename, quality=95)

def get_available_targets(output_root: Path) -> List[str]:
    """Elenca le cartelle di output disponibili."""
    if not output_root.is_dir(): return []
    return sorted([p.name for p in output_root.iterdir() if p.is_dir()])

def run_inference(target_folder: str):
    """Esegue l'inferenza usando il modello HAT."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Utilizzo dispositivo: {device}")

    # Percorsi di output
    BASE_OUT = OUTPUT_ROOT / target_folder
    OUTPUT_TIFF = BASE_OUT / "inference_results_tiff"
    OUTPUT_PREVIEW = BASE_OUT / "inference_previews"
    CHECKPOINT_DIR = BASE_OUT / "checkpoints"
    
    # Ricerca del miglior checkpoint disponibile
    checkpoints = list(CHECKPOINT_DIR.glob("best_gan_model.pth"))
    if not checkpoints:
        checkpoints = list(CHECKPOINT_DIR.glob("latest_checkpoint.pth"))
    
    if not checkpoints:
        print("Nessun checkpoint trovato.")
        return
    
    CHECKPOINT_PATH = checkpoints[0]
    print(f"Caricamento modello: {CHECKPOINT_PATH}")

    # Inizializzazione HAT Engine (configurazione standard x4)
    model = HAT(
        img_size=128, 
        in_chans=1,
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
        if 'net_g' in state_dict: state_dict = state_dict['net_g']
        elif 'params' in state_dict: state_dict = state_dict['params']
            
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
    OUTPUT_TIFF.mkdir(parents=True, exist_ok=True)
    OUTPUT_PREVIEW.mkdir(parents=True, exist_ok=True)
    
    # --- PREPARAZIONE DATASET ---
    targets = target_folder.split('_')
    print(f"Target rilevati per dataset di test: {targets}")
    
    all_test_pairs = []
    for t in targets:
        json_path = DATA_ROOT / t / "8_dataset_split" / "splits_json" / "test.json"
        if json_path.exists():
            with open(json_path, 'r') as f:
                data = json.load(f)
                print(f" - {t}: trovate {len(data)} immagini di test")
                all_test_pairs.extend(data)
        else:
            print(f" [WARN] File test.json non trovato per target: {t}")

    if not all_test_pairs:
        print("Nessun dato di test trovato. Impossibile procedere.")
        return

    # Crea file temporaneo per il Dataset
    temp_json_path = BASE_OUT / "temp_inference_list.json"
    with open(temp_json_path, 'w') as f:
        json.dump(all_test_pairs, f)

    # DataLoader
    test_ds = AstronomicalDataset(temp_json_path, base_path=PROJECT_ROOT, augment=False)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    print(f"\nAvvio inferenza su {len(test_ds)} immagini...")
    print(f"TIFF Output: {OUTPUT_TIFF}")
    print(f"Preview Output: {OUTPUT_PREVIEW}")

    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader, desc="Inference")):
            lr = batch['lr'].to(device)
            hr = batch['hr'].to(device) # Necessario per il confronto

            # Inferenza
            sr = model(lr)

            # Salvataggio TIFF 16-bit
            save_name_tiff = OUTPUT_TIFF / f"result_{i:04d}_sr.tiff"
            save_as_tiff16(sr, save_name_tiff)

            # Salvataggio Tris Comparativo (LR | SR | HR)
            save_name_preview = OUTPUT_PREVIEW / f"compare_{i:04d}.jpg"
            save_comparison(lr, sr, hr, save_name_preview)

    print("\nInferenza completata. Controlla la cartella 'inference_previews' per i confronti.")

if __name__ == "__main__":
    available = get_available_targets(OUTPUT_ROOT)
    if available:
        print("Cartelle di addestramento trovate:", available)
        sel = input("Inserisci il nome della cartella target: ").strip()
        if sel in available:
            run_inference(sel)
        else:
            print("Cartella non valida o non trovata.")
    else:
        print(f"Nessun output trovato in {OUTPUT_ROOT}")
