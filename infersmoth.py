import sys
import torch
import numpy as np
import json
import os
import re
from pathlib import Path
from torch.utils.data import DataLoader
from PIL import Image
from tqdm import tqdm
from typing import List

CURRENT_SCRIPT = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_SCRIPT.parent
OUTPUT_ROOT = PROJECT_ROOT / "outputs"
ROOT_DATA_DIR = PROJECT_ROOT / "data"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Assicurati che models/architecture.py sia presente
try:
    from models.architecture import SwinIR
    from dataset.astronomical_dataset import AstronomicalDataset
    from utils.metrics import TrainMetrics
except ImportError as e:
    sys.exit(f"Errore Import: {e}")

torch.backends.cudnn.benchmark = True

def save_as_tiff16(tensor, path):
    """Salva il tensore come immagine TIFF a 16-bit."""
    arr = tensor.squeeze().float().cpu().numpy()
    arr = np.clip(arr, 0, 1)
    arr_u16 = (arr * 65535).astype(np.uint16)
    Image.fromarray(arr_u16, mode='I;16').save(path)

def inference_tta(model, img, device):
    """
    Test-Time Augmentation (TTA) per 'smoothing' e riduzione rumore.
    Esegue l'inferenza su 8 versioni dell'immagine (rotazioni + flip) e ne fa la media.
    """
    output_list = []
    # 8 combinazioni: 4 rotazioni * 2 flip
    for rot in [0, 1, 2, 3]:
        for flip in [False, True]:
            # 1. Augment
            img_aug = torch.rot90(img, k=rot, dims=[2, 3])
            if flip:
                img_aug = torch.flip(img_aug, dims=[3])
            
            # 2. Inference
            with torch.no_grad():
                out_aug = model(img_aug)
            
            # 3. De-augment (inverso)
            if flip:
                out_aug = torch.flip(out_aug, dims=[3])
            out_aug = torch.rot90(out_aug, k=-rot, dims=[2, 3])
            
            output_list.append(out_aug)
    
    # 4. Media delle predizioni (Smoothing)
    return torch.stack(output_list, dim=0).mean(dim=0)

def detect_model_params(state_dict):
    """
    Tenta di dedurre i parametri del modello (embed_dim, depths) dal state_dict.
    Utile se il modello è stato addestrato con parametri diversi dal default.
    """
    params = {
        'embed_dim': 96,        # Default SwinIR standard
        'depths': [6, 6, 6, 6], # Default SwinIR standard
        'num_heads': [6, 6, 6, 6]
    }
    
    # 1. Rileva embed_dim da conv_first.weight [C_out, C_in, K, K]
    if 'conv_first.weight' in state_dict:
        params['embed_dim'] = state_dict['conv_first.weight'].shape[0]
        print(f" [Auto-Config] Rilevato embed_dim: {params['embed_dim']}")
    
    # 2. Rileva numero di layers (depths) basandosi sugli indici 'layers.X'
    max_layer_idx = -1
    for k in state_dict.keys():
        if k.startswith('layers.'):
            try:
                # Esempio chiave: layers.5.residual_group...
                parts = k.split('.')
                idx = int(parts[1])
                if idx > max_layer_idx:
                    max_layer_idx = idx
            except:
                pass
    
    num_layers = max_layer_idx + 1
    if num_layers > 0:
        # Ricostruisce una configurazione standard simmetrica per i layers trovati
        # Nota: SwinIR usa spesso 6 blocchi per layer
        params['depths'] = [6] * num_layers
        params['num_heads'] = [6] * num_layers
        print(f" [Auto-Config] Rilevati {num_layers} layers (depths={params['depths']})")
    
    return params

def get_available_targets(output_root: Path) -> List[str]:
    if not output_root.is_dir(): return []
    return sorted([p.name for p in output_root.iterdir() if p.is_dir()])

def run_test(target_model_folder: str):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Modalità SMOOTHING attiva (TTA x8) - Inferenza più lenta ma più pulita.")

    OUTPUT_DIR = OUTPUT_ROOT / target_model_folder / "test_results_smooth"
    CHECKPOINT_DIR = OUTPUT_ROOT / target_model_folder / "checkpoints"
    
    # Crea la cartella di output se non esiste
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    checkpoints = list(CHECKPOINT_DIR.glob("best_gan_model.pth"))
    if not checkpoints:
        checkpoints = list(CHECKPOINT_DIR.glob("latest_checkpoint.pth"))
    
    if not checkpoints:
        print("Nessun checkpoint trovato.")
        return
    
    CHECKPOINT_PATH = checkpoints[0]
    print(f"Loading checkpoint: {CHECKPOINT_PATH.name}")

    # Caricamento del dizionario pesi
    try:
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    except Exception as e:
        print(f"Errore lettura file .pth: {e}")
        return

    if 'net_g' in checkpoint: state_dict = checkpoint['net_g']
    elif 'model_state_dict' in checkpoint: state_dict = checkpoint['model_state_dict']
    else: state_dict = checkpoint

    # Pulizia chiavi (rimozione prefisso 'module.' se presente)
    clean_state_dict = {}
    for k, v in state_dict.items():
        name = k.replace("module.", "") 
        clean_state_dict[name] = v

    # --- AUTO-CONFIGURAZIONE MODELLO ---
    detected_params = detect_model_params(clean_state_dict)

    # Inizializzazione Modello con parametri rilevati
    model = SwinIR(
        upscale=4, 
        in_chans=1, 
        img_size=128, 
        window_size=8,
        img_range=1.0, 
        upsampler='pixelshuffle', 
        resi_connection='1conv',
        mlp_ratio=2,
        # Parametri dinamici
        embed_dim=detected_params['embed_dim'],
        depths=detected_params['depths'],
        num_heads=detected_params['num_heads']
    ).to(device)

    # Caricamento pesi nel modello
    try:
        model.load_state_dict(clean_state_dict, strict=True)
        print("Pesi caricati correttamente (Strict Mode).")
    except RuntimeError as e:
        print(f"\n[WARN] Strict loading fallito. Riprovo con strict=False.\nErrore parziale: {e}")
        model.load_state_dict(clean_state_dict, strict=False)

    model.eval()

    # --- PREPARAZIONE DATASET ---
    print("\nRicerca dataset di test...")
    # Ricava i target dal nome della cartella (es. "M1_M33_DDP_SwinIR" -> ["M1", "M33"])
    folder_clean = target_model_folder.replace("_DDP_SwinIR", "")
    targets = folder_clean.split("_")
    
    all_test_data = []
    found_any = False

    for t in targets:
        # Cerca il file test.json generato da prepare_data.py
        test_json_path = ROOT_DATA_DIR / t / "8_dataset_split" / "splits_json" / "test.json"
        if test_json_path.exists():
            print(f" -> Trovato test set per {t}")
            with open(test_json_path, 'r') as f:
                all_test_data.extend(json.load(f))
            found_any = True

    if not found_any or not all_test_data:
        print("Nessun dato di test trovato. Assicurati di aver eseguito prepare_data.py.")
        return

    # Crea file JSON temporaneo combinato
    TEMP_JSON = OUTPUT_DIR / "temp_test_combined.json"
    with open(TEMP_JSON, 'w') as f:
        json.dump(all_test_data, f)

    # Dataset e DataLoader
    test_ds = AstronomicalDataset(TEMP_JSON, base_path=PROJECT_ROOT, augment=False)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)
    
    metrics = TrainMetrics()
    print(f"Inizio inferenza su {len(test_ds)} immagini...\n")

    # --- CICLO DI INFERENZA ---
    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader, desc="Inference")):
            lr = batch['lr'].to(device)
            hr = batch['hr'].to(device)
            
            # Smoothing TTA (Media di 8 predizioni)
            sr = inference_tta(model, lr, device)
            
            # Calcolo Metriche
            sr_clamped = torch.clamp(sr, 0, 1)
            metrics.update(sr_clamped, hr)
            
            # Salvataggio
            save_as_tiff16(sr_clamped, OUTPUT_DIR / f"test_{i:04d}_sr_smooth.tiff")

    avg_psnr = metrics.psnr / metrics.count if metrics.count > 0 else 0
    print(f"\nTEST COMPLETATO. PSNR Medio: {avg_psnr:.2f} dB")
    print(f"Immagini salvate in: {OUTPUT_DIR}")

if __name__ == "__main__":
    targets = get_available_targets(OUTPUT_ROOT)
    if targets:
        print("Cartelle trovate:", targets)
        sel = input("Scrivi nome cartella: ")
        run_test(sel)
    else:
        print("Nessun output trovato.")
