import sys
import torch
import numpy as np
import shutil
import json
import os
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from PIL import Image
from tqdm import tqdm
from typing import List, Optional, Dict

CURRENT_SCRIPT = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_SCRIPT.parent
OUTPUT_ROOT = PROJECT_ROOT / "outputs"
ROOT_DATA_DIR = PROJECT_ROOT / "data"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Assicurati che i file esistano
from models.architecture import SwinIR
from dataset.astronomical_dataset import AstronomicalDataset
from utils.metrics import TrainMetrics

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
    output = torch.stack(output_list, dim=0).mean(dim=0)
    return output

def get_available_targets(output_root: Path) -> List[str]:
    if not output_root.is_dir(): return []
    return sorted([p.name for p in output_root.iterdir() if p.is_dir()])

def run_test(target_model_folder: str):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Modalità SMOOTHING attiva (TTA x8) - L'elaborazione sarà più lenta ma più pulita.")

    OUTPUT_DIR = OUTPUT_ROOT / target_model_folder / "test_results_smooth"
    CHECKPOINT_DIR = OUTPUT_ROOT / target_model_folder / "checkpoints"
    
    # Crea la cartella di output
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    checkpoints = list(CHECKPOINT_DIR.glob("best_gan_model.pth"))
    if not checkpoints:
        checkpoints = list(CHECKPOINT_DIR.glob("latest_checkpoint.pth"))
    
    if not checkpoints:
        print("Nessun checkpoint trovato.")
        return
    
    CHECKPOINT_PATH = checkpoints[0]
    print(f"Loading: {CHECKPOINT_PATH}")

    # --- CONFIGURAZIONE MODELLO ---
    model = SwinIR(upscale=4, in_chans=1, img_size=128, window_size=8,
                   img_range=1.0, depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                   mlp_ratio=2, upsampler='pixelshuffle', resi_connection='1conv').to(device)

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

    # --- PREPARAZIONE DATASET ---
    print("\nRicerca dataset di test...")
    folder_clean = target_model_folder.replace("_DDP_SwinIR", "")
    targets = folder_clean.split("_")
    
    all_test_data = []
    found_any = False

    for t in targets:
        test_json_path = ROOT_DATA_DIR / t / "8_dataset_split" / "splits_json" / "test.json"
        if test_json_path.exists():
            print(f" -> Trovato test set per {t}: {test_json_path}")
            with open(test_json_path, 'r') as f:
                data = json.load(f)
                all_test_data.extend(data)
            found_any = True
        else:
            print(f" [!] ATTENZIONE: Nessun test.json trovato per {t}")

    if not found_any or not all_test_data:
        print("Nessun dato di test trovato.")
        return

    TEMP_JSON = OUTPUT_DIR / "temp_test_combined.json"
    with open(TEMP_JSON, 'w') as f:
        json.dump(all_test_data, f)

    test_ds = AstronomicalDataset(TEMP_JSON, base_path=PROJECT_ROOT, augment=False)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)
    
    metrics = TrainMetrics()
    print(f"Inizio inferenza (con Smoothing TTA) su {len(test_ds)} immagini...\n")

    # --- CICLO DI INFERENZA CON TTA ---
    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader, desc="Processing")):
            lr = batch['lr'].to(device)
            hr = batch['hr'].to(device)
            
            # Usa la funzione TTA invece della chiamata diretta al modello
            sr = inference_tta(model, lr, device)
            
            # Calcolo Metriche
            sr_clamped = torch.clamp(sr, 0, 1)
            metrics.update(sr_clamped, hr)
            
            # Salvataggio
            filename = f"test_{i:04d}_sr_smooth.tiff"
            save_path = OUTPUT_DIR / filename
            save_as_tiff16(sr_clamped, save_path)

    avg_psnr = metrics.psnr / metrics.count if metrics.count > 0 else 0
    
    print("\n" + "="*40)
    print(f"TEST COMPLETATO.")
    print(f"Immagini salvate in: {OUTPUT_DIR}")
    print(f"Average PSNR: {avg_psnr:.2f} dB")
    print("="*40)

if __name__ == "__main__":
    targets = get_available_targets(OUTPUT_ROOT)
    if targets:
        print("Cartelle trovate:", targets)
        sel = input("Scrivi nome cartella: ")
        run_test(sel)
    else:
        print("Nessun output trovato.")
