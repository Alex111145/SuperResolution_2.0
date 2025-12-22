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

# Assicurati che i moduli esistano
try:
    from models.architecture import SwinIR
    from dataset.astronomical_dataset import AstronomicalDataset
    from utils.metrics import TrainMetrics
except ImportError as e:
    sys.exit(f"Errore Import: {e}. Verifica la struttura delle cartelle.")

torch.backends.cudnn.benchmark = True

def save_as_tiff16(tensor, path):
    """Salva il tensore come immagine TIFF a 16-bit."""
    arr = tensor.squeeze().float().cpu().numpy()
    arr = np.clip(arr, 0, 1)
    arr_u16 = (arr * 65535).astype(np.uint16)
    Image.fromarray(arr_u16, mode='I;16').save(path)

def inference_tta(model, img, device):
    """
    Test-Time Augmentation (TTA) per Smoothing.
    Esegue l'inferenza su 8 varianti (rotazioni/flip) e ne fa la media
    per rimuovere il rumore granuloso e gli artefatti geometrici.
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
            
            # 3. De-augment
            if flip:
                out_aug = torch.flip(out_aug, dims=[3])
            out_aug = torch.rot90(out_aug, k=-rot, dims=[2, 3])
            
            output_list.append(out_aug)
    
    # 4. Media (Smoothing)
    return torch.stack(output_list, dim=0).mean(dim=0)

def detect_model_params(state_dict):
    """
    Rileva i parametri specifici per l'architettura Swin 4.0.
    In questa versione, 'layers' è una lista di stadi, e ogni stadio è una lista di blocchi.
    Keys tipiche: layers.0.0.norm1... (Stadio 0, Blocco 0)
    """
    params = {
        'embed_dim': 96,        # Default
        'depths': [6, 6, 6],    # Default generico
        'num_heads': [6, 6, 6]  # Default generico
    }
    
    # 1. Rileva embed_dim da conv_first
    if 'conv_first.weight' in state_dict:
        params['embed_dim'] = state_dict['conv_first.weight'].shape[0]
        print(f" [Auto-Config] Rilevato embed_dim: {params['embed_dim']}")
    
    # 2. Rileva struttura layers (Stadi e Blocchi)
    stage_block_counts = {}
    
    for k in state_dict.keys():
        if k.startswith('layers.'):
            # Formato atteso Swin 4.0: layers.STAGE.BLOCK.param...
            parts = k.split('.')
            if len(parts) >= 3 and parts[1].isdigit() and parts[2].isdigit():
                stage = int(parts[1])
                block = int(parts[2])
                
                current_max = stage_block_counts.get(stage, -1)
                if block > current_max:
                    stage_block_counts[stage] = block
    
    if stage_block_counts:
        num_stages = max(stage_block_counts.keys()) + 1
        new_depths = []
        for i in range(num_stages):
            # +1 perché l'indice parte da 0 (es. max index 5 significa 6 blocchi)
            count = stage_block_counts.get(i, -1) + 1 
            new_depths.append(count)
        
        params['depths'] = new_depths
        # Assumiamo num_heads coerente col numero di stadi
        params['num_heads'] = [6] * num_stages
        
        print(f" [Auto-Config] Rilevati {num_stages} stadi. Depths: {new_depths}")
    
    return params

def get_available_targets(output_root: Path) -> List[str]:
    if not output_root.is_dir(): return []
    return sorted([p.name for p in output_root.iterdir() if p.is_dir()])

def run_test(target_model_folder: str):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Modalità SMOOTHING attiva (TTA x8).")

    OUTPUT_DIR = OUTPUT_ROOT / target_model_folder / "test_results_smooth"
    CHECKPOINT_DIR = OUTPUT_ROOT / target_model_folder / "checkpoints"
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    checkpoints = list(CHECKPOINT_DIR.glob("best_gan_model.pth"))
    if not checkpoints:
        checkpoints = list(CHECKPOINT_DIR.glob("latest_checkpoint.pth"))
    
    if not checkpoints:
        print("Nessun checkpoint trovato.")
        return
    
    CHECKPOINT_PATH = checkpoints[0]
    print(f"Loading checkpoint: {CHECKPOINT_PATH.name}")

    # Caricamento dizionario
    try:
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    except Exception as e:
        print(f"Errore lettura file .pth: {e}")
        return

    if 'net_g' in checkpoint: state_dict = checkpoint['net_g']
    elif 'model_state_dict' in checkpoint: state_dict = checkpoint['model_state_dict']
    else: state_dict = checkpoint

    # Pulizia chiavi
    clean_state_dict = {}
    for k, v in state_dict.items():
        name = k.replace("module.", "") 
        clean_state_dict[name] = v

    # --- AUTO-CONFIGURAZIONE ---
    detected_params = detect_model_params(clean_state_dict)

    # Inizializzazione Modello Swin 4.0
    model = SwinIR(
        upscale=4, 
        in_chans=1, 
        img_size=128, 
        window_size=8,
        # Parametri dinamici
        embed_dim=detected_params['embed_dim'],
        depths=detected_params['depths'],
        num_heads=detected_params['num_heads'],
        # Altri default
        mlp_ratio=2, 
        upsampler='pixelshuffle', 
        resi_connection='1conv'
    ).to(device)

    # Caricamento Pesi
    try:
        model.load_state_dict(clean_state_dict, strict=True)
        print("Pesi caricati correttamente (Strict Mode).")
    except RuntimeError as e:
        print(f"\n[WARN] Strict loading fallito. Riprovo con strict=False.\nErrore parziale: {e}")
        model.load_state_dict(clean_state_dict, strict=False)

    model.eval()

    # --- DATASET ---
    print("\nRicerca dataset di test...")
    folder_clean = target_model_folder.replace("_DDP_SwinIR", "")
    targets = folder_clean.split("_")
    
    all_test_data = []
    found_any = False

    for t in targets:
        test_json_path = ROOT_DATA_DIR / t / "8_dataset_split" / "splits_json" / "test.json"
        if test_json_path.exists():
            print(f" -> Trovato test set per {t}")
            with open(test_json_path, 'r') as f:
                all_test_data.extend(json.load(f))
            found_any = True

    if not found_any or not all_test_data:
        print("Nessun dato di test trovato (hai eseguito prepare_data.py?).")
        return

    TEMP_JSON = OUTPUT_DIR / "temp_test_combined.json"
    with open(TEMP_JSON, 'w') as f: json.dump(all_test_data, f)

    test_ds = AstronomicalDataset(TEMP_JSON, base_path=PROJECT_ROOT, augment=False)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)
    
    metrics = TrainMetrics()
    print(f"Inizio inferenza su {len(test_ds)} immagini...\n")

    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader, desc="Inference")):
            lr = batch['lr'].to(device)
            hr = batch['hr'].to(device)
            
            # Smoothing TTA
            sr = inference_tta(model, lr, device)
            
            sr_clamped = torch.clamp(sr, 0, 1)
            metrics.update(sr_clamped, hr)
            
            save_as_tiff16(sr_clamped, OUTPUT_DIR / f"test_{i:04d}_sr_smooth.tiff")

    avg_psnr = metrics.psnr / metrics.count if metrics.count > 0 else 0
    print(f"\nTEST COMPLETATO. PSNR Medio: {avg_psnr:.2f} dB")
    print(f"Risultati salvati in: {OUTPUT_DIR}")

if __name__ == "__main__":
    targets = get_available_targets(OUTPUT_ROOT)
    if targets:
        print("Cartelle trovate:", targets)
        sel = input("Scrivi nome cartella: ")
        run_test(sel)
    else:
        print("Nessun output trovato.")
