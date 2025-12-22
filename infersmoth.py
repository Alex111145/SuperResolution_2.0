import sys
import torch
import numpy as np
import json
import os
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

# Importazione moduli aggiornati per HAT
try:
    from models.hat_arch import HAT
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
    Test-Time Augmentation (TTA) per Smoothing con HAT.
    Esegue l'inferenza su 8 varianti (rotazioni/flip) e ne fa la media.
    """
    output_list = []
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

def detect_hat_params(state_dict):
    """
    Rileva i parametri specifici per l'architettura HAT.
    HAT usa RHAG (layers) che contengono HAB (blocks).
    Pattern chiavi: layers.i.residual_group.blocks.j...
    """
    params = {
        'embed_dim': 180,
        'depths': [6, 6, 6, 6, 6, 6],
        'num_heads': [6, 6, 6, 6, 6, 6]
    }
    
    # 1. Rileva embed_dim da conv_first
    if 'conv_first.weight' in state_dict:
        params['embed_dim'] = state_dict['conv_first.weight'].shape[0]
        print(f" [Auto-Config] Rilevato embed_dim: {params['embed_dim']}")
    
    # 2. Rileva struttura layers (RHAG e Blocks)
    rhag_block_counts = {}
    for k in state_dict.keys():
        if k.startswith('layers.'):
            # Pattern: layers.0.residual_group.blocks.0.norm1.weight
            parts = k.split('.')
            if len(parts) >= 5 and parts[1].isdigit() and parts[4].isdigit():
                rhag_idx = int(parts[1])
                block_idx = int(parts[4])
                current_max = rhag_block_counts.get(rhag_idx, -1)
                if block_idx > current_max:
                    rhag_block_counts[rhag_idx] = block_idx
    
    if rhag_block_counts:
        num_rhags = max(rhag_block_counts.keys()) + 1
        new_depths = [rhag_block_counts.get(i, -1) + 1 for i in range(num_rhags)]
        params['depths'] = new_depths
        params['num_heads'] = [6] * num_rhags # Default per HAT
        print(f" [Auto-Config] Rilevati {num_rhags} RHAG. Depths: {new_depths}")
    
    return params

def get_available_targets(output_root: Path) -> List[str]:
    if not output_root.is_dir(): return []
    return sorted([p.name for p in output_root.iterdir() if p.is_dir()])

def run_test(target_model_folder: str):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device} | HAT Smoothing Active")

    OUTPUT_DIR = OUTPUT_ROOT / target_model_folder / "test_results_hat_smooth"
    CHECKPOINT_DIR = OUTPUT_ROOT / target_model_folder / "checkpoints"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    checkpoints = list(CHECKPOINT_DIR.glob("best_gan_model.pth"))
    if not checkpoints: checkpoints = list(CHECKPOINT_DIR.glob("latest_checkpoint.pth"))
    if not checkpoints:
        print("Nessun checkpoint trovato.")
        return
    
    CHECKPOINT_PATH = checkpoints[0]
    try:
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    except Exception as e:
        print(f"Errore lettura file: {e}")
        return

    # Estrazione state_dict
    state_dict = checkpoint.get('net_g', checkpoint.get('model_state_dict', checkpoint))
    clean_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    # Auto-Configurazione per HAT
    detected = detect_hat_params(clean_state_dict)

    # Inizializzazione HAT Engine
    model = HAT(
        img_size=128,
        in_chans=1,
        upscale=4,
        window_size=7, # Default HAT
        embed_dim=detected['embed_dim'],
        depths=detected['depths'],
        num_heads=detected['num_heads'],
        mlp_ratio=2,
        upsampler='pixelshuffle',
        resi_connection='1conv'
    ).to(device)

    model.load_state_dict(clean_state_dict, strict=True)
    model.eval()

    # --- DATI ---
    folder_clean = target_model_folder.split("_DDP")[0]
    targets = folder_clean.split("_")
    all_test_data = []
    for t in targets:
        test_json = ROOT_DATA_DIR / t / "8_dataset_split" / "splits_json" / "test.json"
        if test_json.exists():
            with open(test_json, 'r') as f:
                all_test_data.extend(json.load(f))

    if not all_test_data:
        print("Nessun dato di test trovato.")
        return

    TEMP_JSON = OUTPUT_DIR / "temp_test.json"
    with open(TEMP_JSON, 'w') as f: json.dump(all_test_data, f)
    test_ds = AstronomicalDataset(TEMP_JSON, base_path=PROJECT_ROOT, augment=False)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)
    
    metrics = TrainMetrics()
    print(f"Inferenza HAT su {len(test_ds)} immagini...")

    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader, desc="HAT-TTA")):
            lr, hr = batch['lr'].to(device), batch['hr'].to(device)
            sr = inference_tta(model, lr, device)
            sr_clamped = torch.clamp(sr, 0, 1)
            metrics.update(sr_clamped, hr)
            save_as_tiff16(sr_clamped, OUTPUT_DIR / f"test_{i:04d}_sr_hat_smooth.tiff")

    print(f"\nTEST HAT COMPLETATO. PSNR: {metrics.psnr/metrics.count:.2f} dB")

if __name__ == "__main__":
    targets = get_available_targets(OUTPUT_ROOT)
    if targets:
        print("Output trovati:", targets)
        run_test(input("Cartella: "))
    else:
        print("Nessun output.")
