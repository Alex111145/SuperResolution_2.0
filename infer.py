import sys
import torch
import torch.nn.functional as F
import torchvision.utils as vutils
import numpy as np
import json
from pathlib import Path
from torch.utils.data import DataLoader
from PIL import Image
from tqdm import tqdm
from typing import List

# Configurazione Percorsi
CURRENT_SCRIPT = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_SCRIPT.parent
OUTPUT_ROOT = PROJECT_ROOT / "outputs"
ROOT_DATA_DIR = PROJECT_ROOT / "data"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import dai moduli del progetto
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

def detect_model_params(state_dict):
    """Tenta di dedurre i parametri del modello dal state_dict."""
    params = {'embed_dim': 96, 'depths': [6, 6, 6, 6], 'num_heads': [6, 6, 6, 6]}
    if 'conv_first.weight' in state_dict:
        params['embed_dim'] = state_dict['conv_first.weight'].shape[0]
    max_layer_idx = -1
    for k in state_dict.keys():
        if k.startswith('layers.'):
            try:
                idx = int(k.split('.')[1])
                if idx > max_layer_idx: max_layer_idx = idx
            except: pass
    num_layers = max_layer_idx + 1
    if num_layers > 0:
        params['depths'] = [6] * num_layers
        params['num_heads'] = [6] * num_layers
    return params

def get_available_targets(output_root: Path) -> List[str]:
    if not output_root.is_dir(): return []
    return sorted([p.name for p in output_root.iterdir() if p.is_dir()])

def run_test(target_model_folder: str):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print("Modalità: Inferenza standard con salvataggio comparativo PNG (Tris)")

    # Percorsi cartelle risultati
    BASE_RESULTS = OUTPUT_ROOT / target_model_folder / "test_results_standard"
    TIFF_DIR = BASE_RESULTS / "tiff_16bit"
    PNG_DIR = BASE_RESULTS / "comparison_png"
    CHECKPOINT_DIR = OUTPUT_ROOT / target_model_folder / "checkpoints"

    TIFF_DIR.mkdir(parents=True, exist_ok=True)
    PNG_DIR.mkdir(parents=True, exist_ok=True)
    
    checkpoints = list(CHECKPOINT_DIR.glob("best_gan_model.pth"))
    if not checkpoints:
        checkpoints = list(CHECKPOINT_DIR.glob("latest_checkpoint.pth"))
    
    if not checkpoints:
        print(f"Nessun checkpoint trovato in {CHECKPOINT_DIR}")
        return
    
    CHECKPOINT_PATH = checkpoints[0]
    print(f"Caricamento checkpoint: {CHECKPOINT_PATH.name}")

    try:
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
        state_dict = checkpoint.get('net_g', checkpoint.get('model_state_dict', checkpoint))
        clean_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        
        det_params = detect_model_params(clean_state_dict)
        model = SwinIR(
            upscale=4, in_chans=1, img_size=128, window_size=8,
            img_range=1.0, upsampler='pixelshuffle', resi_connection='1conv',
            mlp_ratio=2, embed_dim=det_params['embed_dim'],
            depths=det_params['depths'], num_heads=det_params['num_heads']
        ).to(device)

        model.load_state_dict(clean_state_dict, strict=True)
        print("Pesi caricati correttamente.")
    except Exception as e:
        print(f"Errore caricamento modello: {e}")
        return

    model.eval()

    # Ricerca JSON di test
    folder_clean = target_model_folder.replace("_DDP_SwinIR", "")
    targets_names = folder_clean.split("_")
    all_test_data = []

    for t in targets_names:
        test_json_path = ROOT_DATA_DIR / t / "8_dataset_split" / "splits_json" / "test.json"
        if test_json_path.exists():
            with open(test_json_path, 'r') as f:
                all_test_data.extend(json.load(f))

    if not all_test_data:
        print("Nessun dato di test trovato.")
        return

    TEMP_JSON = BASE_RESULTS / "temp_test.json"
    with open(TEMP_JSON, 'w') as f: json.dump(all_test_data, f)

    test_ds = AstronomicalDataset(str(TEMP_JSON), base_path=PROJECT_ROOT, augment=False)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=2)
    
    metrics = TrainMetrics()
    
    print(f"Esecuzione su {len(test_ds)} immagini...")

    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader, desc="Inference")):
            lr = batch['lr'].to(device)
            hr = batch['hr'].to(device)
            
            # 1. Inferenza
            sr = model(lr)
            sr_clamped = torch.clamp(sr, 0, 1)
            
            # 2. Creazione Comparativa "Tris" (LR Upscalato | SR | HR)
            # Upsample LR con nearest per matchare le dimensioni di SR e HR
            lr_up = F.interpolate(lr, size=sr_clamped.shape[2:], mode='nearest')
            
            # Concatenazione orizzontale (dim=3)
            comparison = torch.cat((lr_up, sr_clamped, hr), dim=3)
            
            # 3. Salvataggi
            save_as_tiff16(sr_clamped, TIFF_DIR / f"test_{i:04d}_sr.tiff")
            vutils.save_image(comparison, PNG_DIR / f"test_{i:04d}_tris.png")
            
            # Aggiornamento metriche
            metrics.update(sr_clamped, hr)

    avg_psnr = metrics.psnr / metrics.count if metrics.count > 0 else 0
    print(f"\nCOMPLETATO. PSNR Medio: {avg_psnr:.2f} dB")
    print(f"TIFF salvati in: {TIFF_DIR}")
    print(f"PNG comparativi salvati in: {PNG_DIR}")

if __name__ == "__main__":
    available = get_available_targets(OUTPUT_ROOT)
    if available:
        print("Cartelle trovate:", available)
        sel = input("Scrivi nome cartella: ")
        run_test(sel)
    else:
        print("Nessun output trovato.")
