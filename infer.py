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

# --- CONFIGURAZIONE PERCORSI ---
CURRENT_SCRIPT = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_SCRIPT.parent
OUTPUT_ROOT = PROJECT_ROOT / "outputs"
ROOT_DATA_DIR = PROJECT_ROOT / "data"

# Aggiunge la root del progetto al path per consentire gli import
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# --- IMPORT MODULI PROGETTO ---
try:
    from models.architecture import SwinIR
    from dataset.astronomical_dataset import AstronomicalDataset
    from utils.metrics import TrainMetrics
except ImportError as e:
    sys.exit(f"Errore Import: {e}. Verifica di eseguire lo script dalla root del progetto o che la struttura delle cartelle sia corretta.")

# Ottimizzazione CUDNN
torch.backends.cudnn.benchmark = True

def save_as_tiff16(tensor, path):
    """
    Salva il tensore (normalizzato 0-1) come immagine TIFF a 16-bit.
    """
    arr = tensor.squeeze().float().cpu().numpy()
    arr = np.clip(arr, 0, 1)
    # Conversione a 16-bit (0-65535)
    arr_u16 = (arr * 65535).astype(np.uint16)
    Image.fromarray(arr_u16, mode='I;16').save(path)

def detect_model_params(state_dict):
    """
    Tenta di dedurre i parametri strutturali del modello (embed_dim, profondità, head)
    analizzando le chiavi dei pesi salvati (state_dict).
    """
    params = {'embed_dim': 96, 'depths': [6, 6, 6, 6], 'num_heads': [6, 6, 6, 6]}
    
    # Rileva embed_dim dalla prima convoluzione
    if 'conv_first.weight' in state_dict:
        params['embed_dim'] = state_dict['conv_first.weight'].shape[0]
    
    # Rileva il numero di layer (RSTB blocks)
    max_layer_idx = -1
    for k in state_dict.keys():
        if k.startswith('layers.'):
            try:
                # Esempio chiave: layers.0.residual_group...
                idx = int(k.split('.')[1])
                if idx > max_layer_idx:
                    max_layer_idx = idx
            except:
                pass
    
    num_layers = max_layer_idx + 1
    if num_layers > 0:
        # Imposta depths e heads in base al numero di layer trovati
        # Nota: questo assume una struttura simmetrica standard (es. tutti 6)
        params['depths'] = [6] * num_layers
        params['num_heads'] = [6] * num_layers
        
    print(f"Parametri rilevati: dim={params['embed_dim']}, layers={num_layers}")
    return params

def get_available_targets(output_root: Path) -> List[str]:
    """Restituisce la lista delle cartelle presenti in 'outputs'."""
    if not output_root.is_dir():
        return []
    return sorted([p.name for p in output_root.iterdir() if p.is_dir()])

def run_test(target_model_folder: str):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device in uso: {device}")
    print("Modalità: Inferenza standard con salvataggio comparativo PNG (Tris)")

    # --- PERCORSI FILES ---
    BASE_RESULTS = OUTPUT_ROOT / target_model_folder / "test_results_standard"
    TIFF_DIR = BASE_RESULTS / "tiff_16bit"
    PNG_DIR = BASE_RESULTS / "comparison_png"
    CHECKPOINT_DIR = OUTPUT_ROOT / target_model_folder / "checkpoints"

    TIFF_DIR.mkdir(parents=True, exist_ok=True)
    PNG_DIR.mkdir(parents=True, exist_ok=True)
    
    # Ricerca checkpoint
    checkpoints = list(CHECKPOINT_DIR.glob("best_gan_model.pth"))
    if not checkpoints:
        checkpoints = list(CHECKPOINT_DIR.glob("latest_checkpoint.pth"))
    
    if not checkpoints:
        print(f"Nessun checkpoint trovato in {CHECKPOINT_DIR}")
        return
    
    CHECKPOINT_PATH = checkpoints[0]
    print(f"Caricamento checkpoint: {CHECKPOINT_PATH.name}")

    # --- CARICAMENTO MODELLO ---
    try:
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
        # Gestione compatibilità nomi chiavi
        state_dict = checkpoint.get('net_g', checkpoint.get('model_state_dict', checkpoint))
        clean_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        
        # Rilevamento automatico parametri
        det_params = detect_model_params(clean_state_dict)
        
        model = SwinIR(
            upscale=4, 
            in_chans=1, 
            img_size=128, 
            window_size=8,
            img_range=1.0, 
            upsampler='pixelshuffle', 
            resi_connection='1conv',
            mlp_ratio=2, 
            embed_dim=det_params['embed_dim'],
            depths=det_params['depths'], 
            num_heads=det_params['num_heads']
        ).to(device)

        model.load_state_dict(clean_state_dict, strict=True)
        print("Pesi caricati correttamente.")
    except Exception as e:
        print(f"Errore critico nel caricamento del modello: {e}")
        return

    model.eval()

    # --- PREPARAZIONE DATASET ---
    # Deduce il nome del dataset originale dal nome della cartella output
    # Es: "Astronomical_DDP_SwinIR" -> cerca dati per "Astronomical"
    folder_clean = target_model_folder.replace("_DDP_SwinIR", "")
    targets_names = folder_clean.split("_")
    all_test_data = []

    print("Ricerca file test.json...")
    for t in targets_names:
        # Percorso adattato alla struttura: data/<nome>/8_dataset_split/splits_json/test.json
        test_json_path = ROOT_DATA_DIR / t / "8_dataset_split" / "splits_json" / "test.json"
        if test_json_path.exists():
            print(f" -> Trovato: {test_json_path}")
            with open(test_json_path, 'r') as f:
                all_test_data.extend(json.load(f))
        else:
            print(f" -> Non trovato per target: {t}")

    if not all_test_data:
        print("Nessun dato di test trovato. Verifica la cartella 'data'.")
        return

    # Salva un JSON temporaneo combinato per il Dataset
    TEMP_JSON = BASE_RESULTS / "temp_test.json"
    with open(TEMP_JSON, 'w') as f:
        json.dump(all_test_data, f)

    test_ds = AstronomicalDataset(str(TEMP_JSON), base_path=PROJECT_ROOT, augment=False)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=2)
    
    metrics = TrainMetrics()
    
    print(f"Inizio inferenza su {len(test_ds)} immagini...")

    # --- LOOP DI INFERENZA ---
    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader, desc="Processing")):
            lr = batch['lr'].to(device)
            hr = batch['hr'].to(device)
            
            # 1. Super-Resolution
            sr = model(lr)
            sr_clamped = torch.clamp(sr, 0, 1)
            
            # 2. Creazione "Tris" (Low Res Upscaled | Super Res | High Res)
            # Upsample Nearest Neighbor della LR per portarla alle dimensioni della HR/SR
            lr_up = F.interpolate(lr, size=sr_clamped.shape[2:], mode='nearest')
            
            # Concatenazione sull'asse larghezza (dim=3 per tensori N,C,H,W)
            comparison = torch.cat((lr_up, sr_clamped, hr), dim=3)
            
            # 3. Salvataggi
            # TIFF 16-bit per analisi scientifica
            save_as_tiff16(sr_clamped, TIFF_DIR / f"test_{i:04d}_sr.tiff")
            # PNG per visualizzazione rapida (comparison)
            vutils.save_image(comparison, PNG_DIR / f"test_{i:04d}_tris.png")
            
            # 4. Calcolo Metriche
            metrics.update(sr_clamped, hr)

    avg_psnr = metrics.psnr / metrics.count if metrics.count > 0 else 0
    
    print("\n" + "="*50)
    print(f"TEST COMPLETATO.")
    print(f"PSNR Medio: {avg_psnr:.2f} dB")
    print(f"TIFF salvati in: {TIFF_DIR}")
    print(f"PNG comparativi in: {PNG_DIR}")
    print("="*50)

if __name__ == "__main__":
    available = get_available_targets(OUTPUT_ROOT)
    if available:
        print("\nCartelle output disponibili:", available)
        sel = input("Scrivi il nome della cartella da testare: ").strip()
        if sel:
            run_test(sel)
        else:
            print("Selezione annullata.")
    else:
        print("Nessuna cartella trovata in 'outputs'.")
