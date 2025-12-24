import sys
import os
import torch
import torch.nn.functional as F
import torchvision.utils as vutils
import numpy as np
import json
import warnings
from pathlib import Path
from torch.utils.data import DataLoader
from PIL import Image
from tqdm import tqdm
from typing import List

# === PATCH TORCHVISION (Necessaria per HAT) ===
import torchvision.transforms.functional as TF_functional
sys.modules['torchvision.transforms.functional_tensor'] = TF_functional

# --- CONFIGURAZIONE PERCORSI ---
CURRENT_SCRIPT = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_SCRIPT.parent
OUTPUT_ROOT = PROJECT_ROOT / "outputs"
ROOT_DATA_DIR = PROJECT_ROOT / "data"

# Aggiunge la root del progetto al path
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# --- IMPORT MODULI PROGETTO ---
try:
    from models.hybridmodels import HybridHATRealESRGAN
    from dataset.astronomical_dataset import AstronomicalDataset
    from utils.metrics import TrainMetrics
except ImportError as e:
    sys.exit(f"❌ Errore Import: {e}. Assicurati di essere nella root del progetto.")

# Ottimizzazione
torch.backends.cudnn.benchmark = True
warnings.filterwarnings("ignore", category=UserWarning)

def save_as_tiff16(tensor, path):
    """
    Salva il tensore (normalizzato 0-1) come immagine TIFF a 16-bit.
    """
    arr = tensor.squeeze().float().cpu().numpy()
    arr = np.clip(arr, 0, 1)
    # Conversione a 16-bit (0-65535)
    arr_u16 = (arr * 65535).astype(np.uint16)
    Image.fromarray(arr_u16, mode='I;16').save(path)

def detect_hybrid_params(state_dict):
    """
    Analizza i pesi salvati per dedurre la configurazione del modello (Small vs Large).
    """
    params = {
        'img_size': 128,
        'in_chans': 1,
        'embed_dim': 180,       # Default Large
        'depths': (6, 6, 6, 6, 6, 6),
        'num_heads': (6, 6, 6, 6, 6, 6),
        'window_size': 8,       # Default HAT moderno
        'upscale': 4,
        'num_rrdb': 23,         # Default Large
        'num_feat': 64,         # Default Large
        'num_grow_ch': 32       # Default Large
    }
    
    print("🔍 Analisi parametri dal checkpoint...")
    
    # 1. Rileva embed_dim (dalla prima conv di HAT)
    if 'hat.conv_first.weight' in state_dict:
        e_dim = state_dict['hat.conv_first.weight'].shape[0]
        params['embed_dim'] = e_dim
        print(f"   • Embed Dim rilevata: {e_dim}")

    # 2. Rileva num_feat (dall'adattatore tra HAT e RRDB)
    if 'conv_adapt.weight' in state_dict:
        n_feat = state_dict['conv_adapt.weight'].shape[0]
        params['num_feat'] = n_feat
        print(f"   • Num Feat rilevati: {n_feat}")

    # 3. Rileva num_grow_ch (dal primo blocco RRDB)
    if 'rrdb_trunk.0.rdb1.conv1.weight' in state_dict:
        g_ch = state_dict['rrdb_trunk.0.rdb1.conv1.weight'].shape[0]
        params['num_grow_ch'] = g_ch
        print(f"   • Growth Channel rilevati: {g_ch}")

    # 4. Rileva numero blocchi RRDB
    max_rrdb_idx = -1
    for k in state_dict.keys():
        if k.startswith('rrdb_trunk.'):
            try:
                # k è tipo rrdb_trunk.10.rdb1...
                idx = int(k.split('.')[1])
                if idx > max_rrdb_idx:
                    max_rrdb_idx = idx
            except: pass
    if max_rrdb_idx >= 0:
        params['num_rrdb'] = max_rrdb_idx + 1
        print(f"   • Blocchi RRDB rilevati: {params['num_rrdb']}")

    # 5. Rileva numero stadi HAT (depths)
    max_stage_idx = -1
    for k in state_dict.keys():
        if k.startswith('hat.layers.'):
            try:
                idx = int(k.split('.')[2]) # hat.layers.0...
                if idx > max_stage_idx:
                    max_stage_idx = idx
            except: pass
            
    if max_stage_idx >= 0:
        num_stages = max_stage_idx + 1
        params['depths'] = tuple([6] * num_stages)
        params['num_heads'] = tuple([6] * num_stages)
        print(f"   • Stadi HAT rilevati: {num_stages} (Depths: {params['depths']})")

    # Fix Window Size: se i pesi Bias Table non matchano, si prova a inferire o si tiene 8
    # HAT usa una relative_position_bias_table di dimensione (2*window_size-1)^2
    # Ma è difficile dedurlo al volo, assumiamo 8 (standard del tuo training)
    
    return params

def get_available_targets(output_root: Path) -> List[str]:
    """Restituisce la lista delle cartelle presenti in 'outputs'."""
    if not output_root.is_dir():
        return []
    return sorted([p.name for p in output_root.iterdir() if p.is_dir()])

def run_test(target_model_folder: str):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🖥️  Device in uso: {device}")
    print("📋 Modalità: Inferenza Hybrid HAT+ESRGAN con output Tris (LR|SR|HR)")

    # --- PERCORSI FILES ---
    # Gestione percorsi: se l'utente seleziona una sottocartella tipo 'checkpoints', risaliamo
    target_path = OUTPUT_ROOT / target_model_folder
    
    BASE_RESULTS = target_path / "test_results_hybrid"
    TIFF_DIR = BASE_RESULTS / "tiff_16bit"
    PNG_DIR = BASE_RESULTS / "comparison_png"
    CHECKPOINT_DIR = target_path / "checkpoints"
    
    # Fallback se non esiste cartella checkpoints standard
    if not CHECKPOINT_DIR.exists():
        CHECKPOINT_DIR = target_path

    TIFF_DIR.mkdir(parents=True, exist_ok=True)
    PNG_DIR.mkdir(parents=True, exist_ok=True)
    
    # --- RICERCA CHECKPOINT ---
    # Priorità: best_hybrid_model.pth -> hybrid_epoch_XXX.pth -> qualsiasi .pth
    checkpoints = list(CHECKPOINT_DIR.glob("best_hybrid_model.pth"))
    if not checkpoints:
        pths = list(CHECKPOINT_DIR.glob("hybrid_epoch_*.pth"))
        if pths:
            # Ordina per numero epoca
            checkpoints = [sorted(pths, key=lambda x: int(x.stem.split('_')[-1]))[-1]]
    
    if not checkpoints:
        # Ultimo tentativo: qualsiasi .pth
        checkpoints = list(CHECKPOINT_DIR.glob("*.pth"))

    if not checkpoints:
        print(f"❌ Nessun checkpoint trovato in {CHECKPOINT_DIR}")
        return
    
    CHECKPOINT_PATH = checkpoints[0]
    print(f"📦 Caricamento checkpoint: {CHECKPOINT_PATH.name}")

    # --- CARICAMENTO MODELLO ---
    try:
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
        
        # Estrazione pesi
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'net_g' in checkpoint:
            state_dict = checkpoint['net_g']
        else:
            state_dict = checkpoint

        clean_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        
        # Rilevamento automatico parametri (Small vs Large)
        model_params = detect_hybrid_params(clean_state_dict)
        
        model = HybridHATRealESRGAN(**model_params).to(device)

        # Caricamento strict=True per verificare che la detection abbia funzionato
        model.load_state_dict(clean_state_dict, strict=True)
        print("✓ Pesi caricati correttamente (Strict Mode).")
        
    except Exception as e:
        print(f"❌ Errore critico nel caricamento del modello: {e}")
        print("   Suggerimento: Verifica che il file models/hybridmodels.py sia aggiornato.")
        return

    model.eval()

    # --- PREPARAZIONE DATASET ---
    # Cerca dataset dai nomi della cartella (es. M1_M33 -> cerca M1 e M33)
    # Rimuove eventuali suffissi comuni
    clean_name = target_model_folder.replace("_Hybrid", "").replace("_SR", "")
    targets_names = clean_name.split("_")
    
    all_test_data = []
    print("\n📂 Ricerca Dataset Test...")
    
    for t in targets_names:
        # Cerca file test.json standard
        json_path = ROOT_DATA_DIR / t / "8_dataset_split" / "splits_json" / "test.json"
        
        if json_path.exists():
            print(f"   -> Trovato: {t}")
            with open(json_path, 'r') as f:
                data = json.load(f)
                # Aggiunge info sul target per debug se necessario
                for item in data: item['target_name'] = t 
                all_test_data.extend(data)
        else:
            # Fallback: prova a cercare immagini direttamente se non c'è il json
            # (Codice semplificato: assume ci sia il json come da training)
            print(f"   ⚠️  Non trovato JSON per: {t}")

    if not all_test_data:
        print("❌ Nessun dato di test trovato. Verifica le cartelle in 'data/'.")
        return

    # Salva JSON temporaneo
    TEMP_JSON = BASE_RESULTS / "temp_test_infer.json"
    with open(TEMP_JSON, 'w') as f:
        json.dump(all_test_data, f)

    test_ds = AstronomicalDataset(str(TEMP_JSON), base_path=PROJECT_ROOT, augment=False)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=2)
    
    metrics = TrainMetrics()
    
    print(f"\n🚀 Inizio inferenza su {len(test_ds)} immagini...")

    # --- LOOP DI INFERENZA ---
    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader, desc="Upscaling", unit="img")):
            lr = batch['lr'].to(device)
            hr = batch['hr'].to(device)
            
            # 1. Super-Resolution
            sr = model(lr)
            sr_clamped = torch.clamp(sr, 0, 1)
            
            # 2. Creazione "Tris" (LR | SR | HR)
            # Upsample Nearest della LR per confronto dimensionale
            lr_up = F.interpolate(lr, size=sr_clamped.shape[2:], mode='nearest')
            comparison = torch.cat((lr_up, sr_clamped, hr), dim=3)
            
            # 3. Salvataggi
            # TIFF 16-bit
            save_as_tiff16(sr_clamped, TIFF_DIR / f"test_{i:04d}_sr.tiff")
            # PNG Tris
            vutils.save_image(comparison, PNG_DIR / f"test_{i:04d}_tris.png")
            
            # 4. Calcolo Metriche
            metrics.update(sr_clamped, hr)

    # Clean
    if TEMP_JSON.exists(): os.remove(TEMP_JSON)

    results = metrics.compute()
    print("\n" + "="*50)
    print(f"✅ TEST COMPLETATO.")
    print(f"📊 PSNR Medio: {results['psnr']:.2f} dB")
    print(f"📊 SSIM Medio: {results['ssim']:.4f}")
    print(f"💾 TIFF salvati in: .../{TIFF_DIR.parent.name}/{TIFF_DIR.name}")
    print(f"🖼️  PNG salvati in:  .../{PNG_DIR.parent.name}/{PNG_DIR.name}")
    print("="*50)

if __name__ == "__main__":
    available = get_available_targets(OUTPUT_ROOT)
    
    if available:
        print("\n==========================================")
        print("      INFERENZA HYBRID HAT (XPixel)       ")
        print("==========================================\n")
        print("Cartelle output disponibili:")
        for idx, name in enumerate(available):
            print(f"   [{idx}] {name}")
            
        sel = input("\nSeleziona numero o nome cartella: ").strip()
        
        target = None
        if sel.isdigit():
            idx = int(sel)
            if 0 <= idx < len(available):
                target = available[idx]
        elif sel in available:
            target = sel
            
        if target:
            run_test(target)
        else:
            print("❌ Selezione non valida.")
    else:
        print("❌ Nessuna cartella trovata in 'outputs'. Assicurati di aver fatto il training.")
