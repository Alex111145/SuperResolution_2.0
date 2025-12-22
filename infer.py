import sys
import torch
import numpy as np
import shutil
import json
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

from models.architecture import SwinIR
from dataset.astronomical_dataset import AstronomicalDataset
from utils.metrics import TrainMetrics

torch.backends.cudnn.benchmark = True

def save_as_tiff16(tensor, path):
    arr = tensor.squeeze().float().cpu().numpy()
    arr = np.clip(arr, 0, 1)
    arr_u16 = (arr * 65535).astype(np.uint16)
    Image.fromarray(arr_u16, mode='I;16').save(path)

def get_available_targets(output_root: Path) -> List[str]:
    if not output_root.is_dir(): return []
    return sorted([p.name for p in output_root.iterdir() if p.is_dir()])

def run_test(target_model_folder: str):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    OUTPUT_DIR = OUTPUT_ROOT / target_model_folder / "test_results_tiff"
    CHECKPOINT_DIR = OUTPUT_ROOT / target_model_folder / "checkpoints"
    
    checkpoints = list(CHECKPOINT_DIR.glob("best_gan_model.pth"))
    if not checkpoints:
        checkpoints = list(CHECKPOINT_DIR.glob("latest_checkpoint.pth"))
    
    if not checkpoints:
        print("Nessun checkpoint trovato.")
        return
    
    CHECKPOINT_PATH = checkpoints[0]
    print(f"Loading: {CHECKPOINT_PATH}")

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
    
    print("Script inferenza pronto. Assicurati di passare un JSON valido a AstronomicalDataset.")

if __name__ == "__main__":
    targets = get_available_targets(OUTPUT_ROOT)
    if targets:
        print("Cartelle trovate:", targets)
        sel = input("Scrivi nome cartella: ")
        run_test(sel)
    else:
        print("Nessun output trovato.")
