import os
import shutil
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from astropy.io import fits
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import sys

CURRENT_SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_SCRIPT_DIR.parent
ROOT_DATA_DIR = PROJECT_ROOT / "data"

USE_LOG_STRETCH = True 
BLACK_CLIP_PERCENTILE = 4.0 

NUM_SAMPLES_PER_IMG = 4000 
BATCH_SIZE = 32
NUM_WORKERS = 4
DEBUG_INTERVAL = 50 
class RawFitsDataset(Dataset):
    """Dataset semplificato per caricare i file FITS per il campionamento statistico."""
    def __init__(self, file_list):
        self.file_list = file_list

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        path = self.file_list[idx]
        try:
            with fits.open(path) as hdul:
                data = hdul[0].data
                data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
                tensor = torch.from_numpy(data.astype(np.float32))
                return tensor
        except Exception:
            return torch.zeros((1, 1))

def calculate_robust_stats(file_list):
    """
    Calcola i percentili robusti (minimo e massimo) per definire il range dinamico.
    Applica il log stretch prima del campionamento se richiesto.
    """
    if not file_list:
        return 0.0, 1.0

    print(f"Campionamento statistico su {len(file_list)} immagini...")
    
    dataset = RawFitsDataset(file_list)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=False)
    
    sampled_pixels = []
    
    for batch in tqdm(loader, desc=" 	 Sampling", ncols=100):
        if USE_LOG_STRETCH:
            batch = torch.log1p(torch.maximum(batch, torch.tensor(0.0)))

        batch_flat = batch.view(-1)
        valid_mask = batch_flat > 1e-5 
        valid_pixels = batch_flat[valid_mask]
        
        if valid_pixels.numel() > 0:
            num_take = min(valid_pixels.numel(), NUM_SAMPLES_PER_IMG * batch.shape[0])
            indices = torch.randperm(valid_pixels.numel())[:num_take]
            sampled_pixels.append(valid_pixels[indices].numpy())
            
    if not sampled_pixels:
        print("Nessun dato valido trovato! Uso fallback 0-1.")
        return 0.0, 1.0

    full_sample = np.concatenate(sampled_pixels)
    
    global_min = np.percentile(full_sample, BLACK_CLIP_PERCENTILE) 
    global_max = np.percentile(full_sample, 99.99)
    
    print(f"Min (Nero) calcolato al {BLACK_CLIP_PERCENTILE}° percentile: {global_min:.4f}")
    
    return global_min, global_max

def save_debug_png(hr_raw, lr_raw, hr_norm, lr_norm, clip_percentile, save_path):
    """Salva una scheda debug per visualizzare il risultato della normalizzazione."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 10), facecolor='#000000')
    
    axes[0,0].imshow(np.log1p(np.maximum(hr_raw, 1e-5)), cmap='inferno')
    axes[0,0].set_title("Hubble RAW (Log)", color='white')
    
    axes[0,1].imshow(np.log1p(np.maximum(lr_raw, 1e-5)), cmap='viridis')
    axes[0,1].set_title("Obs RAW (Log)", color='white')
    
    axes[1,0].imshow(hr_norm, cmap='gray', vmin=0, vmax=65535)
    axes[1,0].set_title(f"Hubble AI Input (Clip {clip_percentile}%)", color='white')
    
    axes[1,1].imshow(lr_norm, cmap='gray', vmin=0, vmax=65535)
    axes[1,1].set_title(f"Obs AI Input (Clip {clip_percentile}%)", color='white')
    
    for ax in axes.flat: ax.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, facecolor='#000000')
    plt.close()

def process_single_normalization(target_dir):
    """Logica principale per la normalizzazione e conversione TIFF su una singola directory."""

    input_dir = target_dir / '6_patches_final'
    output_dir = target_dir / '7_dataset_ready_LOG'
    debug_dir = target_dir / '7_dataset_debug_png'

    if output_dir.exists(): shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    if debug_dir.exists(): shutil.rmtree(debug_dir)
    debug_dir.mkdir(parents=True)

    all_pairs = sorted(list(input_dir.glob("pair_*")))
    hubble_files = [p / "hubble.fits" for p in all_pairs if (p/"hubble.fits").exists()]
    obs_files = [p / "observatory.fits" for p in all_pairs if (p/"observatory.fits").exists()]

    if not hubble_files: 
        print("Nessuna coppia di patch FITS trovata in 6_patches_final. Salto.")
        return

    print(f"\nNORMALIZZAZIONE DEEP BLACK (Clip {BLACK_CLIP_PERCENTILE}%)")
    
    print("\n--- Analisi Hubble (HR) ---")
    h_min, h_max = calculate_robust_stats(hubble_files)

    print("\n--- Analisi Observatory (LR) ---")
    o_min, o_max = calculate_robust_stats(obs_files)
    
    print(f"\n--- Generazione TIFF 16-bit ({len(all_pairs)} coppie) ---")
    for i, pair_path in enumerate(tqdm(all_pairs, ncols=100)):
        try:
            with fits.open(pair_path / "hubble.fits") as h: d_h = np.nan_to_num(h[0].data)
            with fits.open(pair_path / "observatory.fits") as o: d_o = np.nan_to_num(o[0].data)

            raw_h_copy = d_h.copy()
            raw_o_copy = d_o.copy()

            if USE_LOG_STRETCH:
                d_h = np.log1p(np.maximum(d_h, 0))
                d_o = np.log1p(np.maximum(d_o, 0))

            d_h_norm = (d_h - h_min) / (h_max - h_min + 1e-8)
            d_o_norm = (d_o - o_min) / (o_max - o_min + 1e-8)

            d_h_norm = np.clip(d_h_norm, 0, 1)
            d_o_norm = np.clip(d_o_norm, 0, 1)

            h_u16 = (d_h_norm * 65535).astype(np.uint16)
            o_u16 = (d_o_norm * 65535).astype(np.uint16)

            p_out = output_dir / pair_path.name
            p_out.mkdir(exist_ok=True)

            Image.fromarray(h_u16, mode='I;16').save(p_out / "hubble.tiff")
            Image.fromarray(o_u16, mode='I;16').save(p_out / "observatory.tiff")

            if i % DEBUG_INTERVAL == 0:
                save_debug_png(raw_h_copy, raw_o_copy, h_u16, o_u16, BLACK_CLIP_PERCENTILE, debug_dir / f"check_{pair_path.name}.png")

        except Exception as e:
            print(f"Errore {pair_path.name}: {e}")

    print("\nCompletato.")
    
    zip_name = target_dir / f"debug_checks_clip_{BLACK_CLIP_PERCENTILE}"
    shutil.make_archive(str(zip_name), 'zip', str(debug_dir))
    print(f"Zippando cartella debug in: {zip_name}.zip")

def select_target_directories(required_subdir='6_patches_final'):
    """
    Consente la selezione di una, più o tutte le directory target che contengono la sottocartella richiesta.
    """
    subdirs = [d for d in ROOT_DATA_DIR.iterdir() if d.is_dir() and d.name not in ['splits', 'logs']]
    valid_subdirs = [d for d in subdirs if (d / required_subdir).exists()]
    
    if not valid_subdirs: 
        print(f"\nNessuna directory valida con '{required_subdir}' trovata.")
        return []
        
    print(f"\nSELEZIONA TARGET (Solo quelle con '{required_subdir}'):")
    valid_subdirs.sort(key=lambda x: x.name)
    for i, d in enumerate(valid_subdirs): print(f" {i+1}: {d.name}")
    print(f" A: TUTTE ({len(valid_subdirs)} in totale)")
    print(" (Es. '1,3,4' per selezione multipla, o 'A' per tutte)")
    
    try:
        raw_val = input("Scelta: ").strip().upper()
        if not raw_val: return []

        if raw_val == 'A':
            return valid_subdirs
        
        selected_indices = []
        for part in raw_val.split(','):
            try:
                idx = int(part) - 1
                if 0 <= idx < len(valid_subdirs):
                    selected_indices.append(idx)
            except ValueError:
                continue

        unique_indices = []
        for idx in selected_indices:
            if idx not in unique_indices:
                unique_indices.append(idx)
                
        return [valid_subdirs[i] for i in unique_indices]
        
    except Exception: 
        return []

def main():
    """Funzione di controllo che gestisce l'input e itera sui target."""
    target_dirs = select_target_directories('6_patches_final')
    if not target_dirs: 
        print("Nessun target selezionato. Esco.")
        return

    total_targets = len(target_dirs)
    
    for i, target_dir in enumerate(target_dirs):
        current_index = i + 1 
        progress_label = f"({current_index}/{total_targets})"
        
        print(f"\n=======================================================")
        print(f"INIZIO LAVORAZIONE TARGET {progress_label}: {target_dir.name}")
        print(f"=======================================================")
        
        try:
            process_single_normalization(target_dir)
        except Exception as e:
            print(f"Errore critico nel processo di {target_dir.name}: {e}")
            
    print("\n\nTUTTE LE NORMALIZZAZIONI COMPLETATE.")

if __name__ == "__main__":
    main()