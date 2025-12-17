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
PROJECT_ROOT = CURRENT_SCRIPT.parent.parent
OUTPUT_ROOT = PROJECT_ROOT / "outputs"
ROOT_DATA_DIR = PROJECT_ROOT / "data"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

print(f"PATH INFO:")
print(f"Project Root: {PROJECT_ROOT}")

if not (PROJECT_ROOT / "src").exists():
    sys.exit("ERRORE CRITICO: Cartella 'src' non trovata nella root.")

print("Importazione moduli...")

try:
    from src.architecture_train import TrainHybridModel
    from src.dataset import AstronomicalDataset
    from src.metrics_train import TrainMetrics
    print("Moduli importati correttamente.")
except ImportError as e:
    print(f"\nERRORE IMPORTAZIONE MODULI PROGETTO (Controlla sys.path e dipendenze): {e}")
    sys.exit(1)

torch.backends.cudnn.benchmark = True

def save_as_tiff16(tensor, path):
    """Salva un tensore PyTorch come immagine TIFF a 16 bit."""
    arr = tensor.squeeze().float().cpu().numpy()
    arr = np.clip(arr, 0, 1)
    arr_u16 = (arr * 65535).astype(np.uint16)
    Image.fromarray(arr_u16, mode='I;16').save(path)

def get_target_list_from_model_name(model_folder_name: str) -> List[str]:
    """
    Estrae la lista dei target dal nome della cartella di output.
    Esempio: 'M33_M42_M8_DDP' -> ['M33', 'M42', 'M8']
    """
    base_name = model_folder_name.replace("_DDP_GAN", "").replace("_DDP", "")
    return [t.strip() for t in base_name.split('_') if t.strip()]

def get_available_targets(output_root: Path) -> List[str]:
    """Elenca le cartelle in 'outputs'."""
    if not output_root.is_dir():
        print(f"La cartella '{output_root.name}' non esiste.")
        return []
    targets = [p.name for p in output_root.iterdir() if p.is_dir()]
    return sorted(targets)

def get_model_info(target_folder_name: str, output_root: Path) -> str:
    """
    Legge i metadati dal file 'latest_checkpoint.pth' per mostrare 
    a che punto è il training e qual è il punteggio migliore.
    """
    ckpt_dir = output_root / target_folder_name / "checkpoints"
    ckpt_path = ckpt_dir / "latest_checkpoint.pth"
    
    has_gan = (ckpt_dir / "best_gan_model.pth").exists()
    has_std = (ckpt_dir / "best_train_model.pth").exists()
    type_label = "GAN" if has_gan else ("STD" if has_std else "N/A")

    if not ckpt_path.exists():
        return f"({type_label} - Nessun Checkpoint Recente)"
    
    try:
        ckpt = torch.load(ckpt_path, map_location='cpu')
        
        epoch = ckpt.get('epoch', '?')
        
        if 'best_ssim' in ckpt:
            score = ckpt['best_ssim']
            label = "SSIM"
        elif 'best_psnr' in ckpt:
            score = ckpt['best_psnr']
            label = "PSNR"
        else:
            score = 0.0
            label = "Score"
            
        return f"[{type_label}] Epoca: {epoch} | Best {label}: {score:.4f}"
        
    except Exception as e:
        return f"(Errore lettura ckpt: {str(e)[:20]})"

def select_target_from_menu(targets: List[str]) -> Optional[str]:
    """Menu interattivo aggiornato con Info Modello."""
    if not targets:
        print("Nessun training trovato.")
        return None
    
    print("\n--- SELEZIONE MODELLO DA TESTARE (Cartelle in outputs/) ---")
    for i, target in enumerate(targets):
        info_str = get_model_info(target, OUTPUT_ROOT)
        
        targets_in_model = get_target_list_from_model_name(target)
        if len(targets_in_model) > 1:
            print(f"[{i+1}] {target} (AGGREGATO su {len(targets_in_model)} target)")
            print(f"{info_str}") 
        else:
            print(f"[{i+1}] {target}")
            print(f"{info_str}") 
    
    while True:
        try:
            choice = input("\nSeleziona il numero (o Invio per uscire): ")
            if not choice:
                return None 
            choice_index = int(choice) - 1
            if 0 <= choice_index < len(targets):
                return targets[choice_index]
            else:
                print("Numero non valido.")
        except ValueError:
            print("Inserisci un numero valido.")

def aggregate_test_data(target_names: List[str]) -> Path:
    """
    Aggrega i dati di test (o validazione) per tutti i target specificati
    e salva il risultato in un unico JSON temporaneo.
    """
    all_test_data: List[Dict] = []
    
    print(f"Aggregazione dati di test per {len(target_names)} target...")
    
    for target_name in target_names:
        splits_dir = ROOT_DATA_DIR / target_name / "8_dataset_split" / "splits_json"
        
        test_path = splits_dir / "test.json"
        if not test_path.exists():
            test_path = splits_dir / "val.json"
            if test_path.exists():
                print(f" -> Usando 'val.json' per {target_name}.")
            else:
                print(f"Nessun file test/val trovato per {target_name}. Salto.")
                continue

        with open(test_path, 'r') as f:
            all_test_data.extend(json.load(f))
    
    if not all_test_data:
        raise FileNotFoundError("Nessun dato di test aggregato trovato.")
        
    temp_dir = OUTPUT_ROOT / "temp_test_data"
    temp_dir.mkdir(exist_ok=True)
    temp_json_path = temp_dir / f"aggregated_test_{'_'.join(target_names)}.json"
    
    with open(temp_json_path, 'w') as f:
        json.dump(all_test_data, f)
        
    print(f"Totale coppie di test aggregate: {len(all_test_data)}")
    return temp_json_path

def run_test(target_model_folder: str):
    print("\nATTENZIONE: Se il training è attivo, la GPU è occupata.")
    use_cpu = input("Vuoi usare la CPU per evitare crash? (s/n) [default: s]: ").strip().lower()
    
    if use_cpu == 'n' and torch.cuda.is_available():
        device = torch.device('cuda')
        print("Tento di usare la GPU...")
    else:
        device = torch.device('cpu')
        print("Uso la CPU (Lento ma sicuro).")

    OUTPUT_DIR = OUTPUT_ROOT / target_model_folder / "test_results_tiff"
    CHECKPOINT_DIR = OUTPUT_ROOT / target_model_folder / "checkpoints"
    
    data_target_names = get_target_list_from_model_name(target_model_folder)
    target_base_name = target_model_folder.replace("_DDP_GAN", "").replace("_DDP", "")
    
    # MODIFICA: Cerca SOLO pesi GAN (Best o Latest/Resume)
    checkpoints = list(CHECKPOINT_DIR.glob("best_gan_model.pth"))
    ckpt_type = "GAN (Best)"
        
    if not checkpoints:
        # Se non c'è il best, proviamo il latest (che contiene i pesi GAN del training corrente)
        checkpoints = list(CHECKPOINT_DIR.glob("latest_checkpoint.pth"))
        ckpt_type = "Latest (GAN Resume)"
    
    # MODIFICA: Rimossa ricerca "best_train_model.pth" e "*.pth"
    
    if not checkpoints:
        print(f"Nessun checkpoint GAN trovato in {CHECKPOINT_DIR}")
        return
    
    CHECKPOINT_PATH = checkpoints[0]
    
    print(f"\n" + "="*60)
    print(f"INFERENZA SU MODELLO: {target_model_folder}")
    print(f"Cartella Checkpoints: {CHECKPOINT_DIR.name}")
    print(f"Tipo Pesi Trovati: {ckpt_type}")
    print(f"FILE SELEZIONATO: {CHECKPOINT_PATH.name}")
    print("="*60 + "\n")

    try:
        temp_json_path = aggregate_test_data(data_target_names)
    except FileNotFoundError as e:
        print(f"ERRORE: {e}")
        return

    (OUTPUT_DIR / "tiff_science").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "png_preview").mkdir(parents=True, exist_ok=True)

    test_ds = AstronomicalDataset(temp_json_path, base_path=PROJECT_ROOT, augment=False)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=2)
    
    print("Caricamento Architettura...")
    
    model = TrainHybridModel(smoothing='none', device=device, output_size=512).to(device)

    try:
        state_dict = torch.load(CHECKPOINT_PATH, map_location=device)
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k.replace("module.", "") if k.startswith("module.") else k
            new_state_dict[name] = v
        
        keys_to_remove = [k for k in new_state_dict.keys() if "s1.weight" in k or "s2.weight" in k or "sf.weight" in k]
        if keys_to_remove:
            print(f"Aggiornamento layer smoothing: rimossi {len(keys_to_remove)} vecchi buffer.")
            for k in keys_to_remove:
                del new_state_dict[k]
        
        model.load_state_dict(new_state_dict, strict=False)
        
        print(f"Pesi caricati con successo da: {CHECKPOINT_PATH.name}")
        
    except Exception as e:
        print(f"Errore caricamento pesi: {e}")
        return

    model.eval()
    metrics = TrainMetrics()
    
    print(f"Elaborazione {len(test_ds)} immagini aggregate...")
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader)):
            lr = batch['lr'].to(device)
            hr = batch['hr'].to(device)
            
            if device.type == 'cuda' and hasattr(torch.amp, 'autocast'):
                with torch.amp.autocast('cuda'):
                    sr = model(lr)
            else:
                sr = model(lr)
            
            metrics.update(sr.float(), hr.float())
            
            save_as_tiff16(sr, OUTPUT_DIR / "tiff_science" / f"sr_{i:04d}.tiff")
            
            lr_up = torch.nn.functional.interpolate(lr, size=(sr.shape[2], sr.shape[3]), mode='nearest')
            if sr.shape != hr.shape:
                hr_preview = torch.nn.functional.interpolate(hr, size=(sr.shape[2], sr.shape[3]), mode='bicubic')
            else:
                hr_preview = hr
            
            comp = torch.cat((lr_up, sr, hr_preview), dim=3).clamp(0,1)
            save_image(comp, OUTPUT_DIR / "png_preview" / f"comp_{i:04d}.png")

    res = metrics.compute()
    print("\nRISULTATI MEDI (Sul dataset aggregato):")
    
    if 'psnr' in res:
        print(f"   PSNR: {res['psnr']:.2f} dB")
    if 'ssim' in res:
        print(f"   SSIM: {res['ssim']:.4f}")

    print("\nCreazione ZIP e Pulizia...")
    
    if temp_json_path.exists():
        temp_json_path.unlink()

    zip_root_dir = OUTPUT_ROOT / target_model_folder
    
    def create_zip(folder_name, suffix):
        path_to_zip = OUTPUT_DIR / folder_name
        if path_to_zip.exists():
            zip_filename = zip_root_dir / f"{target_base_name}_{suffix}"
            shutil.make_archive(
                base_name=str(zip_filename), 
                format='zip', 
                root_dir=OUTPUT_DIR.parent, 
                base_dir=Path("test_results_tiff") / folder_name
            )
            print(f"{suffix}: {zip_filename.name}.zip")

    create_zip("tiff_science", "results_tiff")
    create_zip("png_preview", "preview_png")
    
    print(f"\nFinito. Risultati in {OUTPUT_DIR.parent}")


if __name__ == "__main__":
    targets = get_available_targets(OUTPUT_ROOT)
    selected_target_folder = select_target_from_menu(targets)
    if selected_target_folder:
        run_test(selected_target_folder)
    else:
        print("\nOperazione annullata.")
