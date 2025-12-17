import json
import sys
import random
from pathlib import Path

CURRENT_SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_SCRIPT_DIR.parent
ROOT_DATA_DIR = PROJECT_ROOT / "data"

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1

def prepare_full_dataset(target_dir_path):
    """
    Trova tutte le coppie di patch in un target, le mescola e le divide in 
    set di Train, Validation e Test, salvando gli elenchi in file JSON.
    """
    target_dir = Path(target_dir_path)
    print(f"\nPREPARAZIONE DATASET COMPLETO (TRAIN): {target_dir.name}")
    
    SOURCE = target_dir / "7_dataset_ready_LOG"
    if not SOURCE.exists(): SOURCE = target_dir / "7_dataset_ready"
    
    SPLITS = target_dir / "8_dataset_split" / "splits_json"
    
    if not SOURCE.exists():
        print(f"Sorgente dati (7_dataset_ready) non trovata in {target_dir.name}.")
        return

    pair_folders = sorted([d for d in SOURCE.glob("pair_*") if d.is_dir()])
    
    if not pair_folders:
        print(f"Nessuna coppia trovata in {SOURCE.name}.")
        return
    
    print(f"Trovate {len(pair_folders)} patch totali.")

    dataset_entries = []
    for pair in pair_folders:
        h_file = pair / "hubble.tiff"
        o_file = pair / "observatory.tiff"
        if h_file.exists() and o_file.exists():
            dataset_entries.append({
                "patch_id": pair.name,
                "hubble_path": str(h_file.resolve()),
                "ground_path": str(o_file.resolve())
            })
    
    random.seed(42) 
    random.shuffle(dataset_entries)
    
    total = len(dataset_entries)
    n_train = int(total * TRAIN_RATIO)
    n_val = int(total * VAL_RATIO)
    
    train_data = dataset_entries[:n_train]
    val_data = dataset_entries[n_train:n_train+n_val]
    test_data = dataset_entries[n_train+n_val:]
    
    print(f"Split: Train={len(train_data)} | Val={len(val_data)} | Test={len(test_data)}")

    SPLITS.mkdir(parents=True, exist_ok=True)
    
    with open(SPLITS / 'train.json', 'w') as f: json.dump(train_data, f, indent=4)
    with open(SPLITS / 'val.json', 'w') as f: json.dump(val_data, f, indent=4)
    with open(SPLITS / 'test.json', 'w') as f: json.dump(test_data, f, indent=4)

    print(f"JSON Generati in {SPLITS.resolve()}")
    print("-" * 40)

def select_target_directories(required_subdir='7_dataset_ready'):
    """
    Consente la selezione di una, più o tutte le directory target 
    che contengono la sottocartella dati necessaria.
    """
    all_subdirs = [d for d in ROOT_DATA_DIR.iterdir() if d.is_dir() and d.name not in ['splits', 'logs']]
    
    valid_subdirs = []
    for d in all_subdirs:
        if (d / '7_dataset_ready_LOG').exists() or (d / '7_dataset_ready').exists():
             valid_subdirs.append(d)

    if not valid_subdirs: 
        print(f"\nNessuna directory valida con dati pronti (7_dataset_ready) trovata.")
        return []

    print("\n--- SELEZIONE DATASET PER TRAINING ---")
    valid_subdirs.sort(key=lambda x: x.name)
    for i, d in enumerate(valid_subdirs): 
        print(f"  [{i+1}] {d.name}")
        
    print(f"  [A] TUTTE ({len(valid_subdirs)} in totale)")
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
    
    if len(sys.argv) > 1: 
        target_dirs = [Path(sys.argv[1])]
    else:
        target_dirs = select_target_directories()
        
    if not target_dirs:
        print("Nessun target selezionato. Esco.")
        return

    for target_dir in target_dirs:
        print(f"\n=======================================================")
        print(f"INIZIO SPLIT PER TARGET: {target_dir.name}")
        print(f"=======================================================")
        
        try:
            prepare_full_dataset(target_dir)
        except Exception as e:
            print(f"Errore critico nello split di {target_dir.name}: {e}")
            
    print("\n\nTUTTI GLI SPLIT COMPLETATI.")

if __name__ == "__main__":
    main()