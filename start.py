import os
import sys
import subprocess
import torch
from pathlib import Path

CURRENT_SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_SCRIPT_DIR
DATA_DIR = PROJECT_ROOT / "data"
SCRIPT_PATH = PROJECT_ROOT / "train_hat.py" 

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def get_available_targets(required_subdir='8_dataset_split'):
    if not DATA_DIR.exists():
        return []
    
    all_subdirs = [d for d in DATA_DIR.iterdir() if d.is_dir() and d.name not in ['splits', 'logs']]
    
    valid_subdirs = [
        d.name for d in all_subdirs 
        if (d / required_subdir / "splits_json" / "train.json").exists()
    ]
    return sorted(valid_subdirs)

def get_available_gpus():
    count = torch.cuda.device_count()
    gpus = []
    for i in range(count):
        name = torch.cuda.get_device_name(i)
        mem = torch.cuda.get_device_properties(i).total_memory / 1024**2
        gpus.append(f"[{i}] {name} ({int(mem)} MB)")
    return gpus

def select_targets_interactive(targets):
    print("Target disponibili per il training HAT (con split pronti):")
    for idx, t in enumerate(targets):
        print(f"   [{idx}] {t}")
    
    print("\nOpzioni di Selezione:")
    print("   [A] Tutti i target listati")
    print("   [ID] Numeri separati da virgola (es. 0,2)")
    
    while True:
        sel = input("\nSeleziona target(s): ").strip().upper()
        if sel == 'A':
            return targets
        
        selected_names = []
        try:
            ids = [x.strip() for x in sel.split(',')]
            for x in ids:
                if x.isdigit():
                    t_idx = int(x)
                    if 0 <= t_idx < len(targets):
                        if targets[t_idx] not in selected_names:
                            selected_names.append(targets[t_idx])
            
            if selected_names:
                return selected_names
            print("Selezione non valida.")
        except:
            print("Formato di input non valido.")

def select_gpus_interactive(gpus):
    print("GPU Disponibili:")
    for g in gpus:
        print(f"   {g}")
    
    while True:
        sel = input("\nScelta GPU (es. 'a' per tutte, o '0,1'): ").strip().lower()
        if sel == 'a':
            gpu_ids = [str(i) for i in range(len(gpus))]
            break
        else:
            try:
                ids = [x.strip() for x in sel.split(',')]
                if all(x.isdigit() and int(x) < len(gpus) for x in ids) and len(ids) > 0:
                    gpu_ids = ids
                    break
                print("ID GPU non validi.")
            except:
                print("Formato non valido.")
    
    return ",".join(gpu_ids), len(gpu_ids)

def main():
    clear_screen()
    print("==========================================")
    print("      ASTRONOMICAL HAT LAUNCHER           ")
    print("==========================================\n")

    if not torch.cuda.is_available():
        print("CUDA non disponibile. Impossibile avviare il training.")
        sys.exit(1)
        
    targets = get_available_targets()
    if not targets:
        print(f"Nessun target trovato in: {DATA_DIR}")
        sys.exit(1)
    
    selected_target_names = select_targets_interactive(targets)
    target_env_string = ",".join(selected_target_names) 

    gpus = get_available_gpus()
    gpu_env_string, nproc = select_gpus_interactive(gpus)
    
    print(f"\nTarget: {target_env_string}")
    print(f"GPU: {gpu_env_string} (Processi: {nproc})\n")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu_env_string
    env["NCCL_P2P_DISABLE"] = "1"
    env["NCCL_IB_DISABLE"] = "1"
    env["OMP_NUM_THREADS"] = "4"

    cmd = [
        sys.executable,
        "-m", "torch.distributed.run",
        f"--nproc_per_node={nproc}",
        "--master_port=29500", 
        str(SCRIPT_PATH),
        "--target", target_env_string 
    ]

    print("Avvio Training HAT Engine...")
    try:
        subprocess.run(cmd, env=env, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\nErrore durante l'addestramento: {e}")
    except KeyboardInterrupt:
        print("\nProcesso interrotto dall'utente.")

if __name__ == "__main__":
    main()
