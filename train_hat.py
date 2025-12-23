import os
import sys
import argparse
import json
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from pathlib import Path

# --- FIX IMPORT PRIORITARIO ---
# Questo blocco deve stare PRIMA di importare i modelli
# Serve a rendere visibile la libreria 'basicsr' contenuta in models/BasicSR
CURRENT_DIR = Path(__file__).resolve().parent
BASICSR_PATH = CURRENT_DIR / "models" / "BasicSR"
if BASICSR_PATH.exists():
    sys.path.insert(0, str(BASICSR_PATH))
    print(f" [SETUP] Aggiunto al path: {BASICSR_PATH}")
else:
    print(f" [ERR] ATTENZIONE: La cartella {BASICSR_PATH} non esiste! Il training potrebbe fallire.")
# ------------------------------

# Imports dai moduli del progetto
# Ora questi import funzioneranno perché il path è stato corretto sopra
from models.hat_arch import HAT
from models.discriminator import UNetDiscriminatorSN
from dataset.astronomical_dataset import AstronomicalDataset
from utils.gan_losses import CombinedGANLoss, DiscriminatorLoss
from utils.metrics import TrainMetrics

# --- CONFIGURAZIONE IPERPARAMETRI ---
BATCH_SIZE = 4       # Se la VRAM si riempie, abbassa a 2
LR_G = 1e-4
LR_D = 1e-4
NUM_EPOCHS = 300

def setup():
    """Inizializza il processo distribuito."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    else:
        # Fallback debug
        print("Attenzione: DDP non rilevato. Esecuzione Single-GPU/CPU.")
        os.environ["LOCAL_RANK"] = "0"
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29500"
        dist.init_process_group(backend="gloo")

def train_worker():
    # 1. PARSING ARGOMENTI (Riceve i target da start.py)
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', type=str, required=True, help='Lista target separati da virgola')
    args, unknown = parser.parse_known_args()

    setup()
    
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = dist.get_rank()
    device = torch.device(f"cuda:{local_rank}")

    if rank == 0:
        print(f"Avvio processo su {device}. Target: {args.target}")

    # 2. PREPARAZIONE DATASET COMBINATO
    # Uniamo i file json dei target selezionati in un unico file temporaneo
    targets = args.target.split(',')
    base_data_path = Path("./data") 
    combined_json_path = "temp_train_combined.json"

    # Solo il Master (Rank 0) crea il file JSON unificato
    if rank == 0:
        all_pairs = []
        found_any = False
        for t in targets:
            # Percorso: data/{TARGET}/8_dataset_split/splits_json/train.json
            json_path = base_data_path / t / "8_dataset_split" / "splits_json" / "train.json"
            if json_path.exists():
                with open(json_path, 'r') as f:
                    data = json.load(f)
                    all_pairs.extend(data)
                print(f" [OK] {t}: Caricati {len(data)} esempi.")
                found_any = True
            else:
                print(f" [ERR] {t}: File non trovato in {json_path}")
        
        if not found_any:
            print("ERRORE CRITICO: Nessun dato trovato. Interruzione.")
            sys.exit(1)

        with open(combined_json_path, 'w') as f:
            json.dump(all_pairs, f)
        print(f" Dataset combinato salvato: {combined_json_path} ({len(all_pairs)} immagini)")

    # Barriera: Tutti i processi aspettano che il file sia pronto
    dist.barrier()

    # 3. CARICAMENTO DATASET
    try:
        train_ds = AstronomicalDataset(combined_json_path, base_path="./")
    except Exception as e:
        if rank == 0: print(f"Errore init Dataset: {e}")
        dist.destroy_process_group()
        return

    train_sampler = DistributedSampler(train_ds)
    train_loader = DataLoader(
        train_ds, 
        batch_size=BATCH_SIZE, 
        sampler=train_sampler, 
        num_workers=4, 
        pin_memory=True,
        persistent_workers=True
    )

    # 4. INIZIALIZZAZIONE MODELLI
    # HAT (Generatore)
    net_g = HAT(
        img_size=128, in_chans=1, embed_dim=180, 
        depths=(6, 6, 6, 6, 6, 6), num_heads=(6, 6, 6, 6, 6, 6),
        window_size=7, upscale=4, upsampler='pixelshuffle'
    ).to(device)
    
    # Discriminatore
    net_d = UNetDiscriminatorSN(num_in_ch=1, num_feat=64).to(device)

    # Wrap DDP
    net_g = DDP(net_g, device_ids=[local_rank])
    net_d = DDP(net_d, device_ids=[local_rank])

    # 5. LOSS E OPTIMIZER
    opt_g = torch.optim.AdamW(net_g.parameters(), lr=LR_G)
    opt_d = torch.optim.AdamW(net_d.parameters(), lr=LR_D)
    
    criterion_g = CombinedGANLoss(pixel_weight=1.0, adversarial_weight=0.005).to(device)
    criterion_d = DiscriminatorLoss().to(device)

    # 6. TRAINING LOOP
    if rank == 0: 
        print("==========================================")
        print("      AVVIO TRAINING HAT ENGINE           ")
        print("==========================================")

    for epoch in range(1, NUM_EPOCHS + 1):
        train_sampler.set_epoch(epoch)
        for i, batch in enumerate(train_loader):
            lr = batch['lr'].to(device)
            hr = batch['hr'].to(device)

            # --- A. TRAINING GENERATORE (HAT) ---
            # Blocca gradienti D per efficienza
            for p in net_d.parameters(): p.requires_grad = False
            opt_g.zero_grad()

            sr = net_g(lr)
            
            # Predizioni per Loss (senza aggiornare D)
            pred_fake = net_d(sr)
            pred_real = net_d(hr).detach() 
            
            loss_g_total, losses_g_dict = criterion_g(sr, hr, pred_real, pred_fake)
            loss_g_total.backward()
            opt_g.step()

            # --- B. TRAINING DISCRIMINATORE ---
            for p in net_d.parameters(): p.requires_grad = True
            opt_d.zero_grad()
            
            # Detach necessario per non intaccare G
            pred_fake_d = net_d(sr.detach())
            pred_real_d = net_d(hr)
            
            loss_d_total, losses_d_dict = criterion_d(pred_real_d, pred_fake_d)
            loss_d_total.backward()
            opt_d.step()

            # --- LOGGING ---
            if rank == 0 and i % 10 == 0:
                print(f"Epoch {epoch} | Batch {i} | G_Loss: {loss_g_total.item():.4f} | D_Loss: {loss_d_total.item():.4f}")

        # Salvataggio Checkpoint (Solo Rank 0)
        if rank == 0 and epoch % 5 == 0:
            save_dir = Path("./outputs/checkpoints")
            save_dir.mkdir(parents=True, exist_ok=True)
            # Salviamo il modulo interno (senza DDP wrapper) per facilitare l'inferenza
            torch.save(net_g.module.state_dict(), save_dir / "latest_checkpoint.pth")
            torch.save(net_g.module.state_dict(), save_dir / "best_gan_model.pth")
            print(f"Checkpoint salvato: Epoch {epoch}")

    dist.destroy_process_group()

if __name__ == "__main__":
    train_worker()
