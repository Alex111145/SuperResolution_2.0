import os
import sys
import argparse
import json
import torch
import numpy as np
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import warnings

# === PULIZIA WARNINGS ===
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision.models._utils")
warnings.filterwarnings("ignore", message="Grad strides do not match bucket view strides")
warnings.filterwarnings("ignore", message="torch.meshgrid: in an upcoming release")

# --- FIX IMPORT PRIORITARIO ---
CURRENT_DIR = Path(__file__).resolve().parent
BASICSR_PATH = CURRENT_DIR / "models" / "BasicSR"
if BASICSR_PATH.exists():
    sys.path.insert(0, str(BASICSR_PATH))
else:
    if os.environ.get("LOCAL_RANK", "0") == "0":
        print(f" [WARN] Cartella BasicSR non trovata in: {BASICSR_PATH}")

from models.hat_arch import HAT
from models.discriminator import UNetDiscriminatorSN
from dataset.astronomical_dataset import AstronomicalDataset
from utils.gan_losses import CombinedGANLoss, DiscriminatorLoss

# --- CONFIGURAZIONE ---
BATCH_SIZE = 4
LR_G = 1e-4
LR_D = 1e-4
NUM_EPOCHS = 300
SAVE_INTERVAL = 5  # Salva modello e foto ogni 5 epoche

def tensor_to_img(tensor):
    """Converte un tensore (C, H, W) in immagine numpy uint8."""
    img = tensor.cpu().detach().squeeze().float().numpy()
    img = np.clip(img, 0, 1)
    return (img * 255).astype(np.uint8)

def save_validation_preview(lr, sr, hr, epoch, save_path):
    """Salva un'immagine di confronto: LR (upscaled) | SR | HR"""
    lr_img = tensor_to_img(lr[0])
    sr_img = tensor_to_img(sr[0])
    hr_img = tensor_to_img(hr[0])
    
    # Ridimensiona LR per affiancarla (interpolazione nearest per vedere i pixel)
    h, w = sr_img.shape
    lr_pil = Image.fromarray(lr_img).resize((w, h), resample=Image.NEAREST)
    lr_img_resized = np.array(lr_pil)

    # Crea immagine combinata
    combined = np.hstack((lr_img_resized, sr_img, hr_img))
    Image.fromarray(combined).save(save_path / f"epoch_{epoch:03d}_preview.jpg")

def setup():
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    else:
        os.environ["LOCAL_RANK"] = "0"
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29500"
        dist.init_process_group(backend="gloo")

def find_latest_checkpoint(ckpt_dir):
    """Trova il checkpoint più recente basandosi sul numero di epoca nel nome file"""
    if not ckpt_dir.exists():
        return None
    
    # Cerca file pattern hat_epoch_XXX.pth
    pths = list(ckpt_dir.glob("hat_epoch_*.pth"))
    if not pths:
        return None
        
    try:
        # Ordina in base al numero estratto dal nome
        latest = sorted(pths, key=lambda x: int(x.stem.split('_')[-1]))[-1]
        return latest
    except Exception:
        return None

def train_worker():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', type=str, required=True)
    parser.add_argument('--resume', type=str, default=None, help='Path checkpoint per resume manuale')
    args, unknown = parser.parse_known_args()

    setup()
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = dist.get_rank()
    device = torch.device(f"cuda:{local_rank}")

    # Configurazione Percorsi Output
    target_folder_name = args.target.replace(',', '_')
    base_output_dir = Path("./outputs") / target_folder_name
    ckpt_dir = base_output_dir / "checkpoints"
    preview_dir = base_output_dir / "previews"

    if rank == 0:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        preview_dir.mkdir(parents=True, exist_ok=True)

    # === LOGICA AUTO-RESUME ===
    if args.resume is None:
        latest_ckpt = find_latest_checkpoint(ckpt_dir)
        if latest_ckpt:
            args.resume = str(latest_ckpt)
            if rank == 0:
                print(f"🔄 Auto-resume attivato: Trovato {latest_ckpt.name}")

    if rank == 0:
        print(f"Avvio su {device}. Target: {args.target}")
        if args.resume:
            print(f"Resume da: {args.resume}")

    # Dataset Combinato
    targets = args.target.split(',')
    combined_json_path = f"temp_train_combined_{rank}.json" # Unique per rank

    if rank == 0:
        all_pairs = []
        for t in targets:
            json_path = Path("./data") / t / "8_dataset_split" / "splits_json" / "train.json"
            if json_path.exists():
                with open(json_path, 'r') as f: all_pairs.extend(json.load(f))
        
        if not all_pairs: sys.exit("Nessun dato trovato.")
        # Usa un nome file comune per il dataset finale
        with open("temp_train_combined_HAT.json", 'w') as f: json.dump(all_pairs, f)

    dist.barrier()

    train_ds = AstronomicalDataset("temp_train_combined_HAT.json", base_path="./")
    train_sampler = DistributedSampler(train_ds)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=train_sampler, 
                              num_workers=4, pin_memory=True, persistent_workers=True)

    # Modelli
    net_g = HAT(img_size=128, in_chans=1, embed_dim=180, depths=(6,6,6,6,6,6), 
                num_heads=(6,6,6,6,6,6), window_size=7, upscale=4, upsampler='pixelshuffle').to(device)
    net_d = UNetDiscriminatorSN(num_in_ch=1, num_feat=64).to(device)

    net_g = DDP(net_g, device_ids=[local_rank])
    net_d = DDP(net_d, device_ids=[local_rank])

    opt_g = torch.optim.AdamW(net_g.parameters(), lr=LR_G)
    opt_d = torch.optim.AdamW(net_d.parameters(), lr=LR_D)
    
    criterion_g = CombinedGANLoss(pixel_weight=1.0, adversarial_weight=0.005).to(device)
    criterion_d = DiscriminatorLoss().to(device)

    # === CARICAMENTO CHECKPOINT (RESUME) ===
    start_epoch = 1
    if args.resume:
        try:
            checkpoint = torch.load(args.resume, map_location=device)
            
            # Gestione compatibilità vecchi checkpoint (solo pesi) vs nuovi (dizionario completo)
            if 'model_state_dict' in checkpoint:
                net_g.module.load_state_dict(checkpoint['model_state_dict'])
                opt_g.load_state_dict(checkpoint['optimizer_state_dict'])
                # opt_d.load_state_dict(checkpoint['optimizer_d_state_dict']) # Se disponibile
                start_epoch = checkpoint['epoch'] + 1
                if rank == 0: print(f"✅ Checkpoint completo caricato. Ripresa da Epoca {start_epoch}")
            else:
                # Fallback per vecchi checkpoint che erano solo state_dict
                net_g.module.load_state_dict(checkpoint)
                if rank == 0: print("⚠️ Caricati solo pesi modello (vecchio formato). Si riparte da Epoca 1.")

        except Exception as e:
            if rank == 0: print(f"❌ Errore caricamento resume: {e}")

    dist.barrier()

    if rank == 0: 
        print("=== TRAINING AVVIATO ===")

    for epoch in range(start_epoch, NUM_EPOCHS + 1):
        train_sampler.set_epoch(epoch)
        if rank == 0:
            loader_bar = tqdm(train_loader, desc=f"Ep {epoch}", unit="bt", leave=True)
        else:
            loader_bar = train_loader

        for i, batch in enumerate(loader_bar):
            lr = batch['lr'].to(device)
            hr = batch['hr'].to(device)

            # Train G
            for p in net_d.parameters(): p.requires_grad = False
            opt_g.zero_grad()
            sr = net_g(lr)
            pred_fake = net_d(sr)
            pred_real = net_d(hr).detach()
            loss_g, _ = criterion_g(sr, hr, pred_real, pred_fake)
            loss_g.backward()
            opt_g.step()

            # Train D
            for p in net_d.parameters(): p.requires_grad = True
            opt_d.zero_grad()
            pred_fake_d = net_d(sr.detach())
            pred_real_d = net_d(hr)
            loss_d, _ = criterion_d(pred_real_d, pred_fake_d)
            loss_d.backward()
            opt_d.step()

            if rank == 0:
                loader_bar.set_postfix({'G': f"{loss_g.item():.3f}", 'D': f"{loss_d.item():.3f}"})

        # --- SALVATAGGIO CHECKPOINT E PREVIEW ---
        if rank == 0 and epoch % SAVE_INTERVAL == 0:
            # Salva Checkpoint Completo (per resume)
            checkpoint_dict = {
                'epoch': epoch,
                'model_state_dict': net_g.module.state_dict(),
                'optimizer_state_dict': opt_g.state_dict(),
                'loss_g': loss_g.item(),
                'loss_d': loss_d.item()
            }
            
            # Salva versione numerata (storico) e versione 'latest' (per compatibilità)
            torch.save(checkpoint_dict, ckpt_dir / f"hat_epoch_{epoch:03d}.pth")
            torch.save(net_g.module.state_dict(), ckpt_dir / "latest_checkpoint.pth") # Legacy format (solo pesi)
            torch.save(net_g.module.state_dict(), ckpt_dir / "best_gan_model.pth")    # Legacy format
            
            # Salva Foto Preview
            save_validation_preview(lr, sr, hr, epoch, preview_dir)
            
            tqdm.write(f" [SAVE] Epoch {epoch}: Checkpoint salvato.")

    if rank == 0 and os.path.exists("temp_train_combined_HAT.json"):
        os.remove("temp_train_combined_HAT.json")

    dist.destroy_process_group()

if __name__ == "__main__":
    train_worker()
