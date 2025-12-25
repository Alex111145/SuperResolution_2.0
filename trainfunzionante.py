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
    Image.fromarray(combined).save(save_path / f"epoch_{epoch}_preview.jpg")

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

def train_worker():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', type=str, required=True)
    args, unknown = parser.parse_known_args()

    setup()
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = dist.get_rank()
    device = torch.device(f"cuda:{local_rank}")

    if rank == 0:
        print(f"Avvio su {device}. Target: {args.target}")

    # Dataset Combinato
    targets = args.target.split(',')
    combined_json_path = "temp_train_combined.json"

    if rank == 0:
        all_pairs = []
        for t in targets:
            json_path = Path("./data") / t / "8_dataset_split" / "splits_json" / "train.json"
            if json_path.exists():
                with open(json_path, 'r') as f: all_pairs.extend(json.load(f))
        
        if not all_pairs: sys.exit("Nessun dato trovato.")
        with open(combined_json_path, 'w') as f: json.dump(all_pairs, f)

    dist.barrier()

    train_ds = AstronomicalDataset(combined_json_path, base_path="./")
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

    if rank == 0: 
        print("=== TRAINING AVVIATO ===")

    for epoch in range(1, NUM_EPOCHS + 1):
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
            # Sostituisce la virgola con underscore per creare il nome cartella (es. target1_target2)
            target_folder_name = args.target.replace(',', '_')
            
            # Percorsi aggiornati: outputs/NOME_TARGET/checkpoints
            base_output_dir = Path("./outputs") / target_folder_name
            ckpt_dir = base_output_dir / "checkpoints"
            preview_dir = base_output_dir / "previews"
            
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            preview_dir.mkdir(parents=True, exist_ok=True)
            
            # Salva Modello
            torch.save(net_g.module.state_dict(), ckpt_dir / "latest_checkpoint.pth")
            torch.save(net_g.module.state_dict(), ckpt_dir / "best_gan_model.pth")
            
            # Salva Foto Preview
            save_validation_preview(lr, sr, hr, epoch, preview_dir)
            
            tqdm.write(f" [SAVE] Epoch {epoch}: Modello e Preview salvati in {base_output_dir}")

    dist.destroy_process_group()

if __name__ == "__main__":
    train_worker()
