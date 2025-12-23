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

# MODIFICA: Importa il nuovo modello ibrido
from models.hybrid_model import HybridHATRealESRGAN
from models.discriminator import UNetDiscriminatorSN
from dataset.astronomical_dataset import AstronomicalDataset
from utils.gan_losses import CombinedGANLoss, DiscriminatorLoss

# --- CONFIGURAZIONE ---
BATCH_SIZE = 2  # Ridotto per modello più pesante
LR_G = 1e-4
LR_D = 1e-4
NUM_EPOCHS = 300
SAVE_INTERVAL = 5
GRADIENT_ACCUMULATION = 2  # Simula batch_size=4 con meno VRAM

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
    
    h, w = sr_img.shape
    lr_pil = Image.fromarray(lr_img).resize((w, h), resample=Image.NEAREST)
    lr_img_resized = np.array(lr_pil)

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
    parser.add_argument('--pretrained_hat', type=str, default=None, 
                       help='Path to pretrained HAT weights (optional)')
    args, unknown = parser.parse_known_args()

    setup()
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = dist.get_rank()
    device = torch.device(f"cuda:{local_rank}")

    if rank == 0:
        print(f"🚀 Avvio Training Ibrido HAT + Real-ESRGAN")
        print(f"   Device: {device} | Target: {args.target}")

    # Dataset Combinato
    targets = args.target.split(',')
    combined_json_path = "temp_train_combined.json"

    if rank == 0:
        all_pairs = []
        for t in targets:
            json_path = Path("./data") / t / "8_dataset_split" / "splits_json" / "train.json"
            if json_path.exists():
                with open(json_path, 'r') as f: 
                    all_pairs.extend(json.load(f))
        
        if not all_pairs: 
            sys.exit("❌ Nessun dato trovato.")
        with open(combined_json_path, 'w') as f: 
            json.dump(all_pairs, f)
        print(f"   Training samples: {len(all_pairs)}")

    dist.barrier()

    train_ds = AstronomicalDataset(combined_json_path, base_path="./")
    train_sampler = DistributedSampler(train_ds)
    train_loader = DataLoader(
        train_ds, 
        batch_size=BATCH_SIZE, 
        sampler=train_sampler, 
        num_workers=4, 
        pin_memory=True, 
        persistent_workers=True
    )

    # === MODELLO IBRIDO HAT + Real-ESRGAN ===
    net_g = HybridHATRealESRGAN(
        img_size=128,
        in_chans=1,
        embed_dim=180,
        depths=(6, 6, 6, 6, 6, 6),
        num_heads=(6, 6, 6, 6, 6, 6),
        window_size=7,
        upscale=4,
        num_rrdb=23,  # Configurazione Real-ESRGAN completa
        num_feat=64,
        num_grow_ch=32
    ).to(device)

    # Carica HAT pre-trained (opzionale)
    if args.pretrained_hat and rank == 0:
        try:
            net_g.load_pretrained_hat(args.pretrained_hat)
        except Exception as e:
            print(f"⚠️  Impossibile caricare HAT pre-trained: {e}")

    net_d = UNetDiscriminatorSN(num_in_ch=1, num_feat=64).to(device)

    net_g = DDP(net_g, device_ids=[local_rank])
    net_d = DDP(net_d, device_ids=[local_rank])

    opt_g = torch.optim.AdamW(net_g.parameters(), lr=LR_G, betas=(0.9, 0.99))
    opt_d = torch.optim.AdamW(net_d.parameters(), lr=LR_D, betas=(0.9, 0.99))
    
    criterion_g = CombinedGANLoss(pixel_weight=1.0, adversarial_weight=0.005).to(device)
    criterion_d = DiscriminatorLoss().to(device)

    if rank == 0: 
        total_params = sum(p.numel() for p in net_g.parameters())
        print(f"   Parametri Generatore: {total_params:,}")
        print("=" * 60)

    for epoch in range(1, NUM_EPOCHS + 1):
        train_sampler.set_epoch(epoch)
        if rank == 0:
            loader_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS}", unit="batch")
        else:
            loader_bar = train_loader

        for i, batch in enumerate(loader_bar):
            lr = batch['lr'].to(device)
            hr = batch['hr'].to(device)

            # === Train Generator ===
            for p in net_d.parameters(): 
                p.requires_grad = False
            
            sr = net_g(lr)
            pred_fake = net_d(sr)
            pred_real = net_d(hr).detach()
            loss_g, loss_dict_g = criterion_g(sr, hr, pred_real, pred_fake)
            
            loss_g = loss_g / GRADIENT_ACCUMULATION
            loss_g.backward()
            
            if (i + 1) % GRADIENT_ACCUMULATION == 0:
                opt_g.step()
                opt_g.zero_grad()

            # === Train Discriminator ===
            for p in net_d.parameters(): 
                p.requires_grad = True
            
            pred_fake_d = net_d(sr.detach())
            pred_real_d = net_d(hr)
            loss_d, loss_dict_d = criterion_d(pred_real_d, pred_fake_d)
            
            loss_d = loss_d / GRADIENT_ACCUMULATION
            loss_d.backward()
            
            if (i + 1) % GRADIENT_ACCUMULATION == 0:
                opt_d.step()
                opt_d.zero_grad()

            if rank == 0:
                loader_bar.set_postfix({
                    'L_G': f"{loss_g.item() * GRADIENT_ACCUMULATION:.4f}",
                    'L_D': f"{loss_d.item() * GRADIENT_ACCUMULATION:.4f}"
                })

        # === SALVATAGGIO CHECKPOINT E PREVIEW ===
        if rank == 0 and epoch % SAVE_INTERVAL == 0:
            target_folder_name = args.target.replace(',', '_')
            base_output_dir = Path("./outputs") / target_folder_name
            ckpt_dir = base_output_dir / "checkpoints"
            preview_dir = base_output_dir / "previews"
            
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            preview_dir.mkdir(parents=True, exist_ok=True)
            
            # Salva Modello
            torch.save({
                'epoch': epoch,
                'model_state_dict': net_g.module.state_dict(),
                'optimizer_state_dict': opt_g.state_dict(),
            }, ckpt_dir / f"hybrid_epoch_{epoch}.pth")
            
            torch.save(net_g.module.state_dict(), ckpt_dir / "best_hybrid_model.pth")
            
            # Salva Preview
            save_validation_preview(lr, sr, hr, epoch, preview_dir)
            
            tqdm.write(f"💾 Epoch {epoch}: Modello Ibrido salvato in {base_output_dir}")

    if rank == 0:
        print("✅ Training completato!")
    
    dist.destroy_process_group()

if __name__ == "__main__":
    train_worker()
