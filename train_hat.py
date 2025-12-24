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

# === PATCH TORCHVISION ===
import torchvision.transforms.functional as TF_functional
sys.modules['torchvision.transforms.functional_tensor'] = TF_functional

# === IMPORT MODELLO IBRIDO ===
from models.hybridmodels import HybridHATRealESRGAN
from models.discriminator import UNetDiscriminatorSN
from dataset.astronomical_dataset import AstronomicalDataset
from utils.gan_losses import CombinedGANLoss, DiscriminatorLoss

# === CONFIGURAZIONE TRAINING ===
BATCH_SIZE = 1 
LR_G = 1e-4
LR_D = 1e-4
NUM_EPOCHS = 300
WARMUP_EPOCHS = 30       # <--- NUOVO: Prime 30 epoche solo L1 Loss (Pixel)
SAVE_INTERVAL_CKPT = 3   
SAVE_INTERVAL_IMG = 10   
GRADIENT_ACCUMULATION = 2  

def tensor_to_img(tensor):
    img = tensor.cpu().detach().squeeze().float().numpy()
    img = np.clip(img, 0, 1)
    return (img * 255).astype(np.uint8)

def save_validation_preview(lr, sr, hr, epoch, save_path):
    lr_img = tensor_to_img(lr[0])
    sr_img = tensor_to_img(sr[0])
    hr_img = tensor_to_img(hr[0])
    h, w = sr_img.shape
    lr_pil = Image.fromarray(lr_img).resize((w, h), resample=Image.NEAREST)
    lr_img_resized = np.array(lr_pil)
    combined = np.hstack((lr_img_resized, sr_img, hr_img))
    Image.fromarray(combined).save(save_path / f"epoch_{epoch:03d}_preview.png")

def setup_distributed():
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
    if not ckpt_dir.exists(): return None
    pths = list(ckpt_dir.glob("hybrid_epoch_*.pth"))
    if not pths: return None
    try:
        return sorted(pths, key=lambda x: int(x.stem.split('_')[-1]))[-1]
    except: return None

def train_worker():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', type=str, required=True)
    parser.add_argument('--pretrained_hat', type=str, default=None)
    parser.add_argument('--resume', type=str, default=None)
    args, unknown = parser.parse_known_args()

    setup_distributed()
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = dist.get_rank()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    target_folder_name = args.target.replace(',', '_')
    base_output_dir = Path("./outputs") / target_folder_name
    ckpt_dir = base_output_dir / "checkpoints"
    preview_dir = base_output_dir / "previews"
    
    if rank == 0:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        preview_dir.mkdir(parents=True, exist_ok=True)

    # Auto-Resume
    if args.resume is None:
        latest_ckpt = find_latest_checkpoint(ckpt_dir)
        if latest_ckpt: args.resume = str(latest_ckpt)

    if rank == 0:
        os.system('cls' if os.name == 'nt' else 'clear')
        print("=" * 70)
        print("🚀 TRAINING MODELLO IBRIDO (con WARMUP)")
        print(f"   • Warmup (L1 only): {WARMUP_EPOCHS} epoche")
        print(f"   • Totale Epoche: {NUM_EPOCHS}")
        print("=" * 70)

    # Dataset
    targets = args.target.split(',')
    if rank == 0:
        all_pairs = []
        for t in targets:
            json_path = Path("./data") / t / "8_dataset_split" / "splits_json" / "train.json"
            if json_path.exists():
                with open(json_path, 'r') as f: all_pairs.extend(json.load(f))
        
        if not all_pairs: sys.exit("❌ Errore: Nessun dato trovato!")
        with open("temp_train_combined.json", 'w') as f: json.dump(all_pairs, f)

    dist.barrier()

    train_ds = AstronomicalDataset("temp_train_combined.json", base_path="./")
    train_sampler = DistributedSampler(train_ds, shuffle=True)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=train_sampler,
                              num_workers=4, pin_memory=True, persistent_workers=True, drop_last=True)

    # Modello SMALL (Configurazione tua)
    net_g = HybridHATRealESRGAN(
        img_size=128, in_chans=1, embed_dim=90, depths=(6, 6, 6, 6),
        num_heads=(6, 6, 6, 6), window_size=8, upscale=4,
        num_rrdb=12, num_feat=48, num_grow_ch=24
    ).to(device)

    net_d = UNetDiscriminatorSN(num_in_ch=1, num_feat=64).to(device)
    net_g = DDP(net_g, device_ids=[local_rank], find_unused_parameters=False)
    net_d = DDP(net_d, device_ids=[local_rank])

    opt_g = torch.optim.AdamW(net_g.parameters(), lr=LR_G, betas=(0.9, 0.99))
    opt_d = torch.optim.AdamW(net_d.parameters(), lr=LR_D, betas=(0.9, 0.99))

    # LOSSES
    criterion_pixel = torch.nn.L1Loss().to(device) # Loss per Warmup
    criterion_gan = CombinedGANLoss(pixel_weight=1.0, adversarial_weight=0.005).to(device)
    criterion_d = DiscriminatorLoss().to(device)

    start_epoch = 1
    if args.resume:
        try:
            checkpoint = torch.load(args.resume, map_location=device)
            net_g.module.load_state_dict(checkpoint['model_state_dict'])
            opt_g.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            if rank == 0: print(f"✅ Resume da epoca {start_epoch}")
        except Exception as e:
            if rank == 0: print(f"❌ Errore resume: {e}")
    
    dist.barrier()

    for epoch in range(start_epoch, NUM_EPOCHS + 1):
        train_sampler.set_epoch(epoch)
        net_g.train()
        net_d.train()
        
        # Determina fase di training
        is_warmup = epoch <= WARMUP_EPOCHS
        desc = f"Epoch {epoch} [WARMUP L1]" if is_warmup else f"Epoch {epoch} [GAN]"

        if rank == 0:
            loader_bar = tqdm(train_loader, desc=desc, unit="bt", ncols=100)
        else:
            loader_bar = train_loader

        for i, batch in enumerate(loader_bar):
            lr = batch['lr'].to(device, non_blocking=True)
            hr = batch['hr'].to(device, non_blocking=True)

            # === TRAINING GENERATOR ===
            # Blocca discriminatore durante update G
            for p in net_d.parameters(): p.requires_grad = False
            
            sr = net_g(lr)
            
            if is_warmup:
                # FASE 1: Solo Pixel Loss (Stabilizza le stelle)
                loss_g = criterion_pixel(sr, hr)
                loss_d = torch.tensor(0.0).to(device) # Loss D fittizia
            else:
                # FASE 2: GAN Completa
                pred_fake = net_d(sr)
                pred_real = net_d(hr).detach()
                loss_g, _ = criterion_gan(sr, hr, pred_real, pred_fake)
            
            (loss_g / GRADIENT_ACCUMULATION).backward()

            if (i + 1) % GRADIENT_ACCUMULATION == 0:
                opt_g.step()
                opt_g.zero_grad()

            # === TRAINING DISCRIMINATOR (Solo dopo Warmup) ===
            if not is_warmup:
                for p in net_d.parameters(): p.requires_grad = True
                
                pred_fake_d = net_d(sr.detach())
                pred_real_d = net_d(hr)
                
                loss_d, _ = criterion_d(pred_real_d, pred_fake_d)
                (loss_d / GRADIENT_ACCUMULATION).backward()

                if (i + 1) % GRADIENT_ACCUMULATION == 0:
                    opt_d.step()
                    opt_d.zero_grad()
            
            if rank == 0:
                loader_bar.set_postfix({'L_G': f"{loss_g.item():.4f}", 'L_D': f"{loss_d.item():.4f}"})

        if rank == 0:
            if epoch % SAVE_INTERVAL_CKPT == 0 or epoch == NUM_EPOCHS:
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': net_g.module.state_dict(),
                    'optimizer_state_dict': opt_g.state_dict(),
                }
                torch.save(checkpoint, ckpt_dir / f"hybrid_epoch_{epoch:03d}.pth")
                torch.save(net_g.module.state_dict(), ckpt_dir / "best_hybrid_model.pth")
                tqdm.write(f"💾 Checkpoint salvato.")

            if epoch % SAVE_INTERVAL_IMG == 0:
                save_validation_preview(lr, sr, hr, epoch, preview_dir)

    if rank == 0: 
        print("\n✅ TRAINING COMPLETATO!")
        if os.path.exists("temp_train_combined.json"): os.remove("temp_train_combined.json")
    
    dist.destroy_process_group()

if __name__ == "__main__":
    train_worker()
