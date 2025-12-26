import os
import sys
import argparse
import json
import torch
import numpy as np
import csv
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="torchvision.models._utils")
warnings.filterwarnings("ignore", message="Grad strides do not match bucket view strides")
warnings.filterwarnings("ignore", message="torch.meshgrid: in an upcoming release")

import torchvision.transforms.functional as TF_functional
sys.modules['torchvision.transforms.functional_tensor'] = TF_functional

from models.hybridmodels import HybridHATRealESRGAN
from models.discriminator import UNetDiscriminatorSN
from dataset.astronomical_dataset import AstronomicalDataset
from utils.gan_losses import CombinedGANLoss, DiscriminatorLoss

BATCH_SIZE = 1 
LR_G = 1e-4
LR_D = 1e-4
NUM_EPOCHS = 300
WARMUP_EPOCHS = 30       
SAVE_INTERVAL_CKPT = 3   
SAVE_INTERVAL_IMG = 10   
GRADIENT_ACCUMULATION = 16

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

def update_ema(model_src, model_ema, decay=0.999):
    with torch.no_grad():
        for p_src, p_ema in zip(model_src.parameters(), model_ema.parameters()):
            p_ema.data.mul_(decay).add_(p_src.data, alpha=1 - decay)

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
    log_path = base_output_dir / "train_log.csv"
    
    if rank == 0:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        preview_dir.mkdir(parents=True, exist_ok=True)
        
        if not log_path.exists():
            with open(log_path, "w", newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Epoch", "G_Total", "L1", "G_Adv", "D_Total", "LR"])

    if args.resume is None:
        latest_ckpt = find_latest_checkpoint(ckpt_dir)
        if latest_ckpt: args.resume = str(latest_ckpt)

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

    net_g = HybridHATRealESRGAN(
        img_size=128, in_chans=1, embed_dim=90, depths=(6, 6, 6, 6),
        num_heads=(6, 6, 6, 6), window_size=8, upscale=4,
        num_rrdb=12, num_feat=48, num_grow_ch=24
    ).to(device)

    net_d = UNetDiscriminatorSN(num_in_ch=1, num_feat=64).to(device)

    net_g_ema = HybridHATRealESRGAN(
        img_size=128, in_chans=1, embed_dim=90, depths=(6, 6, 6, 6),
        num_heads=(6, 6, 6, 6), window_size=8, upscale=4,
        num_rrdb=12, num_feat=48, num_grow_ch=24
    ).to(device)
    for p in net_g_ema.parameters():
        p.requires_grad = False

    net_g = DDP(net_g, device_ids=[local_rank], find_unused_parameters=False)
    net_d = DDP(net_d, device_ids=[local_rank])

    opt_g = torch.optim.AdamW(net_g.parameters(), lr=LR_G, betas=(0.9, 0.99))
    opt_d = torch.optim.AdamW(net_d.parameters(), lr=LR_D, betas=(0.9, 0.99))

    criterion_pixel = torch.nn.L1Loss().to(device) 
    criterion_gan = CombinedGANLoss(pixel_weight=1.0, adversarial_weight=0.005).to(device)
    criterion_d = DiscriminatorLoss().to(device)

    start_epoch = 1
    
    if args.resume:
        try:
            checkpoint = torch.load(args.resume, map_location=device)
            net_g.module.load_state_dict(checkpoint['model_state_dict'])
            opt_g.load_state_dict(checkpoint['optimizer_state_dict'])
            
            for param_group in opt_g.param_groups:
                if 'initial_lr' not in param_group: param_group['initial_lr'] = LR_G
            
            net_g_ema.load_state_dict(net_g.module.state_dict())
            start_epoch = checkpoint['epoch'] + 1
            if rank == 0: print(f"✅ Resume da epoca {start_epoch}")
        except Exception as e:
            if rank == 0: print(f"❌ Errore resume: {e}")
    else:
        net_g_ema.load_state_dict(net_g.module.state_dict())

    for param_group in opt_d.param_groups:
         if 'initial_lr' not in param_group: param_group['initial_lr'] = LR_D

    last_epoch_scheduler = start_epoch - 2 if start_epoch > 1 else -1
    scheduler_g = CosineAnnealingLR(opt_g, T_max=NUM_EPOCHS, eta_min=1e-7, last_epoch=last_epoch_scheduler)
    scheduler_d = CosineAnnealingLR(opt_d, T_max=NUM_EPOCHS, eta_min=1e-7, last_epoch=last_epoch_scheduler)

    if rank == 0:
        os.system('cls' if os.name == 'nt' else 'clear')
        print("=" * 70)
        print(f"🚀 TRAINING HYBRID (Scheduler Attivo & EMA & Logger)")
        print(f"   • Start Epoch: {start_epoch}")
        print(f"   • LR G Attuale: {scheduler_g.get_last_lr()[0]:.2e}")
        print(f"   • Accumulation: {GRADIENT_ACCUMULATION}")
        print(f"   • Log File: {log_path}")
        print("=" * 70)

    dist.barrier()

    for epoch in range(start_epoch, NUM_EPOCHS + 1):
        train_sampler.set_epoch(epoch)
        net_g.train()
        net_d.train()
        
        is_warmup = epoch <= WARMUP_EPOCHS
        desc = f"Ep {epoch} [WARMUP]" if is_warmup else f"Ep {epoch} [GAN]"
        current_lr = scheduler_g.get_last_lr()[0]

        if rank == 0:
            loader_bar = tqdm(train_loader, desc=desc, unit="bt", ncols=100)
        else:
            loader_bar = train_loader
        
        ep_g_total = 0.0
        ep_l1 = 0.0 
        ep_g_adv = 0.0
        ep_d_total = 0.0
        steps = 0

        for i, batch in enumerate(loader_bar):
            lr = batch['lr'].to(device, non_blocking=True)
            hr = batch['hr'].to(device, non_blocking=True)

            for p in net_d.parameters(): p.requires_grad = False
            
            sr = net_g(lr)
            
            l1_loss = criterion_pixel(sr, hr)
            l1_val = l1_loss.item()
            
            g_adv_val = 0.0
            loss_d_val = 0.0
            
            if is_warmup:
                loss_g = l1_loss
                loss_d = torch.tensor(0.0).to(device)
            else:
                pred_fake = net_d(sr)
                pred_real = net_d(hr).detach()
                
                loss_g, loss_dict_g = criterion_gan(sr, hr, pred_real, pred_fake)
                g_adv_val = loss_dict_g.get('adversarial', torch.tensor(0.0)).item()
            
            (loss_g / GRADIENT_ACCUMULATION).backward()

            if (i + 1) % GRADIENT_ACCUMULATION == 0:
                opt_g.step()
                opt_g.zero_grad()
                update_ema(net_g.module, net_g_ema)

            if not is_warmup:
                for p in net_d.parameters(): p.requires_grad = True
                
                pred_fake_d = net_d(sr.detach())
                pred_real_d = net_d(hr)
                
                loss_d, _ = criterion_d(pred_real_d, pred_fake_d)
                loss_d_val = loss_d.item()
                
                (loss_d / GRADIENT_ACCUMULATION).backward()

                if (i + 1) % GRADIENT_ACCUMULATION == 0:
                    opt_d.step()
                    opt_d.zero_grad()
            
            steps += 1
            ep_g_total += loss_g.item()
            ep_l1 += l1_val
            ep_g_adv += g_adv_val
            ep_d_total += loss_d_val

            if rank == 0:
                loader_bar.set_postfix({
                    'LG': f"{loss_g.item():.3f}", 
                    'L1': f"{l1_val:.3f}", 
                    'Adv': f"{g_adv_val:.3f}",
                    'LD': f"{loss_d_val:.3f}", 
                    'LR': f"{current_lr:.1e}"
                })

        scheduler_g.step()
        scheduler_d.step()

        if rank == 0:
            avg_g = ep_g_total / steps if steps > 0 else 0
            avg_l1 = ep_l1 / steps if steps > 0 else 0
            avg_adv = ep_g_adv / steps if steps > 0 else 0
            avg_d = ep_d_total / steps if steps > 0 else 0

            with open(log_path, "a", newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    epoch, 
                    f"{avg_g:.4f}", 
                    f"{avg_l1:.4f}", 
                    f"{avg_adv:.4f}", 
                    f"{avg_d:.4f}", 
                    f"{current_lr:.2e}"
                ])

            if epoch % SAVE_INTERVAL_CKPT == 0 or epoch == NUM_EPOCHS:
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': net_g.module.state_dict(),
                    'optimizer_state_dict': opt_g.state_dict(),
                }
                torch.save(checkpoint, ckpt_dir / f"hybrid_epoch_{epoch:03d}.pth")
                torch.save(net_g.module.state_dict(), ckpt_dir / "best_hybrid_model.pth")
                torch.save(net_g_ema.state_dict(), ckpt_dir / "best_hybrid_model_EMA.pth")
                tqdm.write(f"💾 Checkpoint e EMA salvati.")

            if epoch % SAVE_INTERVAL_IMG == 0:
                save_validation_preview(lr, sr, hr, epoch, preview_dir)

    if rank == 0: 
        print("\n✅ TRAINING COMPLETATO!")
        if os.path.exists("temp_train_combined.json"): os.remove("temp_train_combined.json")
    
    dist.destroy_process_group()

if __name__ == "__main__":
    train_worker()
