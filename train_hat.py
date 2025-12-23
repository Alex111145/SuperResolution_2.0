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

# === IMPORT MODELLO IBRIDO ===
from models.hybrid_model import HybridHATRealESRGAN
from models.discriminator import UNetDiscriminatorSN
from dataset.astronomical_dataset import AstronomicalDataset
from utils.gan_losses import CombinedGANLoss, DiscriminatorLoss

# === CONFIGURAZIONE TRAINING ===
BATCH_SIZE = 2  # Ridotto per modello pesante (HAT + 23 RRDB)
LR_G = 1e-4
LR_D = 1e-4
NUM_EPOCHS = 300
SAVE_INTERVAL = 5
GRADIENT_ACCUMULATION = 2  # Simula batch_size effettivo = 4

def tensor_to_img(tensor):
    """Converte tensore PyTorch in immagine numpy uint8"""
    img = tensor.cpu().detach().squeeze().float().numpy()
    img = np.clip(img, 0, 1)
    return (img * 255).astype(np.uint8)

def save_validation_preview(lr, sr, hr, epoch, save_path):
    """Salva preview comparativa: LR | SR | HR"""
    lr_img = tensor_to_img(lr[0])
    sr_img = tensor_to_img(sr[0])
    hr_img = tensor_to_img(hr[0])
    
    h, w = sr_img.shape
    lr_pil = Image.fromarray(lr_img).resize((w, h), resample=Image.NEAREST)
    lr_img_resized = np.array(lr_pil)

    combined = np.hstack((lr_img_resized, sr_img, hr_img))
    Image.fromarray(combined).save(save_path / f"epoch_{epoch:03d}_preview.png")

def setup_distributed():
    """Setup training distribuito (multi-GPU)"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    else:
        # Single GPU / CPU fallback
        os.environ["LOCAL_RANK"] = "0"
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29500"
        dist.init_process_group(backend="gloo")

def train_worker():
    parser = argparse.ArgumentParser(description='Training Ibrido HAT + Real-ESRGAN')
    parser.add_argument('--target', type=str, required=True,
                       help='Nome target(s) separati da virgola (es: M1,M33)')
    parser.add_argument('--pretrained_hat', type=str, default=None,
                       help='Path a checkpoint HAT pre-trained (opzionale)')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path a checkpoint per riprendere training')
    args, unknown = parser.parse_known_args()

    setup_distributed()
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    if rank == 0:
        print("=" * 70)
        print("🚀 TRAINING MODELLO IBRIDO HAT + REAL-ESRGAN")
        print("=" * 70)
        print(f"📊 Configurazione:")
        print(f"   • Device: {device}")
        print(f"   • World Size: {world_size}")
        print(f"   • Batch Size (per GPU): {BATCH_SIZE}")
        print(f"   • Batch Size (effettivo): {BATCH_SIZE * GRADIENT_ACCUMULATION * world_size}")
        print(f"   • Target(s): {args.target}")
        print("=" * 70)

    # === PREPARAZIONE DATASET ===
    targets = args.target.split(',')
    combined_json_path = "temp_train_combined.json"

    if rank == 0:
        all_pairs = []
        for t in targets:
            json_path = Path("./data") / t / "8_dataset_split" / "splits_json" / "train.json"
            if json_path.exists():
                with open(json_path, 'r') as f:
                    pairs = json.load(f)
                    all_pairs.extend(pairs)
                    print(f"   ✓ Caricati {len(pairs)} samples da {t}")
            else:
                print(f"   ⚠️  Warning: {json_path} non trovato")
        
        if not all_pairs:
            sys.exit("❌ Errore: Nessun dato di training trovato!")
        
        with open(combined_json_path, 'w') as f:
            json.dump(all_pairs, f)
        print(f"   📦 Totale training samples: {len(all_pairs)}")

    dist.barrier()

    train_ds = AstronomicalDataset(combined_json_path, base_path="./")
    train_sampler = DistributedSampler(train_ds, shuffle=True)
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        drop_last=True
    )

    # === INIZIALIZZAZIONE MODELLI ===
    if rank == 0:
        print("\n🏗️  Inizializzazione modelli...")

    # Generatore: Modello Ibrido HAT + Real-ESRGAN
    net_g = HybridHATRealESRGAN(
        img_size=128,
        in_chans=1,
        embed_dim=180,
        depths=(6, 6, 6, 6, 6, 6),
        num_heads=(6, 6, 6, 6, 6, 6),
        window_size=7,
        upscale=4,
        num_rrdb=23,  # Real-ESRGAN completo
        num_feat=64,
        num_grow_ch=32
    ).to(device)

    # Carica HAT pre-trained (opzionale)
    if args.pretrained_hat and rank == 0:
        try:
            net_g.load_pretrained_hat(args.pretrained_hat)
        except Exception as e:
            print(f"⚠️  Impossibile caricare HAT pre-trained: {e}")

    # Discriminatore
    net_d = UNetDiscriminatorSN(num_in_ch=1, num_feat=64).to(device)

    # Distributed Data Parallel
    net_g = DDP(net_g, device_ids=[local_rank], find_unused_parameters=False)
    net_d = DDP(net_d, device_ids=[local_rank])

    # === OTTIMIZZATORI ===
    opt_g = torch.optim.AdamW(net_g.parameters(), lr=LR_G, betas=(0.9, 0.99), weight_decay=0)
    opt_d = torch.optim.AdamW(net_d.parameters(), lr=LR_D, betas=(0.9, 0.99), weight_decay=0)

    # === LOSS FUNCTIONS ===
    criterion_g = CombinedGANLoss(pixel_weight=1.0, adversarial_weight=0.005).to(device)
    criterion_d = DiscriminatorLoss().to(device)

    # === RESUME TRAINING (opzionale) ===
    start_epoch = 1
    if args.resume and rank == 0:
        try:
            checkpoint = torch.load(args.resume, map_location=device)
            net_g.module.load_state_dict(checkpoint['model_state_dict'])
            opt_g.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            print(f"✓ Training ripreso dall'epoca {start_epoch}")
        except Exception as e:
            print(f"⚠️  Impossibile riprendere training: {e}")

    if rank == 0:
        total_params_g = sum(p.numel() for p in net_g.parameters())
        total_params_d = sum(p.numel() for p in net_d.parameters())
        print(f"\n📈 Parametri:")
        print(f"   • Generatore (HAT+RRDB): {total_params_g:,}")
        print(f"   • Discriminatore: {total_params_d:,}")
        print("=" * 70)

    # === TRAINING LOOP ===
    for epoch in range(start_epoch, NUM_EPOCHS + 1):
        train_sampler.set_epoch(epoch)
        net_g.train()
        net_d.train()

        if rank == 0:
            loader_bar = tqdm(
                train_loader,
                desc=f"Epoch {epoch}/{NUM_EPOCHS}",
                unit="batch",
                ncols=100
            )
        else:
            loader_bar = train_loader

        for i, batch in enumerate(loader_bar):
            lr = batch['lr'].to(device, non_blocking=True)
            hr = batch['hr'].to(device, non_blocking=True)

            # === TRAIN GENERATOR ===
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

            # === TRAIN DISCRIMINATOR ===
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

            # === LOGGING ===
            if rank == 0:
                loader_bar.set_postfix({
                    'L_G': f"{loss_g.item() * GRADIENT_ACCUMULATION:.4f}",
                    'L_D': f"{loss_d.item() * GRADIENT_ACCUMULATION:.4f}"
                })

        # === SALVATAGGIO CHECKPOINT ===
        if rank == 0 and epoch % SAVE_INTERVAL == 0:
            target_folder_name = args.target.replace(',', '_')
            base_output_dir = Path("./outputs") / target_folder_name
            ckpt_dir = base_output_dir / "checkpoints"
            preview_dir = base_output_dir / "previews"

            ckpt_dir.mkdir(parents=True, exist_ok=True)
            preview_dir.mkdir(parents=True, exist_ok=True)

            # Salva checkpoint completo
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': net_g.module.state_dict(),
                'optimizer_state_dict': opt_g.state_dict(),
                'loss_g': loss_g.item() * GRADIENT_ACCUMULATION,
                'loss_d': loss_d.item() * GRADIENT_ACCUMULATION
            }
            torch.save(checkpoint, ckpt_dir / f"hybrid_epoch_{epoch:03d}.pth")
            torch.save(net_g.module.state_dict(), ckpt_dir / "best_hybrid_model.pth")

            # Salva preview
            save_validation_preview(lr, sr, hr, epoch, preview_dir)

            tqdm.write(f"💾 Epoch {epoch}: Checkpoint salvato in {base_output_dir}")

    if rank == 0:
        print("\n" + "=" * 70)
        print("✅ TRAINING COMPLETATO!")
        print("=" * 70)

    dist.destroy_process_group()

if __name__ == "__main__":
    train_worker()
