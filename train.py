import os
import argparse
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from tqdm import tqdm
import warnings
import time
import gc

warnings.filterwarnings("ignore")

# --- MODIFICA PATH: Setup Root ---
# Il file è ora nella root, quindi il parent è la directory del progetto stessa
CURRENT_SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_SCRIPT_DIR
ROOT_DATA_DIR = PROJECT_ROOT / "data"

# Aggiungiamo la root al sys.path per poter importare i pacchetti (models, utils, dataset)
sys.path.insert(0, str(PROJECT_ROOT))

try:
    # --- MODIFICA IMPORT: Nuova Struttura ---
    from models.architecture import TrainHybridModel
    from models.discriminator import UNetDiscriminatorSN
    from dataset.astronomical_dataset import AstronomicalDataset
    from utils.gan_losses import CombinedGANLoss, DiscriminatorLoss
    from utils.metrics import TrainMetrics
except ImportError as e:
    sys.exit(f"Errore Import: {e}. Assicurati di aver spostato i file nelle cartelle 'models', 'dataset', 'utils'.")

BATCH_SIZE = 1        
ACCUM_STEPS = 4       
LR_G = 1e-4           
LR_D = 1e-4           
TOTAL_EPOCHS = 300 
LOG_INTERVAL = 1 
IMAGE_INTERVAL = 1

def setup():
    """Inizializza il gruppo di processi per DDP"""
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def cleanup():
    dist.destroy_process_group()

def train_worker():
    setup()
    
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{local_rank}")
    is_master = (rank == 0)

    parser = argparse.ArgumentParser()
    parser.add_argument('--target', type=str, required=True, help="Nome target (es. M1,M33)")
    args = parser.parse_args()

    target_names = [t.strip() for t in args.target.split(',') if t.strip()]
    target_output_name = "_".join(target_names)

    out_dir = PROJECT_ROOT / "outputs" / f"{target_output_name}_DDP_GAN"
    save_dir = out_dir / "checkpoints"
    img_dir = out_dir / "images"
    log_dir = out_dir / "tensorboard"
    splits_dir_temp = out_dir / "temp_splits"

    old_l1_dir = PROJECT_ROOT / "outputs" / f"{target_output_name}_DDP"
    old_l1_weights = old_l1_dir / "checkpoints" / "best_train_model.pth"

    latest_ckpt_path = save_dir / "latest_checkpoint.pth"
    best_weights_path = save_dir / "best_gan_model.pth"

    if is_master:
        for d in [save_dir, img_dir, log_dir, splits_dir_temp]: d.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(str(log_dir))
        print(f"[Master] Training GAN su: {target_output_name} | GPUs: {world_size}")

    dist.barrier() 

    all_train_data = []
    all_val_data = []
    
    for t_name in target_names:
        s_dir = ROOT_DATA_DIR / t_name / "8_dataset_split" / "splits_json"
        try:
            with open(s_dir / "train.json") as f: all_train_data.extend(json.load(f))
            with open(s_dir / "val.json") as f: all_val_data.extend(json.load(f))
        except FileNotFoundError:
            if is_master: print(f"Dati non trovati per {t_name}, salto.")

    ft_path = splits_dir_temp / f"temp_train_r{rank}.json"
    fv_path = splits_dir_temp / f"temp_val_r{rank}.json"
    with open(ft_path, 'w') as f: json.dump(all_train_data, f)
    with open(fv_path, 'w') as f: json.dump(all_val_data, f)

    train_ds = AstronomicalDataset(ft_path, base_path=PROJECT_ROOT, augment=True)
    val_ds = AstronomicalDataset(fv_path, base_path=PROJECT_ROOT, augment=False)
    
    train_sampler = DistributedSampler(train_ds, shuffle=True)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False, 
                              num_workers=4, pin_memory=True, sampler=train_sampler, drop_last=True)
    
    val_sampler = DistributedSampler(val_ds, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2, sampler=val_sampler)

    net_g = TrainHybridModel(smoothing='none', device=device).to(device)
    net_g = nn.SyncBatchNorm.convert_sync_batchnorm(net_g)
    net_g = DDP(net_g, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
    
    net_d = UNetDiscriminatorSN(num_in_ch=1, num_feat=64).to(device)
    net_d = nn.SyncBatchNorm.convert_sync_batchnorm(net_d)
    net_d = DDP(net_d, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    opt_g = optim.AdamW(net_g.parameters(), lr=LR_G, weight_decay=1e-4, betas=(0.9, 0.99))
    opt_d = optim.AdamW(net_d.parameters(), lr=LR_D, weight_decay=1e-4, betas=(0.9, 0.99))

    sched_g = optim.lr_scheduler.CosineAnnealingLR(opt_g, T_max=TOTAL_EPOCHS, eta_min=1e-7)
    sched_d = optim.lr_scheduler.CosineAnnealingLR(opt_d, T_max=TOTAL_EPOCHS, eta_min=1e-7)

    criterion_g = CombinedGANLoss(gan_type='ragan', pixel_weight=1.0, perceptual_weight=0.1, adversarial_weight=5e-3).to(device)
    criterion_d = DiscriminatorLoss(gan_type='ragan').to(device)
    
    scaler = torch.amp.GradScaler('cuda')

    start_epoch = 1
    best_psnr = 0.0
    map_location = {'cuda:%d' % 0: 'cuda:%d' % local_rank}
    
    status_msg = "NESSUN PESO TROVATO. Inizio training DA ZERO (Random Init)."
    loaded_files = []

    if latest_ckpt_path.exists():
        try:
            checkpoint = torch.load(latest_ckpt_path, map_location=map_location)
            
            net_g.module.load_state_dict(checkpoint['net_g'])
            opt_g.load_state_dict(checkpoint['opt_g'])
            sched_g.load_state_dict(checkpoint['sched_g'])
            
            if 'net_d' in checkpoint:
                net_d.module.load_state_dict(checkpoint['net_d'])
                opt_d.load_state_dict(checkpoint['opt_d'])
                sched_d.load_state_dict(checkpoint['sched_d'])
            
            if 'scaler' in checkpoint:
                scaler.load_state_dict(checkpoint['scaler'])
            
            start_epoch = checkpoint['epoch'] + 1
            best_psnr = checkpoint.get('best_psnr', 0.0)
            
            status_msg = f"RESUME GAN COMPLETATO da Epoca {start_epoch}."
            loaded_files.append(latest_ckpt_path.name)
            
        except Exception as e:
            if is_master: print(f"Errore lettura checkpoint {latest_ckpt_path.name}: {e}")

    elif best_weights_path.exists():
        try:
            state_dict = torch.load(best_weights_path, map_location=map_location)
            net_g.module.load_state_dict(state_dict)
            status_msg = f"WARM START da Best GAN Model. D riparte da zero."
            loaded_files.append(best_weights_path.name)
        except Exception as e:
            if is_master: print(f"Errore lettura best weights: {e}")
            
    elif old_l1_weights.exists():
         try:
            state_dict = torch.load(old_l1_weights, map_location=map_location)
            if 'model_state_dict' in state_dict:
                 state_dict = state_dict['model_state_dict']
            
            net_g.module.load_state_dict(state_dict, strict=False) 
            status_msg = f"WARM START da Vecchio Training L1 ({old_l1_weights.parent.parent.name}). D riparte da zero."
            loaded_files.append(old_l1_weights.name)
         except Exception as e:
            if is_master: print(f"Errore lettura L1 weights ({old_l1_weights}): {e}")

    if is_master:
        print("\n" + "="*80)
        print(f"REPORT CARICAMENTO PESI:")
        print(f"File trovati: {loaded_files if loaded_files else 'Nessuno'}")
        print(f"Stato: {status_msg}")
        if not loaded_files:
            print(f"Ho cercato anche qui: {old_l1_weights}")
        print("="*80 + "\n")

    epoch_pbar = tqdm(range(start_epoch, TOTAL_EPOCHS + 1), desc="Training Total", unit="ep", initial=start_epoch, total=TOTAL_EPOCHS) if is_master else range(start_epoch, TOTAL_EPOCHS + 1)

    for epoch in epoch_pbar:
        start_time = time.time()
        train_sampler.set_epoch(epoch)
        net_g.train()
        net_d.train()
        
        acc_loss_g = 0.0
        acc_loss_d = 0.0
        acc_loss_adv = 0.0
        
        opt_g.zero_grad(set_to_none=True)
        opt_d.zero_grad(set_to_none=True)
        
        loader_iter = tqdm(train_loader, desc=f"Ep {epoch} [GAN]", ncols=100, leave=False) if is_master else train_loader

        for i, batch in enumerate(loader_iter):
            lr_img = batch['lr'].to(device, non_blocking=True)
            hr_img = batch['hr'].to(device, non_blocking=True)
            
            for p in net_g.parameters(): p.requires_grad = False
            for p in net_d.parameters(): p.requires_grad = True

            with torch.amp.autocast('cuda'):
                with torch.no_grad():
                    sr_img = net_g(lr_img)
                
                d_real = net_d(hr_img)
                d_fake = net_d(sr_img.detach()) 
                
                loss_d, _ = criterion_d(d_real, d_fake)
                loss_d = loss_d / ACCUM_STEPS

            scaler.scale(loss_d).backward()
            
            del sr_img, d_real, d_fake
            
            if (i + 1) % ACCUM_STEPS == 0:
                scaler.unscale_(opt_d)
                torch.nn.utils.clip_grad_norm_(net_d.parameters(), 1.0)
                scaler.step(opt_d)
                opt_d.zero_grad(set_to_none=True)
            
            for p in net_g.parameters(): p.requires_grad = True
            for p in net_d.parameters(): p.requires_grad = False
            
            with torch.amp.autocast('cuda'):
                sr_img_g = net_g(lr_img)
                
                d_fake_for_g = net_d(sr_img_g)
                d_real_for_g = net_d(hr_img).detach()

                loss_g, loss_dict_g = criterion_g(sr_img_g, hr_img, d_real_for_g, d_fake_for_g)
                loss_g = loss_g / ACCUM_STEPS

            scaler.scale(loss_g).backward()
            
            if (i + 1) % ACCUM_STEPS == 0:
                scaler.unscale_(opt_g)
                torch.nn.utils.clip_grad_norm_(net_g.parameters(), 0.5)
                scaler.step(opt_g)
                
                scaler.update()
                opt_g.zero_grad(set_to_none=True)
                opt_d.zero_grad(set_to_none=True)

            if not torch.isnan(loss_g):
                acc_loss_g += loss_g.item() * ACCUM_STEPS
                acc_loss_d += loss_d.item() * ACCUM_STEPS
                acc_loss_adv += loss_dict_g.get('adversarial', torch.tensor(0)).item()
            
            del sr_img_g, d_fake_for_g, d_real_for_g, loss_g, loss_d, lr_img, hr_img

        sched_g.step()
        sched_d.step()
        
        epoch_duration = time.time() - start_time
        
        if epoch % LOG_INTERVAL == 0 or epoch == TOTAL_EPOCHS:
            dist.barrier()
            torch.cuda.empty_cache() 
            
            metrics_tensor = torch.tensor([acc_loss_g, acc_loss_d, acc_loss_adv], device=device)
            dist.all_reduce(metrics_tensor, op=dist.ReduceOp.SUM)
            metrics_tensor /= (len(train_loader) * world_size)
            avg_loss_g, avg_loss_d, avg_loss_adv = metrics_tensor.tolist()

            net_g.eval()
            local_metrics = TrainMetrics()
            
            val_iter = tqdm(val_loader, desc="Val", ncols=50, leave=False) if is_master else val_loader

            with torch.inference_mode():
                for v_batch in val_iter:
                    v_lr = v_batch['lr'].to(device)
                    v_hr = v_batch['hr'].to(device)
                    with torch.amp.autocast('cuda'):
                        v_pred = net_g(v_lr)
                    v_pred = torch.nan_to_num(v_pred)
                    local_metrics.update(v_pred.float(), v_hr.float())
            
            total_psnr = torch.tensor(local_metrics.psnr, device=device)
            total_ssim = torch.tensor(local_metrics.ssim, device=device)
            total_count = torch.tensor(local_metrics.count, device=device)
            
            dist.all_reduce(total_psnr, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_ssim, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_count, op=dist.ReduceOp.SUM)
            
            global_psnr = total_psnr.item() / total_count.item()
            global_ssim = total_ssim.item() / total_count.item()

            if is_master:
                writer.add_scalar('Train/Loss_G', avg_loss_g, epoch)
                writer.add_scalar('Train/Loss_D', avg_loss_d, epoch)
                writer.add_scalar('Val/PSNR', global_psnr, epoch)
                writer.add_scalar('Val/SSIM', global_ssim, epoch)
                
                print(f" Ep {epoch:04d} | G: {avg_loss_g:.4f} | D: {avg_loss_d:.4f} | PSNR: {global_psnr:.2f} dB | SSIM: {global_ssim:.4f} | Time: {epoch_duration:.0f}s")

                if global_psnr > best_psnr:
                    best_psnr = global_psnr
                    print("Nuovo Best PSNR (GAN)!")
                    torch.save(net_g.module.state_dict(), save_dir / "best_gan_model.pth")
                
                if epoch % IMAGE_INTERVAL == 0:
                    v_lr_up = torch.nn.functional.interpolate(v_lr, size=(512,512), mode='nearest')
                    comp = torch.cat((v_lr_up, v_pred, v_hr), dim=3).clamp(0,1)
                    vutils.save_image(comp, img_dir / f"gan_epoch_{epoch}.png")

        if is_master:
            checkpoint_dict = {
                'epoch': epoch,
                'net_g': net_g.module.state_dict(),
                'net_d': net_d.module.state_dict(),
                'opt_g': opt_g.state_dict(),
                'opt_d': opt_d.state_dict(),
                'sched_g': sched_g.state_dict(),
                'sched_d': sched_d.state_dict(),
                'scaler': scaler.state_dict(),
                'best_psnr': best_psnr
            }
            torch.save(checkpoint_dict, latest_ckpt_path)

        dist.barrier()
        net_g.train()

    if is_master:
        try:
            if ft_path.exists(): ft_path.unlink()
            if fv_path.exists(): fv_path.unlink()
        except: pass
        
    cleanup()

if __name__ == "__main__":
    train_worker()
