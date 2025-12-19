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
from copy import deepcopy

warnings.filterwarnings("ignore")

# --- PATH SETUP ---
CURRENT_SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_SCRIPT_DIR
ROOT_DATA_DIR = PROJECT_ROOT / "data"
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from models.architecture import TrainHybridModel
    from models.discriminator import UNetDiscriminatorSN
    from dataset.astronomical_dataset import AstronomicalDataset
    from utils.gan_losses import CombinedGANLoss, DiscriminatorLoss
    from utils.metrics import TrainMetrics
except ImportError as e:
    sys.exit(f"Errore Import: {e}. Verifica la struttura delle cartelle.")

# --- CONFIGURAZIONE ---
BATCH_SIZE = 1          # Consigliato: aumentare se la VRAM lo permette (es. 2 o 4)
ACCUM_STEPS = 4         # Gradient Accumulation
LR_G = 5e-5             # Ridotto leggermente per stabilità
LR_D = 5e-5             # Ridotto leggermente per stabilità
TOTAL_EPOCHS = 400 
LOG_INTERVAL = 1   
IMAGE_INTERVAL = 1      # Salva immagini meno frequentemente per risparmiare spazio
EMA_DECAY = 0.999       # Decay per l'EMA

# --- UTILS ---
class ModelEMA:
    """ Exponential Moving Average per i pesi del Generatore """
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.register()

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}

def check_nan(loss_value, label="Loss"):
    """ Controlla se la loss è valida """
    if torch.isnan(loss_value) or torch.isinf(loss_value):
        return True
    return False

def setup():
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def cleanup():
    dist.destroy_process_group()

# --- TRAINING LOOP ---
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

    # Paths
    out_dir = PROJECT_ROOT / "outputs" / f"{target_output_name}_DDP_GAN"
    save_dir = out_dir / "checkpoints"
    img_dir = out_dir / "images"
    log_dir = out_dir / "tensorboard"
    splits_dir_temp = out_dir / "temp_splits"

    # Load paths
    latest_ckpt_path = save_dir / "latest_checkpoint.pth"
    best_weights_path = save_dir / "best_gan_model.pth"
    # Fallback weights (L1 pretraining)
    old_l1_dir = PROJECT_ROOT / "outputs" / f"{target_output_name}_DDP"
    old_l1_weights = old_l1_dir / "checkpoints" / "best_train_model.pth"

    if is_master:
        for d in [save_dir, img_dir, log_dir, splits_dir_temp]: d.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(str(log_dir))
        print(f"[Master] Training GAN su: {target_output_name} | GPUs: {world_size}")

    dist.barrier() 

    # --- DATASET PREP ---
    # (Logica identica all'originale per compatibilità)
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

    # Dataset & Loader
    train_ds = AstronomicalDataset(ft_path, base_path=PROJECT_ROOT, augment=True)
    val_ds = AstronomicalDataset(fv_path, base_path=PROJECT_ROOT, augment=False)
    
    train_sampler = DistributedSampler(train_ds, shuffle=True)
    # Increase drop_last=True is crucial for BatchNorm stability with small batches
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False, 
                              num_workers=4, pin_memory=True, sampler=train_sampler, drop_last=True)
    val_sampler = DistributedSampler(val_ds, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2, sampler=val_sampler)

    # --- MODELS ---
    net_g = TrainHybridModel(smoothing='none', device=device).to(device)
    # Nota: SyncBatchNorm con BS=1 è instabile. Se possibile, usa GroupNorm o InstanceNorm nell'architettura.
    net_g = nn.SyncBatchNorm.convert_sync_batchnorm(net_g)
    net_g = DDP(net_g, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
    
    net_d = UNetDiscriminatorSN(num_in_ch=1, num_feat=64).to(device)
    net_d = nn.SyncBatchNorm.convert_sync_batchnorm(net_d)
    net_d = DDP(net_d, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    # Inizializza EMA
    ema_g = ModelEMA(net_g.module, decay=EMA_DECAY)

    # Optimizers & Schedulers
    opt_g = optim.AdamW(net_g.parameters(), lr=LR_G, weight_decay=1e-4, betas=(0.9, 0.99))
    opt_d = optim.AdamW(net_d.parameters(), lr=LR_D, weight_decay=1e-4, betas=(0.9, 0.99))

    sched_g = optim.lr_scheduler.CosineAnnealingLR(opt_g, T_max=TOTAL_EPOCHS, eta_min=1e-7)
    sched_d = optim.lr_scheduler.CosineAnnealingLR(opt_d, T_max=TOTAL_EPOCHS, eta_min=1e-7)

    # Losses
    # Riduciamo adversarial_weight inizialmente se il training è instabile (es. 1e-3)
    criterion_g = CombinedGANLoss(gan_type='ragan', pixel_weight=1.0, perceptual_weight=0.1, adversarial_weight=5e-3).to(device)
    criterion_d = DiscriminatorLoss(gan_type='ragan').to(device)
    
    scaler = torch.cuda.amp.GradScaler() # Usa la sintassi standard compatibile

    # --- RESUME / LOAD WEIGHTS ---
    start_epoch = 1
    best_psnr = 0.0
    map_location = {'cuda:%d' % 0: 'cuda:%d' % local_rank}
    
    status_msg = "Random Init."
    
    if latest_ckpt_path.exists():
        try:
            checkpoint = torch.load(latest_ckpt_path, map_location=map_location)
            net_g.module.load_state_dict(checkpoint['net_g'])
            net_d.module.load_state_dict(checkpoint['net_d'])
            opt_g.load_state_dict(checkpoint['opt_g'])
            opt_d.load_state_dict(checkpoint['opt_d'])
            sched_g.load_state_dict(checkpoint['sched_g'])
            sched_d.load_state_dict(checkpoint['sched_d'])
            scaler.load_state_dict(checkpoint['scaler'])
            start_epoch = checkpoint['epoch'] + 1
            best_psnr = checkpoint.get('best_psnr', 0.0)
            status_msg = f"RESUMED from Ep {start_epoch}."
            # Resume EMA if present, else re-register
            if 'ema_shadow' in checkpoint:
                ema_g.shadow = checkpoint['ema_shadow']
        except Exception as e:
            if is_master: print(f"Errore lettura checkpoint: {e}")

    elif best_weights_path.exists():
        state_dict = torch.load(best_weights_path, map_location=map_location)
        net_g.module.load_state_dict(state_dict)
        ema_g.register() # Re-init EMA from best weights
        status_msg = "Warm start from Best GAN weights."

    elif old_l1_weights.exists():
         try:
            state_dict = torch.load(old_l1_weights, map_location=map_location)
            if 'model_state_dict' in state_dict: state_dict = state_dict['model_state_dict']
            net_g.module.load_state_dict(state_dict, strict=False) 
            ema_g.register() # Init EMA
            status_msg = "Warm start from L1 pretraining."
         except Exception as e:
            if is_master: print(f"Errore L1 weights: {e}")

    if is_master:
        print(f"Status: {status_msg}")

    # --- LOOP ---
    for epoch in range(start_epoch, TOTAL_EPOCHS + 1):
        start_time = time.time()
        train_sampler.set_epoch(epoch)
        net_g.train()
        net_d.train()
        
        # Metrics Accumulators
        accum_g = 0.0
        accum_d = 0.0
        accum_adv = 0.0
        accum_pix = 0.0
        accum_vgg = 0.0
        valid_batches = 0
        
        opt_g.zero_grad()
        opt_d.zero_grad()
        
        loader_iter = tqdm(train_loader, desc=f"Ep {epoch} [GAN]", ncols=100, leave=False) if is_master else train_loader

        for i, batch in enumerate(loader_iter):
            current_iter = (epoch - 1) * len(train_loader) + i
            lr_img = batch['lr'].to(device, non_blocking=True)
            hr_img = batch['hr'].to(device, non_blocking=True)
            
            # -------------------------------------------------------------------
            # 1. Update Discriminator
            # -------------------------------------------------------------------
            for p in net_d.parameters(): p.requires_grad = True
            for p in net_g.parameters(): p.requires_grad = False
            
            with torch.cuda.amp.autocast():
                with torch.no_grad():
                    sr_img = net_g(lr_img)
                
                d_real = net_d(hr_img)
                d_fake = net_d(sr_img.detach()) 
                
                loss_d, _ = criterion_d(d_real, d_fake)
                loss_d = loss_d / ACCUM_STEPS

            # NaN Check D
            if check_nan(loss_d):
                if is_master: print(f"[WARN] NaN in D Loss (Ep {epoch}, It {i}). Skipping batch.")
                scaler.update() # Force update scaler to skip bad stats
                opt_d.zero_grad()
                opt_g.zero_grad()
                continue

            scaler.scale(loss_d).backward()
            
            if (i + 1) % ACCUM_STEPS == 0:
                scaler.unscale_(opt_d)
                torch.nn.utils.clip_grad_norm_(net_d.parameters(), 1.0) # Clipping
                scaler.step(opt_d)
                opt_d.zero_grad()

            # -------------------------------------------------------------------
            # 2. Update Generator
            # -------------------------------------------------------------------
            for p in net_d.parameters(): p.requires_grad = False
            for p in net_g.parameters(): p.requires_grad = True
            
            with torch.cuda.amp.autocast():
                sr_img_g = net_g(lr_img)
                d_fake_for_g = net_d(sr_img_g)
                d_real_for_g = net_d(hr_img).detach()

                loss_g_total, loss_dict_g = criterion_g(sr_img_g, hr_img, d_real_for_g, d_fake_for_g)
                loss_g = loss_g_total / ACCUM_STEPS

            # NaN Check G
            if check_nan(loss_g):
                if is_master: print(f"[WARN] NaN in G Loss (Ep {epoch}, It {i}). Skipping batch.")
                scaler.update()
                opt_g.zero_grad()
                continue

            scaler.scale(loss_g).backward()
            
            if (i + 1) % ACCUM_STEPS == 0:
                scaler.unscale_(opt_g)
                torch.nn.utils.clip_grad_norm_(net_g.parameters(), 1.0) # Clipping anche su G
                scaler.step(opt_g)
                scaler.update()
                opt_g.zero_grad()
                
                # Update EMA
                ema_g.update()

            # Accumula metriche solo se valido
            valid_batches += 1
            accum_g += loss_g_total.item()
            accum_d += loss_d.item() * ACCUM_STEPS # Revert scale
            accum_adv += loss_dict_g.get('adversarial', torch.tensor(0)).item()
            accum_pix += loss_dict_g.get('pixel', torch.tensor(0)).item()
            accum_vgg += loss_dict_g.get('perceptual', torch.tensor(0)).item()

        # Update Schedulers
        sched_g.step()
        sched_d.step()
        
        # --- VALIDATION ---
        if epoch % LOG_INTERVAL == 0 or epoch == TOTAL_EPOCHS:
            if valid_batches == 0: valid_batches = 1 # Avoid div by zero
            
            # All reduce metrics
            metrics = torch.tensor([accum_g, accum_d, accum_adv, accum_pix, accum_vgg, valid_batches], device=device)
            dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
            
            total_batches = metrics[5].item()
            if total_batches == 0: total_batches = 1
            
            avg_g = metrics[0].item() / total_batches
            avg_d = metrics[1].item() / total_batches
            avg_adv = metrics[2].item() / total_batches
            avg_pix = metrics[3].item() / total_batches
            avg_vgg = metrics[4].item() / total_batches

            # Validation with EMA weights
            ema_g.apply_shadow() # Load EMA weights
            net_g.eval()
            local_metrics = TrainMetrics()
            
            val_iter = tqdm(val_loader, desc="Val", ncols=50, leave=False) if is_master else val_loader

            with torch.inference_mode():
                for v_batch in val_iter:
                    v_lr = v_batch['lr'].to(device)
                    v_hr = v_batch['hr'].to(device)
                    with torch.cuda.amp.autocast():
                        v_pred = net_g(v_lr)
                    v_pred = torch.nan_to_num(v_pred).float().clamp(0, 1) # Safety clamp
                    local_metrics.update(v_pred, v_hr.float())
            
            ema_g.restore() # Restore training weights
            
            # Reduce Validation Metrics
            total_psnr = torch.tensor(local_metrics.psnr, device=device)
            total_ssim = torch.tensor(local_metrics.ssim, device=device)
            total_count = torch.tensor(local_metrics.count, device=device)
            dist.all_reduce(total_count, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_psnr, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_ssim, op=dist.ReduceOp.SUM)
            
            if total_count.item() > 0:
                global_psnr = total_psnr.item() / total_count.item()
                global_ssim = total_ssim.item() / total_count.item()
            else:
                global_psnr, global_ssim = 0.0, 0.0

            if is_master:
                epoch_duration = time.time() - start_time
                writer.add_scalar('Loss/G_Total', avg_g, epoch)
                writer.add_scalar('Loss/D_Total', avg_d, epoch)
                writer.add_scalar('Loss/G_Adv', avg_adv, epoch)
                writer.add_scalar('Loss/G_Pixel', avg_pix, epoch)
                writer.add_scalar('Loss/G_VGG', avg_vgg, epoch)
                writer.add_scalar('Metrics/PSNR', global_psnr, epoch)
                writer.add_scalar('Metrics/SSIM', global_ssim, epoch)
                
                print(f" Ep {epoch:04d} | G: {avg_g:.4f} (Adv: {avg_adv:.4f}) | D: {avg_d:.4f} | PSNR: {global_psnr:.2f} | Time: {epoch_duration:.0f}s")

                if global_psnr > best_psnr:
                    best_psnr = global_psnr
                    print("Nuovo Best PSNR (EMA)!")
                    # Save EMA model as best
                    ema_g.apply_shadow()
                    torch.save(net_g.module.state_dict(), save_dir / "best_gan_model.pth")
                    ema_g.restore()
                
                if epoch % IMAGE_INTERVAL == 0:
                    with torch.no_grad():
                        ema_g.apply_shadow()
                        v_pred_vis = net_g(v_lr).float().clamp(0, 1)
                        ema_g.restore()
                        v_lr_up = torch.nn.functional.interpolate(v_lr, size=v_pred_vis.shape[2:], mode='nearest')
                        comp = torch.cat((v_lr_up, v_pred_vis, v_hr), dim=3)
                        vutils.save_image(comp, img_dir / f"gan_epoch_{epoch}.png")

        # Save Checkpoint
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
                'best_psnr': best_psnr,
                'ema_shadow': ema_g.shadow
            }
            torch.save(checkpoint_dict, latest_ckpt_path)

        dist.barrier()
    
    cleanup()

if __name__ == "__main__":
    train_worker()
