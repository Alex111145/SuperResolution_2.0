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
from copy import deepcopy

warnings.filterwarnings("ignore")

CURRENT_SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_SCRIPT_DIR
ROOT_DATA_DIR = PROJECT_ROOT / "data"
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from models.architecture import SwinIR 
    from models.discriminator import UNetDiscriminatorSN
    from dataset.astronomical_dataset import AstronomicalDataset
    from utils.gan_losses import CombinedGANLoss, DiscriminatorLoss
    from utils.metrics import TrainMetrics
except ImportError as e:
    sys.exit(f"Errore Import: {e}. Verifica la struttura delle cartelle.")

BATCH_SIZE = 1          
ACCUM_STEPS = 4         
LR_G = 1e-4             
LR_D = 1e-4             
TOTAL_EPOCHS = 300 
LOG_INTERVAL = 1   
IMAGE_INTERVAL = 1       
EMA_DECAY = 0.999        

class ModelEMA:
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
    if torch.isnan(loss_value) or torch.isinf(loss_value):
        return True
    return False

def setup():
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

    out_dir = PROJECT_ROOT / "outputs" / f"{target_output_name}_DDP_SwinIR"
    save_dir = out_dir / "checkpoints"
    img_dir = out_dir / "images"
    log_dir = out_dir / "tensorboard"
    splits_dir_temp = out_dir / "temp_splits"

    latest_ckpt_path = save_dir / "latest_checkpoint.pth"
    best_weights_path = save_dir / "best_gan_model.pth"

    if is_master:
        for d in [save_dir, img_dir, log_dir, splits_dir_temp]: d.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(str(log_dir))
        print(f"[Master] Training SwinIR-GAN su: {target_output_name} | GPUs: {world_size}")

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

    net_g = SwinIR(upscale=4, in_chans=1, img_size=128, window_size=8,
                   img_range=1.0, depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                   mlp_ratio=2, upsampler='pixelshuffle', resi_connection='1conv')
    
    net_g = net_g.to(device)
    net_g = DDP(net_g, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    
    net_d = UNetDiscriminatorSN(num_in_ch=1, num_feat=64).to(device)
    net_d = nn.SyncBatchNorm.convert_sync_batchnorm(net_d)
    net_d = DDP(net_d, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    ema_g = ModelEMA(net_g.module, decay=EMA_DECAY)

    opt_g = optim.AdamW(net_g.parameters(), lr=LR_G, weight_decay=0, betas=(0.9, 0.99))
    opt_d = optim.AdamW(net_d.parameters(), lr=LR_D, weight_decay=0, betas=(0.9, 0.99))

    sched_g = optim.lr_scheduler.CosineAnnealingLR(opt_g, T_max=TOTAL_EPOCHS, eta_min=1e-7)
    sched_d = optim.lr_scheduler.CosineAnnealingLR(opt_d, T_max=TOTAL_EPOCHS, eta_min=1e-7)

    criterion_g = CombinedGANLoss(gan_type='ragan', pixel_weight=1.0, perceptual_weight=0.5, adversarial_weight=0.005).to(device)
    criterion_d = DiscriminatorLoss(gan_type='ragan').to(device)
    
    scaler = torch.cuda.amp.GradScaler() 

    start_epoch = 1
    best_psnr = 0.0
    map_location = {'cuda:%d' % 0: 'cuda:%d' % local_rank}
    
    if latest_ckpt_path.exists():
        try:
            checkpoint = torch.load(latest_ckpt_path, map_location=map_location)
            net_g.module.load_state_dict(checkpoint['net_g'])
            net_d.module.load_state_dict(checkpoint['net_d'])
            opt_g.load_state_dict(checkpoint['opt_g'])
            opt_d.load_state_dict(checkpoint['opt_d'])
            start_epoch = checkpoint['epoch'] + 1
            best_psnr = checkpoint.get('best_psnr', 0.0)
            if 'ema_shadow' in checkpoint: ema_g.shadow = checkpoint['ema_shadow']
            if is_master: print(f"Resumed from Epoch {start_epoch}")
        except Exception as e:
            if is_master: print(f"Errore resume: {e}")

    for epoch in range(start_epoch, TOTAL_EPOCHS + 1):
        start_time = time.time()
        train_sampler.set_epoch(epoch)
        net_g.train()
        net_d.train()
        
        accum_g = 0.0
        accum_d = 0.0
        valid_batches = 0
        
        opt_g.zero_grad()
        opt_d.zero_grad()
        
        loader_iter = tqdm(train_loader, desc=f"Ep {epoch} [Swin]", ncols=100, leave=False) if is_master else train_loader

        for i, batch in enumerate(loader_iter):
            lr_img = batch['lr'].to(device, non_blocking=True)
            hr_img = batch['hr'].to(device, non_blocking=True)
            
            for p in net_d.parameters(): p.requires_grad = True
            for p in net_g.parameters(): p.requires_grad = False
            
            with torch.cuda.amp.autocast():
                with torch.no_grad():
                    sr_img = net_g(lr_img)
                
                d_real = net_d(hr_img)
                d_fake = net_d(sr_img.detach()) 
                loss_d, _ = criterion_d(d_real, d_fake)
                loss_d = loss_d / ACCUM_STEPS

            if check_nan(loss_d):
                scaler.update()
                opt_d.zero_grad()
                continue

            scaler.scale(loss_d).backward()
            
            if (i + 1) % ACCUM_STEPS == 0:
                scaler.step(opt_d)
                opt_d.zero_grad()

            for p in net_d.parameters(): p.requires_grad = False
            for p in net_g.parameters(): p.requires_grad = True
            
            with torch.cuda.amp.autocast():
                sr_img_g = net_g(lr_img)
                d_fake_for_g = net_d(sr_img_g)
                d_real_for_g = net_d(hr_img).detach()

                loss_g_total, _ = criterion_g(sr_img_g, hr_img, d_real_for_g, d_fake_for_g)
                loss_g = loss_g_total / ACCUM_STEPS

            if check_nan(loss_g):
                scaler.update()
                opt_g.zero_grad()
                continue

            scaler.scale(loss_g).backward()
            
            if (i + 1) % ACCUM_STEPS == 0:
                scaler.step(opt_g)
                scaler.update()
                opt_g.zero_grad()
                ema_g.update()

            valid_batches += 1
            accum_g += loss_g_total.item()
            accum_d += loss_d.item() * ACCUM_STEPS

        sched_g.step()
        sched_d.step()
        
        if epoch % LOG_INTERVAL == 0:
            metrics = torch.tensor([accum_g, accum_d, valid_batches], device=device)
            dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
            
            total_b = metrics[2].item()
            if total_b == 0: total_b = 1
            avg_g = metrics[0].item() / total_b
            avg_d = metrics[1].item() / total_b

            ema_g.apply_shadow()
            net_g.eval()
            local_metrics = TrainMetrics()
            
            with torch.inference_mode():
                for v_batch in val_loader:
                    v_lr = v_batch['lr'].to(device)
                    v_hr = v_batch['hr'].to(device)
                    with torch.cuda.amp.autocast():
                        v_pred = net_g(v_lr)
                    v_pred = torch.nan_to_num(v_pred).float().clamp(0, 1)
                    local_metrics.update(v_pred, v_hr.float())
            
            ema_g.restore()
            
            t_psnr = torch.tensor(local_metrics.psnr, device=device)
            t_ssim = torch.tensor(local_metrics.ssim, device=device)
            t_cnt = torch.tensor(local_metrics.count, device=device)
            dist.all_reduce(t_cnt, op=dist.ReduceOp.SUM)
            dist.all_reduce(t_psnr, op=dist.ReduceOp.SUM)
            dist.all_reduce(t_ssim, op=dist.ReduceOp.SUM)
            
            g_psnr = t_psnr.item() / t_cnt.item() if t_cnt.item() > 0 else 0
            g_ssim = t_ssim.item() / t_cnt.item() if t_cnt.item() > 0 else 0

            if is_master:
                print(f" Ep {epoch:04d} | G: {avg_g:.4f} | D: {avg_d:.4f} | PSNR: {g_psnr:.2f} | Time: {time.time()-start_time:.0f}s")
                writer.add_scalar('Metrics/PSNR', g_psnr, epoch)
                
                if g_psnr > best_psnr:
                    best_psnr = g_psnr
                    ema_g.apply_shadow()
                    torch.save(net_g.module.state_dict(), save_dir / "best_gan_model.pth")
                    ema_g.restore()
                
                checkpoint_dict = {
                    'epoch': epoch,
                    'net_g': net_g.module.state_dict(),
                    'net_d': net_d.module.state_dict(),
                    'opt_g': opt_g.state_dict(),
                    'opt_d': opt_d.state_dict(),
                    'best_psnr': best_psnr,
                    'ema_shadow': ema_g.shadow
                }
                torch.save(checkpoint_dict, latest_ckpt_path)

                if epoch % IMAGE_INTERVAL == 0:
                    with torch.no_grad():
                        ema_g.apply_shadow()
                        v_pred_vis = net_g(v_lr).float().clamp(0, 1)
                        ema_g.restore()
                        v_lr_up = torch.nn.functional.interpolate(v_lr, size=v_pred_vis.shape[2:], mode='nearest')
                        comp = torch.cat((v_lr_up, v_pred_vis, v_hr), dim=3)
                        vutils.save_image(comp, img_dir / f"swin_epoch_{epoch}.png")

    cleanup()

if __name__ == "__main__":
    train_worker()
