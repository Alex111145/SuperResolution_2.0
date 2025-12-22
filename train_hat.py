import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from models.hat_arch import HAT
from models.discriminator import UNetDiscriminatorSN
from dataset.astronomical_dataset import AstronomicalDataset
from utils.gan_losses import CombinedGANLoss, DiscriminatorLoss
from utils.metrics import TrainMetrics

# Configurazione Iperparametri
BATCH_SIZE = 2
LR_G = 1e-4
EMA_DECAY = 0.999

def setup():
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def train_worker():
    setup()
    device = torch.device(f"cuda:{os.environ['LOCAL_RANK']}")
    rank = dist.get_rank()

    # Inizializzazione HAT (Il nuovo Motore)
    net_g = HAT(
        img_size=128, in_chans=1, embed_dim=180, 
        depths=(6, 6, 6, 6, 6, 6), num_heads=(6, 6, 6, 6, 6, 6),
        window_size=7, upscale=4, upsampler='pixelshuffle'
    ).to(device)
    net_g = DDP(net_g, device_ids=[int(os.environ["LOCAL_RANK"])])

    # Inizializzazione Discriminatore
    net_d = UNetDiscriminatorSN(num_in_ch=1, num_feat=64).to(device)
    net_d = DDP(net_d, device_ids=[int(os.environ["LOCAL_RANK"])])

    # Dataset e Loader
    train_ds = AstronomicalDataset("train.json", base_path="./")
    train_sampler = DistributedSampler(train_ds)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=train_sampler)

    # Loss e Optimizer
    opt_g = torch.optim.AdamW(net_g.parameters(), lr=LR_G)
    criterion_g = CombinedGANLoss(pixel_weight=1.0, adversarial_weight=0.005).to(device)

    for epoch in range(1, 301):
        train_sampler.set_epoch(epoch)
        for i, batch in enumerate(train_loader):
            lr = batch['lr'].to(device)
            hr = batch['hr'].to(device)

            # Step HAT (Generatore)
            sr = net_g(lr)
            # Calcolo Loss (Pixel + Perceptual + GAN)
            # ... Logica di backprop e update ...

            if rank == 0 and i % 10 == 0:
                print(f"Epoch {epoch} | Batch {i}")

if __name__ == "__main__":
    train_worker()