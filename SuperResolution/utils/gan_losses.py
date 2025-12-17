import torch
import torch.nn as nn
import torch.nn.functional as F
from .losses_train import VGGLoss

class GANLoss(nn.Module):
    """
    Loss GAN standard con supporto per varianti:
    - Vanilla GAN (BCE)
    - Least Squares GAN (MSE)
    - Relativistic Average GAN (RaGAN)
    """
    def __init__(self, gan_type='ragan', real_label=1.0, fake_label=0.0):
        super().__init__()
        self.gan_type = gan_type.lower()
        self.real_label = real_label
        self.fake_label = fake_label
        
        if self.gan_type == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif self.gan_type in ['lsgan', 'ragan']:
            self.loss = nn.MSELoss()
        else:
            raise ValueError(f"GAN type {gan_type} non supportato. Usa: vanilla, lsgan, ragan")
    
    def _get_target_tensor(self, prediction, target_is_real):
        """Crea tensor di label (real o fake) della stessa dimensione della prediction."""
        if target_is_real:
            target = torch.full_like(prediction, self.real_label)
        else:
            target = torch.full_like(prediction, self.fake_label)
        return target
    
    def forward(self, prediction, target_is_real):
        """
        Calcola la loss GAN.
        
        Args:
            prediction: Output del discriminatore [B, 1, H, W]
            target_is_real: True se l'immagine è reale, False se fake
        """
        target = self._get_target_tensor(prediction, target_is_real)
        loss = self.loss(prediction, target)
        return loss


class RelativeGANLoss(nn.Module):
    """
    Relativistic Average GAN Loss (RaGAN).
    Migliora la stabilità del training valutando la "relatività" tra real e fake.
    
    Paper: "The relativistic discriminator: a key element missing from standard GAN"
    https://arxiv.org/abs/1807.00734
    """
    def __init__(self):
        super().__init__()
        self.loss = nn.BCEWithLogitsLoss()
    
    def forward(self, real_pred, fake_pred, for_discriminator=True):
        """
        Args:
            real_pred: Discriminatore su immagini reali
            fake_pred: Discriminatore su immagini generate
            for_discriminator: True per D loss, False per G loss
        """
        if for_discriminator:
            real_loss = self.loss(
                real_pred - torch.mean(fake_pred),
                torch.ones_like(real_pred)
            )
            fake_loss = self.loss(
                fake_pred - torch.mean(real_pred),
                torch.zeros_like(fake_pred)
            )
            return (real_loss + fake_loss) / 2
        else:
            fake_loss = self.loss(
                fake_pred - torch.mean(real_pred),
                torch.ones_like(fake_pred)
            )
            real_loss = self.loss(
                real_pred - torch.mean(fake_pred),
                torch.zeros_like(real_pred)
            )
            return (fake_loss + real_loss) / 2


class GradientPenalty(nn.Module):
    """
    Gradient Penalty per WGAN-GP.
    Opzionale, utile se si vuole usare Wasserstein GAN.
    """
    def __init__(self, lambda_gp=10.0):
        super().__init__()
        self.lambda_gp = lambda_gp
    
    def forward(self, discriminator, real_data, fake_data):
        batch_size = real_data.size(0)
        
        alpha = torch.rand(batch_size, 1, 1, 1, device=real_data.device)
        interpolates = (alpha * real_data + (1 - alpha) * fake_data).requires_grad_(True)
        
        disc_interpolates = discriminator(interpolates)
        
        gradients = torch.autograd.grad(
            outputs=disc_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones_like(disc_interpolates),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        
        gradients = gradients.view(batch_size, -1)
        gradient_norm = gradients.norm(2, dim=1)
        penalty = self.lambda_gp * ((gradient_norm - 1) ** 2).mean()
        
        return penalty


class TextureLoss(nn.Module):
    """
    Texture Loss basata su Gram Matrix (stile Neural Style Transfer).
    Utile per preservare texture astronomiche complesse.
    """
    def __init__(self, layer_weights=None):
        super().__init__()
        self.vgg = VGGLoss(feature_layer=35) 
        self.layer_weights = layer_weights or [1.0, 1.0, 1.0, 1.0]
    
    def _gram_matrix(self, features):
        """Calcola Gram Matrix per catturare correlazioni tra feature."""
        b, c, h, w = features.size()
        features = features.view(b, c, h * w)
        gram = torch.bmm(features, features.transpose(1, 2))
        return gram / (c * h * w)
    
    def forward(self, pred, target):
        return self.vgg(pred, target)


class HighFrequencyLoss(nn.Module):
    """
    Loss che penalizza la differenza nelle componenti ad alta frequenza.
    Critico per texture astronomiche (stelle, dettagli fini).
    """
    def __init__(self):
        super().__init__()
        laplacian_kernel = torch.tensor([
            [0, -1, 0],
            [-1, 4, -1],
            [0, -1, 0]
        ], dtype=torch.float32).view(1, 1, 3, 3)
        
        self.register_buffer('laplacian', laplacian_kernel)
    
    def forward(self, pred, target):
        pred_edges = F.conv2d(pred, self.laplacian, padding=1)
        target_edges = F.conv2d(target, self.laplacian, padding=1)
        
        return F.l1_loss(pred_edges, target_edges)


class CombinedGANLoss(nn.Module):
    """
    Loss combinata per il training GAN del Generatore.
    Include: Adversarial + Perceptual + Pixel + Texture + High-Frequency
    """
    def __init__(self, 
                 gan_type='ragan',
                 pixel_weight=1.0,
                 perceptual_weight=0.1,
                 adversarial_weight=0.005,
                 texture_weight=0.05,
                 hf_weight=0.1):
        super().__init__()
        
        if gan_type == 'ragan':
            self.gan_loss = RelativeGANLoss()
        else:
            self.gan_loss = GANLoss(gan_type=gan_type)
            
        self.pixel_loss = nn.L1Loss()
        self.perceptual_loss = VGGLoss(feature_layer=35)
        self.texture_loss = TextureLoss()
        self.hf_loss = HighFrequencyLoss()
        
        self.pixel_weight = pixel_weight
        self.perceptual_weight = perceptual_weight
        self.adversarial_weight = adversarial_weight
        self.texture_weight = texture_weight
        self.hf_weight = hf_weight
        
        self.gan_type = gan_type
    
    def forward(self, pred, target, real_pred=None, fake_pred=None):
        """
        Calcola loss totale per il generatore.
        
        Args:
            pred: Immagine generata [B, 1, H, W]
            target: Immagine ground truth [B, 1, H, W]
            real_pred: Discriminatore su immagini reali (per RaGAN)
            fake_pred: Discriminatore su immagini generate (per RaGAN)
        """
        losses = {}
        
        losses['pixel'] = self.pixel_loss(pred, target) * self.pixel_weight
        
        losses['perceptual'] = self.perceptual_loss(pred, target) * self.perceptual_weight
        
        losses['texture'] = self.texture_loss(pred, target) * self.texture_weight
        
        losses['hf'] = self.hf_loss(pred, target) * self.hf_weight
        
        if fake_pred is not None:
            if self.gan_type == 'ragan' and real_pred is not None:
                losses['adversarial'] = self.gan_loss(real_pred, fake_pred, for_discriminator=False) * self.adversarial_weight
            else:
                losses['adversarial'] = self.gan_loss(fake_pred, target_is_real=True) * self.adversarial_weight
        else:
            losses['adversarial'] = torch.tensor(0.0, device=pred.device)
        
        losses['total'] = sum(losses.values())
        
        return losses['total'], losses


class DiscriminatorLoss(nn.Module):
    """
    Loss per il training del Discriminatore.
    """
    def __init__(self, gan_type='ragan', use_gradient_penalty=False, lambda_gp=10.0):
        super().__init__()
        
        if gan_type == 'ragan':
            self.gan_loss = RelativeGANLoss()
        else:
            self.gan_loss = GANLoss(gan_type=gan_type)
            
        self.gan_type = gan_type
        self.use_gp = use_gradient_penalty
        
        if use_gradient_penalty:
            self.gp = GradientPenalty(lambda_gp=lambda_gp)
    
    def forward(self, real_pred, fake_pred, discriminator=None, real_data=None, fake_data=None):
        """
        Calcola loss del discriminatore.
        
        Args:
            real_pred: Discriminatore su immagini reali
            fake_pred: Discriminatore su immagini generate
            discriminator: Modello discriminatore (per gradient penalty)
            real_data: Immagini reali (per gradient penalty)
            fake_data: Immagini generate (per gradient penalty)
        """
        losses = {}
        
        if self.gan_type == 'ragan':
            losses['adversarial'] = self.gan_loss(real_pred, fake_pred, for_discriminator=True)
        else:
            real_loss = self.gan_loss(real_pred, target_is_real=True)
            fake_loss = self.gan_loss(fake_pred, target_is_real=False)
            losses['adversarial'] = (real_loss + fake_loss) / 2
        
        if self.use_gp and discriminator is not None and real_data is not None and fake_data is not None:
            losses['gp'] = self.gp(discriminator, real_data, fake_data)
        else:
            losses['gp'] = torch.tensor(0.0, device=real_pred.device)
        
        losses['total'] = losses['adversarial'] + losses['gp']
        
        return losses['total'], losses


if __name__ == "__main__":
    print("Testing GAN Losses...")
    
    pred = torch.randn(2, 1, 512, 512)
    target = torch.randn(2, 1, 512, 512)
    real_disc = torch.randn(2, 1, 30, 30)
    fake_disc = torch.randn(2, 1, 30, 30)
    
    combined_loss = CombinedGANLoss(gan_type='ragan')
    total, losses = combined_loss(pred, target, real_disc, fake_disc)
    print(f"CombinedGANLoss: Total={total.item():.4f}")
    for k, v in losses.items():
        print(f"{k}: {v.item():.4f}")
    
    disc_loss_fn = DiscriminatorLoss(gan_type='ragan')
    d_total, d_losses = disc_loss_fn(real_disc, fake_disc)
    print(f"DiscriminatorLoss: Total={d_total.item():.4f}")
    for k, v in d_losses.items():
        print(f"{k}: {v.item():.4f}")
