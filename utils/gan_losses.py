import torch
import torch.nn as nn
import torch.nn.functional as F
from .losses_train import VGGLoss, CharbonnierLoss
import torchvision.models as models

class GANLoss(nn.Module):
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
            raise ValueError(f"GAN type {gan_type} non supportato.")
    
    def _get_target_tensor(self, prediction, target_is_real):
        if target_is_real:
            return torch.full_like(prediction, self.real_label)
        return torch.full_like(prediction, self.fake_label)
    
    def forward(self, prediction, target_is_real):
        target = self._get_target_tensor(prediction, target_is_real)
        return self.loss(prediction, target)

class RelativeGANLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.BCEWithLogitsLoss()
    
    def forward(self, real_pred, fake_pred, for_discriminator=True):
        if for_discriminator:
            # D Loss
            return (self.loss(real_pred - torch.mean(fake_pred), torch.ones_like(real_pred)) +
                    self.loss(fake_pred - torch.mean(real_pred), torch.zeros_like(fake_pred))) / 2
        else:
            # G Loss
            return (self.loss(fake_pred - torch.mean(real_pred), torch.ones_like(fake_pred)) +
                    self.loss(real_pred - torch.mean(fake_pred), torch.zeros_like(real_pred))) / 2

class TextureLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.vgg = models.vgg19(pretrained=True).features[:35]
        for p in self.vgg.parameters(): p.requires_grad = False
        self.register_buffer('mean', torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def _gram_matrix(self, x):
        b, c, h, w = x.size()
        features = x.view(b, c, h * w)
        gram = torch.bmm(features, features.transpose(1, 2))
        return gram / (c * h * w)
    
    def forward(self, pred, target):
        if pred.shape[1] == 1: pred = pred.repeat(1, 3, 1, 1)
        if target.shape[1] == 1: target = target.repeat(1, 3, 1, 1)
        pred = (pred - self.mean) / self.std
        target = (target - self.mean) / self.std
        f_pred = self.vgg(pred)
        f_target = self.vgg(target)
        return F.mse_loss(self._gram_matrix(f_pred), self._gram_matrix(f_target.detach()))

class HighFrequencyLoss(nn.Module):
    """Calcola la loss sui dettagli ad alta frequenza (bordi/stelle) usando un filtro Laplaciano."""
    def __init__(self):
        super().__init__()
        # Kernel Laplaciano per estrarre i bordi
        kernel = torch.tensor([[-1, -1, -1],
                               [-1,  8, -1],
                               [-1, -1, -1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        self.register_buffer('kernel', kernel)

    def forward(self, pred, target):
        b, c, h, w = pred.shape
        # Replica il kernel per ogni canale (funziona sia per grayscale che RGB)
        kernel = self.kernel.repeat(c, 1, 1, 1)
        
        # padding=1 mantiene le dimensioni
        p_hf = F.conv2d(pred, kernel, padding=1, groups=c)
        t_hf = F.conv2d(target, kernel, padding=1, groups=c)
        
        return F.l1_loss(p_hf, t_hf)

class LogL1Loss(nn.Module):
    """L1 Loss in spazio logaritmico per dare peso ai pixel scuri (stelle deboli)."""
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        return F.l1_loss(torch.log(pred.clamp(min=0) + self.eps), 
                         torch.log(target.clamp(min=0) + self.eps))

class CombinedGANLoss(nn.Module):
    def __init__(self, gan_type='ragan', pixel_weight=1.0, perceptual_weight=1.0, 
                 adversarial_weight=0.005, texture_weight=0, hf_weight=0, log_weight=0):
        super().__init__()
        
        self.gan_type = gan_type
        if gan_type == 'ragan':
            self.gan_loss = RelativeGANLoss()
        else:
            self.gan_loss = GANLoss(gan_type=gan_type)
            
        self.pixel_loss = nn.L1Loss() 
        self.perceptual_loss = VGGLoss()
        self.texture_loss = TextureLoss() if texture_weight > 0 else None
        self.hf_loss = HighFrequencyLoss() if hf_weight > 0 else None
        self.log_loss = LogL1Loss() if log_weight > 0 else None
        
        self.pixel_weight = pixel_weight
        self.perceptual_weight = perceptual_weight
        self.adversarial_weight = adversarial_weight
        self.texture_weight = texture_weight
        self.hf_weight = hf_weight
        self.log_weight = log_weight
    
    def forward(self, pred, target, real_pred=None, fake_pred=None):
        losses = {}
        
        losses['pixel'] = self.pixel_loss(pred, target) * self.pixel_weight
        losses['perceptual'] = self.perceptual_loss(pred, target) * self.perceptual_weight
        
        if self.texture_loss:
            losses['texture'] = self.texture_loss(pred, target) * self.texture_weight
        
        if self.hf_loss:
            losses['hf'] = self.hf_loss(pred, target) * self.hf_weight
            
        if self.log_loss:
            losses['log'] = self.log_loss(pred, target) * self.log_weight
        
        if fake_pred is not None:
            if self.gan_type == 'ragan' and real_pred is not None:
                losses['adversarial'] = self.gan_loss(real_pred, fake_pred, for_discriminator=False) * self.adversarial_weight
            else:
                losses['adversarial'] = self.gan_loss(fake_pred, target_is_real=True) * self.adversarial_weight
        
        losses['total'] = sum(losses.values())
        return losses['total'], losses

class DiscriminatorLoss(nn.Module):
    def __init__(self, gan_type='ragan'):
        super().__init__()
        if gan_type == 'ragan':
            self.gan_loss = RelativeGANLoss()
        else:
            self.gan_loss = GANLoss(gan_type=gan_type)
        self.gan_type = gan_type
    
    def forward(self, real_pred, fake_pred):
        losses = {}
        if self.gan_type == 'ragan':
            losses['adversarial'] = self.gan_loss(real_pred, fake_pred, for_discriminator=True)
        else:
            losses['adversarial'] = (self.gan_loss(real_pred, True) + self.gan_loss(fake_pred, False)) / 2
        
        losses['total'] = losses['adversarial']
        return losses['total'], losses
