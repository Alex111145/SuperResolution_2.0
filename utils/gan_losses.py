import torch
import torch.nn as nn
import torch.nn.functional as F
from .losses_train import VGGLoss, CharbonnierLoss

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
    """
    Texture/Style Loss corretta basata su Gram Matrix.
    """
    def __init__(self):
        super().__init__()
        # VGG per estrarre feature (layer intermedio per texture)
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
        # Gestione Grayscale -> RGB
        if pred.shape[1] == 1: pred = pred.repeat(1, 3, 1, 1)
        if target.shape[1] == 1: target = target.repeat(1, 3, 1, 1)

        # Norm
        pred = (pred - self.mean) / self.std
        target = (target - self.mean) / self.std

        f_pred = self.vgg(pred)
        f_target = self.vgg(target)
        
        return F.mse_loss(self._gram_matrix(f_pred), self._gram_matrix(f_target.detach()))

class CombinedGANLoss(nn.Module):
    """
    Loss SwinIR-GAN ottimizzata.
    """
    def __init__(self, gan_type='ragan', pixel_weight=1.0, perceptual_weight=1.0, 
                 adversarial_weight=0.005, texture_weight=0, hf_weight=0):
        super().__init__()
        
        self.gan_type = gan_type
        if gan_type == 'ragan':
            self.gan_loss = RelativeGANLoss()
        else:
            self.gan_loss = GANLoss(gan_type=gan_type)
            
        # Per SwinIR si preferisce L1 o Charbonnier rispetto a MSE
        self.pixel_loss = nn.L1Loss() 
        self.perceptual_loss = VGGLoss()
        self.texture_loss = TextureLoss() if texture_weight > 0 else None
        
        self.pixel_weight = pixel_weight
        self.perceptual_weight = perceptual_weight
        self.adversarial_weight = adversarial_weight
        self.texture_weight = texture_weight
    
    def forward(self, pred, target, real_pred=None, fake_pred=None):
        losses = {}
        
        losses['pixel'] = self.pixel_loss(pred, target) * self.pixel_weight
        losses['perceptual'] = self.perceptual_loss(pred, target) * self.perceptual_weight
        
        if self.texture_loss:
            losses['texture'] = self.texture_loss(pred, target) * self.texture_weight
        
        if fake_pred is not None:
            if self.gan_type == 'ragan' and real_pred is not None:
                losses['adversarial'] = self.gan_loss(real_pred, fake_pred, for_discriminator=False) * self.adversarial_weight
            else:
                losses['adversarial'] = self.gan_loss(fake_pred, target_is_real=True) * self.adversarial_weight
        
        losses['total'] = sum(losses.values())
        return losses['total'], losses

# Import necessario per TextureLoss nel file
import torchvision.models as models

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