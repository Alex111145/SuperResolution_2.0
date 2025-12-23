import torch
import torch.nn as nn
from .losses_train import VGGLoss

class RelativeGANLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.BCEWithLogitsLoss()
    
    def forward(self, real_pred, fake_pred, for_discriminator=True):
        if for_discriminator:
            return (self.loss(real_pred - torch.mean(fake_pred), torch.ones_like(real_pred)) +
                    self.loss(fake_pred - torch.mean(real_pred), torch.zeros_like(fake_pred))) / 2
        else:
            return (self.loss(fake_pred - torch.mean(real_pred), torch.ones_like(fake_pred)) +
                    self.loss(real_pred - torch.mean(fake_pred), torch.zeros_like(real_pred))) / 2

class CombinedGANLoss(nn.Module):
    def __init__(self, gan_type='ragan', pixel_weight=1.0, perceptual_weight=1.0, adversarial_weight=0.005):
        super().__init__()
        self.gan_loss = RelativeGANLoss()
        self.pixel_loss = nn.L1Loss() 
        self.perceptual_loss = VGGLoss()
        self.pixel_weight = pixel_weight
        self.perceptual_weight = perceptual_weight
        self.adversarial_weight = adversarial_weight
    
    def forward(self, pred, target, real_pred=None, fake_pred=None):
        losses = {}
        losses['pixel'] = self.pixel_loss(pred, target) * self.pixel_weight
        losses['perceptual'] = self.perceptual_loss(pred, target) * self.perceptual_weight
        if fake_pred is not None and real_pred is not None:
            losses['adversarial'] = self.gan_loss(real_pred, fake_pred, for_discriminator=False) * self.adversarial_weight
        losses['total'] = sum(losses.values())
        return losses['total'], losses

class DiscriminatorLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.gan_loss = RelativeGANLoss()
    def forward(self, real_pred, fake_pred):
        loss = self.gan_loss(real_pred, fake_pred, for_discriminator=True)
        return loss, {'adversarial': loss}
