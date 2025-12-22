import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class VGGLoss(nn.Module):
    def __init__(self, feature_layer=35):
        super(VGGLoss, self).__init__()
        vgg19 = models.vgg19(pretrained=True)
        self.features = nn.Sequential(*list(vgg19.features.children())[:feature_layer+1])
        for p in self.features.parameters(): p.requires_grad = False
        
        self.register_buffer('mean', torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x, y):
        if x.shape[1] == 1: x = x.repeat(1, 3, 1, 1) # Da Mono a RGB per VGG
        if y.shape[1] == 1: y = y.repeat(1, 3, 1, 1)
        x = (x - self.mean) / self.std
        y = (y - self.mean) / self.std
        return F.l1_loss(self.features(x), self.features(y).detach())

class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps
    def forward(self, x, y):
        return torch.sum(torch.sqrt((x - y)**2 + self.eps))