import torch
import torch.nn.functional as F
from math import exp

def ssim_torch(img1, img2, window_size=11):
    c = img1.size(1)
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*1.5**2)) for x in range(window_size)])
    win = (gauss/gauss.sum()).unsqueeze(1).mm((gauss/gauss.sum()).unsqueeze(0)).unsqueeze(0).unsqueeze(0).expand(c,1,window_size,window_size).type_as(img1)
    mu1, mu2 = F.conv2d(img1, win, groups=c), F.conv2d(img2, win, groups=c)
    sigma1_sq = F.conv2d(img1*img1, win, groups=c) - mu1.pow(2)
    sigma2_sq = F.conv2d(img2*img2, win, groups=c) - mu2.pow(2)
    sigma12 = F.conv2d(img1*img2, win, groups=c) - mu1*mu2
    return (((2*mu1*mu2 + 0.01**2)*(2*sigma12 + 0.03**2))/((mu1.pow(2) + mu2.pow(2) + 0.01**2)*(sigma1_sq + sigma2_sq + 0.03**2))).mean()

class TrainMetrics:
    def __init__(self): self.reset()
    def reset(self): self.psnr = 0.0; self.ssim = 0.0; self.count = 0
    def update(self, p, t):
        p, t = p.clamp(0, 1), t.clamp(0, 1)
        mse = F.mse_loss(p, t, reduction='none').mean(dim=[1,2,3])
        self.psnr += (10 * torch.log10(1.0 / (mse + 1e-8))).sum().item()
        self.ssim += ssim_torch(p, t).item() * p.size(0)
        self.count += p.size(0)
    def compute(self): 
        return {'psnr': self.psnr/self.count, 'ssim': self.ssim/self.count} if self.count else {'psnr':0, 'ssim':0}