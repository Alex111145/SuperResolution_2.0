import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import time

# Importa i tuoi modelli (adatta al tuo codice)
# from models.swin_model import SwinIR
# from models.hat_model import HAT
# from models.hybrid_model import HybridModel

class ModelEvaluator:
    def __init__(self, model, model_name, weights_path, device='cuda'):
        self.model = model.to(device)
        self.model_name = model_name
        self.device = device
        
        # Carica i pesi
        checkpoint = torch.load(weights_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model.eval()
        self.results = {
            'psnr': [],
            'ssim': [],
            'inference_time': []
        }
    
    @torch.no_grad()
    def evaluate_image(self, lr_image, hr_image):
        """Valuta una singola immagine"""
        # Preprocessing
        transform = transforms.ToTensor()
        lr_tensor = transform(lr_image).unsqueeze(0).to(self.device)
        
        # Inferenza con timing
        start_time = time.time()
        sr_tensor = self.model(lr_tensor)
        inference_time = time.time() - start_time
        
        # Converti a numpy per calcolo metriche
        sr_image = sr_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)
        hr_image_np = np.array(hr_image) / 255.0
        
        # Calcola metriche
        psnr_val = psnr(hr_image_np, sr_image, data_range=1.0)
        ssim_val = ssim(hr_image_np, sr_image, channel_axis=2, data_range=1.0)
        
        self.results['psnr'].append(psnr_val)
        self.results['ssim'].append(ssim_val)
        self.results['inference_time'].append(inference_time)
        
        return sr_image, psnr_val, ssim_val
    
    def get_average_metrics(self):
        """Restituisce medie delle metriche"""
        return {
            'model': self.model_name,
            'avg_psnr': np.mean(self.results['psnr']),
            'avg_ssim': np.mean(self.results['ssim']),
            'avg_time': np.mean(self.results['inference_time']),
            'std_psnr': np.std(self.results['psnr']),
            'std_ssim': np.std(self.results['ssim'])
        }

# ==================== CONFIGURAZIONE ====================

# Percorsi ai pesi dei modelli
MODEL_CONFIGS = {
    'Swin5.0': {
        'weights': 'path/to/swin5.0/best_model.pth',
        'model_class': None  # Inserisci la classe del modello
    },
    'HYBRID_9': {
        'weights': 'path/to/HYBRID_9/best_model.pth',
        'model_class': None
    },
    'PURE_HAT_8_1LIGHT': {
        'weights': 'path/to/PURE_HAT_8_1LIGHT/best_model.pth',
        'model_class': None
    }
}

# Percorso al dataset di test
TEST_DATA_PATH = 'path/to/test/dataset'
LR_FOLDER = f'{TEST_DATA_PATH}/LR'  # Immagini low-resolution
HR_FOLDER = f'{TEST_DATA_PATH}/HR'  # Immagini high-resolution (ground truth)

# ==================== VALUTAZIONE ====================

def run_comparison():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    evaluators = {}
    
    # Inizializza gli evaluator per ogni modello
    for name, config in MODEL_CONFIGS.items():
        model = config['model_class']()  # Inizializza il modello
        evaluators[name] = ModelEvaluator(model, name, config['weights'], device)
    
    # Carica le immagini di test
    lr_images = sorted(Path(LR_FOLDER).glob('*.png'))
    hr_images = sorted(Path(HR_FOLDER).glob('*.png'))
    
    print(f"Trovate {len(lr_images)} immagini di test")
    
    # Valuta ogni immagine con tutti i modelli
    for lr_path, hr_path in zip(lr_images, hr_images):
        lr_img = Image.open(lr_path).convert('RGB')
        hr_img = Image.open(hr_path).convert('RGB')
        
        print(f"\nProcessing: {lr_path.name}")
        
        for name, evaluator in evaluators.items():
            sr_img, psnr_val, ssim_val = evaluator.evaluate_image(lr_img, hr_img)
            print(f"  {name}: PSNR={psnr_val:.2f}dB, SSIM={ssim_val:.4f}")
    
    # Raccogli risultati
    results = [eval.get_average_metrics() for eval in evaluators.values()]
    df_results = pd.DataFrame(results)
    
    print("\n" + "="*60)
    print("RISULTATI FINALI:")
    print("="*60)
    print(df_results.to_string(index=False))
    
    # Salva risultati
    df_results.to_csv('model_comparison_results.csv', index=False)
    
    return df_results, evaluators

# ==================== VISUALIZZAZIONE ====================

def plot_comparison(df_results):
    """Crea grafici comparativi"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    models = df_results['model'].values
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    
    # PSNR Comparison
    axes[0, 0].bar(models, df_results['avg_psnr'], color=colors, alpha=0.7)
    axes[0, 0].errorbar(models, df_results['avg_psnr'], 
                        yerr=df_results['std_psnr'], fmt='none', 
                        ecolor='black', capsize=5)
    axes[0, 0].set_ylabel('PSNR (dB)', fontsize=11)
    axes[0, 0].set_title('Peak Signal-to-Noise Ratio', fontweight='bold')
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    # SSIM Comparison
    axes[0, 1].bar(models, df_results['avg_ssim'], color=colors, alpha=0.7)
    axes[0, 1].errorbar(models, df_results['avg_ssim'], 
                        yerr=df_results['std_ssim'], fmt='none', 
                        ecolor='black', capsize=5)
    axes[0, 1].set_ylabel('SSIM', fontsize=11)
    axes[0, 1].set_title('Structural Similarity Index', fontweight='bold')
    axes[0, 1].set_ylim([0, 1])
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    # Inference Time
    axes[1, 0].bar(models, df_results['avg_time']*1000, color=colors, alpha=0.7)
    axes[1, 0].set_ylabel('Tempo (ms)', fontsize=11)
    axes[1, 0].set_title('Tempo di Inferenza', fontweight='bold')
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    # Scatter PSNR vs Time
    axes[1, 1].scatter(df_results['avg_time']*1000, df_results['avg_psnr'], 
                      s=300, c=colors, alpha=0.6, edgecolors='black', linewidth=2)
    for i, model in enumerate(models):
        axes[1, 1].annotate(model, 
                           (df_results['avg_time'].iloc[i]*1000, 
                            df_results['avg_psnr'].iloc[i]),
                           xytext=(10, 10), textcoords='offset points',
                           fontsize=9, fontweight='bold')
    axes[1, 1].set_xlabel('Tempo Inferenza (ms)', fontsize=11)
    axes[1, 1].set_ylabel('PSNR (dB)', fontsize=11)
    axes[1, 1].set_title('Trade-off Qualità/Velocità', fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('complete_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_visual_comparison(evaluators, test_image_path):
    """Confronto visivo su un'immagine specifica"""
    
    lr_img = Image.open(test_image_path).convert('RGB')
    
    fig, axes = plt.subplots(1, len(evaluators)+1, figsize=(20, 5))
    
    # Immagine LR originale
    axes[0].imshow(lr_img)
    axes[0].set_title('Low Resolution\n(Input)', fontweight='bold')
    axes[0].axis('off')
    
    # Output di ogni modello
    for idx, (name, evaluator) in enumerate(evaluators.items(), 1):
        with torch.no_grad():
            transform = transforms.ToTensor()
            lr_tensor = transform(lr_img).unsqueeze(0).to(evaluator.device)
            sr_tensor = evaluator.model(lr_tensor)
            sr_img = sr_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)
        
        axes[idx].imshow(np.clip(sr_img, 0, 1))
        axes[idx].set_title(f'{name}\nSuper-Resolution', fontweight='bold')
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig('visual_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

# ==================== ESECUZIONE ====================

if __name__ == '__main__':
    # Esegui valutazione completa
    df_results, evaluators = run_comparison()
    
    # Crea grafici
    plot_comparison(df_results)
    
    # Confronto visivo (opzionale - su un'immagine specifica)
    # plot_visual_comparison(evaluators, 'path/to/test_image.png')
