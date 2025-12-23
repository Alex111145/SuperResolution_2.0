import torch
import numpy as np
from pathlib import Path
from PIL import Image
import argparse
from tqdm import tqdm

from models.hybrid_model import HybridHATRealESRGAN

def load_model(checkpoint_path, device):
    """Carica il modello ibrido da checkpoint"""
    model = HybridHATRealESRGAN(
        img_size=128,
        in_chans=1,
        embed_dim=180,
        depths=(6, 6, 6, 6, 6, 6),
        num_heads=(6, 6, 6, 6, 6, 6),
        window_size=7,
        upscale=4,
        num_rrdb=23
    ).to(device)
    
    state_dict = torch.load(checkpoint_path, map_location=device)
    if 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']
    
    model.load_state_dict(state_dict)
    model.eval()
    return model

def process_image(img_path, model, device):
    """Processa singola immagine"""
    img = Image.open(img_path).convert('L')
    img_np = np.array(img).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0).to(device)
    
    with torch.no_grad():
        sr = model(img_tensor)
    
    sr_np = sr.squeeze().cpu().numpy()
    sr_np = np.clip(sr_np * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(sr_np)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='Input image or folder')
    parser.add_argument('--output', type=str, required=True, help='Output folder')
    parser.add_argument('--checkpoint', type=str, required=True, help='Model checkpoint')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Caricamento modello ibrido HAT+Real-ESRGAN...")
    model = load_model(args.checkpoint, device)
    print(f"✅ Modello caricato su {device}")
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if input_path.is_file():
        images = [input_path]
    else:
        images = list(input_path.glob('*.png')) + list(input_path.glob('*.jpg'))
    
    print(f"📁 Trovate {len(images)} immagini")
    
    for img_path in tqdm(images, desc="Upscaling"):
        sr_img = process_image(img_path, model, device)
        out_name = f"{img_path.stem}_SR_hybrid.png"
        sr_img.save(output_path / out_name)
    
    print(f"✅ Completato! Immagini salvate in {output_path}")

if __name__ == "__main__":
    main()
