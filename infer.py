import torch
import numpy as np
from pathlib import Path
from PIL import Image
import argparse
from tqdm import tqdm

from models.hybrid_model import HybridHATRealESRGAN

def load_model(checkpoint_path, device):
    """Carica modello ibrido da checkpoint"""
    print(f"📦 Caricamento modello da {checkpoint_path}...")
    
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
    
    # Carica state dict
    state_dict = torch.load(checkpoint_path, map_location=device)
    
    # Gestisci formati checkpoint diversi
    if 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']
    
    # Rimuovi prefisso "module." se presente (da DDP)
    state_dict_cleaned = {}
    for k, v in state_dict.items():
        k_cleaned = k.replace('module.', '')
        state_dict_cleaned[k_cleaned] = v
    
    model.load_state_dict(state_dict_cleaned)
    model.eval()
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Modello caricato ({total_params:,} parametri)")
    
    return model

def process_image(img_path, model, device, tile_size=None, tile_overlap=32):
    """
    Processa singola immagine con tiling opzionale per immagini grandi
    
    Args:
        img_path: Path immagine input
        model: Modello PyTorch
        device: Device (cuda/cpu)
        tile_size: Se None, processa intera immagine. Altrimenti usa tiling.
        tile_overlap: Overlap tra tiles (default 32px)
    """
    img = Image.open(img_path).convert('L')
    img_np = np.array(img).astype(np.float32) / 255.0
    
    if tile_size is None:
        # Processamento immagine intera
        img_tensor = torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0).to(device)
        
        with torch.no_grad():
            sr = model(img_tensor)
        
        sr_np = sr.squeeze().cpu().numpy()
        sr_np = np.clip(sr_np * 255, 0, 255).astype(np.uint8)
        return Image.fromarray(sr_np)
    
    else:
        # Tiling per immagini grandi (TODO: implementare se necessario)
        raise NotImplementedError("Tiling non ancora implementato")

def main():
    parser = argparse.ArgumentParser(
        description='Inferenza con Modello Ibrido HAT + Real-ESRGAN'
    )
    parser.add_argument('--input', type=str, required=True,
                       help='Path immagine o cartella input')
    parser.add_argument('--output', type=str, required=True,
                       help='Path cartella output')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path checkpoint modello')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device per inferenza')
    parser.add_argument('--suffix', type=str, default='_SR_hybrid',
                       help='Suffisso file output')
    args = parser.parse_args()
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("⚠️  CUDA non disponibile, uso CPU")
    
    print("=" * 70)
    print("🔮 INFERENZA MODELLO IBRIDO HAT + REAL-ESRGAN")
    print("=" * 70)
    
    # Carica modello
    model = load_model(args.checkpoint, device)
    
    # Setup paths
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Trova immagini
    if input_path.is_file():
        images = [input_path]
    else:
        images = list(input_path.glob('*.png')) + \
                 list(input_path.glob('*.jpg')) + \
                 list(input_path.glob('*.tif')) + \
                 list(input_path.glob('*.tiff'))
    
    if not images:
        print(f"❌ Nessuna immagine trovata in {input_path}")
        return
    
    print(f"📁 Trovate {len(images)} immagini")
    print(f"💾 Output: {output_path}")
    print("=" * 70)
    
    # Processa immagini
    for img_path in tqdm(images, desc="Upscaling", unit="img"):
        try:
            sr_img = process_image(img_path, model, device)
            out_name = f"{img_path.stem}{args.suffix}.png"
            sr_img.save(output_path / out_name)
        except Exception as e:
            tqdm.write(f"⚠️  Errore su {img_path.name}: {e}")
            continue
    
    print("=" * 70)
    print(f"✅ Completato! {len(images)} immagini processate")
    print("=" * 70)

if __name__ == "__main__":
    main()
