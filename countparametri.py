import os
import sys
import torch
import locale
from pathlib import Path

# --- FIX IMPORT (Lo stesso di train_hat.py per caricare BasicSR) ---
CURRENT_DIR = Path(__file__).resolve().parent
BASICSR_PATH = CURRENT_DIR / "models" / "BasicSR"
if BASICSR_PATH.exists():
    sys.path.insert(0, str(BASICSR_PATH))
else:
    print(f" [WARN] Cartella BasicSR non trovata in: {BASICSR_PATH}")

# Import dei Modelli
try:
    from models.hat_arch import HAT
    from models.discriminator import UNetDiscriminatorSN
except ImportError as e:
    print(f"Errore Import Modelli: {e}")
    sys.exit(1)

def count_parameters(model):
    """Conta i parametri allenabili di un modello PyTorch."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def format_count(n):
    """Formatta il numero in Milioni (M) o Migliaia (K)."""
    if n > 1e6:
        return f"{n / 1e6:.2f} M"
    elif n > 1e3:
        return f"{n / 1e3:.2f} K"
    else:
        return str(n)

def main():
    print("==========================================")
    print("      CONTEGGIO PARAMETRI MODELLI         ")
    print("==========================================\n")

    # 1. Configurazione GENERATORE (HAT)
    # Questi valori devono corrispondere a quelli in train_hat.py
    print("Configurazione HAT in uso:")
    config_hat = {
        'img_size': 128,
        'in_chans': 1,
        'embed_dim': 180,
        'depths': (6, 6, 6, 6, 6, 6),
        'num_heads': (6, 6, 6, 6, 6, 6),
        'window_size': 7,
        'upscale': 4,
        'upsampler': 'pixelshuffle'
    }
    for k, v in config_hat.items():
        print(f"  - {k}: {v}")

    try:
        net_g = HAT(**config_hat)
        params_g = count_parameters(net_g)
        print(f"\n[GENERATORE] Parametri HAT: {format_count(params_g)} ({params_g:,})")
    except Exception as e:
        print(f"\n[ERRORE] Impossibile istanziare HAT: {e}")

    print("-" * 40)

    # 2. Configurazione DISCRIMINATORE
    print("Configurazione Discriminatore in uso:")
    config_d = {
        'num_in_ch': 1,
        'num_feat': 64
    }
    for k, v in config_d.items():
        print(f"  - {k}: {v}")

    try:
        net_d = UNetDiscriminatorSN(**config_d)
        params_d = count_parameters(net_d)
        print(f"\n[DISCRIMINATORE] Parametri UNet: {format_count(params_d)} ({params_d:,})")
    except Exception as e:
        print(f"\n[ERRORE] Impossibile istanziare Discriminatore: {e}")

    print("\n==========================================")

if __name__ == "__main__":
    main()
  
