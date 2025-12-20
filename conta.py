import torch
import os
import sys
import re
import ast

# Aggiunge la directory corrente al path per trovare i moduli
sys.path.append(os.getcwd())

try:
    from models.architecture import SwinIR
except ImportError:
    print("❌ ERRORE: Non trovo 'models/architecture.py'.")
    sys.exit(1)

def extract_value_from_file(content, key, default_value):
    """
    Cerca nel testo di train.py un pattern tipo 'key = valore' 
    specifico per la chiamata SwinIR.
    """
    # Pattern regex: cerca "key=valore" o "key = valore"
    # Gestisce numeri interi (es. 256) e liste (es. [6, 6, 6])
    pattern = rf"{key}\s*=\s*(\[[0-9, ]+\]|[0-9]+)"
    
    match = re.search(pattern, content)
    if match:
        val_str = match.group(1)
        try:
            # Converte la stringa (es. "[6,6,6]") in oggetto python reale
            return ast.literal_eval(val_str)
        except:
            return default_value
    return default_value

def get_config_from_train():
    file_path = 'train.py'
    if not os.path.exists(file_path):
        print(f"❌ Errore: '{file_path}' non trovato.")
        sys.exit(1)

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    print(f"📖 Leggo configurazione da '{file_path}'...")

    # Cerchiamo la sezione dove viene definito SwinIR
    # Limitiamo la ricerca intorno alla chiamata SwinIR per evitare falsi positivi
    swin_match = re.search(r'SwinIR\s*\(.*?\)', content, re.DOTALL)
    
    search_content = content
    if swin_match:
        # Se troviamo il blocco SwinIR, cerchiamo i parametri solo lì dentro
        search_content = swin_match.group(0)
    else:
        print("⚠️ Attenzione: Non ho trovato una chiamata esplicita a 'SwinIR(...)' nel file.")
        print("   Cercherò i parametri in tutto il file, ma potrebbero essere imprecisi.")

    # --- ESTRAZIONE PARAMETRI ---
    # Default = valori base di SwinIR se non trovati nel file
    config = {
        'img_size': extract_value_from_file(search_content, 'img_size', 64),
        'patch_size': extract_value_from_file(search_content, 'patch_size', 1), # Di solito è 1 in training SR
        'in_chans': extract_value_from_file(search_content, 'in_chans', 3),     # 3 RGB, 1 Gray
        'embed_dim': extract_value_from_file(search_content, 'embed_dim', 96),
        'depths': extract_value_from_file(search_content, 'depths', [6, 6, 6, 6]),
        'num_heads': extract_value_from_file(search_content, 'num_heads', [6, 6, 6, 6]),
        'window_size': extract_value_from_file(search_content, 'window_size', 7),
        'mlp_ratio': extract_value_from_file(search_content, 'mlp_ratio', 4),
        'upscale': extract_value_from_file(search_content, 'upscale', 4),
        'upsampler': 'pixelshuffle', # Difficile da leggere via regex se stringa, usiamo default comune
        'resi_connection': '1conv'
    }
    
    # Check speciale per in_chans (se l'utente usa 1 per B/N)
    # A volte è hardcodato o passato diversamente, qui assumiamo che se non trovato è 3, 
    # ma se il training è astronomico B/N potrebbe essere 1.
    # Controlliamo se nel file c'è 'in_chans=1'
    if 'in_chans=1' in search_content.replace(" ", ""):
        config['in_chans'] = 1

    return config

def main():
    print("\n--- CONTATORE PARAMETRI DINAMICO ---")

    # 1. Ottieni Configurazione
    conf = get_config_from_train()

    print("\n⚙️  Configurazione Rilevata:")
    print(f"   Input Patch: {conf['img_size']} px")
    print(f"   Canali Input: {conf['in_chans']} (1=Grigio, 3=RGB)")
    print(f"   Embed Dim:   {conf['embed_dim']} (Larghezza)")
    print(f"   Profondità:  {conf['depths']} (Blocchi)")
    print(f"   Upscale:     x{conf['upscale']}")

    # 2. Creazione Modello
    try:
        model = SwinIR(**conf)
    except Exception as e:
        print(f"\n❌ Errore nella creazione del modello con questi parametri.")
        print(f"   Dettaglio: {e}")
        print("   Suggerimento: Controlla che in train.py i parametri siano numeri o liste esplicite.")
        return

    # 3. Conteggio
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_m = total_params / 1_000_000
    
    print(f"\n📊 RISULTATI:")
    print(f"   Parametri Totali:  {total_params:,}")
    print(f"   In Milioni (M):    {total_m:.2f} M")

    # 4. Stima VRAM (Approssimativa per Training)
    # Una stima grezza: (Model Weights + Gradients + Optimizer States + Activations)
    # Con SwinIR le attivazioni occupano molto a causa dell'Attention.
    
    # Peso modello (FP32)
    model_mem_mb = (total_params * 4) / (1024**2)
    
    # Stima empirica per training (Batch size 1, mixed precision o fp32)
    # SwinIR "Heavy" (256px, 240dim) prende circa 7-10GB.
    # SwinIR "Light" (64px, 96dim) ne prende <2GB.
    
    print(f"\n💾 STIMA MEMORIA (Solo Pesi): {model_mem_mb:.2f} MB")
    
    # Calcolo euristico basato su esperienza con SwinIR
    # Fattore di moltiplicazione per l'overhead di training (Attention map enormi)
    img_pixels = conf['img_size'] ** 2
    dim_factor = conf['embed_dim'] / 96.0
    depth_factor = sum(conf['depths']) / 24.0
    
    estimated_training_vram = (img_pixels / (64*64)) * dim_factor * depth_factor * 1.5
    
    print(f"⚠️  STIMA VRAM Training (Batch=1): ~{estimated_training_vram:.1f} GB")
    print("   (Questa è una stima molto approssimativa. Affidati a nvidia-smi)")

    # 5. Verifica File Esistente
    possible_paths = ["outputs/best_model.pth", "experiments/train_SwinIR_SRx4_scratch/models/net_g_best.pth"]
    for path in possible_paths:
        if os.path.exists(path):
            print(f"\n🔎 Trovato checkpoint: {path}")
            try:
                ckpt = torch.load(path, map_location='cpu')
                param_keys = ckpt['params'].keys() if 'params' in ckpt else ckpt.keys()
                # Check veloce sul primo tensore
                first_key = list(param_keys)[0]
                # Non facciamo load_state_dict completo per velocità, ma se il file c'è, è buon segno.
                print("✅ Il file sembra valido (formato PyTorch rilevato).")
            except:
                print("❌ Il file esiste ma sembra corrotto o illeggibile.")

if __name__ == "__main__":
    main()