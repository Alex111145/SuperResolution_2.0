import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import os
from pathlib import Path
from PIL import Image
from tqdm import tqdm

# --- CONFIGURAZIONE PERCORSI ---
# Adatta questi percorsi se necessario
CURRENT_SCRIPT = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_SCRIPT.parent.parent  # Assumendo che infer.py sia in una sottocartella, altrimenti usa .parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Percorsi Input/Output e Modello
INPUT_FOLDER = PROJECT_ROOT / "inputs"       # Metti qui le immagini da elaborare
OUTPUT_FOLDER = PROJECT_ROOT / "outputs_inf" # Qui verranno salvati i risultati
MODEL_PATH = PROJECT_ROOT / "outputs" / "checkpoints" / "best_model.pth" # Percorso del tuo .pth

# Assicurati che le cartelle esistano
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
if not INPUT_FOLDER.exists():
    INPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    print(f"ATTENZIONE: Creata cartella {INPUT_FOLDER}. Inserisci qui le immagini e riavvia.")
    sys.exit(0)

# --- IMPORTAZIONI MODULI CUSTOM ---
try:
    from src.architecture_train import RRDBNet, HAT_Arch
    print("Moduli architettura importati.")
except ImportError as e:
    print(f"Errore importazione src: {e}")
    sys.exit(1)

# --- 1. SELEZIONE DISPOSITIVO ---
def select_device():
    print("\n" + "="*30)
    print("   CONFIGURAZIONE DISPOSITIVO")
    print("="*30)
    if not torch.cuda.is_available():
        print(" [!] Nessuna GPU rilevata. Uso CPU.")
        return torch.device('cpu')

    print(f" GPU Rilevata: {torch.cuda.get_device_name(0)}")
    print(" 1. Usa GPU (Veloce)")
    print(" 2. Usa CPU (Lento ma stabile)")
    
    while True:
        choice = input(" > Scelta (1/2): ").strip()
        if choice == '1': return torch.device('cuda')
        elif choice == '2': return torch.device('cpu')

# --- 2. CLASSE MODELLO (WRAPPER) ---
class DynamicHybridModel(nn.Module):
    def __init__(self, output_size=512, hat_embed_dim=480, hat_depths=[8]*8): 
        super().__init__()
        self.output_size = output_size
        
        # Stage 1: RRDBNet
        self.stage1 = RRDBNet(num_in_ch=1, num_out_ch=1, num_feat=128, num_block=27, scale=2)
        
        # Stage 2: HAT (Se disponibile)
        self.has_hat = False
        if HAT_Arch:
            self.stage2 = HAT_Arch(
                img_size=128, in_chans=1, embed_dim=hat_embed_dim, depths=hat_depths,
                num_heads=[8]*len(hat_depths), upscale=2
            )
            self.has_hat = True

        # Refinement Layers
        self.refine = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2), 
            nn.Conv2d(64, 1, kernel_size=3, padding=1)
        )
        # Upscaler finale se necessario
        self.smart_upscale = nn.Sequential(
            nn.Conv2d(1, 4, 3, 1, 1), nn.PixelShuffle(2), nn.PReLU()
        )

    def forward(self, x):
        feat_1 = self.stage1(x)
        out = feat_1
        
        if self.has_hat:
            hat_out = self.stage2(feat_1)
            if hat_out.shape[-1] != feat_1.shape[-1]:
                feat_1_up = F.interpolate(feat_1, size=hat_out.shape[-2:], mode='nearest')
                out = hat_out + feat_1_up
            else:
                out = hat_out + feat_1

        # Gestione dimensioni output (Resize/Smart Upscale)
        # Nota: In inferenza "tiled", questo potrebbe essere bypassato se gestiamo le tile manualmente,
        # ma lo lasciamo per coerenza con l'architettura.
        if out.shape[-1] != self.output_size:
             # Semplice logica adattiva
             if out.shape[-1] < self.output_size:
                 out = F.interpolate(out, size=(self.output_size, self.output_size), mode='bicubic')
        
        return self.refine(out) + out

# --- 3. RILEVAMENTO PARAMETRI ---
def detect_model_params(state_dict):
    params = {'hat_embed_dim': 480, 'hat_depths': [8, 8, 8, 8, 8, 8, 8, 8]}
    
    if 'stage2.conv_first.weight' in state_dict:
        params['hat_embed_dim'] = state_dict['stage2.conv_first.weight'].shape[0]

    max_idx = -1
    for k in state_dict.keys():
        if "stage2.layers." in k:
            parts = k.split(".")
            if len(parts) > 2 and parts[2].isdigit():
                max_idx = max(max_idx, int(parts[2]))
    
    if max_idx != -1:
        detected_len = max_idx + 1
        if detected_len != len(params['hat_depths']):
            params['hat_depths'] = [6] * detected_len # Default fallback
            
    return params

# --- 4. FUNZIONE CORE: INFERENZA A FINESTRE (TILED) ---
def predict_tiled(model, img_tensor, tile_size=512, overlap=32, device='cuda'):
    """
    Esegue l'inferenza dividendo l'immagine in tile per evitare OOM su GPU.
    img_tensor: [1, 1, H, W] range 0-1
    """
    b, c, h, w = img_tensor.shape
    scale = 2 # Fattore di upscale del modello
    
    # Calcolo dimensioni output attese
    out_h, out_w = h * scale, w * scale
    
    # Inizializza canvas output e maschera pesi
    output = torch.zeros((1, 1, out_h, out_w), device=device)
    output_mask = torch.zeros((1, 1, out_h, out_w), device=device)

    # Loop sulle tile
    for y in range(0, h, tile_size - overlap):
        for x in range(0, w, tile_size - overlap):
            # Coordinate input
            y_end = min(y + tile_size, h)
            x_end = min(x + tile_size, w)
            
            # Estrai patch input
            patch = img_tensor[:, :, y:y_end, x:x_end].to(device)
            
            # Inferenza sulla patch
            with torch.no_grad():
                # Nota: Il modello si aspetta output_size fisso nel __init__, 
                # ma qui lavoriamo su patch variabili. 
                # Bypassiamo il resize interno forzando l'architettura a processare la patch pura.
                # Richiede che il forward del modello sia robusto.
                
                # Hack temporaneo: Se il modello ha output fisso hardcoded, 
                # potrebbe fallire qui se non gestiamo il resize nel forward.
                # Assumiamo che stage1 e stage2 gestiscano dimensioni arbitrarie (sono CNN).
                pred_patch = model.stage1(patch)
                if model.has_hat:
                    hat_out = model.stage2(pred_patch)
                    if hat_out.shape != pred_patch.shape:
                        pred_patch = F.interpolate(pred_patch, size=hat_out.shape[-2:])
                    pred_patch = hat_out + pred_patch
                pred_patch = model.refine(pred_patch) + pred_patch
            
            # Calcolo coordinate output
            y_out, x_out = y * scale, x * scale
            h_patch_out, w_patch_out = pred_patch.shape[2], pred_patch.shape[3]
            
            # Inserimento nel canvas (Gestione semplice somma per ora, ideale sarebbe blending)
            output[:, :, y_out:y_out+h_patch_out, x_out:x_out+w_patch_out] += pred_patch
            output_mask[:, :, y_out:y_out+h_patch_out, x_out:x_out+w_patch_out] += 1.0

    # Normalizza nelle zone di sovrapposizione
    output = output / (output_mask + 1e-8)
    return output

# --- 5. MAIN LOOP ---
def main():
    device = select_device()
    
    # 1. Caricamento Pesi
    if not MODEL_PATH.exists():
        sys.exit(f"ERRORE: Modello non trovato in {MODEL_PATH}")
    
    print(f"Caricamento checkpoint: {MODEL_PATH.name}...")
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    
    # 2. Rilevamento Parametri e Build Modello
    model_params = detect_model_params(state_dict)
    print(f"Parametri rilevati: {model_params}")
    
    # Importante: output_size qui è fittizio, useremo tiling
    model = DynamicHybridModel(output_size=512, **model_params) 
    
    # Carica pesi (strict=False per evitare errori su chiavi accessorie)
    try:
        model.load_state_dict(state_dict, strict=False)
    except Exception as e:
        print(f"Warning nel caricamento pesi: {e}")
        
    model.to(device)
    model.eval()
    
    # 3. Processamento Immagini
    img_files = list(INPUT_FOLDER.glob("*.tif")) + list(INPUT_FOLDER.glob("*.png")) + list(INPUT_FOLDER.glob("*.jpg"))
    
    if not img_files:
        print(f"Nessuna immagine trovata in {INPUT_FOLDER}")
        return

    print(f"\nInizio inferenza su {len(img_files)} immagini...")
    
    for img_path in tqdm(img_files):
        try:
            # Caricamento Immagine (Monocromatica per astronomia 1ch)
            # Usa cv2 per supporto 16bit nativo
            img_np = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
            
            if img_np is None:
                print(f"Errore lettura {img_path.name}")
                continue
                
            # Gestione 16 bit vs 8 bit
            if img_np.dtype == np.uint16:
                img_norm = img_np.astype(np.float32) / 65535.0
            else:
                img_norm = img_np.astype(np.float32) / 255.0
            
            # Se l'immagine è 2D (H, W), aggiungi dimensioni
            if len(img_norm.shape) == 2:
                img_tensor = torch.from_numpy(img_norm).unsqueeze(0).unsqueeze(0)
            elif len(img_norm.shape) == 3:
                # Converti HWC a CHW e prendi solo 1 canale se RGB
                img_tensor = torch.from_numpy(img_norm).permute(2, 0, 1).unsqueeze(0)
                img_tensor = img_tensor[:, 0:1, :, :] # Forza 1 canale
            
            # INFERENZA
            # Se l'immagine è piccola (<1024px) fai inferenza diretta, altrimenti Tiled
            h, w = img_tensor.shape[2], img_tensor.shape[3]
            if h < 1024 and w < 1024:
                img_tensor = img_tensor.to(device)
                with torch.no_grad():
                    output = model(img_tensor)
            else:
                # Usa la funzione Tiled per immagini grandi
                output = predict_tiled(model, img_tensor, tile_size=512, overlap=64, device=device)
            
            # SALVATAGGIO
            save_path = OUTPUT_FOLDER / f"{img_path.stem}_SR.tif"
            
            # Converti output in numpy uint16
            out_np = output.squeeze().cpu().clamp(0, 1).numpy()
            out_u16 = (out_np * 65535).astype(np.uint16)
            
            # Salva usando PIL per compatibilità TIFF
            Image.fromarray(out_u16, mode='I;16').save(save_path)
            
        except Exception as e:
            print(f"Errore su {img_path.name}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\nFinito! Risultati in: {OUTPUT_FOLDER}")

if __name__ == "__main__":
    main()
