Riassunto delle modifiche da applicare
Ecco il blocco completo da sostituire/aggiornare in train.py per applicare tutto in un colpo solo:

Linea ~38 (Configurazione):

Python

BATCH_SIZE = 2          # Usa la VRAM extra per il batch size
ACCUM_STEPS = 2         # Bilancia l'aumento del batch
Linea ~120 (Istanza SwinIR):

Python

# img_size aumentato a 192 per sfruttare i 7GB liberi
net_g = SwinIR(upscale=4, in_chans=1, img_size=192, window_size=8,
               img_range=1.0, depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
               mlp_ratio=2, upsampler='pixelshuffle', resi_connection='1conv')
Linea ~140 (Losses):

Python

# Pesi aggressivi per i dettagli
criterion_g = CombinedGANLoss(gan_type='ragan', 
                              pixel_weight=0.05,      # Molto basso
                              perceptual_weight=1.0,  # Alto
                              adversarial_weight=0.05 # Medio-Alto
                              ).to(device)
