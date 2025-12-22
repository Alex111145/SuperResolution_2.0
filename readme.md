# Aggiunto hf_weight=1.0 (per i dettagli) e log_weight=0.5 (per amplificare la luce debole)
criterion_g = CombinedGANLoss(
    gan_type='ragan', 
    pixel_weight=1.0, 
    perceptual_weight=0.5, 
    adversarial_weight=0.005, 
    hf_weight=1.0,    # Forza il recupero delle stelle puntiformi
    log_weight=0.5    # (Opzionale) Aumenta l'importanza delle zone scure/deboli
).to(device)

Nuova funzione inference_tta: Questa funzione prende l'immagine in input, crea 8 copie (4 rotazioni x 2 flip), le passa al modello, riporta le uscite alla rotazione originale e ne fa la media.

Riduzione Granulosità: Facendo la media di 8 predizioni, gli artefatti "granulosi" (che cambiano leggermente a seconda di come è orientata l'immagine) si annullano a vicenda, lasciando l'immagine più pulita ("smooth").

Output: I file verranno salvati in una nuova cartella test_results_smooth per non sovrascrivere quelli vecchi.





python -m venv venv

source venv/bin/activate

pip install -r requirements.txt



python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt


pkill -9 python
