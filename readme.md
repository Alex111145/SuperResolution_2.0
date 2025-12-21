# Aggiunto hf_weight=1.0 (per i dettagli) e log_weight=0.5 (per amplificare la luce debole)
criterion_g = CombinedGANLoss(
    gan_type='ragan', 
    pixel_weight=1.0, 
    perceptual_weight=0.5, 
    adversarial_weight=0.005, 
    hf_weight=1.0,    # Forza il recupero delle stelle puntiformi
    log_weight=0.5    # (Opzionale) Aumenta l'importanza delle zone scure/deboli
).to(device)


python -m venv venv

source venv/bin/activate

pip install -r requirements.txt



python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt


pkill -9 python
