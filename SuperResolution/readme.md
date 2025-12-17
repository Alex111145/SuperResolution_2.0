SuperResolution-Nuova_Struttura_Main/   <-- ROOT DEL PROGETTO
│
├── .venv/                              <-- [NUOVO] Cartella dell'ambiente virtuale
│   ├── Lib/
│   ├── Scripts/
│   └── ... (file di sistema generati automaticamente)
│
├── dataset/                            <-- [NUOVO] Gestione Dati
│   ├── __init__.py                     (Opzionale)
│   └── astronomical_dataset.py         (ex src/dataset.py)
│
├── models/                             <-- [NUOVO] Reti Neurali e Architetture
│   ├── BasicSR/                        (Cartella libreria esterna spostata qui)
│   ├── HAT/                            (Cartella libreria esterna spostata qui)
│   ├── architecture.py                 (ex src/architecture_train.py)
│   ├── discriminator.py                (ex src/discriminator.py)
│   └── __init__.py                     (Opzionale)
│
├── utils/                              <-- [NUOVO] Utility, Metriche, Loss
│   ├── env_setup.py                    (ex src/env_setup.py - MODIFICATO)
│   ├── gan_losses.py                   (ex src/gan_losses.py)
│   ├── losses_train.py                 (ex src/losses_train.py)
│   ├── metrics.py                      (ex src/metrics_train.py)
│   └── __init__.py                     (Opzionale)
│
├── misc/                               <-- [NUOVO] Script secondari e preprocessing
│   ├── download_data.py                (ex scripts/scaricaautomatica.py)
│   ├── prepare_data.py                 (ex scripts/Train_Prepare_0.py)
│   ├── count_params.py                 (ex scripts/contaparametri.py)
│   ├── Dataset_1.py
│   └── ... (altri script di appoggio)
│
├── data/                               <-- Cartella dei dati (NON toccare la struttura interna)
│   ├── M1/
│   ├── M33/
│   └── ...
│
├── outputs/                            <-- Cartella dove il training salva pesi e immagini
│
├── train.py                            (ex scripts/train_core.py - MODIFICATO)
├── start.py                            (ex scripts/start.py - MODIFICATO)
├── infer.py                            (ex scripts/Train_s_Inference_4.py)
├── requirements.txt                    (Invariato)
├── .gitignore                          (Da aggiornare per escludere .venv)
└── README.md