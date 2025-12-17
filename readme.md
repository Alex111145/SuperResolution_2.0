# Guida alla Configurazione del Progetto SuperResolution

## 1. Configurazione dell'Ambiente Virtuale (Venv)

È fondamentale utilizzare un ambiente virtuale per isolare le dipendenze del progetto.

### Creazione del Venv

Esegui questo comando nella directory principale del progetto (`SuperResolution-Main/`):

```bash
python3 -m venv venv
```

Questo creerà una sottocartella chiamata `venv` che contiene l'ambiente isolato.

### Attivazione del Venv

**Su Linux/macOS:**
```bash
source venv/bin/activate
```

**Su Windows (Command Prompt/PowerShell):**
```bash
.\venv\Scripts\activate
```

Una volta attivato, il prompt del terminale mostrerà `(venv)` all'inizio.

### Installazione delle Dipendenze

Dopo aver attivato l'ambiente, installa i pacchetti necessari utilizzando il file `requirements.txt` aggiornato.

> **Nota:** L'installazione di PyTorch con supporto CUDA richiede il flag `--extra-index-url` (già incluso nel file) per scaricare le wheel pre-compilate.

```bash
pip install -r requirements.txt
o
pip install --no-cache-dir -r requirements.txt
```

## 2. Struttura della Directory

La struttura del progetto richiede la presenza di cartelle specifiche per i modelli esterni e per i pesi pre-allenati.

### 2.1 Architetture Esterne (models/)

È necessario creare la cartella `models/` nella directory principale del progetto e inserirvi i repository dei modelli di base (RRDBNet, HAT, ecc.) come segue:

```
SuperResolution-Main/
├── scripts/
├── src/
├── models/
│   ├── BasicSR/  <-- Contiene l'implementazione di RRDBNet, ecc.
│   └── HAT/      <-- Contiene l'implementazione di HAT_Arch.
└── ...
```

### 2.2 Pesi e Checkpoint Pre-allenati (outputs/ e Weights)

Se desideri riprendere il training (resume) o eseguire l'inferenza con un modello pre-allenato, devi inserire i file `.pth` nella struttura creata dal codice di training.

- **Directory Base:** La cartella `outputs/` viene creata automaticamente durante il training e contiene i risultati.
- **Struttura Checkpoint:** I pesi devono essere collocati all'interno di una sottocartella che rispecchi il nome del target di training (es. M42).

#### Esempio di struttura attesa per l'inferenza o il resume:

Se il tuo modello è stato allenato sul target **M42** e il file di pesi è `best_train_model.pth`, la struttura corretta è:

```
SuperResolution-Main/
└── outputs/
    └── M42_DDP/              <-- Nome del target di training (es. M42) con suffisso _DDP
        └── checkpoints/
            └── best_train_model.pth  <-- Il tuo file di pesi
```

#### Nomi dei file importanti:

- **`latest_checkpoint.pth`**: Usato dal training script (`train_core.py`) per riprendere l'addestramento da un'epoca specifica.
- **`best_train_model.pth`**: Usato come fallback per un warm-start (carica solo i pesi migliori) e per l'inferenza.
