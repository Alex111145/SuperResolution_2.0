PULIRE CODICEEEEEEEEEEEE


HAT E basic 
cd ~/SuperResolution/models

# Clona il repository HAT ufficiale
git clone https://github.com/XPixelGroup/HAT.git

# Verifica la struttura
ls -la HAT/
Installazione dei Requisiti HAT
bash
cd ~/SuperResolution

# Attiva il virtual environment se non già attivo
source venv/bin/activate

# Installa le dipendenze HAT
pip install timm einops


df -h

Il comando df -h mostra una situazione particolare: la tua partizione principale (/) ha solo 16GB totali, di cui 14GB disponibili. Anche se sembrano sufficienti, l'installazione di PyTorch e delle dipendenze NVIDIA può fallire perché pip estrae i pacchetti in una directory temporanea (solitamente /tmp o all'interno del venv) che potrebbe saturare rapidamente quei 14GB durante il processo di compilazione/estrazione.

Tuttavia, vedo che hai un disco enorme montato su percorsi insoliti (probabilmente un'istanza cloud con storage di rete o NVMe):

/dev/nvme0n1 con 1.7TB liberi.

/dev/nvme1n1p2 con 836GB liberi.

Per risolvere l'errore ed evitare che il disco si riempia di nuovo, segui questi passaggi:

1. Forza PIP a usare il disco grande per i file temporanei
Invece di usare la partizione da 16GB, punta a una cartella nel disco da 1.7TB.

Bash

# Crea una cartella temporanea nel disco capiente
mkdir -p /etc/hosts_tmp  # Uso un percorso basato su dove è montato il disco grande

# Imposta le variabili d'ambiente per questa sessione
export TMPDIR=/etc/hosts_tmp
export PIP_CACHE_DIR=/etc/hosts_tmp/pip_cache

# Riprova l'installazione
pip install -r requirements.txt
