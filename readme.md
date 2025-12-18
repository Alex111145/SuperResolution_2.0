# Guida alla Configurazione del Progetto SuperResolution
apt update
apt install unzip
pip install tensorboard
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

```bash
pip install -r requirements.txt
o
pip install --no-cache-dir -r requirements.txt
```
tensorboard --logdir=outputs --port=6007 --bind_all


1. Sistema Git LFS (Consigliato per scaricare tutto subito)
Questo comando installerà lo strumento necessario e scaricherà automaticamente le versioni reali di HAT.zip, Real-ESRGAN.zip, BasicSR.zip e della cartella data.

Bash

# Entra nella cartella del progetto
cd ~/SuperResolution

# Installa Git LFS
apt update && apt install git-lfs -y

# Inizializza e scarica i file reali
git lfs install
git lfs pull
2. Alternativa: Download manuale (Se vuoi solo i file specifici)
Se preferisci scaricarli uno ad uno usando wget (saltando Git LFS), usa questi comandi:

Bash

# Scarica HAT
wget -O ~/SuperResolution/models/HAT.zip https://github.com/Alex111145/SuperResolution/raw/main/models/HAT.zip

# Scarica Real-ESRGAN
wget -O ~/SuperResolution/models/Real-ESRGAN.zip https://github.com/Alex111145/SuperResolution/raw/main/models/Real-ESRGAN.zip

# Scarica BasicSR
wget -O ~/SuperResolution/models/BasicSR.zip https://github.com/Alex111145/SuperResolution/raw/main/models/BasicSR.zip

# Scarica data.zip (nella cartella principale)
wget -O ~/SuperResolution/data.zip https://github.com/Alex111145/SuperResolution/raw/main/data.zip
3. Come unzippare tutto in un colpo solo
Una volta scaricati (verifica con ls -lh che pesino megabyte e non byte!), puoi scompattarli tutti così:

Bash

# Unzip dei modelli
cd ~/SuperResolution/models
unzip HAT.zip
unzip Real-ESRGAN.zip
unzip BasicSR.zip

# Unzip dei dati
cd ~/SuperResolution
unzip data.zip
