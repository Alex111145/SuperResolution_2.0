PULIRE CODICEEEEEEEEEEEE


HAT E basic 


# Attiva il venv se non è attivo
source venv/bin/activate

# Aggiorna pip
pip install --upgrade pip

# Installa BasicSR e dipendenze
pip install git+https://github.com/XPixelGroup/BasicSR.git
pip install einops timm addict

# Fix rapido per il problema torchvision (se persiste dopo l'installazione)
python fix_basicsr.py

