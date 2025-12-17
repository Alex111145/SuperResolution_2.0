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

```bash
pip install -r requirements.txt
o
pip install --no-cache-dir -r requirements.txt
```
