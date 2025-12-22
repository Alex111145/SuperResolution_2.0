import subprocess
import time
import os
import sys
from datetime import datetime

# --- CONFIGURAZIONE AGGIORNATA ---
SERVER_IP = "217.171.200.22"  # Nuovo IP Pubblico
SSH_PORT = 34640              # Porta mappata per SSH (22 -> 34640)
SSH_USER = "root"             # Utente standard per queste istanze
REMOTE_PATH = "/root/SuperResolution/outputs" # Percorso remoto (Verifica se è corretto per la nuova istanza)
LOCAL_PATH = r"F:\outputs"
INTERVAL = 15 * 60            # 15 minuti

def download_job():
    now = datetime.now().strftime("%H:%M:%S")
    print(f"[{now}] Controllo nuovi file dal server...")
    
    # Costruisce la stringa sorgente per SCP (es. root@IP:/path/*)
    source = f"{SSH_USER}@{SERVER_IP}:{REMOTE_PATH}/*"
    
    # Crea la cartella locale se non esiste
    if not os.path.exists(LOCAL_PATH):
        try:
            os.makedirs(LOCAL_PATH)
        except OSError as e:
            print(f"Errore creazione cartella {LOCAL_PATH}: {e}")
            return

    # Comando SCP. Nota: su Windows potrebbe servire il path completo di scp.exe se non è nel PATH,
    # ma solitamente funziona così.
    # L'argomento -o StrictHostKeyChecking=no evita il prompt yes/no alla prima connessione
    cmd = ["scp", "-P", str(SSH_PORT), "-o", "StrictHostKeyChecking=no", "-r", source, LOCAL_PATH]

    try:
        print(f"Tentativo di connessione a {SERVER_IP}:{SSH_PORT}...")
        
        # Esegue il comando. shell=True serve su Windows per alcuni comandi di sistema.
        subprocess.run(cmd, check=True, shell=True)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ Download completato su {LOCAL_PATH}")
        
    except subprocess.CalledProcessError as e:
        print(f"⚠️ Errore SCP (Codice {e.returncode}).") 
        print("Possibili cause: Password errata, percorso remoto inesistente o chiave SSH mancante.")
    except Exception as e:
        print(f"❌ Errore Generico: {e}")

def main():
    os.system('cls' if os.name == 'nt' else 'clear')
    print("--- AUTO-DOWNLOADER (Linux -> Windows) ---")
    print(f"Server: {SSH_USER}@{SERVER_IP}:{SSH_PORT}")
    print(f"Remoto: {REMOTE_PATH}")
    print(f"Locale: {LOCAL_PATH}")
    print("------------------------------------------")
    
    # Primo download immediato
    download_job()

    while True:
        try:
            print(f"\nAttesa {INTERVAL/60:.0f} minuti per il prossimo controllo...")
            time.sleep(INTERVAL)
            download_job()
        except KeyboardInterrupt:
            print("\nInterrotto dall'utente.")
            sys.exit(0)

if __name__ == "__main__":
    main()