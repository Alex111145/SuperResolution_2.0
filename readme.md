
    hf_weight=15,    # Forza il recupero delle stelle puntiformi
    log_weight=4    # (Opzionale) Aumenta l'importanza delle zone scure/deboli


Per risolvere questo problema durante l'inferenza senza dover riaddestrare, la tecnica migliore è la TTA (Test-Time Augmentation). Questa tecnica esegue l'inferenza 8 volte per ogni immagine (ruotandola e specchiandola) e ne fa la media. Questo processo "pulisce" il rumore casuale del GAN e rende l'immagine molto più morbida ("smoothing") pur mantenendo i dettagli reali delle stelle.
