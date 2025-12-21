Smart Upscale (PixelShuffle): Ho inserito la classe Upsample che utilizza nn.PixelShuffle. A differenza del ridimensionamento standard, questo modulo utilizza i canali per ricostruire spazialmente l'immagine, eliminando l'effetto sfocato dell'interpolazione.

Fix Crash Dimensionale: All'interno di SwinTransformerBlock, il sistema ora verifica se la risoluzione dell'input è inferiore alla window_size. In tal caso, riduce dinamicamente la finestra per evitare errori di calcolo.

Gestione Dinamica (F.pad): Nel metodo forward di SwinIR, l'immagine viene automaticamente "paddata" (completata) per essere divisibile per la dimensione della finestra, prevenendo i tipici RuntimeError durante il partizionamento.
