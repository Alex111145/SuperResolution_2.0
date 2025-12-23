# models/hat_arch.py - VERSIONE CORRETTA
from basicsr.archs.hat_arch import HAT as HAT_Base

class HAT(HAT_Base):
    """
    Wrapper HAT per immagini astronomiche grayscale.
    Eredita da BasicSR e adatta la configurazione.
    """
    def __init__(self, img_size=128, in_chans=1, embed_dim=180, 
                 depths=(6,6,6,6,6,6), num_heads=(6,6,6,6,6,6),
                 window_size=7, upscale=4, upsampler='pixelshuffle', **kwargs):
        super().__init__(
            img_size=img_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            upscale=upscale,
            upsampler=upsampler,
            **kwargs
        )
