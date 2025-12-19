import torch
from torch.utils.data import Dataset
from pathlib import Path
import json
import numpy as np
import random
from PIL import Image

class AstronomicalDataset(Dataset):
    def __init__(self, split_file, base_path, augment=True):
        self.base_path = Path(base_path)
        self.augment = augment
        
        with open(split_file, 'r') as f:
            self.pairs = json.load(f)
            
        print(f"Dataset caricato: {len(self.pairs)} coppie da {Path(split_file).name}")

    def _fix_path(self, path_str):
        if '/data/' in path_str:
            relative_part = path_str.split('/data/', 1)[1]
            return self.base_path / "data" / relative_part
        return self.base_path / path_str

    def _load_tiff_as_tensor(self, path):
        try:
            if not path.exists():
                return None
            
            # Apertura e caricamento forzato per verificare l'integrità del file
            img = Image.open(path)
            img.load() 
            
            arr = np.array(img, dtype=np.float32)
            arr = arr / 65535.0  # Normalizzazione 16-bit
            tensor = torch.from_numpy(arr)
            
            if tensor.ndim == 2:
                tensor = tensor.unsqueeze(0)
            return tensor
        except Exception:
            return None

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        path_lr = self._fix_path(str(pair['ground_path']))
        path_hr = self._fix_path(str(pair['hubble_path']))

        lr_tensor = self._load_tiff_as_tensor(path_lr)
        hr_tensor = self._load_tiff_as_tensor(path_hr)

        # Se il file è corrotto (errore PIL) o mancante, pesca un nuovo indice
        if lr_tensor is None or hr_tensor is None:
            new_idx = random.randint(0, len(self.pairs) - 1)
            return self.__getitem__(new_idx)

        if self.augment:
            if random.random() > 0.5:
                lr_tensor = torch.flip(lr_tensor, [-1])
                hr_tensor = torch.flip(hr_tensor, [-1])
            if random.random() > 0.5:
                lr_tensor = torch.flip(lr_tensor, [-2])
                hr_tensor = torch.flip(hr_tensor, [-2])
            k = random.randint(0, 3)
            if k > 0:
                lr_tensor = torch.rot90(lr_tensor, k, [-2, -1])
                hr_tensor = torch.rot90(hr_tensor, k, [-2, -1])
        
        return {'lr': lr_tensor.contiguous(), 'hr': hr_tensor.contiguous()}

    def __len__(self):
        return len(self.pairs)