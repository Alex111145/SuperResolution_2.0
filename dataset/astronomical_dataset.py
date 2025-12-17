import torch
from torch.utils.data import Dataset
from pathlib import Path
import json
import numpy as np
import random
from PIL import Image

class AstronomicalDataset(Dataset):
    """
    Dataset per caricare coppie LR-HR da file TIFF 16-bit.
    Include logica di 'Path Repair' per gestire lo spostamento del progetto tra cartelle diverse.
    """
    def __init__(self, split_file, base_path, augment=True):
        self.base_path = Path(base_path)
        self.augment = augment
        
        with open(split_file, 'r') as f:
            self.pairs = json.load(f)
            
        print(f"Dataset caricato: {len(self.pairs)} coppie da {Path(split_file).name}")

    def _fix_path(self, path_str):
        """
        Corregge i percorsi assoluti vecchi (es. /home/gfrattini/...) 
        adattandoli alla posizione attuale del progetto.
        """

        if '/data/' in path_str:
            relative_part = path_str.split('/data/', 1)[1]
            new_path = self.base_path / "data" / relative_part
            return new_path
        
        return self.base_path / path_str

    def _load_tiff_as_tensor(self, path, expected_size=128):
        """
        Carica un TIFF 16-bit e lo converte in Tensore Float [0-1].
        In caso di errore, ritorna un tensore nero della dimensione ATTESA.
        """
        try:
            if not path.exists():
                raise FileNotFoundError(f"File non trovato: {path}")

            img = Image.open(path)
            arr = np.array(img, dtype=np.float32)
            arr = arr / 65535.0 
            tensor = torch.from_numpy(arr)
            
            if tensor.ndim == 2:
                tensor = tensor.unsqueeze(0)
            
            return tensor
            
        except Exception as e:
            return torch.zeros(1, expected_size, expected_size)

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        
        raw_lr = str(pair['ground_path'])
        raw_hr = str(pair['hubble_path'])
        
        path_lr = self._fix_path(raw_lr)
        path_hr = self._fix_path(raw_hr)

        lr_tensor = self._load_tiff_as_tensor(path_lr, expected_size=128)
        hr_tensor = self._load_tiff_as_tensor(path_hr, expected_size=512)

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
        
        if lr_tensor.stride()[0] < 0: lr_tensor = lr_tensor.contiguous()
        if hr_tensor.stride()[0] < 0: hr_tensor = hr_tensor.contiguous()

        if torch.isnan(lr_tensor).any(): lr_tensor = torch.nan_to_num(lr_tensor)
        if torch.isnan(hr_tensor).any(): hr_tensor = torch.nan_to_num(hr_tensor)

        return {'lr': lr_tensor, 'hr': hr_tensor}

    def __len__(self):
        return len(self.pairs)