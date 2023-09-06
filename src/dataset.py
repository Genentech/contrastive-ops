import os
import warnings
from typing import Optional, List, Dict

import lmdb
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from src.constants import Column
from memory_profiler import profile
from copy import deepcopy
from itertools import cycle

class OPSdataset(Dataset):
    def __init__(self, 
                 metadata_df: pd.DataFrame,
                 dataset_path: str,
                 plate_list: list,
                 stat_path: Optional[str],
                 num_channels: Optional[int] = 4,
                 crop_size: Optional[int] = 100,
                 cell_cycle_stage = ['interphase', 'mitotic'],
                 label_maps: Dict[str, Dict] = [],
                 batch_correction: bool=False,
                 preprocess = None,
                    *args, **kwargs,
                 ):
        self.dataset_path = dataset_path
        self.plate_list = plate_list
        self.cell_cycle_stage = cell_cycle_stage
        self.preprocess = preprocess
        self.crop_size = crop_size
        self.num_channels = num_channels
        self.df = metadata_df
        self.label_maps = label_maps
        self.num_samples = len(self.df)
        if len(self.label_maps) > 0:
            self.label = list(self.label_maps.keys())

        # load statistics for per-well batch correction
        self.batch_correction = batch_correction
        if self.batch_correction:
            dfs = pd.read_pickle(stat_path)
            self.per_well_median = dfs['median']
            self.per_well_MAD = dfs['MAD']

        # create dictionary to store lmdb environment
        self._env_dict = {}

    def read_single_cell_image(self, df_index):
        row = self.df.iloc[df_index]
        index = row[Column.index.value]
        plate = row[Column.plate.value]
        well = row[Column.well.value]
        tile = str(row[Column.tile.value])
        gene = row[Column.gene.value]
        uid = row[Column.uid.value]
        # stage = row[Column.gene.cell_cycle_stage.value]

        key = f'{uid}_{plate}_{well}_{tile}_{gene}_{index}'
        _env_name = 'ntc' if gene == 'nontargeting' else 'perturbed'

        if _env_name in self._env_dict:
            env = self._env_dict[_env_name]
        else:
            path = self.dataset_path[_env_name]
            if os.path.exists(path):
                env = lmdb.Environment(path, readonly=True, readahead=False, lock=False)
                self._env_dict[_env_name] = env
            else:
                warnings.warn(f"LMDB dataset for {_env_name} doesn't exist")
                env = None

        with env.begin(write=False, buffers=True) as txn:
            buf = txn.get(key.encode())
            arr = np.frombuffer(buf, dtype='uint16')
        cell_image = arr.reshape((self.num_channels, self.crop_size, self.crop_size))

        if self.batch_correction:
            batch = plate+well
            cell_image = (cell_image - self.per_well_median[batch])/self.per_well_MAD[batch]
        return cell_image
    
    def __setstate__(self, state):
        self.__dict__ = state

    def __getstate__(self):
        return {
            "crop_size": self.crop_size,
            "num_channels": self.num_channels,
            "dataset_path": deepcopy(self.dataset_path),
            "df": deepcopy(self.df),
            "_env_dict": {},
        }

    def __del__(self):
        for _, env in self._env_dict.items():
            env.close()

    def __len__(self):
        'Denotes the total number of samples'
        return self.num_samples
    
    def __getitem__(self, idx):
        'Generates one sample of data'
        image = self.read_single_cell_image(idx)
        if self.preprocess:
            image = self.preprocess(image=image)
        return image
        
class OPSwithlabel(OPSdataset):
    def __init__(self, metadata_df, *args, **kwargs):
        super().__init__(metadata_df=metadata_df, *args, **kwargs)

    def read_single_cell_label(self, df_index, label_names):
        row = self.df.iloc[df_index]
        label = np.zeros(len(label_names))
        for i, cat in enumerate(label_names):
            label[i] = self.label_maps[cat][row[cat]]
        return label

    def __getitem__(self, idx):
        'Generates one sample of data'
        image = self.read_single_cell_image(idx)
        label = self.read_single_cell_label(idx, self.label)
        image, label = self.preprocess(image=image, label=label)
        return image, label

class PairedDataset(Dataset):
    def __init__(self, background_ds, target_ds):
        self.background_ds = background_ds
        self.target_ds = target_ds
        self.length = max(len(background_ds), len(target_ds))

    def __getitem__(self, index):
        return self.build_pair(self.background_ds[index % len(self.background_ds)], 
                               self.target_ds[index % len(self.target_ds)])
            
    def __len__(self):
        return self.length
    
    @staticmethod
    def build_pair(bg_samples, tg_samples):
        if len(bg_samples) == 2:
            bg_x, bg_y = bg_samples
            tg_x, tg_y = tg_samples
            if len(bg_y) == 1:
                return {'background': bg_x, 'target': tg_x, 'background_label': bg_y, 'target_label':tg_y}
            elif len(bg_y) == 2:
                return {'background': bg_x, 'target': tg_x, 'background_label': bg_y[0], 
                            'target_label': tg_y[0], 'background_batch': bg_y[1], 'target_batch': tg_y[1]}
        else:
            return {'background': bg_samples, 'target': tg_samples}