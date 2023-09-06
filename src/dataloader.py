import os
from itertools import cycle
from typing import Dict, Optional, List
import numpy as np

import pandas as pd
import lightning as L

import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from src.constants import Column, PH_DIMS
from src.dataset import OPSdataset, OPSwithlabel, PairedDataset
from src.transformation import Preprocess
    
class BaseDataModule(L.LightningDataModule):
    def __init__(self, 
                 dataset_path: str,
                 plate_list: List[str],
                 loader_param: Dict,
                 save_dir: str,
                 transform,
                 data_name: str='contrastive',
                 test_ratio: float=0.3,
                 label: List[str]=None,
                 stat_path: str=None,
                 batch_correction: bool=False,
                 crop_size: Optional[int] = 100,
                 *args, **kwargs):
        super().__init__()

        self.data_name = data_name
        self.test_ratio = test_ratio
        self.save_dir = save_dir
        self.crop_size = crop_size
        self.loader_param = loader_param
        self.label = label
        self.transform = transform  # per batch augmentation_kornia

        self.datamodule = {'label':OPSwithlabel, 'nolabel':OPSdataset}
        self.data_param = {'dataset_path':dataset_path, 'plate_list':plate_list, 'stat_path':stat_path, 
                      'batch_correction':batch_correction, 'preprocess':Preprocess()}
        
        # filter data and read in entire dataframe
        if not label:
            print('no label provided')
            self.modulename = 'nolabel'
        else:
            self.modulename = 'label'
            metadata = pd.concat([pd.read_pickle(os.path.join(save_dir, 'perturbed_filtered.pkl')),
                                pd.read_pickle(os.path.join(save_dir, 'ntc_filtered.pkl'))], axis=0)
            label_maps = dict()
            for cat in label:
                label_maps[cat] = {k: i for i, k in enumerate(metadata[cat].unique())}
            self.data_param['label_maps'] = label_maps
            del metadata
        self.save_hyperparameters()

    def prepare_data(self):
        save_dir = self.save_dir
        # train/val/test split
        if not os.path.exists(os.path.join(save_dir, 'perturbed_filtered.pkl')):
            metadata_df = self.get_filtered_df(self.data_param['dataset_path'], 
                                            self.data_param['plate_list'], 
                                            crop_size=self.crop_size,
                                            )
            for key in metadata_df:
                metadata_df[key].to_pickle(f'{save_dir}/{key}_filtered.pkl')
            print('saved new dataframes!')
        else:
            metadata_df = self.read_in_dict_of_df(save_dir)
        if not os.path.exists(os.path.join(save_dir, 'ntc_train.pkl')):
            metadata_df = self.read_in_dict_of_df(save_dir)
            splits = self.split_dataframe(metadata_df, self.test_ratio, shuffle=False)
            for key, val in splits.items():
                val.to_pickle(f'{save_dir}/{key}.pkl')
            print('saved new data splits')

    def setup(self, stage=None):        
        # Assign train datasets for use in dataloader
        if stage == 'fit':
            background = pd.read_pickle(f'{self.save_dir}/ntc_train.pkl')
            target = pd.read_pickle(f'{self.save_dir}/perturbed_train.pkl')
            # mixed = pd.concat([background, target], axis=0)
            mixed = self.mix_df(target, background)
            self.train = OPSdataset(mixed, **self.data_param)
            # validation set
            background = pd.read_pickle(f'{self.save_dir}/ntc_val.pkl')
            target = pd.read_pickle(f'{self.save_dir}/perturbed_val.pkl')
            # mixed = pd.concat([background, target], axis=0)
            mixed = self.mix_df(target, background)
            self.val = OPSdataset(mixed, **self.data_param)

        # Assign test dataset for use in dataloader
        if stage =='test':
            background = pd.read_pickle(f'{self.save_dir}/ntc_test.pkl')
            target= pd.read_pickle(f'{self.save_dir}/perturbed_test.pkl')
            # mixed = pd.concat([background, target], axis=0)
            mixed = self.mix_df(target, background)
            self.test = OPSdataset(mixed, **self.data_param)

        if stage == 'embed':
            df_all = pd.concat([pd.read_pickle(f'{self.save_dir}/perturbed_filtered.pkl'),
                                pd.read_pickle(f'{self.save_dir}/ntc_filtered.pkl')], axis=0)
            self.ds_all = OPSdataset(df_all, **self.data_param)

    def train_dataloader(self):
        return DataLoader(self.train,shuffle=False, drop_last=True, pin_memory=True, persistent_workers=True, **self.loader_param)
    
    def val_dataloader(self):
        return DataLoader(self.val, drop_last=True, **self.loader_param)
    
    def test_dataloader(self):
        return DataLoader(self.test, drop_last=True, **self.loader_param)
    
    def all_dataloader(self, loader_param=None):
        if loader_param is not None:
            for key, val in loader_param.items():
                self.loader_param[key] = val
        return DataLoader(self.ds_all, shuffle=False, drop_last=False, **self.loader_param)
        
    def on_after_batch_transfer(self, batch, dataloader_idx):
        # if self.trainer.training:
        return self.transform(batch)  # GPU/Batched data augmentation
    
    @staticmethod
    def mix_df(df1, df2):
        '''insert df2 into df1 at random positions'''
        # Generate random unique indices in df1 where you want to insert the rows from df2
        random_indices = np.random.choice(len(df1) + len(df2), size=len(df2), replace=False)

        # Create an index array for df1
        df1_index = np.delete(np.arange(len(df1) + len(df2)), random_indices)

        # Assign the indices to the DataFrames
        df1['position'] = df1_index
        df2['position'] = random_indices

        # Concatenate the two DataFrames
        combined_df = pd.concat([df1, df2])

        # Sort by the position index and then by the secondary key, and reset the DataFrame index
        combined_df = combined_df.sort_values(['position']).reset_index(drop=True)

        # Drop the position and df2_order columns if they are no longer needed
        combined_df.drop(columns=['position'], inplace=True)
        return combined_df

    @staticmethod
    def read_in_dict_of_df(file_path):
        new_dict = {}
        new_dict['ntc'] = pd.read_pickle(os.path.join(file_path, f'ntc_filtered.pkl'))
        new_dict['perturbed'] = pd.read_pickle(os.path.join(file_path, f'perturbed_filtered.pkl'))
        return new_dict

    @staticmethod
    def get_filtered_df(dataset_path: Dict[str, str], 
                        plate_list, 
                        crop_size: float):
        # read in metadata, used for sampling cells
        df_all = {}
        for key, val in dataset_path.items():
            df = pd.read_csv(f'{val}/key.csv', dtype={'UID': str})

            # only keep cells from the pre-defined plate list
            df = df[df[Column.plate.value].isin(plate_list)]

            # remove patches from edge due to smaller size
            radius = crop_size/2
            df = df[df[Column.cell_y.value].between(radius, PH_DIMS[0] - radius) &
                                df[Column.cell_x.value].between(radius, PH_DIMS[1] - radius)]
            
            # add batch column as concatenation of plate and well label
            df['batch'] = df['plate'] + df['well']
            df_all[key] = df

        # add cell cycle stage
        # interphase_metadata[Column.cell_cycle_stage.value] = 'interphase'
        # mitosis_metadata[Column.cell_cycle_stage.value] = 'mitotic'
        # df_all = pd.concat([interphase_metadata,mitosis_metadata])
        return df_all
    
    @staticmethod
    def split_dataframe(df: Dict[str, pd.DataFrame], 
                        test_ratio: List[float],
                        shuffle: bool=False) -> Dict[str, pd.DataFrame]:
        split = {}
        _, val, test = test_ratio
        for key, df_val in df.items():
            split[f'{key}_train'], remainder = train_test_split(df_val, test_size=val+test, shuffle=shuffle)
            split[f'{key}_val'], split[f'{key}_test'] = train_test_split(remainder, test_size=test/(val+test), shuffle=shuffle)
        return split


class ContrastiveDataModule(BaseDataModule):
    """
    Iterator for background and target dataloader pairs as found in the contrastive
    analysis setting.
    Each iteration of this iterator returns a dictionary with two elements:
    "background", containing one batch of data from the background dataloader, and
    "target", containing one batch of data from the target dataloader.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def setup(self, stage=None):
        # Assign train datasets for use in dataloader
        if stage == 'fit':
            background = pd.read_pickle(f'{self.save_dir}/ntc_train.pkl')
            target = pd.read_pickle(f'{self.save_dir}/perturbed_train.pkl')
            self.train_background = self.datamodule[self.modulename](background, **self.data_param)
            self.train_target = self.datamodule[self.modulename](target, **self.data_param)
            # validation set
            background = pd.read_pickle(f'{self.save_dir}/ntc_val.pkl')
            target = pd.read_pickle(f'{self.save_dir}/perturbed_val.pkl')
            self.val_background = self.datamodule[self.modulename](background, **self.data_param)
            self.val_target = self.datamodule[self.modulename](target, **self.data_param)

        # Assign test dataset for use in dataloader
        if stage =='test':
            background = pd.read_pickle(f'{self.save_dir}/ntc_test.pkl')
            target= pd.read_pickle(f'{self.save_dir}/perturbed_test.pkl')
            self.test_background = self.datamodule[self.modulename](background, **self.data_param)
            self.test_target = self.datamodule[self.modulename](target, **self.data_param)

        if stage == 'embed':
            df_all = pd.concat([pd.read_pickle(f'{self.save_dir}/perturbed_filtered.pkl'),
                                pd.read_pickle(f'{self.save_dir}/ntc_filtered.pkl')], axis=0)
            self.ds_all = OPSwithlabel(df_all, **self.data_param)

    def train_dataloader(self):
        return DataLoader(PairedDataset(self.train_background, self.train_target), collate_fn=self.collate_fn, 
                                   shuffle=False, drop_last=True, pin_memory=True, persistent_workers=True, **self.loader_param)
    
    def val_dataloader(self):
        return DataLoader(PairedDataset(self.val_background, self.val_target), collate_fn=self.collate_fn, 
                                   drop_last=True, **self.loader_param)
    
    def test_dataloader(self):
        return DataLoader(PairedDataset(self.test_background, self.test_target), collate_fn=self.collate_fn, 
                                   drop_last=True, **self.loader_param)
    
    def all_dataloader(self, loader_param=None):
        if loader_param is not None:
            for key, val in loader_param.items():
                self.loader_param[key] = val
        return DataLoader(self.ds_all, shuffle=False, drop_last=False, **self.loader_param)
    
    @staticmethod
    def collate_fn(batch):
        batch_dict = {}
        for name in batch[0].keys():
            batch_dict[name] = torch.stack([item[name] for item in batch])
        return batch_dict
    
class DualDatasetGenerator:
    def __init__(self, background, target):
        # If one subset is smaller, cycle through it indefinitely
        if len(background) < len(target):
            background = cycle(background)
            self.num_items = len(target)
        elif len(target) < len(background):
            target = cycle(target)
            self.num_items = len(background)

        self.background = background
        self.target = target
        self.iter = iter(zip(self.background, self.target))
        
    def __len__(self):
        return self.num_items

    def __iter__(self):
        while True:
            try:
                yield DualDatasetGenerator.build_pair(next(self.iter))
            except StopIteration:
                self.iter = iter(zip(self.background, self.target))
    
    @staticmethod
    def build_pair(samples):
        bg_samples, tg_samples = samples
        if len(bg_samples) == 2:
            bg_x, bg_y = bg_samples
            tg_x, tg_y = tg_samples
            if bg_y.shape[1] == 2:
                bg_label, bg_batch = bg_y[:,0], bg_y[:,1]
                tg_label, tg_batch = tg_y[:,0], tg_y[:,1]
                return {'background': bg_x, 'target': tg_x, 'background_label': bg_label, 'target_label': tg_label, 'background_batch': bg_batch, 'target_batch': tg_batch}
            else:
                return {'background': bg_x, 'target': tg_x, 'background_label': bg_y, 'target_label':tg_y}
        else:
            return {'background': bg_samples, 'target': tg_samples}


