import sys
import numpy as np
import matplotlib.pyplot as plt
from typing import List
import torch

import pandas as pd
from src.constants import Column
from src.models import *
from src.scvae import SplitContrastiveVAE
from src.dataloader import *
from src.transformation import DataAugmentation, ContrastiveDataAugmentation
from tqdm import tqdm


def embed_images(model, data_module, stage='embed', loader_param=None, modelname='contrastive'):
    # Encode all images in the data_loader using model, and return encodings
    data_module.prepare_data()
    data_module.setup(stage=stage)
    dataloader = data_module.all_dataloader(loader_param)
    transform = DataAugmentation(data_module.transform.transforms)
    model.eval()
    embed_list = []

    pbar = tqdm(total=len(dataloader), file=sys.stdout, desc="Encoding images", leave=False, mininterval=300)
    if modelname == 'contrastive':
        for imgs, labels in dataloader:
            imgs = transform(imgs.to(model.device))
            labels = labels.to(model.device)
            with torch.inference_mode():
                embed_list.append(model.get_image_embedding(imgs, labels[:,0]))
            pbar.update()
    else:
        for imgs in dataloader:
            imgs = transform(imgs.to(model.device))
            with torch.inference_mode():
                embed_list.append(model.get_image_embedding(imgs))
            pbar.update()
    pbar.close()
    embedding = torch.cat(embed_list, dim=0)
    embedding_df = pd.DataFrame(embedding.cpu().numpy())
    embedding_df = pd.concat([data_module.ds_all.df[[Column.sgRNA.value,Column.gene.value]]
                            .reset_index(drop=True), embedding_df], axis=1)
    return embedding_df

def display_patch(image):
    # visualize cells
    _, axes = plt.subplots(1, 4, figsize=(12, 4))
    for i in range(4):
        axes[i].imshow(image[i], cmap='gray')
        axes[i].grid(False)

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # GPU operation have separate seed
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        # Ensure that all operations are deterministic on GPU (if used) for reproducibility
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    # device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    # print("Device:", device)

def get_images(num, dataloader, transform):
    # x = torch.stack([dataset[i] for i in range(num)], dim=0)
    batch = next(iter(dataloader))
    if type(batch) is dict:
        x = {}
        for key, val in batch.items():
            x[key] = val[:num,...]
    elif isinstance(batch, list):
        x = []
        for val in batch:
            x.append(val[:num,...])
    else:
        x = batch[0:num,...]
    return transform(x)

def get_module(key, module_name):
    modules = dict(
        model={
            'ae': AEmodel,
            'vae': VAEmodel,
            'ctvae': ContrastiveVAEmodel,
            },
        dataloader={
            'base': BaseDataModule,
            'contrastive': ContrastiveDataModule,
            },
        augmentation={
            'base': DataAugmentation,
            'contrastive': ContrastiveDataAugmentation,
            },
            )

    return modules[module_name][key]
