import pandas as pd
import numpy as np
from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import pickle
import os
import random
import sys
from joblib import Parallel, delayed, dump, load
from src.constants import Column

# Initialize an IncrementalPCA object
n_components = 650  # Or however many components you wish to keep
ipca = IncrementalPCA(n_components=n_components)

# Initialize a StandardScaler object
scaler = StandardScaler()

# Specify features to use
DATA_DIR = '/projects/site/gred/resbioai/comp_vision/cellpaint-ai/ops/datasets/funk22/funk22-phenotype-profiles'
xls = pd.ExcelFile('/home/wangz222/cellprofiler/extracted_image_features.xlsx')
image_features = xls.parse('(B) Extracted image features', header=1)['feature']

# Get all parquet files
folder_path = f'{DATA_DIR}/mitotic-reclassified_cp_phenotype_normalized.parquet'
mitotic_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.parquet')]
folder_path = f'{DATA_DIR}/interphase-reclassified_cp_phenotype_normalized.parquet'
parquet_files = mitotic_files + [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.parquet')]

# # shuffle order in which data is fed into incremental pca
# random.shuffle(parquet_files)

# accumulator = None
# # For each parquet file in the folder
# for file_path in tqdm(parquet_files, file=sys.stdout, mininterval=120):
#     chunk = pd.read_parquet(file_path)
#     chunk = chunk.replace([np.inf, -np.inf], np.nan).dropna(how="any")

#     # Accumulate
#     if accumulator is None:
#         accumulator = chunk
#     else:
#         accumulator = pd.concat([accumulator, chunk])

#     # Partially fit the scaler and transform the chunk
#     if len(accumulator) >= n_components*2:
#         scaler.partial_fit(accumulator[image_features])
#         chunk_scaled = scaler.transform(accumulator[image_features])
        
#         # Perform incremental PCA on the scaled chunk
#         ipca.partial_fit(chunk_scaled)

#         # Reset accumulator
#         accumulator = None

# # Save models to disk
# dump(ipca, '/home/wangz222/scratch/embedding/cellprofiler/ipca_fixederror.joblib')
# dump(scaler, '/home/wangz222/scratch/embedding/cellprofiler/scaler_fixederror.joblib')
ipca = load('/home/wangz222/scratch/embedding/cellprofiler/ipca_fixederror.joblib')
scaler = load('/home/wangz222/scratch/embedding/cellprofiler/scaler_fixederror.joblib')

features = [
    Column.sgRNA.value,
    Column.gene.value,
    Column.plate.value,
    Column.tile.value,
    Column.well.value,
    "class_predict_1",
    "cell_i",
    "cell_j",
    "cell_area",
    "cell_perimeter",
    "cell_convex_area",
    "cell_form_factor",
    "cell_solidity",
    "cell_extent",
    "cell_euler_number",
    "cell_eccentricity",
    "cell_major_axis",
    "cell_minor_axis",
    "cell_median_radius",
    "cell_mean_radius",
    "cell_number_neighbors_1",
    "nucleus_number_neighbors_1",
    "cell_percent_touching_1",
    "nucleus_percent_touching_1"
]

# gene = ['ARF4','BCAR1','COP1','CRKL','CSE1L','FERMT2','ILK','ITGAV','ITGB1','ITGB5','KANSL1','KANSL2',
#  'KANSL3','KAT8','KPNB1','LIMS1','MARK2','MCRS1','NCBP1','NCBP2','PTK2','PXN','RAB10','RAC1',
#  'RAPGEF1','RBM8A','RSU1','SEC61A1','SEC61G','SFPQ','SRRT','TLK2','TLN1','TNS3','XPO1','ACTR2',
#  'ACTR3','ARPC2','ARPC3','ARPC4','BRK1','CYFIP1','DDX3X','NCKAP1','NUP214','NUP88','NUTF2','RANBP2',
#  'RANGAP1', 'CCT2', 'CCT3', 'CCT4', 'CCT5', 'CCT6A', 'CCT7', 'CCT8', 'TBCC', 'TCP1', 'TUBA1B', 'TUBA1C', 'TUBB','nontargeting']

def pca_transform(file_path, ipca, scaler, n_components, aggregate=True):
    # open the Parquet file
    chunk = pd.read_parquet(file_path).reset_index()
    chunk = chunk.replace([np.inf, -np.inf], np.nan).dropna(how="any").reset_index(drop=True)

    if chunk.shape[0] > 0:
        chunk_scaled = scaler.transform(chunk[image_features])  
        
        # transform data using fitted PCs
        pc_col = [f'PC{i}' for i in range(n_components)]
        transformed = pd.DataFrame(ipca.transform(chunk_scaled)[:,:n_components], columns=pc_col)
        chunk = pd.concat([chunk[features], transformed], axis=1)

        # perform your grouping and averaging operation (we do aggregation to save memory)
        if aggregate:
            agg_dict = {col: 'sum' for col in pc_col}
            agg_dict['sgRNA_0'] = 'count'
            agg_dict['gene_symbol_0'] = 'first'
            chunk_grouped = chunk.groupby('sgRNA_0').agg(agg_dict)
            chunk_grouped.columns = [col if col != 'sgRNA_0' else 'count' for col in chunk_grouped.columns]
            chunk = chunk_grouped.reset_index()
        return chunk

# Create the list of delayed tasks
aggregate = False
n_components = 650
tasks = [delayed(pca_transform)(file_path, ipca, scaler, n_components, aggregate) for file_path in parquet_files]
results = Parallel(n_jobs=6)(tqdm(tasks))

# perform final grouping and averaging over all chunks
results = pd.concat(results)
if aggregate:
    pc_col = [f'PC{i}' for i in range(n_components)]
    agg_dict = {col: 'sum' if col!='gene_symbol_0' else 'first' for col in results.columns}
    results = results.groupby('sgRNA_0').agg(agg_dict)
    results[pc_col] = results[pc_col].div(results['count'], axis='index')
    results = results.groupby('gene_symbol_0')[pc_col].mean()

with open(f'/home/wangz222/scratch/embedding/cellprofiler/singlecell_ipca_fixederror_{n_components}pc.pkl', 'wb') as f:
    pickle.dump(results, f)