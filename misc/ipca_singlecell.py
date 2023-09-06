import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle
import os
from joblib import Parallel, delayed, load

n_components = 650
# Specify features to use
DATA_DIR = '/projects/site/gred/resbioai/comp_vision/cellpaint-ai/ops/datasets/funk22/funk22-phenotype-profiles'
xls = pd.ExcelFile('/home/wangz222/cellprofiler/extracted_image_features.xlsx')
image_features = xls.parse('(B) Extracted image features', header=1)['feature']

# Save models to disk
ipca = load('/home/wangz222/scratch/embedding/cellprofiler/ipca_fixederror.joblib')
scaler = load('/home/wangz222/scratch/embedding/cellprofiler/scaler_fixederror.joblib')

def pca_transform(file_path, ipca, scaler):
    # open the Parquet file
    chunk = pd.read_parquet(file_path).reset_index()
    chunk = chunk.replace([np.inf, -np.inf], np.nan).dropna(how="any").reset_index(drop=True)

    if chunk.shape[0] > 0:
        chunk_scaled = scaler.transform(chunk[image_features])  
        
        # transform data using fitted PCs
        pc_col = [f'PC{i}' for i in range(n_components)]
        transformed = pd.DataFrame(ipca.transform(chunk_scaled), columns=pc_col)
        chunk = pd.concat([chunk[['sgRNA_0','gene_symbol_0','plate','well','i_0','j_0']], transformed], axis=1)
        return chunk

# Create the list of delayed tasks
# Get all parquet files
folder_path = f'{DATA_DIR}/mitotic-reclassified_cp_phenotype_normalized.parquet'
mitotic_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.parquet')]
folder_path = f'{DATA_DIR}/interphase-reclassified_cp_phenotype_normalized.parquet'
interphase_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.parquet')]
tasks = [delayed(pca_transform)(file_path, ipca, scaler) for file_path in interphase_files]
results = Parallel(n_jobs=6)(tqdm(tasks))
aggregated_interphase = pd.concat(results)
aggregated_interphase['stage'] = 'interphase'

tasks = [delayed(pca_transform)(file_path, ipca, scaler) for file_path in mitotic_files]
results = Parallel(n_jobs=6)(tqdm(tasks))
# perform final grouping and averaging over all chunks
aggregated_mitotic = pd.concat(results)
aggregated_mitotic['stage'] = 'mitotic'
aggregated = pd.concat([aggregated_interphase, aggregated_mitotic])

with open('/home/wangz222/scratch/embedding/cellprofiler/singlecell_ipca_fixederror.pkl', 'wb') as f:
    pickle.dump(aggregated, f)