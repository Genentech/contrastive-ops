import pandas as pd
import numpy as np
from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import StandardScaler
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import pickle
import os

# Initialize an IncrementalPCA object
n_components = 150  # Or however many components you wish to keep
ipca = IncrementalPCA(n_components=n_components)

# Initialize a StandardScaler object
scaler = StandardScaler()

DATA_DIR = '/projects/site/gred/resbioai/comp_vision/cellpaint-ai/ops/datasets/funk22/funk22-phenotype-profiles'
folder_path = f'{DATA_DIR}/interphase-reclassified_cp_phenotype_normalized.parquet'
xls = pd.ExcelFile('/home/wangz222/cellprofiler/extracted_image_features.xlsx')
image_features = xls.parse('(B) Extracted image features', header=1)['feature']


# Get all parquet files
parquet_files = [f for f in os.listdir(folder_path) if f.endswith('.parquet')][:10]

# For each parquet file in the folder
for file in tqdm(parquet_files[:10]):
    file_path = os.path.join(folder_path, file)
    chunk = pd.read_parquet(file_path)
    chunk = chunk.replace([np.inf, -np.inf], np.nan).dropna(how="any")

    # Partially fit the scaler and transform the chunk
    scaler.partial_fit(chunk[image_features])
    chunk_scaled = scaler.transform(chunk[image_features])
    
    # Perform incremental PCA on the scaled chunk
    ipca.partial_fit(chunk_scaled)


# Now ipca contains your trained PCA model
def pca_transform(file):
    # specify the file path
    file_path = os.path.join(folder_path, file)

    # open the Parquet file
    chunk = pd.read_parquet(file_path)
    chunk = chunk.replace([np.inf, -np.inf], np.nan).dropna(how="any")

    # Remember to scale any data before transforming it:
    chunk_scaled = scaler.transform(chunk[image_features])  
    
    # Assume ipca is your trained IncrementalPCA object and data is the data you want to transform
    new_col = [f'PC{i}' for i in range(n_components)]
    transformed = pd.DataFrame(ipca.transform(chunk_scaled), columns=new_col)
    chunk = pd.concat([chunk['gene_symbol_0'].reset_index(), transformed], axis=1)

    # perform your grouping and averaging operation
    chunk_grouped = chunk.groupby('gene_symbol_0')[new_col].mean()
    return chunk_grouped


with ProcessPoolExecutor(max_workers=10) as executor:
    # apply the function to each file in parallel
    results = list(tqdm(executor.map(pca_transform, parquet_files), total=len(parquet_files)))

# perform final grouping and averaging over all chunks
aggregated = pd.concat(results)
new_col = [f'PC{i}' for i in range(n_components)]
final_result = aggregated.groupby('gene_symbol_0')[new_col].mean()

with open('/home/wangz222/embedding/cp_interphase_ipca.pkl', 'wb') as f:
    pickle.dump(final_result, f)