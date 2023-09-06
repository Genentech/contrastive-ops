import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import pickle
from concurrent.futures import ProcessPoolExecutor
import random

# root data directory
DATA_DIR = '/projects/site/gred/resbioai/comp_vision/cellpaint-ai/ops/datasets/funk22/funk22-phenotype-profiles'
xls = pd.ExcelFile('/home/wangz222/cellprofiler/extracted_image_features.xlsx')
image_features = xls.parse('(B) Extracted image features', header=1)['feature'].to_list()

# specify the directory containing your parquet files
directory = f'{DATA_DIR}/interphase-reclassified_cp_phenotype_normalized.parquet'

# Get all parquet files
folder_path = f'{DATA_DIR}/mitotic-reclassified_cp_phenotype_normalized.parquet'
mitotic_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.parquet')]
folder_path = f'{DATA_DIR}/interphase-reclassified_cp_phenotype_normalized.parquet'
parquet_files = mitotic_files + [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.parquet')]
random.shuffle(parquet_files)

# define an empty DataFrame to store intermediate results
aggregated = pd.DataFrame()

def pseudo_bulk(file):
    # ensure we're only working with .parquet files
    if file.endswith('.parquet'):
        # specify the file path
        file_path = os.path.join(directory, file)
        
        # open the Parquet file
        chunk = pd.read_parquet(file_path).reset_index()
        chunk = chunk.replace([np.inf, -np.inf], np.nan).dropna(how="any")
        chunk = chunk[['sgRNA_0','gene_symbol_0']+image_features]

        # perform your grouping and averaging operation
        # Create the dictionary for aggregation
        agg_dict = {col: 'sum' for col in image_features}
        agg_dict['sgRNA_0'] = 'count'
        agg_dict['gene_symbol_0'] = 'first'
        chunk_grouped = chunk.groupby('sgRNA_0').agg(agg_dict)
        chunk_grouped.columns = [col if col != 'sgRNA_0' else 'count' for col in chunk_grouped.columns]
        return chunk_grouped.reset_index()

with ProcessPoolExecutor(max_workers=10) as executor:
    # apply the function to each file in parallel
    results = list(tqdm(executor.map(pseudo_bulk, parquet_files), total=len(parquet_files), mininterval=20))
    
# perform final grouping and averaging over all chunks
aggregated = pd.concat(results)
agg_dict = {col: 'sum' if col!='gene_symbol_0' else 'first' for col in aggregated.columns}
aggregated = aggregated.groupby('sgRNA_0').agg(agg_dict)
aggregated[image_features] = aggregated[image_features].div(aggregated['count'], axis='index')
final_result = aggregated.groupby('gene_symbol_0')[image_features].mean()

with open('/home/wangz222/embedding/cp_all.pkl', 'wb') as f:
    pickle.dump(final_result, f)