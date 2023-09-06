import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import pickle
from concurrent.futures import ProcessPoolExecutor

# root data directory
DATA_DIR = '/projects/site/gred/resbioai/comp_vision/cellpaint-ai/ops/datasets/funk22/funk22-phenotype-profiles'
xls = pd.ExcelFile('/home/wangz222/cellprofiler/extracted_image_features.xlsx')
image_features = xls.parse('(B) Extracted image features', header=1)['feature']

# specify the directory containing your parquet files
directory = f'{DATA_DIR}/interphase-reclassified_cp_phenotype_normalized.parquet'

# get a list of all files in the directory
files = os.listdir(directory)

# define an empty DataFrame to store intermediate results
aggregated = pd.DataFrame()

def pseudo_bulk(file):
    # ensure we're only working with .parquet files
    if file.endswith('.parquet'):
        # specify the file path
        file_path = os.path.join(directory, file)
        
        # open the Parquet file
        chunk = pd.read_parquet(file_path)
        chunk = chunk.replace([np.inf, -np.inf], np.nan).dropna(how="any")

        # perform your grouping and averaging operation
        chunk_grouped = chunk.groupby('gene_symbol_0')[image_features].mean()
        return chunk_grouped

with ProcessPoolExecutor(max_workers=10) as executor:
    # apply the function to each file in parallel
    results = list(tqdm(executor.map(pseudo_bulk, files), total=len(files)))
    
# perform final grouping and averaging over all chunks
aggregated = pd.concat(results)
final_result = aggregated.groupby('gene_symbol_0')[image_features].mean()

with open('/home/wangz222/embedding/cp_interphase.pkl', 'wb') as f:
    pickle.dump(final_result, f)