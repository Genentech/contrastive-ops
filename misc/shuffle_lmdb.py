import lmdb
import pandas as pd
import warnings
import os, sys
from tqdm import tqdm
from src.constants import Column
import glob

# Your list of LMDB folders
root = '/projects/site/gred/resbioai/comp_vision/cellpaint-ai/ops/datasets/funk22'
dataset_root = {'interphase': os.path.join(root, 'funk22_lmdb_by_plate_interphase'), 'mitotic': os.path.join(root, 'funk22_lmdb_by_plate_mitotic')}
plate_list = ['20200202_6W-LaC024A','20200202_6W-LaC024D','20200202_6W-LaC024E',
              '20200202_6W-LaC024F','20200206_6W-LaC025A','20200206_6W-LaC025B']

def write_to_lmdb(lmdb_path, key, value):
    env = lmdb.open(lmdb_path, writemap=True, map_size=3000*1024*1024*1024) #3TB
    key = key.encode('utf-8')
    with env.begin(write=True) as txn:
        txn.put(key, value)  # Ensure the key is in bytes
    env.close()

def shuffle_dataframe(df):
    df = df.sample(frac=1).reset_index(drop=True)
    df['UID'] = df.index.map(lambda x: str(f'{x:08}')) # add in UID which will be used to enforce this particular key ordering
    return df

def shuffle_lmdb(keys_df, new_lmdb_folder_base):
    # keys_df = pd.read_csv(f'{new_lmdb_folder_base}/{plate}/key.csv')
    cell_cycle_stage = ['mitotic', 'interphase']
    txn_dict = {key: {stage: None for stage in cell_cycle_stage} for key in plate_list}

    for plate in plate_list:
        for stage in cell_cycle_stage:
            if os.path.exists(os.path.join(dataset_root[stage], plate)):
                env = lmdb.Environment(os.path.join(dataset_root[stage], plate), readonly=True, readahead=False, lock=False)
                txn_dict[plate][stage] = env.begin(write=False, buffers=True)
            else:
                warnings.warn(f"LMDB dataset for {stage} plate {plate} doesn't exist")

    new_lmdb_folder = f'{new_lmdb_folder_base}/'

    if not os.path.exists(new_lmdb_folder):
        os.makedirs(new_lmdb_folder)
    
    for _, row in tqdm(keys_df.iterrows(), mininterval=60*10, total=len(keys_df)):
        index = row[Column.index.value]
        plate = row[Column.plate.value]
        well = row[Column.well.value]
        tile = str(row[Column.tile.value])
        gene = row[Column.gene.value]
        stage = row[Column.gene.cell_cycle_stage.value]
        key = f'{plate}_{well}_{tile}_{gene}_{index}'
        value = txn_dict[plate][stage].get(key.encode())
        
        if value is not None:
            write_to_lmdb(new_lmdb_folder, str(row['UID']) + '_' + key, value)
    return keys_df

def get_lmdb_keys(dir):
    env = lmdb.Environment(dir, readonly=True, 
                           readahead=False, lock=False)
    with env.begin(write=False, buffers=True) as txn:
        length = txn.stat()['entries']
        keys = list(tqdm(txn.cursor().iternext(values=False), leave=False, total=length))
    return [key.tobytes().decode() for key in keys]

# print('reading in original dataframe')
# # Read in dataframe of keys
# interphase = pd.read_pickle('/home/wangz222/data/metadata_ordered_interphase.csv')
# mitotic = pd.read_pickle('/home/wangz222/data/metadata_ordered_mitotic.csv')
# keys_df = pd.concat([interphase,mitotic])

# # shuffle dataframe of keys
# ntc_df = keys_df[keys_df['gene_symbol_0']=='nontargeting']
# # # perturbed_df = keys_df[keys_df['gene_symbol_0']!='nontargeting']
# shuffled_df = shuffle_dataframe(ntc_df)
# #specify where to save database
save_dir = f'{root}/funk22_lmdb_shuffled/perturbed'
print(save_dir)
# shuffled_df.to_csv(f'{save_dir}/key.csv', index=False)

# created shuffled database
key_df = pd.read_csv(os.path.join(save_dir, 'key.csv'), dtype={'UID': str})
for filename in glob.glob(os.path.join(save_dir, '*.mdb')):
    print(f'deleting existing {filename}')
    os.remove(filename)
shuffle_lmdb(key_df, save_dir)
print('database succesfully created!')


# verify key-value pairs order matches key.csv
key_df['key'] = key_df[['UID', 'plate', 'well', 'tile', 'gene_symbol_0', 'index']].astype(str).apply('_'.join, axis=1)
actual = get_lmdb_keys(save_dir)

print(len(actual))
print(len(key_df))
ismatch = list(key_df['key']) == actual
if ismatch:
    print('Orders of lmdb matches dataframe')
else:
    print('Orders of lmdb do not match dataframe')