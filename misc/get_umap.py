import pickle
from umap import UMAP
import pandas as pd
from sklearn.preprocessing import StandardScaler

print('finished importing packages')
model_name = {
              'mcVAE': 'ctvae-conditional_wasserstein8_fxk3lmiz',
              }

fname = model_name['mcVAE']
with open(f'/home/wangz222/scratch/embedding/{fname}.pkl', "rb") as f:
        data = pickle.load(f)

# Creating the UMAP transformer
umap_transformer = UMAP(metric='cosine', n_jobs=-1)

scaler = StandardScaler()

metadata = data.iloc[::10,:2]
embedding = data.iloc[::10,2:34]
embedding_standardized = pd.DataFrame(scaler.fit_transform(embedding), 
                                   columns=embedding.columns)

# Fitting and transforming the data
print('running UMAP')
embedding = umap_transformer.fit_transform(embedding_standardized)
final_df = pd.concat((metadata.reset_index(), 
                      pd.DataFrame(embedding, 
                                   columns=['UMAP1','UMAP2'])), axis=1)

with open(f'/home/wangz222/scratch/UMAP/{fname}_salient.pkl', 'wb') as f:
    pickle.dump(final_df, f)
print('file saved')