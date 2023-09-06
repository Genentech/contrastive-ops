import sys
# Add the directory containing the 'multivariate' module to the Python path
sys.path.append('/home/wangz222/contrastive-ops/src')
import os
import pandas as pd
import numpy as np
import multivariate
import pickle
import matplotlib.pyplot as plt
from constants import Column
from scipy.spatial import distance

def multivariate_metric(fname, reference=None, mode='all', save=True, cellprofiler=False, **kwargs):
    # Load the data
    df_sc = pd.read_pickle(f'/home/wangz222/scratch/embedding/{fname}.pkl')
    db_path = reference

    METADATA_COLS = [Column.sgRNA.value, Column.gene.value]
    if mode == 'all':
        if cellprofiler:
            FEATURE_COLS = [i for i in df_sc.columns if i not in METADATA_COLS][10:]
        else:
            FEATURE_COLS = [i for i in df_sc.columns if i not in METADATA_COLS]
    elif mode == 'salient':
        FEATURE_COLS = [i for i in df_sc.columns if i not in METADATA_COLS][:32]
    elif mode == 'background':
        FEATURE_COLS = [i for i in df_sc.columns if i not in METADATA_COLS][32:]
    print(METADATA_COLS)
    print(FEATURE_COLS)
    df_sc = df_sc[METADATA_COLS + FEATURE_COLS]

    df = df_sc.groupby(Column.sgRNA.value).agg({col: 'mean' if col != Column.gene.value else 'first' for col in df_sc.columns[1:]}
                                                ).groupby(Column.gene.value).mean()
    df = df - df.loc['nontargeting']

    df_standardized = standardize(df.reset_index(), FEATURE_COLS, [Column.gene.value])

    if reference is None:
        sig_gene = load_pickle('/home/wangz222/contrastive-ops/eval/sig_gene_cp.pkl')['99percentile']
        sig_df = df_standardized[df_standardized[Column.gene.value].isin(sig_gene)]
        order_df = pd.DataFrame({Column.gene.value: sig_gene})
        sig_df = order_df.merge(sig_df, on=Column.gene.value, how='left') # reorder to match sig_gene order

        dist_mat = distance.cdist(sig_df[FEATURE_COLS], sig_df[FEATURE_COLS], "cosine")
        gene_list = sig_df[Column.gene.value].values
        return gene_list, dist_mat

    # Run multivaraite analysis
    perturbations = df_standardized[Column.gene.value].values
    gene_graph, gene_sets = multivariate.enrichr_to_gene_graph(db_path, perturbations)
    full_genes = perturbations
    inds = []
    for gene in list(gene_graph.nodes):
        for idx, rxrx_gene in enumerate(full_genes):
            if gene == rxrx_gene:
                inds.append(idx)

    df_sub = df_standardized.iloc[inds]
    print(len(df_sub))
    ret_vals, dist_mat = multivariate.evaluate(embeddings=df_sub[FEATURE_COLS].values,
        gene_graph=gene_graph,
        gene_sets=gene_sets,
        percentile_range=[80, 100],
    )
    ret_df = []
    for k, v in ret_vals["metrics_by_percentile"].items():
        row = {"percentile": k}
        row.update(v)
        ret_df.append(row)
    ret_df = pd.DataFrame(ret_df)

    directory_path = f'/home/wangz222/evaluation/{fname}_{mode}'
    if save:
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
        with open(f"{directory_path}/multivariate.pkl", "wb") as f:
            pickle.dump(ret_df, f)
        print(ret_df[ret_df['percentile']==95])
        print(ret_df[ret_df['percentile']==90])

    plt.figure(dpi=100)
    plt.plot(ret_df[['recall']], ret_df[['precision']], 'o-')
    plt.title('CORUM complex correspondence with embedding clusters')
    plt.xlabel('Recall of complex')
    plt.ylabel('Precision of complex')

    print(f'{fname}_{mode}')

    return ret_df


def standardize(df, feature_names, metadata_names):
    values_gpu = np.asarray(df[feature_names])
    values_gpu = (values_gpu - values_gpu.mean(axis=0)) / values_gpu.std(axis=0)
    df_features = pd.DataFrame(values_gpu, columns=feature_names)
    df = pd.concat([df[metadata_names].reset_index(drop=True), df_features], axis=1)
    return df

def load_pickle(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)