from abc_atlas_access.abc_atlas_cache import AbcProjectCache
import scanpy as sc
import numpy as np
import pandas as pd

def load_allen_data(cache_dir='./data'):
    cache = AbcProjectCache.from_cache_dir(cache_dir)
    cells_df = cache.get_metadata("Developing-Mouse-Vis-Cortex-10X", "cells")
    adata = cache.get_expression_matrix("Developing-Mouse-Vis-Cortex-10X")
    return cells_df, adata

def preprocess(cells_df, adata):

    cells_df = cells_df[(cells_df['n_genes'] > 200) & (cells_df['percent_mito'] < 0.2)]
    adata = adata[cells_df.index, :]

    adata.X = np.log1p(adata.X)

    sc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor='seurat')
    adata = adata[:, adata.var['highly_variable']]

    sc.pp.pca(adata, n_comps=50)
    X = adata.obsm['X_pca']
    y = cells_df['class'].astype('category').cat.codes.values

    stages = cells_df['age_group']   # pre‑defined age group column
    return X, y, stages