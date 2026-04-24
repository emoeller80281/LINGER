import os
import scanpy as sc
import pandas as pd
import warnings
import sys
import argparse
import scipy.sparse as sparse
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

# Import necessary modules from linger
sys.path.insert(0, '/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.GRN_BENCHMARKING.MOELLER/LINGER')
from linger.preprocess import *
from linger.pseudo_bulk import *

# Filter warnings about copying objects from AnnData
warnings.filterwarnings("ignore", message="Received a view of an AnnData. Making a copy.")
warnings.filterwarnings("ignore", message="Trying to modify attribute `.obs` of view, initializing view as actual.")

def convert_dense_matrix_to_10x_format(rna_data, atac_data, cell_type_label):
    """
    Converts dense RNA and ATAC data matrices into a format compatible with 10x Genomics' Cell Ranger
    - Sparse matrix for the combined data
    - Features that lists the gene and peak names along with their types
    - Barcodes that lists the cell barcodes
    - Label that assigns cell type labels to each barcode.
    """
    
    # Update the index to replace '-' with ':' for the chromosome coordinates chr-start-stop -> chr:start-stop
    if not atac_data.index.str.contains(':').all():
        logging.info(atac_data.index[0])
        atac_data.index = atac_data.index.str.replace('-', ':', n=1)
        logging.info(atac_data.index[0])

        atac_data.to_csv(args.atac_data_path, sep=',')

    # Create the data matrix by concatenating the RNA and ATAC data by their indices
    matrix = csc_matrix(pd.concat([rna_data, atac_data], axis=0).values)
    features = pd.DataFrame({
        0: rna_data.index.tolist() + atac_data.index.tolist(),  # Combine RNA and ATAC feature names
        1: ['Gene Expression'] * len(rna_data.index) + ['Peaks'] * len(atac_data.index)  # Assign types
    })
    logging.info(features)
    barcodes = pd.DataFrame(rna_data.columns.values, columns=[0])

    label = pd.DataFrame({
        'barcode_use': barcodes[0].values,  # Use the same barcodes as in the RNA and ATAC data
        'label': [cell_type_label] * len(barcodes)  # Set the label to the specified cell type for all cells
    })

    return matrix, features, barcodes, label

def intersect_barcodes(adata_RNA, adata_ATAC):
    """
    Intersect barcodes between RNA and ATAC datasets.
    """
    selected_barcode = adata_RNA.obs['barcode'][adata_RNA.obs['barcode'].isin(adata_ATAC.obs['barcode'])].tolist()

    rna_barcode_idx = pd.DataFrame(range(adata_RNA.shape[0]), index=adata_RNA.obs['barcode'].values)
    atac_barcode_idx = pd.DataFrame(range(adata_ATAC.shape[0]), index=adata_ATAC.obs['barcode'].values)

    adata_RNA = adata_RNA[rna_barcode_idx.loc[selected_barcode][0]].copy()
    adata_ATAC = adata_ATAC[atac_barcode_idx.loc[selected_barcode][0]].copy()

    return adata_RNA, adata_ATAC

def remove_bad_cells_and_values(adata):
    """
    Remove cells with zero counts and replace NaN/inf values in the data matrix with zeros. 
    Also checks for negative values in the matrix, which are not expected in raw count data.
    """
    X = adata.X

    if sparse.issparse(X):
        X = X.tocsr(copy=True)
        # Replace NaN/inf in sparse stored values
        X.data = np.nan_to_num(X.data, nan=0.0, posinf=0.0, neginf=0.0)

        # Optional hard check: reject negatives for RNA counts
        if (X.data < 0).any():
            raise ValueError("Matrix contains negative values; expected raw counts.")

        row_sums = np.asarray(X.sum(axis=1)).ravel()
        adata = adata[row_sums > 0].copy()
        adata.X = X[row_sums > 0]
    else:
        X = np.asarray(X, dtype=np.float64)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        if (X < 0).any():
            raise ValueError("Matrix contains negative values; expected raw counts.")

        row_sums = X.sum(axis=1)
        keep = row_sums > 0
        adata = adata[keep].copy()
        adata.X = X[keep]

    return adata

# Define command-line arguments
parser = argparse.ArgumentParser(description="Process scRNA-seq and scATAC-seq data for pseudo-bulk analysis.")

# Add arguments for file paths and directories
parser.add_argument("--rna_data_path", required=True, help="Path to RNA data CSV file")
parser.add_argument("--atac_data_path", required=True, help="Path to ATAC data CSV file")
parser.add_argument("--data_dir", required=True, help="Directory to save processed data")
parser.add_argument("--sample_data_dir", required=True, help="Output directory for LINGER-generated data files")
parser.add_argument("--organism", required=True, help='Enter "mouse" or "human"')
parser.add_argument("--genome", required=True, help="Organism genome code")
parser.add_argument("--cell_type", required=True, help="Cell type of interest")
parser.add_argument("--method", required=True, help="Training method")

# Parse arguments
args = parser.parse_args()

output_dir = args.sample_data_dir + "/"

if not os.path.exists(args.sample_data_dir):
    os.makedirs(args.sample_data_dir)

logging.info('\tReading in cell labels...')
# Load scRNA-seq data
rna_data = pd.read_csv(args.rna_data_path, sep=',', index_col=0)
atac_data = pd.read_csv(args.atac_data_path, sep=',', index_col=0)

matrix, features, barcodes, label = convert_dense_matrix_to_10x_format(rna_data, atac_data, args.cell_type)

logging.info('\nConverting dense matrix to AnnData format')
adata_RNA, adata_ATAC = get_adata(matrix, features, barcodes, label)

logging.info(f'\tscRNAseq Dataset: {adata_RNA.shape[1]} genes, {adata_RNA.shape[0]} cells')
logging.info(f'\tscATACseq Dataset: {adata_ATAC.shape[1]} peaks, {adata_ATAC.shape[0]} cells')

# Remove low count cells and genes
logging.info('\nFiltering Data')
sc.pp.filter_cells(adata_RNA, min_genes=200)
sc.pp.filter_genes(adata_RNA, min_cells=3)
sc.pp.filter_cells(adata_ATAC, min_genes=200)
sc.pp.filter_genes(adata_ATAC, min_cells=3)

logging.info('\nShape of the dataset after QC filtering')
logging.info(f'\tscRNAseq Dataset: {adata_RNA.shape[1]} genes, {adata_RNA.shape[0]} cells')
logging.info(f'\tscATACseq Dataset: {adata_ATAC.shape[1]} peaks, {adata_ATAC.shape[0]} cells')

logging.info(f'\nCombining RNA and ATAC seq barcodes')
adata_RNA, adata_ATAC = intersect_barcodes(adata_RNA, adata_ATAC)

logging.info(f"\nNormalizing and log-transforming the data")
# Normalize and log-transform the RNA data (NOTE: only RNA is normalized)
sc.pp.normalize_total(adata_RNA, target_sum=1e4)

sc.pp.log1p(adata_RNA)
sc.pp.log1p(adata_ATAC)

adata_RNA.raw=adata_RNA
adata_ATAC.raw=adata_ATAC

# Subset to highly variable genes
sc.pp.highly_variable_genes(adata_RNA, min_mean=0.0125, max_mean=3, min_disp=0.5)
sc.pp.highly_variable_genes(adata_ATAC, min_mean=0.0125, max_mean=3, min_disp=0.5)

adata_RNA = adata_RNA[:, adata_RNA.var.highly_variable]
adata_ATAC = adata_ATAC[:, adata_ATAC.var.highly_variable]

# Scale the data
sc.pp.scale(adata_RNA, max_value=10)
sc.pp.scale(adata_ATAC, max_value=10, zero_center=True)

# Calculate PCA for RNA data
sc.tl.pca(adata_RNA, n_comps=15,svd_solver="arpack")
sc.tl.pca(adata_ATAC, n_comps=15,svd_solver="arpack")

pca_RNA=adata_RNA.obsm['X_pca']
pca_ATAC=adata_ATAC.obsm['X_pca']

# Concatenate the PCA embeddings from RNA and ATAC
pca = np.concatenate((pca_RNA,pca_ATAC), axis=1)

# Assign the combined PCA embeddings back to the AnnData objects
adata_RNA.obsm['pca']=pca
adata_ATAC.obsm['pca']=pca

num_neighbors = 20

# Generate the KNN graph based on the combined PCA embeddings
sc.pp.neighbors(adata_RNA, n_neighbors=num_neighbors, n_pcs=30,use_rep='pca')

logging.info(f'\nGenerating pseudo-bulk / metacells')
samplelist = list(set(adata_ATAC.obs['sample'].values))

TG_pseudobulk = pd.DataFrame([])
RE_pseudobulk = pd.DataFrame([])

# Only use one cell from each sample if there are more than 100 samples
singlepseudobulk: bool = (adata_RNA.obs['sample'].unique().shape[0] * adata_RNA.obs['sample'].unique().shape[0] > 100)
for tempsample in samplelist:
    adata_RNAtemp = adata_RNA[adata_RNA.obs['sample'] == tempsample].copy()
    adata_ATACtemp = adata_ATAC[adata_ATAC.obs['sample'] == tempsample].copy()
    
    adata_RNAtemp = remove_bad_cells_and_values(adata_RNAtemp)
    adata_ATACtemp = remove_bad_cells_and_values(adata_ATACtemp)

    TG_pseudobulk_temp, RE_pseudobulk_temp = pseudo_bulk(
        adata_RNAtemp, adata_ATACtemp, singlepseudobulk, num_neighbors=num_neighbors
        )

    TG_pseudobulk = pd.concat([TG_pseudobulk, TG_pseudobulk_temp], axis=1)
    RE_pseudobulk = pd.concat([RE_pseudobulk, RE_pseudobulk_temp], axis=1)

    RE_pseudobulk[RE_pseudobulk > 100] = 100
    
TG_pseudobulk = TG_pseudobulk.fillna(0)
RE_pseudobulk = RE_pseudobulk.fillna(0)

logging.info(f'Writing out pseudobulk...')
TG_pseudobulk.to_csv(f'{args.sample_data_dir}/TG_pseudobulk.tsv', sep='\t', index=True)
RE_pseudobulk.to_csv(f'{args.sample_data_dir}/RE_pseudobulk.tsv', sep='\t', index=True)

logging.info(f'Writing adata_ATAC.h5ad and adata_RNA.h5ad')
adata_ATAC.write_h5ad(f'{args.sample_data_dir}/adata_ATAC.h5ad')
adata_RNA.write_h5ad(f'{args.sample_data_dir}/adata_RNA.h5ad')

logging.info(f'Writing out peak gene ids')
pd.DataFrame(adata_ATAC.var['gene_ids']).to_csv(f'{args.sample_data_dir}/Peaks.txt', header=None, index=None)

