import numpy as np
import pandas as pd
import random
import scanpy as sc

np.random.seed(42)

def tfidf(ATAC):
    """
    Calculate the TF-IDF matrix for the given ATAC-seq data.
    
    TF-IDF is a common technique used in text mining to evaluate the importance of a term 
    in a document relative to a corpus. In the context of ATAC-seq data, we can treat peaks as "terms" 
    and cells as "documents". The TF-IDF matrix can help to highlight important peaks that are 
    specific to certain cells while down-weighting peaks that are common across many cells.
    """
    O = 1 * (ATAC > 0)
    tf1 = O / (np.ones((O.shape[0], 1)) * np.log(1 + np.sum(O, axis=0))[np.newaxis,:])
    idf = np.log(1 + O.shape[1] / (1 + np.sum(O > 0, axis=1)))
    O1 = tf1 * (idf[:, np.newaxis] * np.ones((1, O.shape[1])))
    O1[np.isnan(O1)] = 0
    RE = O1.T
    return RE

def find_neighbors(adata_RNA,adata_ATAC):
    import scanpy as sc
    
    # Normalize and log-transform the RNA data
    sc.pp.normalize_total(adata_RNA, target_sum=1e4)
    sc.pp.log1p(adata_RNA)
    
    sc.pp.highly_variable_genes(adata_RNA, min_mean=0.0125, max_mean=3, min_disp=0.5)
    
    # Save the raw data before filtering
    adata_RNA.raw=adata_RNA
    
    # Subset to highly variable genes
    adata_RNA = adata_RNA[:, adata_RNA.var.highly_variable]
    
    # Zero-center and scale the RNA data
    sc.pp.scale(adata_RNA, max_value=10)
    
    # Calculate PCA for RNA data
    sc.tl.pca(adata_RNA, n_comps=15,svd_solver="arpack")
    pca_RNA=adata_RNA.obsm['X_pca']
    
    # Normalize and log-transform the ATAC data
    sc.pp.log1p(adata_ATAC)
    sc.pp.highly_variable_genes(adata_ATAC, min_mean=0.0125, max_mean=3, min_disp=0.5)
    
    # Save the raw data before filtering and select highly variable peaks
    adata_ATAC.raw=adata_ATAC
    adata_ATAC = adata_ATAC[:, adata_ATAC.var.highly_variable]
    
    # Zero-center and scale the ATAC data
    sc.pp.scale(adata_ATAC, max_value=10, zero_center=True)
    
    # Calculate PCA for ATAC data
    sc.tl.pca(adata_ATAC, n_comps=15,svd_solver="arpack")
    
    # Combine PCA embeddings from RNA and ATAC
    pca_ATAC=adata_ATAC.obsm['X_pca']
    pca = np.concatenate((pca_RNA,pca_ATAC), axis=1)
    
    # Assign the combined PCA embeddings back to the AnnData objects
    adata_RNA.obsm['pca']=pca
    adata_ATAC.obsm['pca']=pca

    return adata_RNA,adata_ATAC


def pseudo_bulk(adata_RNA, adata_ATAC, singlepseudobulk: bool):
    num_neighbors = 20
    # Preprocess the data and find neighbors
    adata_RNA, adata_ATAC = find_neighbors(adata_RNA,adata_ATAC)

    # Generate the KNN graph based on the combined PCA embeddings
    sc.pp.neighbors(adata_RNA, n_neighbors=num_neighbors, n_pcs=30,use_rep='pca')
    
    # Select the connectivities for the KNN graph
    connectivities=(adata_RNA.obsp['distances']>0)
    
    # Get the cell-type label information from the RNA data
    label=pd.DataFrame(adata_RNA.obs['label'])
    
    # Set the index of the label DataFrame to match the barcodes of the RNA data
    label.columns=['label']
    label.index=adata_RNA.obs['barcode'].tolist()
    
    cell_types=list(set(label['label'].values))
    
    # Sample cells from each cell type for creating pseudo-bulk profiles. The number of cells sampled is 
    # determined by the square root of the total number of cells in the cell type, with a minimum of 1 cell if singlepseudobulk is set to True.
    allindex=[]
    for i in range(len(cell_types)):
        
        # Select the indices of cells that belong to the current cell type
        cells_in_label = label[label['label'] == cell_types[i]].index
        
        num_cells = len(cells_in_label)
        if num_cells >= 10:
            
            # Determine the number of cells to sample based on the square root of the total number of cells
            n_samples = int(np.floor(np.sqrt(num_cells))) + 1
            
            # If there is only one pseudo-bulk profile to be generated, set n_samples to 1
            if singlepseudobulk == True:
                n_samples = 1
                
            # Randomly sample n_samples elements from the cells in the current cell type
            sampled_elements = random.sample(range(num_cells), n_samples)
            cells_in_label = cells_in_label[sampled_elements]
            
            # Add the sampled cells to the list of all indices for pseudo-bulk profiles
            allindex = allindex + cells_in_label.tolist()
    
    # Create the pseudo-bulk profiles by averaging the gene expression and peak accessibility values across the sampled cells for each cell type.
    # Get the connectivities from the KNN graph
    connectivities=pd.DataFrame(connectivities.toarray(), index=adata_RNA.obs['barcode'].tolist())
    connectivities=connectivities.loc[allindex].values
    
    # Multiply the neighbor connections by the neighbor gene expression and peak accessibility values
    # This step aggregates the gene expression and peak accessibility values from the neighboring cells to create the pseudo-bulk profiles.
    gene_neighbor_weight = (connectivities @ adata_RNA.raw.X.toarray())
    peak_neighbor_weight = (connectivities @ adata_ATAC.raw.X.toarray())
    
    # Normalize the pseudo-bulk profiles by the number of neighbors to get the average expression and accessibility values for each cell type.
    gene_neighbor_weight_norm = gene_neighbor_weight / (num_neighbors-1)
    peak_neighbor_weight_norm = peak_neighbor_weight / (num_neighbors-1)
    
    # Create DataFrames for the normalized pseudo-bulk profiles
    TG_filter1=pd.DataFrame(gene_neighbor_weight_norm.T, columns=allindex, index=adata_RNA.raw.var['gene_ids'].tolist())
    RE_filter1=pd.DataFrame(peak_neighbor_weight_norm.T, columns=allindex, index=adata_ATAC.raw.var['gene_ids'].tolist())
    
    return TG_filter1,RE_filter1
