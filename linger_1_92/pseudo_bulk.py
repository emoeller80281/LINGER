import numpy as np
import pandas as pd
import random
import scanpy as sc

np.random.seed(42)

def pseudo_bulk(adata_RNA, adata_ATAC, singlepseudobulk: bool, num_neighbors: int = 20):
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
    gene_neighbor_agg = (connectivities @ adata_RNA.raw.X.toarray())
    peak_neighbor_agg = (connectivities @ adata_ATAC.raw.X.toarray())
    
    # Normalize the pseudo-bulk profiles by the number of neighbors to get the average expression and accessibility values for each cell type.
    gene_neighbor_agg_norm = gene_neighbor_agg / (num_neighbors-1)
    peak_neighbor_agg_norm = peak_neighbor_agg / (num_neighbors-1)
    
    # Create DataFrames for the normalized pseudo-bulk profiles
    # DataFrame: Data = Aggregated neighbor expression/accessibility, Columns = cell barcodes, Index = gene/peak IDs
    TG_pseudobulk_df = pd.DataFrame(gene_neighbor_agg_norm.T, columns=allindex, index=adata_RNA.raw.var['gene_ids'].tolist())
    RE_pseudobulk_df = pd.DataFrame(peak_neighbor_agg_norm.T, columns=allindex, index=adata_ATAC.raw.var['gene_ids'].tolist())
    
    return TG_pseudobulk_df, RE_pseudobulk_df
