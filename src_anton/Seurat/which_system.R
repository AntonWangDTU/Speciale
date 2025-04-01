suppressMessages(library(Seurat))

# Load the PBMC dataset
pbmc.data <- Read10X(data.dir = "C:/Users/hostp/Desktop/data/filtered_gene_bc_matrices/hg19/")
# Initialize the Seurat object with the raw (non-normalized data).
pbmc <- CreateSeuratObject(counts = pbmc.data, project = "pbmc3k", min.cells = 3, min.features = 200)

random_ids <- paste0("A", seq_len(length(Features(pbmc))))
random_df <- data.frame(original = Features(pbmc),
                        new_ids = random_ids)

pbmc[["RNA"]] <- AddMetaData(object = pbmc[["RNA"]], metadata = random_df, col.name = "new_ids")

head(pbmc@assays$RNA@meta.data)
#>   new_ids
#> 1      A1
#> 2      A2
#> 3      A3
#> 4      A4
#> 5      A5
#> 6      A6