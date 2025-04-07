library(Seurat)


cao_data <- readRDS('C:/Users/hostp/Desktop/data/gene_count_sampled.RDS')
df_cell <- readRDS("C:/Users/hostp/Desktop/data/df_cell.RDS")
df_gene <- readRDS("C:/Users/hostp/Desktop/data/df_gene.RDS")


cao <- CreateSeuratObject(counts = cao_data, meta.data = df_cell)

# Optional: Add gene metadata if applicable
cao[["RNA"]] <- AddMetaData(cao[["RNA"]], metadata = df_gene)




cao_subsample <- subset(cao, cells = sample(Cells(cao), 10000))
# Find variable features
cao_subsample <- FindVariableFeatures(cao_subsample, selection.method = "vst", nfeatures = 2000)

# Identify the 10 most highly variable genes
top10 <- head(VariableFeatures(cao_subsample), 10)

# Get gene short names for the top 10 variable genes
# Mapping the Ensembl IDs in top10 to gene short names
gene_short_names <- cao[["RNA"]]@meta.data$gene_short_name[match(top10, rownames(cao[["RNA"]]))]

# Plot variable features without labels
plot1 <- VariableFeaturePlot(cao_subsample)

# Label the points using gene short names
plot2 <- LabelPoints(plot = plot1, points = top10, labels = gene_short_names, repel = TRUE)

# Display both plots
plot1
plot2