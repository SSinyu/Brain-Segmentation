# BS


### Dataset
---
LGG Segmentation Dataset from The Cancer Imaging Archive(TCIA)
110 patients included in The Cancer Genome Atlas (TCGA) lower-grade glioma collection with at least fluid-attenuated inversion recovery (FLAIR) sequence and genomic cluster data available.

Download it from [LGG Segmentation Dataset](https://www.kaggle.com/mateuszbuda/lgg-mri-segmentation)

All images are provided in `.tif` format with 3 channels per image (pre-contrast, FLAIR, post-contrast). Masks are binary, 1-channel images. They segment FLAIR abnormality present in the FLAIR sequence.

The dataset is organized into 110 folders named after case ID that contains information about source institution. Each folder contains MR images with the following naming convention:

`TCGA_<institution-code>_<patient-id>_<slice-number>.tif`

Corresponding masks have a `_mask` suffix.


The dataset placing should look like:

    lgg-mri-segmentation
    ├── TCGA_CS_4941_19960909
    │   ├── TCGA_CS_4941_19960909_1.tif
    │   ├── TCGA_CS_4941_19960909_1_mask.tif
    │   ├── TCGA_CS_4941_19960909_2.tif
    │   ├── TCGA_CS_4941_19960909_2_mask.tif
    │   └── ...
    │           
    ├── TCGA_CS_4942_19970222
    │   ├── TCGA_CS_4942_19970222_1.tif
    │   ├── TCGA_CS_4942_19970222_1_mask.tif
    │   ├── TCGA_CS_4942_19970222_2.tif
    │   ├── TCGA_CS_4942_19970222_2_mask.tif
    │   └── ...
    ...
    │
    └── TCGA_HT_A616_19991226
        ├── TCGA_HT_A616_19991226_1.tif
        ├── TCGA_HT_A616_19991226_1_mask.tif
        ├── TCGA_HT_A616_19991226_2.tif
        ├── TCGA_HT_A616_19991226_2_mask.tif
        └── ...

-------



### Model
---
* U-Net / Attention U-Net / U-Net++ / U-Net+++
* DeepLabV3+
* V-Net
