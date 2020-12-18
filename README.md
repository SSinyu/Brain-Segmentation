# Brain-Segmentation
Performance experiment and reporting of several semantic segmentation models in Low-Grade Glioma segmentation task.

<br/>
<br/>

### Results

|Model|Encoder<br/>(backbone)|Loss|Dice coefficient|IoU|
|:----:|:----:|:----:|:----:|:----:|
|[U-Net](https://arxiv.org/abs/1505.04597)|-|*bce* + *dice*|0.746|0.724|
|-|-|1*bce* + 2*dice*|0.721|0.734|
|[Attention U-Net](https://arxiv.org/abs/1804.03999)|-|*bce* + *dice*|0.782|0.778|
|-|-|1*bce* + 2*dice*|0.769|0.726|
|Attention U-Net using [CBAM](https://arxiv.org/abs/1807.06521)|-|*bce* + *dice*|0.823|0.810|
|[DeepLabV3+](https://arxiv.org/abs/1802.02611)|xception|*bce* + *dice*|0.813|0.814|
|-|-|1*bce* + 2*dice*|0.853|0.814|
|-|-|*dice*|**0.860**|**0.817**|

<br/>

#### Work in progress
- [ ] U-Net++
- [ ] U-squared Net
- [ ] V-Net
- [ ] U-Net+++

<br/>
<br/>

### Dataset
LGG Segmentation Dataset from The Cancer Imaging Archive(TCIA)
110 patients included in The Cancer Genome Atlas (TCGA) lower-grade glioma collection with at least fluid-attenuated inversion recovery (FLAIR) sequence and genomic cluster data available.

Download it from [LGG Segmentation Dataset](https://www.kaggle.com/mateuszbuda/lgg-mri-segmentation)


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
