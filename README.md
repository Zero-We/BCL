# A Bayesian Collaborative Learning Framework for Whole-Slide Image Classification

<img src="https://github.com/Zero-We/BCL/blob/main/docs/bcl_framework.png">


## Introduction
Whole-slide image (WSI) classification is fundamental to computational pathology, which is challenging in extra-high resolution, expensive manual annotation, data heterogeneity, etc. Multiple instance learning (MIL) provides a promising way to tackle these challenges. Nevertheless,
due to the gigapixel resolutions, MIL for WSI classification inherently suffers from the memory bottleneck issue, which remains largely unresolved in the literature. Towards this end, this paper presents a Bayesian Collaborative Learning (BCL) framework for gigapixel WSI classification. The
basic idea of BCL is to introduce an auxiliary patch classifier, along with the target MIL classifier to be learned, so that the two classifiers can collaborate with each other to achieve joint learning of the feature encoder and the MIL aggregator in the MIL classifier, free of the memory bottleneck issue. Such a collaborative learning procedure is formulated under a principled Bayesian probabilistic framework and an Expectation-Maximization algorithm is developed to infer the optimal model parameters iteratively. As an implementation of the E-step, an effective quality-
aware pseudo labeling approach is further proposed. Our BCL is extensively evaluated on three publicly available WSI datasets, i.e., CAMELYON16, TCGA-NSCLC and TCGA-RCC, achieving an AUC of 95.6%, 96.0% and 97.5% respectively, which consistently outperforms all the compared MIL methods. Comprehensive analysis and discussion will also be presented for in-depth understanding of the proposed BCL.

## Model
The trained model weights and are provided here （[[Google Drive]](https://drive.google.com/drive/folders/1kfib8H-4jhNzwj-_LDmUGVtjCv3Lg6zT?usp=sharing) | [[Baidu Cloud]](https://pan.baidu.com/s/1OQJM8Tp7y1RlRIPUKdjqIA) (fzts)）. You can download these files and drag `bcl_model.pth` and  to  the `model` directory.

## Dataset
* **Camelyon16**  
Camelyon16 is a public challenge dataset of sentinel lymph
node biopsy of early-stage breast cancer, which includes 270 H&E-stained WSIs for training and 129 for testing (48 LNM-positive and 81 LNM-negative), collected from two medical centers.   
Download from [here](https://camelyon17.grand-challenge.org/Data/).

* **TCGA**  
TCGA-NSCLC and TCGA-RCC are public available datasets for non-small-cell lung carcinoma(NSCLC) subtyping and renal cell carcinoma(RCC) subtyping respectively. They can be download from [here](https://portal.gdc.cancer.gov/)   

## Evaluation


## Visualization
The key to BCL is to introduce an auxiliary patch classifier to improve the MIL classifier collaboratively. Here we further visualize some representative results to show this point intuitively. More precisely, we choose two exemplary WSIs (named “test 021” and “test 122”) from CAMEL YON16 for visualization. We depict the MIL-classifier heat maps obtained by BCL at the first and last iteration steps and also by other methods and the patch-classifier heat maps obtained by BCL at various iteration steps. For the learned patch classifier, we also visualize its Class
Activation Maps (CAMs) generated over some patches selected from these two WSIs.  

<img src="https://github.com/Zero-We/BCL/blob/main/vis/attn-map.png" width="350px">


## Reference  
If you find our work useful in your research or if you use parts of this code please consider citing our paper.  
