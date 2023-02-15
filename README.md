

# VegAnn

![logo]()

** VegAnn: Vegetation Annotation of a large multi-crop RGB Dataset acquired under diverse conditions for image semantic segmentation**


# Table of contents
1. [Keypoints](#Keypoints)
1. [Abstract](#📚-Abstract)
3. [Citing](#Citing)
3. [Pytorch Data Loader](#loader)
3. [Baseline Results](#base)



## ⏳ Keypoints <a name="Keypoints"></a>

- Dataset can be found @ https://doi.org/10.5281/zenodo.7636408 
- VegAnn contains 3775 images 
- Images are 512*512 pixels 
- Corresponding binary masks is 0 for soil + crop residues (background) 255 for Vegetation (foreground)
- VegAnn is constituting of 26+ crop species (not represented homogeneously)
- Different acquisition system and configuration has been used to build VegAnn
- Check the dataset paper for more information (papier in review stage) @ :

## 📚 Abstract <a name="Abstract"></a>

  Applying deep learning to images of cropping systems provides new knowledge and insights in research and commercial applications. Semantic segmentation or pixel-wise classification, of RGB images acquired at the ground level, into vegetation and background is a critical step in the estimation of several canopy traits. Current state of the art methodologies based on convolutional neural networks (CNNs) are trained on datasets acquired under controlled or indoor environments. These models are unable to generalize to real-world images and hence need to be fine-tuned using new labelled datasets. This motivated the creation of the VegAnn - **Veg**etation **Ann**otation - dataset, a collection of 3795 multi-crop RGB images acquired for different phenological stages using different systems and platforms in diverse illumination conditions. We anticipate that VegAnn will help improving segmentation algorithm performances, facilitate benchmarking and promote large-scale crop vegetation segmentation research.



## ⏳ Useful information <a name="Useful information"></a>

## Baseline Results <a name="base"></a>
15/02/2023 Dataset is now open

## 📝 Citing

If you find this dataset useful, please cite:

@article{madec2023,
  title={VegAnn: Vegetation Annotation of multi-crop RGB images acquired under diverse conditions for segmentation},
  author={Madec, Simon  and Irfan, Kamran and Velumani, Kaaviya and Baret, Frederic and David, Etienne  and Daubige, Gaetan  and Samatan, Lucas   and Serouart, Mario and Smith, Daniel  and James, Chris  and Camacho, Fernando  and Guo, Wei and De Solan, Benoit  and Chapman, Scott and Weiss, Marie },
  url={https://doi.org/10.5281/zenodo.7636408},
  year={2023}
}
#### Paper <a name="Paper"></a>
In review stage


## ☸️ How to use

