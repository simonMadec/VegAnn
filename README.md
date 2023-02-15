

# VegAnn : Vegetation Annotation of a large multi-crop RGB Dataset acquired under diverse conditions for image semantic segmentation

![logo](https://i.ibb.co/QC6rfCG/2023-02-15-17-01-57-VSEG-Online-La-Te-X-Editor-Overleaf.png)

# Table of contents
1. [Keypoints](#key)
2. [Abstract](#abs)
3. [Pytorch Data Loader](#loader)
4. [Baseline Results](#res)
5. [Citing](#cite)
6. [Paper](#paper)
7. [Meta-Information](#meta)

## ⏳ Keypoints <a name="key"></a>

- The dataset can be accessed at https://doi.org/10.5281/zenodo.7636408.
- VegAnn contains 3775 images 
- Images are 512*512 pixels 
- Corresponding binary masks is 0 for soil + crop residues (background) 255 for Vegetation (foreground)
- The dataset includes images of 26+ crop species, which are not evenly represented
- VegAnn was constructed using various acquisition systems and configurations
- For more information about VegAnn, details and potential uses see @

## 📚 Abstract <a name="abs"></a>

  Applying deep learning to images of cropping systems provides new knowledge and insights in research and commercial applications. Semantic segmentation or pixel-wise classification, of RGB images acquired at the ground level, into vegetation and background is a critical step in the estimation of several canopy traits. Current state of the art methodologies based on convolutional neural networks (CNNs) are trained on datasets acquired under controlled or indoor environments. These models are unable to generalize to real-world images and hence need to be fine-tuned using new labelled datasets. This motivated the creation of the VegAnn - **Veg**etation **Ann**otation - dataset, a collection of 3795 multi-crop RGB images acquired for different phenological stages using different systems and platforms in diverse illumination conditions. We anticipate that VegAnn will help improving segmentation algorithm performances, facilitate benchmarking and promote large-scale crop vegetation segmentation research.

## 📦 Pytorch Data Loader <a name="loader"></a>
We provide Python dataloader that load the data as PyTorch tensors. With the dataloader, users can select desired images with the metadata information such as species, camera system, and training/validation/test sets. 

### 🍲 Example use : 

Here is an example use case of the dataloader with our custom dataset class:

```
    from segmentation_models_pytorch.encoders import get_preprocessing_fn
    from utils.dataset import DatasetVegAnn
    from torch.utils.data import DataLoader

    train_dataset = DatasetVegAnn(images_dir = veganpath,species = ["Wheat","Maize"], system = ["Handeld Cameras","Phone Camera"], tvt="Training")    
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True,pin_memory=False, num_workers=10)
```
By using this dataloader, you can easily load the desired images with metadata information as PyTorch tensors for training, validation, or testing purposes.

## 👀 Baseline Results <a name="res"></a>

Metrics are computed at the dataset level for the 5 Test sets of VegAnn

Method               | Encoder | IOU | F1 
---                  | ---  | ---   | ---                  
Unet       |   ResNet34  | 89.7 ±1.4  |  94.5 ±0.8
DeepLabV3  |   ResNet34  | 89.5 ±0.2  |  94.5 ±0.2


## 📝 Citing  <a name="cite"></a>

If you find this dataset useful, please cite:

@article{madec2023,
  title={VegAnn: Vegetation Annotation of multi-crop RGB images acquired under diverse conditions for segmentation},
  author={Madec, Simon  and Irfan, Kamran and Velumani, Kaaviya and Baret, Frederic and David, Etienne  and Daubige, Gaetan  and Samatan, Lucas   and Serouart, Mario and Smith, Daniel  and James, Chris  and Camacho, Fernando  and Guo, Wei and De Solan, Benoit  and Chapman, Scott and Weiss, Marie },
  url={https://doi.org/10.5281/zenodo.7636408},
  year={2023}
}
## 📖 Paper <a name="paper"></a>
In review stage

## ☸️ Model inference 
Docker image in construction
