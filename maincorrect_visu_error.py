import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from utils.dataset import DatasetVegAnn
import time
from utils.model import VegAnnModel
import pytorch_lightning as pl
from random import randrange
from segmentation_models_pytorch.encoders import get_preprocessing_fn
from utils.visu import colorTransform_VegGround
from PIL import Image
from pathlib import Path
import cv2
import os, shutil
preprocess_input = get_preprocessing_fn('resnet34', pretrained='imagenet')

acc = []
iou = []
f1 = []
acc_im = []
iou_im = []
f1_im = []

veganpath = "/home/simon/Project/segmentation/DatasetFinal/VegAnn_dataset"

for split in [1,2,3,4,5]:

    train_dataset = DatasetVegAnn(images_dir = veganpath,preprocess = preprocess_input,tvt="Training",split=split)    
    test_dataset = DatasetVegAnn(images_dir = veganpath, preprocess = preprocess_input,tvt="Test",split=split)    
    valid_dataset = DatasetVegAnn(images_dir = veganpath, preprocess = preprocess_input,tvt="Validation",split=split)

    print(f"Train size: {len(train_dataset)}")
    print(f"Test size: {len(test_dataset)}")
    print("loading dataset ..")

    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True,pin_memory=False, num_workers=10) # 8 ok False attention
    valid_dataloader = DataLoader(valid_dataset, batch_size=16, shuffle=True,pin_memory=False, num_workers=10) # 8 ok
    test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=True,pin_memory=False, num_workers=10) # 8 ok

    print("finish dataset ..")

    start_time = time.time()
    first = 0

    # initialize model
    model = VegAnnModel("Unet", "resnet34", in_channels=3, out_classes=1)

    # training configuration
    trainer = pl.Trainer(
        gpus=1, 
        max_epochs=15,
        # early_stop_callback=EarlyStopping(monitor="valid_dataset_iou", patience=3, verbose=True, mode="max"),
        log_every_n_steps=15,
    )

    # data loaders
    trainer.fit(
        model, 
        train_dataloaders=train_dataloader, 
        val_dataloaders=valid_dataloader,
    )

    for batch in iter(test_dataloader):
        with torch.no_grad():
            model.eval()
            logits = model(batch["image"])
        pr_masks = logits.sigmoid()
        
        for image, gt_mask, pr_mask, name, species in zip(batch["image"], batch["mask"], pr_masks, batch["name"], batch["species"]):
            
            imageread = cv2.imread(str(  Path(veganpath) / "images" / name))
            imageread = cv2.cvtColor(imageread, cv2.COLOR_BGR2RGB)

            pred = (pr_mask > 0.5).numpy().astype(np.uint8) 
            im1_pred = colorTransform_VegGround(imageread,pred[0],0.8,0.2)
            im2_pred = colorTransform_VegGround(imageread,pred[0],0.2,0.8)
            
            mask = cv2.imread(str(  Path(veganpath) / "annotations" / name),cv2.IMREAD_GRAYSCALE)/255
            im1_true = colorTransform_VegGround(imageread,mask,0.8,0.2)
            im2_true = colorTransform_VegGround(imageread,mask,0.2,0.8)

            nameshort = Path(name).stem + f"_{split}"  

            print(f"doing {nameshort}")
            cv2.imwrite(f"{nameshort}_image.JPG",cv2.cvtColor(imageread, cv2.COLOR_RGB2BGR))
            cv2.imwrite(f"{nameshort}_image_veg_pred.JPG",cv2.cvtColor(im1_pred, cv2.COLOR_RGB2BGR))
            cv2.imwrite(f"{nameshort}_image_ground_pred.JPG",cv2.cvtColor(im2_pred, cv2.COLOR_RGB2BGR))

            cv2.imwrite(f"{nameshort}_image_veg_true.JPG",cv2.cvtColor(im1_true, cv2.COLOR_RGB2BGR))
            cv2.imwrite(f"{nameshort}_image_ground_true.JPG",cv2.cvtColor(im2_true, cv2.COLOR_RGB2BGR))


            # plt.subplot(1,2,1)s
            # plt.imshow(im1) # for visualization we have to transpose back to HWC
            # plt.subplot(1,2,2)
            # plt.imshow(im2)  # for visualization we have to remove 3rd dimension of mask
            # plt.savefig(f"{Path(name).stem}_visumask.png")
            # mask = mask*255
            # cv2.imwrite(f"{Path(name).name}",mask.transpose(1,2,0))








