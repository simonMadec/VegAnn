import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from utils.dataset import DatasetVSEG
import time
from utils.model import VSEGModel
import pytorch_lightning as pl
from random import randrange
from segmentation_models_pytorch.encoders import get_preprocessing_fn
from utils.visu import colorTransform_VegGround

preprocess_input = get_preprocessing_fn('resnet34', pretrained='imagenet')

acc = []
iou = []
f1 = []
acc_im = []
iou_im = []
f1_im = []

for split in range(1,6):
    print(f"split {split}")
    train_dataset = DatasetVSEG(images_dir = "/home/simon/DATA/VSEG/VSEG2308",preprocess = preprocess_input, tvt="Training",split=split)    
    test_dataset = DatasetVSEG(images_dir = "/home/simon/DATA/VSEG/VSEG2308", preprocess = preprocess_input,tvt="Test",split=split)
    valid_dataset = DatasetVSEG(images_dir = "/home/simon/DATA/VSEG/VSEG2308", preprocess = preprocess_input,tvt="Validation",split=split)    
    print(f"Train size: {len(train_dataset)}")
    print(f"Test size: {len(test_dataset)}")
    print("loading dataset ..")
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True,pin_memory=False, num_workers=10) # 8 ok
    valid_dataloader = DataLoader(valid_dataset, batch_size=16, shuffle=False,pin_memory=False, num_workers=10) # 8 ok
    test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False,pin_memory=False, num_workers=10) # 8 ok
    print("finish dataset ..")

    start_time = time.time()
    first = 0

    # save 1 png image to check annotation 
    sample = train_dataset[randrange(len(train_dataset))]
    plt.subplot(1,2,1)
    plt.imshow(sample["image"].transpose(1, 2, 0)) # for visualization we have to transpose back to HWC
    plt.subplot(1,2,2)
    plt.imshow(sample["mask"].squeeze())  # for visualization we have to remove 3rd dimension of mask
    plt.show()
    plt.savefig("1.png")

    # initialize model
    model = VSEGModel("Unet", "resnet34", in_channels=3, out_classes=1)

    # training configuration
    trainer = pl.Trainer(
        gpus=1, 
        max_epochs=15,
        # early_stop_callback=EarlyStopping(monitor="valid_dataset_iou", patience=3, verbose=True, mode="max"),
        log_every_n_steps=1,
    )

    # data loaders
    trainer.fit(
        model, 
        train_dataloaders=train_dataloader, 
        val_dataloaders=valid_dataloader,
    )

    # run test dataset on VegAN
    test_metrics = trainer.test(model, dataloaders=test_dataloader, verbose=False)
    
    # store Accuracy and IOU
    acc.append(test_metrics[0]['test_dataset_acc'])
    iou.append(test_metrics[0]['test_dataset_iou'])
    f1.append(test_metrics[0]['test_dataset_f1'])

    acc_im.append(test_metrics[0]['test_per_image_acc'])
    iou_im.append(test_metrics[0]['test_per_image_iou'])
    f1_im.append(test_metrics[0]['test_per_image_f1'])



# results to CSV
dd = {"OA-dt":[np.mean(acc)],"IOU-dt":[np.mean(iou)],"f1-dt":[np.mean(f1)],"OA-dt_std":[np.std(acc)],"IOU-dt_std":[np.std(iou)],"f1-dt_std":[np.std(f1)],
    "OA-im":[np.mean(acc_im)],"IOU-im":[np.mean(iou_im)],"f1-im":[np.mean(f1_im)],"OA-im_std":[np.std(acc_im)],"IOU-im_std":[np.std(iou_im)],"f1-im_std":[np.std(f1_im)]}
    
df = pd.DataFrame(data=dd)

dst = f"results_resnet_fullmetrics.csv"
df.to_csv(path_or_buf=str(dst),index=False)


