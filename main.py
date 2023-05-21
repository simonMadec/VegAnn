import logging
from typing import List, Dict
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from vegann.trainer import VeganTrainer

logger = logging.getLogger(__name__)


def metrics_to_csv(split_metrics: List[Dict], out_path: str):
    
    acc, iou, f1, acc_im, iou_im, f1_im = [], [], [], [], [], []

    for test_metrics in split_metrics:
        # store Accuracy and IOU
        acc.append(test_metrics[0]["test_dataset_acc"])
        iou.append(test_metrics[0]["test_dataset_iou"])
        f1.append(test_metrics[0]["test_dataset_f1"])

        acc_im.append(test_metrics[0]["test_per_image_acc"])
        iou_im.append(test_metrics[0]["test_per_image_iou"])
        f1_im.append(test_metrics[0]["test_per_image_f1"])

    # results to CSV
    dd = {
        "OA-dt": [np.mean(acc)],
        "IOU-dt": [np.mean(iou)],
        "f1-dt": [np.mean(f1)],
        "OA-dt_std": [np.std(acc)],
        "IOU-dt_std": [np.std(iou)],
        "f1-dt_std": [np.std(f1)],
        "OA-im": [np.mean(acc_im)],
        "IOU-im": [np.mean(iou_im)],
        "f1-im": [np.mean(f1_im)],
        "OA-im_std": [np.std(acc_im)],
        "IOU-im_std": [np.std(iou_im)],
        "f1-im_std": [np.std(f1_im)],
    }

    df = pd.DataFrame(data=dd)

    df.to_csv(path_or_buf=str(out_path), index=False)

def main(config_path: str):
    config = OmegaConf.load(config_path)

    n_split = config.dataset.n_split
    split_metrics = []
    
    for split_id in range(1, n_split + 1):
        # initialize vegantrainer
        Vtrainer = VeganTrainer(config=config, expt_dir=Path(config.expt_dir) / f"split_{split_id}")
        Vtrainer.setup_dataloaders(split_id=split_id)

        logger.info(f"Train size: {len(Vtrainer.train_dataset)}")
        logger.info(f"Test size: {len(Vtrainer.test_dataset)}")
        logger.info("loading dataset ..")

        # launch training
        Vtrainer.train()

        # get test metrics
        test_metrics = Vtrainer.test()
        split_metrics.append(test_metrics)
        
    out_path = Path(config.expt_dir) / f"results_{Vtrainer.model_}_{Vtrainer.encoder_}_fullmetrics.csv"
    metrics_to_csv(split_metrics, out_path=out_path)


if __name__ == "__main__":
    config_path = "./resources/config.yaml"
    main(config_path)
