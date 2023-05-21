from omegaconf.dictconfig import DictConfig
from typing import Dict
from dataclasses import dataclass
import logging

import numpy as np
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.callbacks import ModelCheckpoint

from segmentation_models_pytorch.encoders import get_preprocessing_fn

from vegann.utils.vegan_dataset import DatasetVegAnn
from vegann.utils.vegan_model import VegAnnModel

logger = logging.getLogger(__name__)


class VeganTrainer:
    def __init__(self, config: DictConfig, expt_dir: str):
        self.config = config
        self.expt_dir = expt_dir
        self.modelconf = config.model
        self.dataconf = config.dataset
        self.trainconf = config.training
        self.setup_model()

    def setup_model(self) -> None:
        self.encoder_ = self.modelconf.encoder
        self.pretrained = self.modelconf.pretrained
        self.model_ = self.modelconf.name

        self.preprocess_input = get_preprocessing_fn(self.encoder_, pretrained=self.pretrained)
        self.model = VegAnnModel(self.model_, self.encoder_, in_channels=3, out_classes=1)
        self.checkpoint_callback = ModelCheckpoint(
            dirpath=self.expt_dir,
            filename="vegann-{epoch:02d}",
            monitor="valid_dataset_acc",
            every_n_epochs=self.trainconf.every_n_epochs
        )

    def setup_dataloaders(self, split_id: int) -> None:
        veganpath = self.config.dataset.VegAnn_path

        self.train_dataset = DatasetVegAnn(
            images_dir=veganpath, preprocess=self.preprocess_input, tvt="Training", split=split_id
        )
        self.test_dataset = DatasetVegAnn(
            images_dir=veganpath, preprocess=self.preprocess_input, tvt="Test", split=split_id
        )
        self.valid_dataset = DatasetVegAnn(
            images_dir=veganpath, preprocess=self.preprocess_input, tvt="Validation", split=split_id
        )

        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.trainconf.batch_size,
            shuffle=True,
            pin_memory=self.trainconf.pin_memory,
            num_workers=self.trainconf.num_workers,
            persistent_workers=True,
        )
        self.valid_dataloader = DataLoader(
            self.valid_dataset,
            batch_size=self.trainconf.batch_size,
            shuffle=False,
            pin_memory=self.trainconf.pin_memory,
            num_workers=self.trainconf.num_workers,
        )

        self.test_dataloader = DataLoader(
            self.test_dataset, batch_size=16, shuffle=False, pin_memory=False, num_workers=8
        )  # 8 ok

    def train(self):
        # training configuration
        self.trainer = pl.Trainer(
            accelerator=self.trainconf.accelerator,
            devices=self.trainconf.devices,
            max_epochs=self.trainconf.max_epochs,
            callbacks=[self.checkpoint_callback],
            # early_stop_callback=EarlyStopping(monitor="valid_dataset_iou", patience=3, verbose=True, mode="max"),
            log_every_n_steps=self.trainconf.log_every_n_steps,
            strategy=DDPStrategy(find_unused_parameters=False),
            limit_val_batches=self.trainconf.limit_val_batches,
            limit_train_batches=self.trainconf.limit_train_batches,
            limit_test_batches=self.trainconf.limit_test_batches
        )

        # data loaders
        self.trainer.fit(
            self.model,
            train_dataloaders=self.train_dataloader,
            val_dataloaders=self.valid_dataloader,
        )

    def test(self) -> Dict:
        # run test dataset on VegAN
        test_metrics = self.trainer.test(
            self.model,
            dataloaders=self.test_dataloader,
            verbose=False,
        )

        return test_metrics
