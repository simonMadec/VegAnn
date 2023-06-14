from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from segmentation_models_pytorch.encoders import get_preprocessing_fn
from torch.utils.data import Dataset as BaseDataset


class DatasetVegAnn(BaseDataset):
    def __init__(
        self,
        images_dir: str,
        tvt: str = "Training",  # Specify which set of images (TVT-split) to select (options: "Training" "Validation" or "Test")
        split: int = 1,  # Specify the split use for Training Validation or Test (integer between 1 or 5)
        species: list = [],  # Species to include in the dataset (example: ["Wheat","Maize"])
        system: list = [],  # System used to acquire the images (options: "Handeld cameras", "DHP","IOT", "UAV", "Phenomobile" or "Phone Camera")
        orientation: list = [],  # Orientation of the images (options: "Nadir", 45 or "DHP")
        alltvt: bool = False, # Whether to use all images (Training/Validation/Test) for Training
        preprocess=None,  # Preprocessing function to use when loading the images
    ):
        # Load metadata from a CSV file
        df = pd.read_csv(
            Path(images_dir) / str(Path(images_dir).stem + ".csv"), delimiter=";"
        )

        # Filter metadata based on the provided parameters
        if alltvt:
            self.metadata = df
            print(f"Selec all Data")
        else:
            self.metadata = df[df[f"TVT-split{split}"] == tvt]
            print(f"selec TVT-split{split} = {tvt}")

        if species:
            if species[0] == "All":
                print("all species")
            else:
                self.metadata = self.metadata[self.metadata["Species"].isin(species)]

        if system:
            self.metadata = self.metadata[self.metadata["System"].isin(system)]

        if orientation:
            self.metadata = self.metadata[
                self.metadata["Orientation"].isin(orientation)
            ]

        self.preprocess = preprocess
        self.pathin = images_dir

    def __getitem__(self, i):
        row = self.metadata.iloc[i]

        # Load the image and corresponding mask
        imname = row["Name"]
        image = cv2.imread(str(Path(self.pathin) / "images" / imname))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # perform preprocessing 
        if self.preprocess:
            image = self.preprocess(image)
            image = image.astype("float32")
            
        # performe HWC -> CHW transformation
        image = image.transpose(2, 0, 1)

        system = row["System"]
        dataset = row["Dataset-Name"]
        species = row["Species"]

        # we divide by 255 to have 0 1 values 0 is background
        mask = (
            cv2.imread(
                str(Path(self.pathin) / "annotations" / imname), cv2.IMREAD_GRAYSCALE
            )
            / 255
        )

        # add new axis and perform HWC->CHW transformation
        mask = mask[..., np.newaxis].transpose(2, 0, 1)

        return {
            "image": image,
            "mask": mask,
            "system": system,
            "dataset": dataset,
            "species": species,
            "name": imname,
        }

    def __len__(self):
        return len(self.metadata)
    
    # species return the set of crop species that are in the dataset
    def species(self):
        return self.metadata["Species"].unique().tolist()
