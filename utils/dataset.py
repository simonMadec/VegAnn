from torch.utils.data import Dataset as BaseDataset
import numpy as np
import pandas as pd
import cv2
from pathlib import Path
from segmentation_models_pytorch.encoders import get_preprocessing_fn


class DatasetVSEG(BaseDataset):
    def __init__(
            self,
            images_dir,
            tvt,
            split,
            Species = None,
            System = None,
            Orientation = None,
            preprocess = None
    ):

        df= pd.read_csv( Path(images_dir) / str(Path(images_dir).stem + ".csv"),delimiter=";")
        self.metadata = df[ df[f"TVT-split{split}"] == tvt]
        
        if not not Species:
            if Species[0]=="All":
                print("all species")
            else:
                self.metadata = self.metadata [self.metadata["Species"].isin(Species)]

        if not not System:
            self.metadata = self.metadata [self.metadata["System"].isin(System)]

        if not not Orientation:
            self.metadata = self.metadata [self.metadata["Orientation"].isin(Orientation)]

        self.preprocess = preprocess
        self.pathin = images_dir

        SPEC = set(self.metadata["Species"].tolist())
        print(F"{tvt} : species is : {SPEC}")

    def __getitem__(self, i):

        imname = self.metadata.iloc[i]["Name"]
        image = cv2.imread(str(Path(self.pathin) / "images" / imname))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if not not self.preprocess:
            image = self.preprocess(image)
            image = image.astype('float32')
        image = image.transpose(2,0,1)
        system =self.metadata.iloc[i]["System"]
        dataset =self.metadata.iloc[i]["Dataset-Name"]
        species =self.metadata.iloc[i]["Species"]
        # we divide by 255 to have 0 1 values 0 is background
        mask = cv2.imread(str(Path(self.pathin) / "annotations" / imname),cv2.IMREAD_GRAYSCALE)/255
        
        return {'image': image, 'mask': mask[..., np.newaxis].transpose(2,0,1), 'system': system, 'dataset': dataset, "species": species} 

    def __len__(self):
        return len(self.metadata)

    def species(self):
        return self.metadata["Species"].unique().tolist()