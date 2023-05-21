import datetime
import glob
import os
import random
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from geojson import Feature, FeatureCollection, Point, dump
from PIL import Image
from skimage.io import imread


def colorTransform_VegGround(
    im: np.ndarray, X_true: np.ndarray, alpha_vert: float, alpha_g: float
):
    """ """

    alpha = alpha_vert
    color = [97, 65, 38]
    # color = [x / 255 for x in color]
    image = np.copy(im)
    for c in range(3):
        image[:, :, c] = np.where(
            X_true == 0, image[:, :, c] * (1 - alpha) + alpha * color[c], image[:, :, c]
        )
    alpha = alpha_g
    color = [34, 139, 34]
    #    color = [x / 255 for x in color]
    for c in range(3):
        image[:, :, c] = np.where(
            X_true == 1, image[:, :, c] * (1 - alpha) + alpha * color[c], image[:, :, c]
        )
    return image
