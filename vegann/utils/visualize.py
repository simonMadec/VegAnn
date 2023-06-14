
import numpy as np


# take Image as input (im) with 0 1 boolean mask (X_true) 
# and alpha_ground, alpha_veg that indicates the degree of transparency
# for ground mask (background) and vegetation mask (vegetation)

def colorTransform_VegGround(
    im: np.ndarray, X_true: np.ndarray, alpha_ground: float, alpha_veg: float
):

    color = [97, 65, 38]
    image = np.copy(im)
    for c in range(3):
        image[:, :, c] = np.where(
            X_true == 0, image[:, :, c] * (1 - alpha_veg) + alpha_veg * color[c], image[:, :, c]
        )
  
    color = [34, 139, 34]
    for c in range(3):
        image[:, :, c] = np.where(
            X_true == 1, image[:, :, c] * (1 - alpha_ground) + alpha_ground * color[c], image[:, :, c]
        )
    return image
