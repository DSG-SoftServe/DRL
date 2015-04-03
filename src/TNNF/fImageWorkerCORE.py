# ---------------------------------------------------------------------#
from PIL import Image, ImageOps, ImageFilter
import numpy as np
from numpy import *
from fDataWorkerCORE import *
# ---------------------------------------------------------------------#


class Graphic(object):
    @staticmethod
    def PicSaver(img, folder, name, color="L"):  # Saves picture to folder. Color "L" or "RGB"
        imsave = Image.fromarray(DataMutate.Normalizer(img))  # Normalizer(img).astype('uint8') for RGB
        imsave = imsave.convert(color)
        imsave.save(folder + name + ".jpg", "JPEG", quality=100)

# ---------------------------------------------------------------------#