#-----------------------------------------------#
# Data preprocessing class.
#-----------------------------------------------#
from binascii import unhexlify
import scipy
from PIL import Image
import numpy as np
#-----------------------------------------------#

class preprocessing:
    def __init__(self):
        self.resized_image_size = 84   # From paper
        self.counter = 0   # For image saving
        self.coeff = 1.0   # (0,1) and (0,255) for demo
        # Debug
        #np.set_printoptions(threshold=np.nan)

    def preprocess(self, image_data, bnw=False, saving=False):   # LReLu if bnw True
        # Crop image
        cropped_img = image_data[160 * 33 * 2 : 160 * 193 * 2]
        # Convert image
        pixels = np.fromstring(unhexlify(cropped_img), dtype=np.uint8)   # TODO Scale to 0-1 if not bin
        pixels = pixels.reshape((160, 160))
        pixels = scipy.misc.imresize(pixels, (84, 84))  # Try without cubic or linear int.
        # Convert to bnw
        if bnw:
            fil = np.zeros(pixels.shape)
            pixels = np.ones(pixels.shape) * self.coeff * (pixels > fil)
        # Saving images
        if saving:
            img = Image.fromarray(pixels)
            img_save = img.convert('RGB')
            img_save.save("../input_img/" + str(self.counter) + ".jpg", "JPEG", quality=100)
            self.counter += 1

        return pixels
        
#-----------------------------------------------#
