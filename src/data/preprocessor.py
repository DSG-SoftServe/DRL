#-----------------------------------------------#
# Data preprocessing class.
#-----------------------------------------------#
from binascii import unhexlify
import scipy
from PIL import Image
import numpy as np
#-----------------------------------------------#

class Preprocessor:
    def __init__(self):
        self.desired_image_size = 84   # From paper
        self.counter = 0   # For image saving
        self.coeff = 1.0   # (0,1) and (0,255) for demo
        # Debug
        #np.set_printoptions(threshold=np.nan)

    def proc(self, image_string, bnw=True, saving=False):   # ReLu if bnw False
        # Crop lines from beginning and end
        input = image_string[160 * 33 * 2 : 160 * 193 * 2]
        # Split input image string into a nparray of hex codes converted to int
        input_p = np.fromstring(unhexlify(input), dtype=np.uint8)   # TODO Scale to 0-1 if not bin
        input_p = input_p.reshape((160, 160))
        input_p = scipy.misc.imresize(input_p, (84,84))
        # Convert to bNw
        if bnw:
            fil = np.zeros(input_p.shape)
            input_p = np.ones(input_p.shape) * self.coeff * (input_p > fil)
        # Saving images
        if saving:
            img = Image.fromarray(input_p)
            img_save = img.convert('RGB')
            img_save.save("../input_img/" + str(self.counter) + ".jpg", "JPEG", quality=100)
            self.counter += 1
        
        raw_pixels = input_p
        return raw_pixels
        
#-----------------------------------------------#
