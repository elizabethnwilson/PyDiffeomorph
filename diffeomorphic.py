#!/usr/bin/python
import numpy as np
from math import cos # cosine used in diffeomoprhic scrambling equation
from PIL import Image

class ArrayImage():
    def __init__(self, input_image):
        with Image.open(input_image).convert('RGB') as im: # Original script does not use alpha channel
            # Make image twice the size per original script
            upscaled_width = im.width * 2
            upscaled_height = im.height * 2
            self._original_image = im.resize((upscaled_width, upscaled_height), resample=Image.Resampling.BILINEAR) # Original
            # Need to get RGB channels into an array of arrays or numpy thing
            
    @property
    def original_image(self):
        return self._original_image
