#!/usr/bin/python
import argparse
from math import sin, cos  # sine, cosine used in diffeomoprhic scrambling equation
import numpy as np
import pathlib
from PIL import Image


class ArrayImage:
    def __init__(self, input_image):
        with Image.open(input_image).convert(
            "RGB"
        ) as im:  # Original script does not use alpha channel
            # Make image twice the size per original script
            upscaled_width = im.width * 2
            upscaled_height = im.height * 2
            self._original_image = im.resize(
                (upscaled_width, upscaled_height), resample=Image.Resampling.BILINEAR
            )  # Original
            # Need to get RGB channels into an array of arrays or numpy thing

    @property
    def original_image(self):
        return self._original_image


def make_new_image(input_image: Image.Image):
    input_image.save(fp=)


def main():
    pass


if __name__ == "__main__":
    main()
