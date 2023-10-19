#!/usr/bin/python
import argparse
from math import sin, cos  # sine, cosine used in diffeomoprhic scrambling equation
import numpy as np
from pathlib import Path
from PIL import Image


class ImageDir:
    def __init__(self, input_dir: Path, output_dir: Path):
        self._input_dir = input_dir
        self._output_dir = output_dir
        self._image_paths = [im for im in input_dir.iterdir() if im.is_file() and str(im).endswith((".jpg", ".png", ".webp"))] # Gets all the images from user-inputted dir, add complete file type list later
        self._images = {}
    
    def save(self):
        """
        Save all files in the output directory

        Possibly implement yield in ArrayImage or a subclass to save memory
        """
        for im_path in self._image_paths:
            with Image.open(im_path) as im:
                output_file: Path = Path(f"{self._output_dir}/diffeomorphed-{im_path.name}")
                im.save(fp=output_file)

    def update_file(self, input_image: Image.Image):
        """
        To be called upon completion of an image while it is still open with .open()
        """
        self._images.update()

class ArrayImage:
    def __init__(self, input_image: Path):
        with Image.open(input_image).convert(
            "RGB"
        ) as im:  # Original script does not use alpha channel
            self._original_image = im
            # Increase image size per original script
            upscaled_width = im.width * 2
            upscaled_height = im.height * 2
            self._upscaled_image = im.resize(
                (upscaled_width, upscaled_height), resample=Image.Resampling.BILINEAR #Linear interpolation was used in the experiment
            ) 
            # Need to get RGB channels into an array of arrays or numpy thing
    
    @property
    def original(self):
        return self._original_image

    @property
    def upscaled(self):
        return self._upscaled_image


def parse_args():
    """
    Called regardless of whether script is called on its own or as a library
    since certain things need to be specified every time
    """
    parser = argparse.ArgumentParser(
        prog="diffeomorphic.py",
        description="Python implementation of Rhodri Cusack and Bobby Stojanoski's diffeomorphic scrambling MATLAB script."
    )
    parser.add_argument("input_dir", metavar="DIRECTORY", type=Path)
    parser.add_argument("output_dir", metavar="OUTPUT_DIRECTORY", type=Path)


def make_new_image(input_image: Image.Image, output_dir: Path):


def main():
    pass


if __name__ == "__main__":
    main()
