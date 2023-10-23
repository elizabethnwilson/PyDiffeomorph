#!/usr/bin/python
import argparse
from math import sin, cos  # sine, cosine used in diffeomoprhic scrambling equation
import numpy as np
from pathlib import Path
from PIL import Image


class ArrayImage:
    def __init__(self, input_image: Path):
        with Image.open(input_image).convert(
            "RGB"
        ) as im:  # Original script does not use alpha channel
            self._original_image = im
            # Increase image size per original script
            self._upscaled_image = im.resize(
                (im.width * 2, im.height * 2),
                resample=Image.Resampling.BILINEAR,  # Linear interpolation was used in the experiment
            )
            # Need to get RGB channels into an array of arrays or numpy thing

    def diffeomorph(self):
        """
        Yields the result of the diffeomorphic scrambling to be saved.
        """
        # For now, just yield the upscaled image for testing.
        yield self._upscaled_image  # We want to yield since we are going to loop over ImageDir.update()

    @property
    def original(self):
        return self._original_image

    @property
    def upscaled(self):
        return self._upscaled_image


class ImageDir:
    def __init__(self, input_dir: Path, output_dir: Path):
        self._input_dir = input_dir
        self._output_dir = output_dir
        self._image_paths: list = [
            im
            for im in input_dir.iterdir()
            if im.is_file() and str(im).endswith((".jpg", ".png", ".webp"))
        ]  # Gets all the images from user-inputted dir, add complete file type list later
        self._count = 0  # Need to keep track of iteration in __next__
        self._images = []

    def update(self, input_image: Image.Image):
        """
        To be called upon completion of an image.
        """
        self._images.append(input_image)

    def save(self):
        """
        Save all files in the output directory
        """
        for im in self._images:
            with im.open():  # Currently saves originals
                output_file: Path = Path(f"{self._output_dir}/diffeomorphed-{im.name}")
                im.save(fp=output_file)

    @property
    def image_paths(self) -> list:
        return self._image_paths


def parse_args():
    """
    Called regardless of whether script is called on its own or as a library
    since certain things need to be specified every time
    """
    parser = argparse.ArgumentParser(
        prog="diffeomorphic.py",
        description="Python implementation of Rhodri Cusack and Bobby Stojanoski's diffeomorphic scrambling MATLAB script.",
    )
    parser.add_argument("input_dir", metavar="DIRECTORY", type=Path)
    parser.add_argument("output_dir", metavar="OUTPUT_DIRECTORY", type=Path)


def main():
    parse_args()
    imdir = ImageDir(input_dir, output_dir)  # Call with results from parse_args()
    for im in imdir.image_paths:
        imdir.update(im)
        imdir.save()


if __name__ == "__main__":
    main()
