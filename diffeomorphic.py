#!/usr/bin/python
import argparse
from math import sin, cos  # sine, cosine used in diffeomoprhic scrambling equation
import numpy as np
from pathlib import Path
from PIL import Image


class ArrayImage:
    def __init__(self, input_image: Path, upscale=False):
        """
        Upscale can be set to true or false for flexibility if using this class as a library.
        """
        with Image.open(input_image).convert("RGB") as im:
            # Original script does not use alpha channel, so only use RGB
            self._original = im
            # Increase image size per original script
            if upscale == True:
                self._upscaled = im.resize(
                    (im.width * 2, im.height * 2),
                    resample=Image.Resampling.BILINEAR,  # Linear interpolation was used in the experiment
                )
            # Need to get RGB channels into an array of arrays or numpy thing

    def diffeomorph(self) -> Image.Image:
        """
        Returns the result of the diffeomorphic scrambling to be saved.
        Done on demand instead of in __init__() to allow for flexibility
        and possible more efficient memory usage.
        """
        ...

    @property
    def original(self) -> Image.Image:
        return self._original

    @property
    def upscaled(self) -> Image.Image:
        return self._upscaled


class ImageDir:
    def __init__(self, input_dir: Path, output_dir: Path):
        self._input_dir = input_dir
        self._output_dir = output_dir
        self._image_paths: list = [
            im_path
            for im_path in input_dir.iterdir()
            if im_path.is_file() and str(im_path).endswith((".jpg", ".png", ".webp"))
        ]  # Gets all the images from user-inputted dir, add complete file type set later
        # We only need image_paths until the images have been generated.
        self._images = []
        for path in self._image_paths:
            im = ArrayImage(path)
            self._images.append(im.upscaled)

    def save(self):
        """
        Save all files in the output directory
        """
        for im in self._images:
            with im.open():  # Currently saves originals
                output_file = Path(f"{self._output_dir}/diffeomorphed-{im.name}")
                im.save(fp=output_file)

    @property
    def image_paths(self) -> list:
        return self._image_paths

    @property
    def images(self) -> list:
        return self._images

    @images.setter
    def images(self, input_image: Image.Image):
        """
        Called every time an image is generated. The object then contains all files to be written when save() is called.
        """
        self._images.append(input_image)


# Called regardless of whether script is called on its own or as a library
# since certain things need to be specified every time
parser = argparse.ArgumentParser(
    prog="diffeomorphic.py",
    description="Python implementation of Rhodri Cusack and Bobby Stojanoski's diffeomorphic scrambling MATLAB script.",
)
# Need option to specify only one file, in which case output directory can still be specified but is assumed to be the same as the file.
# Output directory if not specified should either make a new directory "{args.input_dir}-diffeomorphed" or just use args.input_dir.
parser.add_argument("input_dir", metavar="DIRECTORY", type=Path)
parser.add_argument("output_dir", metavar="OUTPUT_DIRECTORY", type=Path)
args = parser.parse_args()


def main():
    imdir = ImageDir(args.input_dir, args.output_dir)

    for im_path in imdir.image_paths:
        # Run the diffeomorph for every image in directory
        im = ArrayImage(im_path)
        imdir.images = im.upscaled  # Currently using upscaled for testing.

    imdir.save()


if __name__ == "__main__":
    main()
