#!/usr/bin/python
import argparse
from math import sin, cos  # sine, cosine used in diffeomoprhic scrambling equation
import numpy as np
import pathlib as pl
from PIL import Image


class ArrayImage:
    def __init__(self, input_image: pl.Path, upscale=True):
        with Image.open(input_image).convert("RGB") as im:
            # Original script does not use alpha channel, so only use RGB
            self._original = im
            # Increase image size per original script
            if upscale == True:
                self._upscaled = im.resize(
                    (im.width * 2, im.height * 2),
                    resample=Image.Resampling.BILINEAR,  # Linear interpolation was used in the experiment
                )
            # Need to get RGB channels into an array of arrays or numpy thing in init because that should be the main property of ArrayImage.

    def _diffeomorph(self) -> Image.Image:
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
        """
        This doesn't do what .diffeomorphed does (generate an image)
        because .upscaled is needed for .diffeomorphed and is therefore included in __init__().
        """
        if self._upscaled:
            return self._upscaled
        else:
            raise SyntaxError("This image has not been upscaled")

    @property
    def diffeomorphed(self) -> Image.Image:
        return self._diffeomorph()


class ArrayImageDir:
    def __init__(self, input_file_or_dir: pl.Path, output_dir: pl.Path):
        """
        Until an operation has been run on self._images, the values are entire ArrayImage objects.
        Afterward, they become images that can be saved.
        """
        self._input_file_or_dir: pl.Path = input_file_or_dir
        self._output_dir: pl.Path = output_dir
        self._accepted_file_types: set = {".jpg", ".png", ".webp"}

        # Creates a dict
        if input_file_or_dir.is_file():
            # Need different logic for giving a file versus a dir
            if input_file_or_dir.suffix in self._accepted_file_types:
                self._images: dict = {
                    input_file_or_dir: ArrayImage(input_file_or_dir)
                }  # Path: ArrayImage
        else:
            for file in input_file_or_dir.iterdir():
                if file.suffix in self._accepted_file_types:
                    self._images = {file: ArrayImage(file)}

    def diffeomorph(self):
        for file in self._images:
            self._images |= {file: file.diffeomorphed}

    def upscale(self):
        """
        ArrayImages are upscaled by default due to the specifications in the original research paper.
        This behavior can be modified by setting upscale=False when instantiating an ArrayImage (such as if you use ArrayImage in a library).
        """
        for file in self._images:
            self._images |= {file: file.upscaled}

    def save(self):
        """
        Save all files in the output directory
        """
        for im in self._images:
            if type(im) == ArrayImage:
                raise TypeError(
                    "No operation has been run on these files; they will not be saved"
                )
            with im.open():
                output_file = pl.Path(f"{self._output_dir}/diffeomorphed-{im.name}")
                im.save(fp=output_file)

    @property
    def images(self) -> dict:
        return self._images


def parse_args():
    """
    Called regardless of whether script is called on its own or as a library
    since certain things need to be specified every time
    """
    parser = argparse.ArgumentParser(
        prog="diffeomorphic.py",
        description="Python implementation of Rhodri Cusack and Bobby Stojanoski's diffeomorphic scrambling MATLAB script.",
    )
    # Need option to specify only one file, in which case output directory can still be specified but is assumed to be the same as the file.
    # Output directory if not specified should either make a new directory "{args.input_dir}-diffeomorphed" or just use args.input_dir.
    parser.add_argument(
        "input_file_or_dir",
        metavar="INPUT_FILE_OR_DIRECTORY",
        type=pl.Path,
        help="Give a path to a file or a directory of files you want to diffeomorph.",
    )
    parser.add_argument(
        "output_dir",
        metavar="OUTPUT_DIRECTORY",
        type=pl.Path,
        help="Specify an output directory for the new file. Defaults to the input directory or the input file's directory.",
        default=None,  # None allows us to set it once we have opened the input file or directory
    )
    return parser.parse_args()


def run_diffeomorph(input_file_or_dir, output_dir):
    """
    Uses an input directory or file and an output directory to find images to run the diffeomorphic scrambling on.
    """
    imdir = ArrayImageDir(input_file_or_dir, output_dir)
    # Run the diffeomorph for every image in directory
    # imdir.diffeomorph()
    imdir.save()


def savetest(input_file_or_dir, output_dir):
    imdir = ArrayImageDir(input_file_or_dir, output_dir)
    imdir.upscale()
    imdir.save()


def main():
    args = parse_args()
    # run_diffeomorph(args.input_file_or_dir, args.output_dir)
    savetest(args.input_file_or_dir, args.output_dir)


if __name__ == "__main__":
    main()
else:
    args = parse_args()
