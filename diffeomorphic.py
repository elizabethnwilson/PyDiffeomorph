#!/usr/bin/python
import argparse
from math import sin, cos  # sine, cosine used in diffeomoprhic scrambling equation
import numpy as np
import pathlib as pl
from PIL import Image


class ArrayImage:
    def __init__(self, filepath: pl.Path, upscale=True):
        with Image.open(filepath).convert("RGB") as im:
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
        and potentially more efficient memory usage.
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
            raise AttributeError("This image has not been upscaled")

    @property
    def diffeomorphed(self) -> Image.Image:
        return self._diffeomorph()


class ArrayImageDir:
    def __init__(self, inputs: list, output_dir: pl.Path):
        """
        Until an operation has been run on self._images, the values are entire ArrayImage objects.
        Afterward, they become images that can be saved.
        """
        self._inputs: list = inputs
        self._output_dir: pl.Path = output_dir
        if self._output_dir != None:
            if not self._output_dir.exists():
                self._output_dir.mkdir(parents=True)
        self._accepted_file_types: set = {".jpg", ".png", ".webp"}
        self._images: dict = {}

        # Updates self._images dict
        for input in self._inputs:
            if input.is_file():
                # Logic for an input file
                if input.suffix in self._accepted_file_types:
                    self._images |= {input: ArrayImage(input)}  # Path: ArrayImage
                else:
                    raise TypeError(
                        f"Unrecognized file type. Supported file types are: {self._accepted_file_types}"
                    )
            else:
                # Logic for an input dir
                for filepath in input.iterdir():
                    if filepath.suffix in self._accepted_file_types:
                        self._images |= {filepath: ArrayImage(filepath)}
                    else:
                        raise TypeError(
                            f"Unrecognized file type. Supported file types are: {self._accepted_file_types}"
                        )

    def diffeomorph(self):
        for filepath, file in self._images.items():
            self._images |= {filepath: file.diffeomorphed}

    def upscale(self):
        """
        ArrayImages are upscaled by default due to the specifications in the original research paper.
        This behavior can be modified by setting upscale=False when instantiating an ArrayImage (such as if you use ArrayImage in a library).
        """
        for filepath, file in self._images.items():
            self._images |= {filepath: file.upscaled}

    def save(self):
        """
        Save all files in the output directory
        """
        for filepath, file in self._images.items():
            if type(file) == ArrayImage:
                raise TypeError(
                    "No operation has been run on these files; they will not be saved."
                )
            file.save(pl.Path(f"{self._output_dir}/diffeomorphed-{str(filepath.name)}"))

    @property
    def images(self) -> dict:
        return self._images


def get_args():
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
        "inputs",
        metavar="FILES_OR_DIRECTORIES",
        type=pl.Path,
        nargs="+",
        help="Give a path to one or more files or one or more directories of files you want to diffeomorph.",
    )
    parser.add_argument(
        "output_dir",
        metavar="OUTPUT_DIRECTORY",
        type=pl.Path,
        help="Specify an output directory for the new file. This is required since you can supply multiple files and directories from different locations.",
    )
    return parser.parse_args()


def run_diffeomorph(inputs: list, output_dir: pl.Path):
    """
    Uses an input directory or file and an output directory to find images to run the diffeomorphic scrambling on.
    """
    imdir = ArrayImageDir(inputs, output_dir)
    # Run the diffeomorph for every image in directory
    # imdir.diffeomorph()
    imdir.save()


def savetest(inputs: list, output_dir: pl.Path):
    imdir = ArrayImageDir(inputs, output_dir)
    imdir.upscale()
    imdir.save()


def main():
    args = get_args()
    # run_diffeomorph(args.inputs, args.output_dir)
    savetest(args.inputs, args.output_dir)


if __name__ == "__main__":
    main()
else:
    args = get_args()  # For if args are supplied to script as a library
