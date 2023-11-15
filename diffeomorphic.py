#!/usr/bin/python
import argparse
import math
import numpy as np
import pathlib as pl
from PIL import Image
from scipy import griddata  # For interpolation


class MatrixImage:
    # Random-number generator for whole class so we don't re-seed every image
    _rand = np.random.default_rng()

    def __init__(self, filepath: pl.Path, maxdistortion, nsteps, upscale=True):
        """
        Initializes a MatrixImage object with necessary properties for diffeomorphic scrambling.

        Although the original study uses upscaling, it can be disabled here if using as a library
        in order for flexibility; not all uses of diffeomorphic scrambling may demand upscaling beforehand.
        """
        with Image.open(filepath) as im:
            if im.mode != "RGBA":
                # Images have 3 color channels and 1 transparency channel. All images will be output with a transparency channel, saved as pngs.
                try:
                    im.convert("RGBA")
                except ValueError as ve:
                    print("ERROR: This image could not be converted")
                    # This really should never run. The only way it would is if the user supplies an image of an accepted type but its data is broken.
                    raise ve
            self._original = im
            # Increase image size per original script
            if upscale == True:
                self._upscaled = im.resize(
                    (im.width * 2, im.height * 2),
                    resample=Image.Resampling.BILINEAR,  # Linear interpolation was used in the experiment
                )
                self._image_matrix = np.asarray(self._upscaled)
                self._width = self._upscaled.width
                self._height = self._upscaled.height
            else:
                self._image_matrix = np.asarray(im)
                self._width = im.width
                self._height = im.height

        self._maxdistortion = maxdistortion
        self._nsteps = nsteps

    def _getdiffeo(self) -> np.ndarray:
        """
        This function is a one-to-one recreation of the getdiffeo function from
        the original script using NumPy.
        """
        ncomp: int = 6  # Amount of computations, 6 is used in original script
        # This is transformed to make the flow field
        mesh = np.mgrid[
            1 : self._width, 1 : self._height
        ]  # mesh[0] is YI, mesh[1] is XI

        # Create diffeomorphic warp field by adding random discrete cosine transformations
        phase = MatrixImage._rand.random(size=(ncomp, ncomp, 4)) * 2 * math.pi
        amplitude_a = MatrixImage._rand.random(size=(ncomp, ncomp)) * 2 * math.pi
        # Separate amplitudes for x and y were implemented in an update to the original script.
        amplitude_b = MatrixImage._rand.random(size=(ncomp, ncomp)) * 2 * math.pi
        xn = np.zeros((self._width, self._height))
        yn = np.zeros((self._width, self._height))

        # The main form field generation
        for xc in range(1, ncomp):
            for yc in range(1, ncomp):
                xn = xn + amplitude_a[xc, yc] * math.cos(
                    xc * mesh[1] / self._width * 2 * math.pi + phase[xc, yc, 1]
                ) * math.cos(
                    yc * mesh[0] / self._height * 2 * math.pi + phase[xc, yc, 2]
                )
                yn = yn + amplitude_b[xc, yc] * math.cos(
                    xc * mesh[1] / self._width * 2 * math.pi + phase[xc, yc, 3]
                ) * math.cos(
                    yc * mesh[0] / self._height * 2 * math.pi + phase[xc, yc, 4]
                )

        # Normalize to root mean square of warps in each direction
        xn = xn / np.sqrt(
            np.mean(xn.ravel() ** 2)
        )  # ravel creates a vectorized array for the squaring operation.
        yn = yn / np.sqrt(np.mean(yn.ravel() ** 2))

        yin = self._maxdistortion * yn / self._nsteps
        xin = self._maxdistortion * xn / self._nsteps

        return np.array(xin, yin)

    def _interpolate_image(self, diffeo_field: np.ndarray):
        """
        Gets matrix of xin, yin to cx, yx.

        The original script creates a figure displaying a continuous circle of images generated in steps.
        Each quadrant of this circle relative to the beginning represents a new method of diffeomorphic scrambling,
        generating four sets of images based on nsteps. However, there is no evidence in the paper or in future
        revisions of the script that any images other than those using a
        basic interpolation using varying degrees of distortion were used in the Mturk perceptual ratings or in the
        HMAX model.

        Therefore, this script uses only the basic implementation, rather than generating multiple flow fields and
        making other transformations (for example, in quadrant 2, the new interpolation would be done using cx = cxf - cxa, etc.)

        Uses SciPy griddata for interpolation.
        """
        cx = diffeo_field[0]
        cy = diffeo_field[1]

        """
        Example:
        Modify based on already having flow fields and other variables.
            # Generate a regular grid
        XI, YI = np.meshgrid(np.arange(1, imsz[1] + 1), np.arange(1, imsz[0] + 1))

        # Interpolate using griddata
        interp_img = griddata((cy.flatten(), cx.flatten()), img.reshape(-1, img.shape[2]),
                              (YI, XI), method='linear', fill_value=0.0)

        # Clip values to [0, 255]
        interp_img = np.clip(interp_img, 0, 255)

        return interp_img.astype(np.uint8)
        """

    def _diffeomorph(self) -> Image.Image:
        """
        Returns the result of the diffeomorphic scrambling to be saved.
        Done on demand instead of in __init__() to allow for flexibility
        and potentially more efficient memory usage.

        Uses _getdiffeo()'s form field to warp the _image_array created in __init__().
        This is split into multiple functions to resemble the original script.

        Ability to return each step may be needed since the original script
        does this by default.
        """
        # Only get one diffeomorphic form field; reasons listed in _interpolate_image()
        self._diffeo_field: np.ndarray = self._getdiffeo()
        diffeo_im: np.ndarray = self._interpolate_image(self._diffeo_field)
        im = Image.fromarray(diffeo_im)
        return im

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

    @property
    def diffeo_field(self) -> np.ndarray:
        if self._diffeo_field:
            return self._diffeo_field
        else:
            raise AttributeError(
                "This image has not been diffeomorphed; there is no diffeomorphic form field to access"
            )


class MatrixImageDir:
    def __init__(self, inputs: list, output_dir: pl.Path):
        """
        Until an operation has been run on self._images, the values are entire MatrixImage objects.
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
                    self._images |= {input: MatrixImage(input)}  # Path: MatrixImage
                else:
                    raise TypeError(
                        f"Unrecognized file type. Supported file types are: {self._accepted_file_types}"
                    )
            else:
                # Logic for an input dir
                for filepath in input.iterdir():
                    if filepath.suffix in self._accepted_file_types:
                        self._images |= {filepath: MatrixImage(filepath)}
                    else:
                        raise TypeError(
                            f"Unrecognized file type. Supported file types are: {self._accepted_file_types}"
                        )

    def diffeomorph(self):
        for filepath, file in self._images.items():
            self._images |= {filepath: file.diffeomorphed}

    def upscale(self):
        """
        MatrixImages are upscaled by default due to the specifications in the original research paper.
        This behavior can be modified by setting upscale=False when instantiating an MatrixImage (such as if you use MatrixImage in a library).
        """
        for filepath, file in self._images.items():
            self._images |= {filepath: file.upscaled}

    def save(self):
        """
        Save all files in the output directory
        """
        for filepath, file in self._images.items():
            if type(file) == MatrixImage:
                raise TypeError(
                    "No operation has been run on these files; they will not be saved."
                )
            file.save(
                pl.Path(f"{self._output_dir}/diffeomorphed-{str(filepath.name)}")
            )  # Make it only pngs

    @property
    def images(self) -> dict:
        return self._images


def setup():
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
    imdir = MatrixImageDir(inputs, output_dir)
    # Run the diffeomorph for every image in directory
    # imdir.diffeomorph()
    imdir.save()


def savetest(inputs: list, output_dir: pl.Path):
    imdir = MatrixImageDir(inputs, output_dir)
    imdir.upscale()
    imdir.save()


def main():
    args = setup()
    # run_diffeomorph(args.inputs, args.output_dir)
    savetest(args.inputs, args.output_dir)


if __name__ == "__main__":
    main()
else:
    args = setup()  # For if args are supplied to script as a library
