import argparse
from math import pi
import numpy as np
import pathlib as pl
from PIL import Image
from scipy.interpolate import RectBivariateSpline


class DiffeoImage:
    # Random-number generator for whole class so we don't re-seed every image
    _rand = np.random.default_rng()

    def __init__(
        self,
        filepath: pl.Path,
        maxdistortion: int,
        nsteps: int,
        save_steps: bool,
        upscale: bool = True,
    ):
        """
        Initializes a DiffeoImage object with necessary properties for diffeomorphic scrambling.

        Although the original study uses upscaling, it can be disabled here for flexibility;
        not all uses of diffeomorphic scrambling may demand upscaling beforehand.
        """

        self._maxdistortion: int = maxdistortion
        self._nsteps: int = nsteps
        self._save_steps: bool = save_steps

        with Image.open(filepath) as im:
            if im.mode != "RGBA":
                # Images have 3 color channels and 1 transparency channel.
                # All images will be output with a transparency channel, saved as pngs.
                # If the image didn't already have transparency,
                # set this to true so resources are not wasted diffeomorphing an empty channel
                self._no_transparency: bool = True
            else:
                self._no_transparency: bool = False

            self._original: Image.Image = im

            # Increase image size per original script
            if upscale == True:
                self._upscaled = im.resize(
                    (im.width * 2, im.height * 2),
                    resample=Image.Resampling.BILINEAR,  # Linear interpolation was used in the experiment
                )

                # Skip padding for square images
                if self._upscaled.width == self._upscaled.height:
                    # Avoids read-only
                    self._image_matrix: np.ndarray = np.asarray(self._upscaled).copy()
                    transparency_check_matrix: np.ndarray = self._image_matrix
                    skip_padding: bool = True
                else:
                    unpadded_image_matrix: np.ndarray = np.asarray(
                        self._upscaled
                    ).copy()
                    transparency_check_matrix: np.ndarray = unpadded_image_matrix
                    skip_padding: bool = False

                # Use longer side for square image size so no data is lost
                if self._upscaled.width >= self._upscaled.height:
                    self._image_size: int = self._upscaled.width
                else:
                    self._image_size: int = self._upscaled.height
            else:
                if im.width == im.height:
                    self._image_matrix: np.ndarray = np.asarray(im).copy()
                    transparency_check_matrix: np.ndarray = self._image_matrix
                    skip_padding: bool = True
                else:
                    unpadded_image_matrix: np.ndarray = np.asarray(im).copy()
                    transparency_check_matrix: np.ndarray = unpadded_image_matrix
                    skip_padding: bool = False

                if im.width >= im.height:
                    self._image_size: int = im.width
                else:
                    self._image_size: int = im.height

        # Checks for not having alpha channel or alpha channel being fully opaque
        if transparency_check_matrix.shape[2] != 4 or np.all(
            transparency_check_matrix[:, :, 3] == 255
        ):
            self._no_transparency: bool = True

        # Will omit following code for padding, exiting __init__() early
        if skip_padding == True:
            # Can only be self._image_matrix because of skip_padding value
            if self._image_matrix.shape[2] != 4:
                alpha_channel: np.ndarray = np.full(
                    (self._image_size, self._image_size), 255
                )
                self._image_matrix = np.dstack((self._image_matrix, alpha_channel))
            return

        # Pad image when not already square
        if self._no_transparency == True:
            self._image_matrix: np.ndarray = np.full(
                (self._image_size, self._image_size, 4), 255
            )  # 255 for white, 127 for grey
        else:
            # Assume if the image is transparent that the padding should also be transparent
            self._image_matrix: np.ndarray = np.zeros(
                (self._image_size, self._image_size, 4)
            )

        x_size, y_size, z_size = unpadded_image_matrix.shape

        # Ensure consistency in channels
        if z_size != 4:
            alpha_channel: np.ndarray = np.full((x_size, y_size), 255)
            unpadded_image_matrix = np.dstack((unpadded_image_matrix, alpha_channel))

        if x_size > y_size:
            # Only pad vertically if width is greater than height
            image_start: int = round((self._image_size - y_size) / 2)
            image_end: int = image_start + y_size
            self._image_matrix[:, image_start:image_end, :] = unpadded_image_matrix
        else:
            # Pad horizontally otherwise
            image_start: int = round((self._image_size - x_size) / 2)
            image_end: int = image_start + x_size
            self._image_matrix[:, image_start:image_end, :] = unpadded_image_matrix

    def _getdiffeo(self):
        """
        This function is a one-to-one recreation of the getdiffeo function from
        the original script using NumPy.

        This function does not return a matrix of the flow field like in the original,
        instead storing xin and yin in the
        x_diffeo_field and y_diffeo_field attributes.
        """
        ncomp: int = 6  # Amount of computations, 6 is used in original script
        # This is transformed to make the flow field
        # mesh[0] is XI, mesh[1] is YI
        mesh: np.ndarray = np.mgrid[0 : self._image_size, 0 : self._image_size]

        # Create diffeomorphic warp field by adding random discrete cosine transformations
        phase: np.ndarray = DiffeoImage._rand.random(size=(ncomp, ncomp, 4)) * 2 * pi
        # Separate amplitudes for x and y were implemented in an update to the original script.
        amplitude_a: np.ndarray = DiffeoImage._rand.random(size=(ncomp, ncomp)) * 2 * pi
        amplitude_b: np.ndarray = DiffeoImage._rand.random(size=(ncomp, ncomp)) * 2 * pi
        xn: np.ndarray = np.zeros((self._image_size, self._image_size))
        yn: np.ndarray = np.zeros((self._image_size, self._image_size))

        # The main form field generation
        for xc in range(ncomp):
            for yc in range(ncomp):
                xn += (
                    amplitude_a[xc, yc]
                    * np.cos(
                        xc * mesh[1] / self._image_size * 2 * pi + phase[xc, yc, 0]
                    )
                    * np.cos(
                        yc * mesh[0] / self._image_size * 2 * pi + phase[xc, yc, 1]
                    )
                )
                yn += (
                    amplitude_b[xc, yc]
                    * np.cos(
                        xc * mesh[0] / self._image_size * 2 * pi + phase[xc, yc, 2]
                    )
                    * np.cos(
                        yc * mesh[1] / self._image_size * 2 * pi + phase[xc, yc, 3]
                    )
                )
        # Normalize to root mean square of warps in each direction
        xn = xn / np.sqrt(np.mean(xn.ravel() ** 2))
        yn = yn / np.sqrt(np.mean(yn.ravel() ** 2))

        self._x_diffeo_field: np.ndarray = self._maxdistortion * xn / self._nsteps
        self._y_diffeo_field: np.ndarray = self._maxdistortion * yn / self._nsteps

    def _interpolate_image(self):
        """
        Gets matrix of getdiffeo's xin, yin to cx, cy.

        The original script creates a figure displaying a continuous circle of images generated in steps.
        Each quadrant of this circle relative to the beginning represents a new method of diffeomorphic scrambling,
        generating four sets of images based on nsteps. However, there is no evidence in the paper or in future
        revisions of the script that any images other than those using a
        basic interpolation using varying degrees of distortion were used in the Mturk perceptual ratings or in the
        HMAX model.

        Therefore, this script uses only the basic implementation, rather than generating multiple flow fields and
        making other transformations (for example, in quadrant 2, the new interpolation would be done using cx = cxf - cxa, etc.)

        Uses scipy.interpolate's RectBivariateSpline for interpolation.
        """
        if self._save_steps == True:
            steps: list = []

        cy: np.ndarray = self._y_diffeo_field
        cx: np.ndarray = self._x_diffeo_field

        # Add meshgrid (calculate points based on vectors generated by _getdiffeo()) and apply mask
        mesh = np.mgrid[0 : self._image_size, 0 : self._image_size]
        cy = mesh[1] + cy
        cx = mesh[0] + cx

        # Make sure interpolation points are not out-of-bounds
        mask = (cx < 1) | (cx > self._image_size) | (cy < 1) | (cy > self._image_size)
        cy[mask] = 1
        cx[mask] = 1

        x_grid = np.arange(0, self._image_size)
        y_grid = np.arange(0, self._image_size)

        # Images will always be output as RGBA; they always have four channels.
        if self._no_transparency == True:
            channel_range: range = range(3)  # Omit transparency layer
        else:
            channel_range: range = range(4)

        # Repeatedly apply diffeo field to image
        for _ in range(self._nsteps):
            # Interpolate using RectBivariateSpline
            for channel in channel_range:
                # Create interpolater
                interpolater = RectBivariateSpline(
                    x_grid, y_grid, self._image_matrix[:, :, channel], kx=1, ky=1
                )

                # Evaluate at points defined by diffeo field
                self._image_matrix[:, :, channel] = interpolater(
                    cx.ravel(), cy.ravel(), grid=False
                ).reshape(self._image_size, self._image_size)

            # Clip values to [0, 255]
            self._image_matrix = np.clip(self._image_matrix, 0, 255)

            if self._save_steps == True:
                steps.append(self._image_matrix.astype(np.uint8))

        if self._save_steps == True:
            return steps
        else:
            # Only return final result if not saving each step
            return self._image_matrix.astype(np.uint8)

    def _diffeomorph(self):
        """
        Returns the result of the diffeomorphic scrambling to be saved.
        Done on demand instead of in __init__() to allow for flexibility
        and potentially more efficient memory usage.

        Uses _getdiffeo()'s form field to warp the image matrix created in __init__().
        """
        # Only get generate 1 diffeomorphic form field; reasons listed in _interpolate_image()
        self._getdiffeo()
        if self._save_steps == False:
            im: Image.Image = Image.fromarray(self._interpolate_image())
            return im
        else:
            # This will be a list on this code path, returns a list of images
            steps = self._interpolate_image()
            return [Image.fromarray(step) for step in steps]

    @property
    def original(self) -> Image.Image:
        return self._original

    @property
    def upscaled(self) -> Image.Image:
        """
        This doesn't do what .diffeomorphed does (generate an image)
        because .upscaled is (usually) needed for .diffeomorphed and is therefore included in __init__().
        """
        if self._upscaled:
            return self._upscaled
        else:
            raise AttributeError("This image has not been upscaled")

    @property
    def diffeomorphed(self):
        return self._diffeomorph()

    @property
    def x_diffeo_field(self) -> np.ndarray:
        if self._x_diffeo_field:
            return self._x_diffeo_field
        else:
            raise AttributeError(
                "This image has not been diffeomorphed; there is no diffeomorphic form field to access"
            )

    @property
    def y_diffeo_field(self) -> np.ndarray:
        if self._y_diffeo_field:
            return self._y_diffeo_field
        else:
            raise AttributeError(
                "This image has not been diffeomorphed; there is no diffeomorphic form field to access"
            )

    @property
    def diffeo_field(self) -> np.ndarray:
        """
        If used as a library, it may be beneficial to get a 3D matrix of the diffeomorphic
        form fields, rather than accessing them independently.
        """
        if self._x_diffeo_field and self._y_diffeo_field:
            return np.dstack((self._x_diffeo_field, self._y_diffeo_field))
        else:
            raise AttributeError(
                "This image has not been diffeomorphed; there is no diffeomorphic form field to access"
            )


class DiffeoImageDir:
    _accepted_file_types: set = {".jpg", ".png", ".webp"}

    def __init__(
        self,
        inputs: list,
        output_dir: pl.Path,
        maxdistortion: int,
        nsteps: int,
        save_steps: bool,
        upscale: bool = True,
    ):
        """
        Until an operation has been run on self._images, the values are entire DiffeoImage objects.
        Afterward, they become images that can be saved.
        """
        self._inputs: list = inputs
        self._output_dir: pl.Path = output_dir
        self._save_steps: bool = save_steps
        if self._output_dir != None:
            if not self._output_dir.exists():
                self._output_dir.mkdir(parents=True)
        self._images: dict = {}

        # Updates self._images dict and initializes DiffeoImage objects
        for input in self._inputs:
            if input.is_file():
                # Logic for an input file
                if input.suffix in DiffeoImageDir._accepted_file_types:
                    self._images |= {
                        input: DiffeoImage(
                            input, maxdistortion, nsteps, save_steps, upscale
                        )
                    }  # Path: DiffeoImage
                else:
                    raise TypeError(
                        f"Unrecognized file type. Supported file types are: {DiffeoImageDir._accepted_file_types}"
                    )
            else:
                # Logic for an input dir
                for filepath in input.iterdir():
                    if filepath.suffix in DiffeoImageDir._accepted_file_types:
                        self._images |= {
                            filepath: DiffeoImage(
                                filepath, maxdistortion, nsteps, upscale
                            )
                        }
                    else:
                        raise TypeError(
                            f"Unrecognized file type. Supported file types are: {DiffeoImageDir._accepted_file_types}"
                        )

    def _save_no_steps(self):
        for filepath, file in self._images.items():
            if type(file) != Image.Image:
                raise TypeError(
                    "No operation has been run on these files; they will not be saved."
                )
            file.save(
                pl.Path(f"{self._output_dir}/diffeomorphed-{str(filepath.stem)}.png")
            )

    def _save_with_steps(self):
        """
        Saves all steps of each image its own directory in output_dir.
        """
        for filepath, steps in self._images.items():
            steps_dir: pl.Path = pl.Path(f"{self._output_dir}/{str(filepath.stem)}")
            if not steps_dir.exists():
                steps_dir.mkdir()
            step_number: int = 0
            for file in steps:
                if type(file) != Image.Image:
                    raise TypeError(
                        "No operation has been run on these files; they will not be saved."
                    )
                step_number += 1
                file.save(
                    pl.Path(
                        f"{steps_dir}/diffeomorphed-{str(filepath.stem)}-{step_number}.png"
                    )
                )

    def diffeomorph(self):
        for filepath, file in self._images.items():
            # file.diffeomorphed will be a single Image when save_steps = False,
            # otherwise it will be a list of Images
            self._images |= {filepath: file.diffeomorphed}

    def upscale(self):
        """
        DiffeoImages are upscaled by default due to the specifications in the original research paper.
        This behavior can be modified by setting upscale=False when instantiating a DiffeoImage or DiffeoImageDir
        (such as if you use DiffeoImage in a library).
        """
        for filepath, file in self._images.items():
            self._images |= {filepath: file.upscaled}

    def save(self):
        """
        Save all files in the output directory
        """
        if self._save_steps == True:
            self._save_with_steps()
        else:
            self._save_no_steps()

    @property
    def images(self) -> dict:
        return self._images


def setup():
    """
    Parses arguments when ran as a script as opposed to a library.
    Use run_diffeomorph() on its own if using as a library.
    """
    parser = argparse.ArgumentParser(
        prog="diffeomorphic.py",
        description="Python implementation of Rhodri Cusack and Bobby Stojanoski's diffeomorphic scrambling MATLAB script. NOTE: Larger images will compute extremely slowly. For best results, try not to exceed about 3000 pixels on one edge (keeping upscaling in mind).",
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
    parser.add_argument(
        "-m",
        "--maxdistortion",
        type=int,
        default=80,
        help="The maximum amount of distortion to allow. Mturk perceptual ratings based on maxdistortion = 80; defaults to 80.",
    )
    parser.add_argument(
        "-n",
        "--nsteps",
        type=int,
        default=20,
        help="The amount of gradual steps to generate. Mturk perceptual ratings based on nsteps = 20. Unlike the original script, ONLY the final step will be saved unless --save-steps is specified. Defaults to 20.",
    )
    parser.add_argument(
        "--save-steps",
        action="store_true",
        default=False,
        help="Supply this argument to the script to save the result of each nstep.",
    )
    parser.add_argument(
        "--no-upscale",
        action="store_true",
        default=False,
        help="Disable upscaling (useful for larger images).",
    )
    return parser.parse_args()


def run_diffeomorph(
    inputs: list,
    output_dir: pl.Path,
    maxdistortion: int,
    nsteps: int,
    save_steps: bool,
    upscale: bool = True,
):
    """
    Uses an input directory or file and an output directory to find images to run the diffeomorphic scrambling on.

    Supply arguments to this function if using as a library for simplest use.
    """
    imdir = DiffeoImageDir(
        inputs, output_dir, maxdistortion, nsteps, save_steps, upscale
    )
    # Run the diffeomorph for every image that will be put in output directory
    imdir.diffeomorph()
    imdir.save()


def main():
    args = setup()
    if args.no_upscale == True:
        upscale: bool = False
    else:
        upscale: bool = True
    run_diffeomorph(
        args.inputs,
        args.output_dir,
        args.maxdistortion,
        args.nsteps,
        args.save_steps,
        upscale,
    )


if __name__ == "__main__":
    main()
