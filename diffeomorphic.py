import argparse
from math import pi
import numpy as np
import pathlib as pl
from PIL import Image
from scipy.interpolate import interpn, RectBivariateSpline
from scipy.ndimage import rotate


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
                    # If the image didn't already have transparency,
                    # set this to true so resources are not wasted diffeomorphing
                    # an empty channel
                    self._no_transparency: bool = True
                except ValueError as ve:
                    print("ERROR: This image could not be converted")
                    # This really should never run. The only way it would is if the user supplies an image of an accepted type but its data is broken.
                    raise ve
            else:
                self._no_transparency: bool = False

            self._original: Image.Image = im

            # Increase image size per original script
            if upscale == True:
                self._upscaled = im.resize(
                    (im.width * 2, im.height * 2),
                    resample=Image.Resampling.BILINEAR,  # Linear interpolation was used in the experiment
                )
                # Copy to avoid array being read-only. Should not be a massive performance hit.
                self._image_matrix: np.ndarray = np.asarray(self._upscaled).copy()
                self._width: int = self._upscaled.width
                self._height: int = self._upscaled.height
            else:
                self._image_matrix: np.ndarray = np.asarray(im).copy()
                self._width: int = im.width
                self._height: int = im.height
        print(self._image_matrix[:, :, 3], self._no_transparency)
        self._maxdistortion: int = maxdistortion
        self._nsteps: int = nsteps

    def _getdiffeo(self):
        """
        This function is a one-to-one recreation of the getdiffeo function from
        the original script using NumPy.

        In the interest of saving memory, this function does not return a 3D matrix
        of the form fields like in the original, instead storing xin and yin in the
        x_diffeo_field and y_diffeo_field attributes. This avoids copying the fields to
        another array just to be indexed in interpolate_image().
        """
        ncomp: int = 6  # Amount of computations, 6 is used in original script
        # This is transformed to make the flow field
        # mesh[0] is XI, mesh[1] is YI
        mesh: np.ndarray = np.mgrid[0 : self._width, 0 : self._height]

        # Create diffeomorphic warp field by adding random discrete cosine transformations
        phase: np.ndarray = MatrixImage._rand.random(size=(ncomp, ncomp, 4)) * 2 * pi
        # Separate amplitudes for x and y were implemented in an update to the original script.
        amplitude_a: np.ndarray = MatrixImage._rand.random(size=(ncomp, ncomp)) * 2 * pi
        amplitude_b: np.ndarray = MatrixImage._rand.random(size=(ncomp, ncomp)) * 2 * pi
        xn: np.ndarray = np.zeros((self._width, self._height))
        yn: np.ndarray = np.zeros((self._width, self._height))

        # The main form field generation
        for xc in range(ncomp):
            for yc in range(ncomp):
                xn += (
                    amplitude_a[xc, yc]
                    * np.cos(xc * mesh[1] / self._width * 2 * pi + phase[xc, yc, 0])
                    * np.cos(yc * mesh[0] / self._height * 2 * pi + phase[xc, yc, 1])
                )
                yn += (
                    amplitude_b[xc, yc]
                    * np.cos(xc * mesh[0] / self._width * 2 * pi + phase[xc, yc, 2])
                    * np.cos(yc * mesh[1] / self._height * 2 * pi + phase[xc, yc, 3])
                )
        print(f"xn = {xn}, yn = {yn}")
        # Normalize to root mean square of warps in each direction
        # ravel creates a vectorized array for the squaring operation.
        xn = xn / np.sqrt(np.mean(xn.ravel() ** 2))
        yn = yn / np.sqrt(np.mean(yn.ravel() ** 2))

        xin: np.ndarray = self._maxdistortion * xn / self._nsteps
        yin: np.ndarray = self._maxdistortion * yn / self._nsteps
        print(f"xn post-RMS = {xin}, yn post-RMS = {yin}")
        print(f"Max = {xin.max()}, Min = {xin.min()}")

        self._x_diffeo_field: np.ndarray = xin
        self._y_diffeo_field: np.ndarray = yin

    # def _map_diffeo(self, output_coordinates: tuple) -> tuple:
    #     """
    #     Used in interpolation to map output coordinates to input coordinates.
    #
    #     Uses instance variable _transform_iterable to track iterations, reset for each channel.
    #     """
    #     input_coordinates: tuple = (
    #         output_coordinates[1]
    #         + self._y_diffeo_field.ravel()[self._transform_iterable],
    #         output_coordinates[0]
    #         + self._x_diffeo_field.ravel()[self._transform_iterable],
    #     )
    #     # print("ti =", self._transform_iterable)
    #     print(input_coordinates)
    #     self._transform_iterable += 1
    #     return input_coordinates

    def _interpolate_image(self):
        """
        Gets matrix of getdiffeo's xin, yin to cx, yx.

        The original script creates a figure displaying a continuous circle of images generated in steps.
        Each quadrant of this circle relative to the beginning represents a new method of diffeomorphic scrambling,
        generating four sets of images based on nsteps. However, there is no evidence in the paper or in future
        revisions of the script that any images other than those using a
        basic interpolation using varying degrees of distortion were used in the Mturk perceptual ratings or in the
        HMAX model.

        Therefore, this script uses only the basic implementation, rather than generating multiple flow fields and
        making other transformations (for example, in quadrant 2, the new interpolation would be done using cx = cxf - cxa, etc.)

        Uses scipy.interpolate's griddata() for interpolation.
        """
        cy: np.ndarray = self._y_diffeo_field
        cx: np.ndarray = self._x_diffeo_field
        # Add meshgrid (calculate points based on vectors generated by _getdiffeo()) and apply mask
        mesh = np.mgrid[0 : self._width, 0 : self._height]
        cy = mesh[1] + cy
        cx = mesh[0] + cx
        # Make sure interpolation points are not out-of-bounds
        mask = (cx < 1) | (cx > self._width) | (cy < 1) | (cy > self._height)
        cy[mask] = 1
        cx[mask] = 1

        x_grid = np.arange(0, self._width)
        y_grid = np.arange(0, self._height)
        # Images will always be output as RGBA, so we have four channels here.
        interp_image: np.ndarray = np.empty((self._width, self._height, 4))
        bg_fill: int = 255
        # Original script casts to double-might need to do this as well to preserve precision
        if self._no_transparency == True:
            print("Using _no_transparency = True")
            channel_range: range = range(3)  # Omit transparency layer
            interp_image[:, :, 3] = self._image_matrix[:, :, 3]
        else:
            channel_range: range = range(4)

        # Interpolate using griddata
        for channel in channel_range:
            # Debug
            print(
                f"self._width = {self._width}, self._height = {self._height}, cx.size = {cx.size}, cy.size = {cy.size}"
            )
            print(
                f"cx.ravel().size = {cx.ravel().size}, cy.ravel().size = {cy.ravel().size}, self._image_matrix[:, :, channel] = {self._image_matrix[:, :, channel]}, self._image_matrix[:, :, channel].ravel().shape = {self._image_matrix[:, :, channel].ravel().shape}"
            )
            # print(
            #     f"mesh.size = {mesh.size}, mesh.shape = {mesh.shape}, mesh[0].ravel().shape = {mesh[0].ravel().shape}"
            # )
            print(f"Channel range = {channel_range}")

            print("Status: Interpolating channel", channel)
            print("Channel array:", self._image_matrix[:, :, channel])
            print("Points length:", len((cy.ravel(), cx.ravel())[0]))
            # print("Mesh length:", len((mesh[1].ravel(), mesh[0].ravel())[0]))
            print("Image matrix length:", len(self._image_matrix[:, :, channel][0]))
            # test points = points.asanyarray() or whatever. test shape, size, etc. based on griddata source code
            # Create interpolater
            interpolater = RectBivariateSpline(
                y_grid, x_grid, self._image_matrix[:, :, channel], kx=1, ky=1
            )
            # Evaluate at points defined by diffeo field
            interp_image[:, :, channel] = interpolater(
                cy.ravel(), cx.ravel(), grid=False
            ).reshape(self._width, self._height)
            # Rotate clockwise since output will be rotated due to allowing for non-square images.

            # interp_image[:, :, channel] = interpn(
            #     (x_grid, y_grid),
            #     self._image_matrix[:, :, channel],
            #     (cx.ravel(), cy.ravel()),
            #     method="linear",
            #     bounds_error=False,
            #     fill_value=bg_fill,
            # ).reshape(self._height, self._width)
            print("Min Value:", np.min(interp_image[:, :, channel]))
            print("Max Value:", np.max(interp_image[:, :, channel]))

        # Clip values to [0, 255] TEST - may or may not be necessary
        interp_image = np.clip(interp_image, 0, 255)

        return interp_image.astype(np.uint8)  # can be used with Image.fromarray()

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
        # Only get generate 1 diffeomorphic form field; reasons listed in _interpolate_image()
        self._getdiffeo()
        diffeo_im: np.ndarray = self._interpolate_image()
        im: Image.Image = Image.fromarray(diffeo_im)
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
        form fields, rather than accessing them independently. This attribute exists for this
        purpose, but will use more memory as a result.
        """
        if self._x_diffeo_field and self._y_diffeo_field:
            return np.dstack((self._x_diffeo_field, self._y_diffeo_field))
        else:
            raise AttributeError(
                "This image has not been diffeomorphed; there is no diffeomorphic form field to access"
            )


class MatrixImageDir:
    def __init__(
        self, inputs: list, output_dir: pl.Path, maxdistortion: int, nsteps: int
    ):
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

        # Updates self._images dict and initializes MatrixImage objects
        for input in self._inputs:
            if input.is_file():
                # Logic for an input file
                if input.suffix in self._accepted_file_types:
                    self._images |= {
                        input: MatrixImage(input, maxdistortion, nsteps)
                    }  # Path: MatrixImage
                else:
                    raise TypeError(
                        f"Unrecognized file type. Supported file types are: {self._accepted_file_types}"
                    )
            else:
                # Logic for an input dir
                for filepath in input.iterdir():
                    if filepath.suffix in self._accepted_file_types:
                        self._images |= {
                            filepath: MatrixImage(filepath, maxdistortion, nsteps)
                        }
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
        default=1,
        help="The amount of gradual steps to generate. Mturk perceptual ratings based on nsteps = 20; however, defaults to 1 to just generate the final image.",
    )
    return parser.parse_args()


def run_diffeomorph(inputs: list, output_dir: pl.Path, maxdistortion: int, nsteps: int):
    """
    Uses an input directory or file and an output directory to find images to run the diffeomorphic scrambling on.
    """
    imdir = MatrixImageDir(inputs, output_dir, maxdistortion, nsteps)
    # Run the diffeomorph for every image in directory
    imdir.diffeomorph()
    imdir.save()


# def savetest(inputs: list, output_dir: pl.Path):
#     imdir = MatrixImageDir(inputs, output_dir, args.maxdistortion, args.nsteps)
#     imdir.upscale()
#     imdir.save()


def main():
    args = setup()
    run_diffeomorph(args.inputs, args.output_dir, args.maxdistortion, args.nsteps)
    # savetest(args.inputs, args.output_dir)


if __name__ == "__main__":
    main()
else:
    args = setup()  # For if args are supplied to script as a library
