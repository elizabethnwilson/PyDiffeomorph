import PySimpleGUI as sg
import diffeomorphic as diffeo
import pathlib as pl
import functools
from inspect import unwrap


def progressupdate(
    func,
    pbar_key: str,
    label_key: str,
    values: dict,
    stage_progress: int,
    num_files: int,
    label: str,
    runs_each_file: bool = False,
):
    """
    Dynamically updates progress bar's status based on number of files.

    stage_progress is the value the progress bar should be in after each
    function has run for the last time. Combine with runs_each_file
    to have progress increment proportionally based on completion of each file.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper


sg.theme("SystemDefault")

layout = [
    [sg.T("")],
    [
        sg.Text("Select files and/or folders to diffeomorph: "),
        sg.Input(key="-INPUTS-", disabled=True),
        sg.FilesBrowse(),
    ],
    [sg.T("")],
    [
        sg.Text("Select an output folder: "),
        sg.Input(key="-OUTPUT-", disabled=True),
        sg.FolderBrowse(),
    ],
    [sg.T("")],
    [
        sg.Text("Set maxdistortion (default is 80): "),
        sg.Input(default_text="80", key="-MAXDISTORTION-", enable_events=True),
    ],
    [
        sg.Text("Set nsteps (default is 20): "),
        sg.Input(default_text="20", key="-NSTEPS-", enable_events=True),
    ],
    [sg.Checkbox("Save each step? ", key="-SAVE_STEPS-")],
    [sg.Checkbox("Disable upscaling?", key="-NO_UPSCALING-")],
    [sg.Text("", key="-ERROR-")],
    [sg.Button("Run")],
]

window = sg.Window("PyDiffeomorph", layout)

while True:
    event, values = window.read()

    if event == sg.WIN_CLOSED:
        break

    # Only allow digits in maxdistortion and nsteps input fields
    if event == "-MAXDISTORTION-":
        if values["-MAXDISTORTION-"][-1] not in ("0123456789"):
            window["-MAXDISTORTION-"].update(values["-MAXDISTORTION-"][:-1])
    if event == "-NSTEPS-":
        if values["-NSTEPS-"][-1] not in ("0123456789"):
            window["-NSTEPS-"].update(values["-NSTEPS-"][:-1])

    if event == "Run":
        # Debug
        print(values)
        # Check that files/folders are supplied for program to run
        if not values["-INPUTS-"] and not values["-OUTPUT-"]:
            window["-ERROR-"].update(
                value="ERROR: One or more files/folders must be supplied as an input; exactly one folder must be supplied as an output",
                text_color="red",
            )
            # Skips diffeo when missing values
            continue
        elif not values["-INPUTS-"]:
            window["-ERROR-"].update(
                value="ERROR: One or more files/folders must be supplied as an input",
                text_color="red",
            )
            continue
        elif not values["-OUTPUT-"]:
            window["-ERROR-"].update(
                value="ERROR: Exactly one folder must be supplied as an output",
                text_color="red",
            )
            continue

        # Clear error
        window["-ERROR-"].update(value="")

        _, progress_values = sg.Window(
            "Running diffeomorph...",
            [
                [sg.Text("", key="-PBARLABEL-")],
                [sg.ProgressBar(100, orientation="horizontal", key="-PBAR-")],
            ],
        ).read(close=True)

        inputs: list = [pl.Path(file) for file in values["-INPUTS-"].split(";")]
        output_dir: pl.Path = pl.Path(values["-OUTPUT-"])
        maxdistortion: int = int(values["-MAXDISTORTION-"])
        nsteps: int = int(values["-NSTEPS-"])
        save_steps: bool = values["-SAVE_STEPS-"]
        upscale: bool = not values["-NO_UPSCALING-"]

        num_files: int = len(inputs)

        # Wrap functions that should update progress bar when run
        diffeo.DiffeoImage.__init__ = progressupdate(
            diffeo.DiffeoImage.__init__, "-PBAR-", "-PBARLABEL-", progress_values, 5
        )
        diffeo.DiffeoImage._getdiffeo = progressupdate(
            diffeo.DiffeoImage._getdiffeo, "-PBAR-", "-PBARLABEL-", progress_values, 20
        )
        diffeo.DiffeoImage._interpolate_image = progressupdate(
            diffeo.DiffeoImage._interpolate_image,
            "-PBAR-",
            "-PBARLABEL-",
            progress_values,
            40,
        )
        diffeo.DiffeoImageDir.save = progressupdate(
            diffeo.DiffeoImageDir.save, "-PBAR-", "-PBARLABEL-", progress_values, 80
        )

        diffeo.run_diffeomorph(
            inputs,
            output_dir,
            maxdistortion,
            nsteps,
            save_steps,
            upscale,
        )

        # Unwrap functions for next iteration of loop since values can change
        # and to avoid wrapping a wrapped function again
        diffeo.DiffeoImage.__init__ = unwrap(diffeo.DiffeoImageDir.__init__)
        diffeo.DiffeoImage._getdiffeo = unwrap(diffeo.DiffeoImage._getdiffeo)
        diffeo.DiffeoImage._interpolate_image = unwrap(
            diffeo.DiffeoImage._interpolate_image
        )
        diffeo.DiffeoImageDir.save = unwrap(diffeo.DiffeoImageDir.save)