# PyDiffeomorph

This is a translation of the "diffeomorph" MATLAB script created by Stojanoski & Cusack (2014) into Python, also providing a GUI (graphical user interface) for ease of use.

This program is primarily intended for use in research for control creation. Stojanoski & Cusack (2014) has more details on the purpose of this, including why diffeomorphic scrambling has advantages over other methods for creating control images in research. A small summary of their findings is that 

Sources:
* Stojanoski, B., & Cusack, R. (2014). Time to wave goodbye to phase-scrambling: Creating controlled scrambled images using diffeomorphic transformations. *Journal of Vision*, *14*(12), 1-16. <https://s3.amazonaws.com/cusacklab/html/pdfs/2014_stojanoski_cusack_wave_goodbye.pdf>
* The original project: <https://github.com/rhodricusack/diffeomorph/>

The original project's license is provided in [this attribution file](ATTRIBUTION). This project is also released under the MIT License.

## Attribution
Seeing as this project is intended to be used in research, if you use this project in your study _please attribute me_ (and of course, the original study). If you use this project, either for research or not, I would love to hear from you! You can reach me at the email on my GitHub profile.

If you want to use my code in your project, don't hesitate to reach out to me! Or, you can simply follow the license. I would be more than happy to contribute to any projects interested in this algorithm.

## Installation
### Windows
Go to this project's [release page](https://github.com/elizabethnwilson/PyDiffeomorph/releases/latest) and download PyDiffeomorph.exe. The other file (PyDiffeomorph) will not run on your machine. Once you have downloaded the file, you can run it like any other program.

The first time you run the program, there is a chance that Windows will stop the program from running. The program is safe to run; simply click on "More info" and then "Run anyway". If you do not trust the program, you may audit my code to verify it is safe or install manually as described below.

### Linux
Go to the [release page](https://github.com/elizabethnwilson/PyDiffeomorph/releases/latest) and download PyDiffeomorph. You can run this file from your terminal (e.g., `./PyDiffeomorph` while in the same directory), or write a .desktop file for it if you want to use it like other applications. Chances are if you are downloading this, you already know what you are doing.

### Manual Installation
This program was written using `pipenv` to manage dependencies. You can install `pipenv` as described by the authors (here)[https://github.com/pypa/pipenv#installation]. Once you have it installed, do the following:

First, clone and enter the repository:
```
git clone https://github.com/elizabethnwilson/PyDiffeomorph.git
cd PyDiffeomorph
```

Next, install the dependencies:
```
pipenv install
```

If you want to build your own executable, you should instead run:
```
pipenv install -d
```
and use PyInstaller to create the executable for your platform. You may need to install additional libraries.

To run the GUI version of the program:
```
pipenv run run-gui.py
```

To run the script without GUI:
```
pipenv run diffeomorphic.py
```

## Using as a Library
I wrote this code so that it can be used in other projects. If you want to do so, simply put diffeomorphic.py in your project (and this project's LICENSE in the same directory) and import it into your project. For example,
```python
import diffeomorphic as diffeo
```
You can then use diffeomorphic's `run_diffeomorph()` function to run a full diffeomorph operation, using the parameters `inputs` (a list of pathlib Path objects), `output_dir` (a pathlib Path object), `maxdistortion` (an int), `nsteps` (an int), `save_steps` (a bool), and `upscale` (a bool).

You can also import just the `DiffeoImage` class if you wish to implement just parts of the process, or want to implement your own method of handling the images once they have been generated. The diffeomorphed image can be accessed from the object's `diffeomorphed` attribute (which effectively runs the opperation).

This section exists in lieu of proper documentation; I may create real documentation in the future if it seems necessary. You can feel free to open an issue if you run into any problems and I can try to help you resolve it.
