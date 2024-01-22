# PyDiffeomorph

This is a translation of the "diffeomorph" MATLAB script created by Stojanoski & Cusack (2014) into Python, also providing a GUI (graphical user interface) for ease of use.

This program is primarily intended for use in research for control creation. Stojanoski & Cusack (2014) has more details on the purpose of this, including why diffeomorphic scrambling has advantages over other methods for creating control images in research. A small summary of their findings is that 

Sources:
* Stojanoski, B., & Cusack, R. (2014). Time to wave goodbye to phase-scrambling: Creating controlled scrambled images using diffeomorphic transformations. *Journal of Vision*, *14*(12), 1-16. <https://s3.amazonaws.com/cusacklab/html/pdfs/2014_stojanoski_cusack_wave_goodbye.pdf>
* The original project: <https://github.com/rhodricusack/diffeomorph/>

The original project's license is provided in [this attribution file](ATTRIBUTION). This project is also released under the MIT License.

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

To run the GUI version of the program:
```
pipenv run run-gui.py
```

To run the script without GUI:
```
pipenv run diffeomorphic.py
```
