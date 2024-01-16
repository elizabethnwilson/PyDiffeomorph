import PyInstaller.__main__


def main():
    PyInstaller.__main__.run(
        ["run-gui.py", "--windowed", "--onefile", "-nPyDiffeomorph"]
    )


if __name__ == "__main__":
    main()
