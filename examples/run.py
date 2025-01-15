"""_summary_."""

import pathlib
import subprocess
import sys


def main() -> None:
    """_summary_."""
    if len(sys.argv) > 1 and "html" in sys.argv[1:]:
        print("Generating HTML for the example notebooks")
        subprocess.run(  # noqa: S603
            [  # noqa: S607
                "jupyter",
                "nbconvert",
                "--to",
                "html",
                "--embed-images",
                "examples/**/*.ipynb",
            ],
            check=False,
        )
    else:
        notebooks = [str(p) for p in pathlib.Path().glob("examples/**/*.ipynb")]
        subprocess.run(  # noqa: S603
            [  # noqa: S607
                "jupyter",
                "execute",
                "--inplace",
                *notebooks,
            ],
            check=False,
        )


if __name__ == "__main__":
    main()
