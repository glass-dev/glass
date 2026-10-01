import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt  # ty: ignore[unresolved-import]

base_font_size = 17.5


def read_timings(
    path: Path,
) -> tuple[str, tuple[int, ...], dict[str, tuple[float, ...]]]:
    """Read one benchmark's timings from a CSV file.

    The CSV must have an ``nside`` column and one column per implementation,
    for example: ``nside,NumPy,JAX (CPU)``.
    """
    with path.open(newline="", encoding="utf-8") as data_file:
        reader = csv.DictReader(data_file)
        if reader.fieldnames is None or "nside" not in reader.fieldnames:
            message = f"{path}: CSV must contain an 'nside' column"
            raise ValueError(message)

        implementations = [name for name in reader.fieldnames if name != "nside"]
        if not implementations:
            message = f"{path}: CSV must contain at least one timing column"
            raise ValueError(message)

        nsides: list[int] = []
        timings = {name: [] for name in implementations}
        for line_number, row in enumerate(reader, start=2):
            try:
                nsides.append(int(row["nside"]))
                for name in implementations:
                    timings[name].append(float(row[name]))
            except (TypeError, ValueError) as error:
                message = (
                    f"{path}:{line_number}: expected an integer nside and numeric"
                    "timings"
                )
                raise ValueError(message) from error

    if not nsides:
        message = f"{path}: CSV contains no timing rows"
        raise ValueError(message)
    if any(value <= 0 for values in timings.values() for value in values):
        message = f"{path}: timings must be positive for the logarithmic plot"
        raise ValueError(message)

    return (
        path.stem.replace("-", " "),
        tuple(nsides),
        {name: tuple(values) for name, values in timings.items()},
    )


def main() -> None:
    """Generate a benchmark plot from a given list of csv files."""
    parser = argparse.ArgumentParser(
        description="Plot benchmark timings from one or more CSV files."
    )
    parser.add_argument(
        "data_files",
        nargs="+",
        type=Path,
        help="""
        CSV files, or dirs containing CSV files, with an nside column and one column
        per implementation. Note that each file name is used as the title for each sub
        plot.
        """,
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="""
        output image path (default: archer2-lensing-benchmark.png without CSVs,
        benchmark.png otherwise)
        """,
    )
    args = parser.parse_args()

    try:
        csv_files = [
            path
            for path in args.data_files
            if not path.is_dir() and path.suffixes[-1] == ".csv"
        ]
        dirs = [path for path in args.data_files if path.is_dir()]
        for dir_path in dirs:
            csv_files += [
                path
                for path in dir_path.iterdir()
                if not path.is_dir() and path.suffixes[-1] == ".csv"
            ]
        plot_data = [read_timings(path) for path in csv_files]
    except (OSError, ValueError) as error:
        parser.error(str(error))
    output_path = args.output or Path("benchmark.png")

    num_plots = len(plot_data)
    fig, axes = plt.subplots(nrows=num_plots, layout="constrained")

    for i, (machine_name, plot_nsides, timings) in enumerate(plot_data):
        ax = axes if num_plots == 1 else axes[i]
        res = ax.grouped_bar(
            timings,
            tick_labels=plot_nsides,
            group_spacing=1,
        )
        for container in res.bar_containers:
            ax.bar_label(container, padding=1)

        if i == num_plots - 1:
            ax.set_xlabel("nside (problem size)", fontsize=base_font_size * 1.5)
        ax.set_ylabel("Time (s)", fontsize=base_font_size * 1.5)
        ax.set_title(machine_name, fontsize=base_font_size * 1.5)
        ax.legend(loc="upper left", ncols=len(timings), fontsize=base_font_size)
        ax.set_yscale("log")
        ax.tick_params(axis="both", labelsize=base_font_size)

    fig.suptitle("GLASS lensing benchmark", fontsize=base_font_size * 2)
    base_plot_size = 5
    fig.set_size_inches(base_plot_size * 4, base_plot_size * num_plots * 1.5)

    fig.savefig(output_path)


if __name__ == "__main__":
    main()
