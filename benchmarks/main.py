"""Entrypoint for GLASS benchmarks."""

import argparse
import importlib.util
import pathlib

from benchmark_utils import configure_backends, set_run_profile


def main() -> None:
    """Parse command line arguments and run requested benchmarks."""
    benchmarks_dir = pathlib.Path(__file__).parent.resolve() / "benchmarks"
    available_benchmarks = [
        f.stem for f in benchmarks_dir.iterdir() if f.is_file() and f.suffix == ".py"
    ]

    parser = argparse.ArgumentParser(
        prog="GLASS-Benchmarks",
        description="Benchmarks for the GLASS library",
    )

    parser.add_argument(
        "-b",
        "--benchmarks",
        type=str,
        nargs="+",
        choices=available_benchmarks,
        help=(
            "String identifiers for the available benchmarks to be ran. "
            "Runs all by default."
        ),
    )

    parser.add_argument(
        "-l",
        "--list",
        action="store_true",
        help="List the available benchmarks",
    )

    parser.add_argument(
        "-x",
        "--array-backend",
        type=str,
        default="numpy",
        choices=["all", "numpy", "jax"],
        help="String identifier(s) for the array backend(s) to benchmark with.",
    )

    parser.add_argument(
        "-p",
        "--profile",
        action="store_true",
        help=(
            "Generate a perfetto profile of the benchmark. "
            "Only available if using JAX as the array backend."
        ),
    )

    args = parser.parse_args()

    set_run_profile(args.profile)

    available_backends = configure_backends(args.array_backend)

    if args.list:
        print(f"Available benchmarks: {', '.join(available_benchmarks)}")  # noqa: T201
        return

    benchmarks = args.benchmarks or available_benchmarks
    for benchmark_name in benchmarks:
        print(f"Running {benchmark_name} benchmark")  # noqa: T201
        for xp in available_backends.values():
            spec = importlib.util.spec_from_file_location(
                benchmark_name,
                benchmarks_dir / f"{benchmark_name}.py",
            )
            benchmark_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(benchmark_module)  # ty: ignore[unresolved-attribute]

            benchmark_module.benchmark(xp)


if __name__ == "__main__":
    main()
