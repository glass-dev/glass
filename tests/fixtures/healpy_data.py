"""Fixtures for handling healpy data within tests."""

from __future__ import annotations

import os
import pathlib
import subprocess
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Generator


@pytest.fixture(scope="session")
def healpy_datapath() -> Generator[str]:
    """
    Pull the healpy data and returns the path to the new folder.

    Also performs cleanup by removing the cloned dir.
    """
    subprocess.run(  # noqa: S602
        """
        git clone --depth 1 https://github.com/healpy/healpy-data;
        cd healpy-data;
        bash download_weights_8192.sh;
        """,  # noqa: S607
        shell=True,
        check=True,
        capture_output=True,
    )
    healpy_datapath = str(pathlib.Path("healpy-data").absolute())

    yield healpy_datapath

    # Teardown
    subprocess.run(  # noqa: S603
        ["rm", "-rf", healpy_datapath],  # noqa: S607
        check=True,
    )


@pytest.fixture
def add_healpy_datapath_to_env(healpy_datapath: str) -> Generator:
    """
    Add the path to the healpy data into the environment.

    Also removes the new env var when finalising.
    """
    # Set path to healpy data in environment
    os.environ["HEALPY_DATAPATH"] = healpy_datapath

    yield

    # Teardown
    os.environ.pop("HEALPY_DATAPATH")


@pytest.fixture
def invalid_healpy_datapath() -> Generator[str]:
    """
    Add an invalid path to the healpy data into the environment.

    Also removes the new env var when finalising.
    """
    # Set path to healpy data in environment
    fake_datapath = "/does/not/exist"
    os.environ["HEALPY_DATAPATH"] = fake_datapath

    yield fake_datapath

    # Teardown
    os.environ.pop("HEALPY_DATAPATH")
