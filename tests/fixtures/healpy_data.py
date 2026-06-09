"""Fixtures for handling healpy data within tests"""

import pytest
import subprocess
import pathlib
import os

@pytest.fixture(scope="session")
def healpy_datapath(request: pytest.FixtureRequest) -> str:
    """
    Pulls the healpy data and returns the path to the new folder.

    Also performs cleanup by removing the cloned dir.
    """
    subprocess.run(
        """
        git clone --depth 1 https://github.com/healpy/healpy-data;
        cd healpy-data;
        bash download_weights_8192.sh;
        """
        ,
        shell=True,
        check=True,
        capture_output=True,
    )
    healpy_datapath = str(pathlib.Path("healpy-data").absolute())

    # Define a finalizer function for teardown
    def finalizer():
      """Deletes the pulled healpy_datapath directory"""
      subprocess.run(["rm","-rf",healpy_datapath])

    # Register the finalizer to ensure cleanup
    request.addfinalizer(finalizer)

    return healpy_datapath
   

@pytest.fixture(scope="function")
def add_healpy_datapath_to_env(request: pytest.FixtureRequest, healpy_datapath: str) -> None:
    """
    Adds the path to the healpy into the environment.

    Also removes the new env var when finalising.
    """
    # Set path to healpy data in environment
    os.environ["HEALPY_DATAPATH"] = healpy_datapath

    # Define a finalizer function for teardown
    def finalizer():
      """Removes healpy_datapath from the env"""
      os.environ.pop("HEALPY_DATAPATH")

    # Register the finalizer to ensure cleanup
    request.addfinalizer(finalizer)