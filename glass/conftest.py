import pytest
from pathlib import Path


@pytest.fixture
def data_path(request):
    return Path(request.node.location[0]).parent / 'data'
