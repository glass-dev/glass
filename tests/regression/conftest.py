from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from glass import _rng
from tests.fixtures.array_backends import xp_available_backends

if TYPE_CHECKING:
    from types import ModuleType

    from glass._types import UnifiedGenerator


@pytest.fixture(
    params=[
        xp
        for name, xp in xp_available_backends.items()
        if name not in {"array_api_strict", "jax.numpy"}
    ],
    scope="session",
)
def xp(request: pytest.FixtureRequest) -> ModuleType:
    """
    Fixture for array backend to be used in regression tests.

    Access array library functions using `xp.` in tests.

    We are excluding array-api-strict and jax for two reasons
    1. Our use of array-api-strict is not for its performance but
       for checking our interface with array libraries. Additionally,
       users are unlikely to use array-api-strict with glass.
       Therefore, it is not worth regression testing with array-api-strict.
    2. We did not previously support jax, therefore it does
       not _yet_ make sense to regression test jax as there is
       nothing to compare against, since jax is not supported by
       the older versions of glass.

    """
    return request.param


@pytest.fixture
def urng(xp: ModuleType) -> UnifiedGenerator:
    """
    Fixture for a unified RNG interface to be used in regression tests.

    Access the relevant RNG using `urng.` in tests.

    Must be used with the `xp` fixture.

    """
    return _rng.rng_dispatcher(xp=xp)
