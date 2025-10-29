from __future__ import annotations

from typing import TYPE_CHECKING, TypeAlias

if TYPE_CHECKING:
    import numpy as np

    import glass._array_api_utils as _utils
    import glass.jax

    UnifiedGenerator: TypeAlias = (
        np.random.Generator | glass.jax.Generator | _utils.Generator
    )
