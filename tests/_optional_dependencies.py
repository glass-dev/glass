import importlib.util

HAVE_ARRAY_API_STRICT = importlib.util.find_spec("array_api_strict") is not None
HAVE_FITSIO = importlib.util.find_spec("fitsio") is not None
HAVE_JAX = importlib.util.find_spec("jax") is not None
