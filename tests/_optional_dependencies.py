import importlib.util

HAVE_FITSIO = importlib.util.find_spec("fitsio") is not None
HAVE_JAX = importlib.util.find_spec("jax") is not None
HAVE_S2FFT = importlib.util.find_spec("s2fft") is not None
