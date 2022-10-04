from unittest.mock import Mock


def test_wrapped_generator():
    from glass.generator import WrappedGenerator
    from collections.abc import Generator

    gen, rec, yie = Mock(), object(), object()

    gen.myattr = object()

    g = WrappedGenerator(gen, rec, yie)
    assert isinstance(g, Generator)
    assert g.generator is gen
    assert g.receives is rec
    assert g.yields is yie
    assert g.myattr is gen.myattr


def test_decorator():

    from glass.generator import generator
    from collections.abc import Generator

    rec, yie = object(), object()

    @generator(receives=rec, yields=yie)
    def f():
        yield 1
        yield 2
        yield 3

    assert f.__name__ == 'f'

    g = f()
    assert isinstance(g, Generator)
    assert g.receives is rec
    assert g.yields is yie
    assert list(g) == [1, 2, 3]
