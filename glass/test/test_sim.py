import pytest


def test_generator_error():

    from glass.core import generator
    from glass.sim import generate, GeneratorError, State

    @generator('# ->')
    def mygenerator():
        while True:
            n = yield
            if n == 3:
                raise ZeroDivisionError

    g = mygenerator()

    with pytest.raises(GeneratorError, match='shell 3: ') as exc_info:
        for shell in generate([g]):
            pass

    e = exc_info.value
    assert type(e.__cause__) == ZeroDivisionError
    assert e.generator is g
    assert isinstance(e.state, State)
