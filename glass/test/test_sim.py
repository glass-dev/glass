import pytest


def test_generator_error():

    from glass.sim import generate, GeneratorError, State

    def mygenerator():
        for n in range(5):
            if n == 3:
                raise ZeroDivisionError
            yield

    g = mygenerator()

    with pytest.raises(GeneratorError, match='shell 3: ') as exc_info:
        for shell in generate([g]):
            pass

    e = exc_info.value
    assert type(e.__cause__) == ZeroDivisionError
    assert e.generator is g
    assert isinstance(e.state, State)


def test_group():

    from unittest.mock import Mock
    from glass.sim import group

    generator = Mock()
    generator.__name__ = 'foo'
    generator.receives = None
    generator.yields = 'bar'

    g = group('test', [generator])

    assert g.receives == 'state'
    assert g.yields == 'test'

    generator.send.return_value = None

    state = g.send(None)

    assert generator.send.called
    assert generator.send.call_args.args == (None,)

    generator.send.return_value = object()

    state = g.send(None)

    assert generator.send.call_count == 2
    assert generator.send.call_args.args == (None,)
    assert state['bar'] is generator.send.return_value

    g.close()

    assert generator.close.called
