def test_generator_class():
    from glass.generator import Generator

    class WrappedGenerator:
        myattr = object()

    gen, rec, yie, ini = WrappedGenerator(), object(), object(), object()

    g = Generator(gen, rec, yie, ini)
    assert isinstance(g, Generator)
    assert g.generator is gen
    assert g.receives is rec
    assert g.yields is yie
    assert g.initial is ini
    assert g.myattr is WrappedGenerator.myattr

    g = Generator(gen)
    assert isinstance(g, Generator)
    assert g.generator is gen
    assert g.receives is None
    assert g.yields is None
    assert g.initial is None

    g2 = Generator(g)
    assert g2 is g
    assert g.generator is gen
    assert g.receives is None
    assert g.yields is None
    assert g.initial is None

    g3 = Generator(g, receives=rec)
    assert g3 is g
    assert g.generator is gen
    assert g.receives is rec
    assert g.yields is None
    assert g.initial is None

    g4 = Generator(g, yields=yie)
    assert g4 is g
    assert g.generator is gen
    assert g.receives is rec
    assert g.yields is yie
    assert g.initial is None

    g5 = Generator(g, initial=ini)
    assert g5 is g
    assert g.generator is gen
    assert g.receives is rec
    assert g.yields is yie
    assert g.initial is ini


def test_wrap_generator():

    from glass.generator import wrap_generator, Generator

    rec, yie, ini = object(), object(), object()

    def f():
        yield 1
        yield 2
        yield 3

    w = wrap_generator(f, receives=rec, yields=yie, initial=ini)
    assert w is not f
    assert w.__name__ == f.__name__

    g = w()
    assert isinstance(g, Generator)
    assert g.receives is rec
    assert g.yields is yie
    assert g.initial is ini
    assert list(g) == [1, 2, 3]

    w2 = wrap_generator(w)
    g = w2()
    assert isinstance(g, Generator)
    assert g.receives is rec
    assert g.yields is yie
    assert g.initial is ini
    assert list(g) == [1, 2, 3]

    def f(self):
        yield self

    w = wrap_generator(f, self=True)
    g = w()
    assert next(g) is g

    w2 = wrap_generator(w)
    g = w2()
    assert next(g) is g

    w = wrap_generator(f)
    w2 = wrap_generator(w, self=True)
    g = w2()
    assert next(g) is g


def test_decorators():

    from glass.generator import receives, yields, initial

    for decorator in receives, yields, initial:

        @decorator()
        def f():
            pass
        g = f()
        assert getattr(g, decorator.__name__) is None

        @decorator(1)
        def f():
            pass
        g = f()
        assert getattr(g, decorator.__name__) == 1

        @decorator(1, 2, 3)
        def f():
            pass
        g = f()
        assert getattr(g, decorator.__name__) == [1, 2, 3]
