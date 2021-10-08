def test_get_annotation():
    from glass.typing import get_annotation
    from typing import Annotated
    from itertools import chain, combinations

    def powerset(iterable):
        s = list(iterable)
        return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

    # no annotation, no problem
    a = get_annotation(int)
    assert a.name is None
    assert a.random is False

    # list of (a, b, c) -- when annotating with `a`, info `b` has value `c`
    tests = [
        ('name:myname', 'name', 'myname'),
        ('random', 'random', True),
    ]

    for annotations in powerset(tests):
        x = Annotated[int, None]
        for t in annotations:
            x = Annotated[x, t[0]]

        a = get_annotation(x)

        for t in annotations:
            assert getattr(a, t[1]) == t[2]


def test_default_names():
    from glass.typing import (
        get_annotation,
        # simulation
        RedshiftBins,
        NumberOfBins,
        Cosmology,
        ClsDict,
        # fields
        Matter,
        Convergence,
        Shear,
    )

    assert get_annotation(RedshiftBins).name == 'zbins'
    assert get_annotation(NumberOfBins).name == 'nbins'
    assert get_annotation(Cosmology).name == 'cosmology'
    assert get_annotation(ClsDict).name == 'cls'

    assert get_annotation(Matter[int]).name == 'matter'
    assert get_annotation(Convergence[int]).name == 'convergence'
    assert get_annotation(Shear[int]).name == 'shear'
