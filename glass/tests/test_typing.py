def test_get_annotation():
    from glass.typing import get_annotation
    from typing import Annotated, Optional

    # no annotation, no problem
    assert get_annotation(int) is None

    # annotations with name
    assert get_annotation(Annotated[int, 'glass:myname']) == 'myname'

    # resolve annotations of Optional[T]
    assert get_annotation(Optional[int]) is None

    # resolve annotations of Optional[Annotated[T]]
    assert get_annotation(Optional[Annotated[int, 'glass:myname']]) == 'myname'

    # foreign annotations should be ignored
    assert get_annotation(Annotated[int, 'myannotation', 'glass:myname']) == 'myname'


def test_annotate():
    from typing import get_type_hints
    from glass.typing import NoneType, Annotated, annotate, get_annotation

    def f():
        pass

    def g() -> Annotated[None, 'glass:myname']:
        pass

    def _a(func):
        return get_annotation(get_type_hints(func, include_extras=True).get('return', NoneType))

    assert _a(annotate(f, 'myname')) == 'myname'
    assert _a(f) is None

    assert _a(annotate(g, 'myothername')) == 'myothername'
    assert _a(g) == 'myname'


def test_default_names():
    from glass.typing import (
        get_annotation,
        # simulation
        WorkDir,
        NSide,
        RedshiftBins,
        NumberOfBins,
        Cosmology,
        # cls
        TheoryCls,
        SampleCls,
        # random fields
        RandomMatterFields,
        RandomConvergenceFields,
        # fields
        MatterFields,
        ConvergenceFields,
        ShearFields,
        # observations
        Visibility,
    )

    assert get_annotation(WorkDir) == 'workdir'
    assert get_annotation(NSide) == 'nside'
    assert get_annotation(RedshiftBins) == 'zbins'
    assert get_annotation(NumberOfBins) == 'nbins'
    assert get_annotation(Cosmology) == 'cosmology'

    assert get_annotation(TheoryCls) == 'theory_cls'
    assert get_annotation(SampleCls) == 'sample_cls'

    assert get_annotation(RandomMatterFields) == 'matter'
    assert get_annotation(RandomConvergenceFields) == 'convergence'

    assert get_annotation(MatterFields) == 'matter'
    assert get_annotation(ConvergenceFields) == 'convergence'
    assert get_annotation(ShearFields) == 'shear'

    assert get_annotation(Visibility) == 'visibility'
