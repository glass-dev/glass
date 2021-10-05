def test_get_default_ref():
    from glass.types import get_default_ref
    from typing import Annotated

    assert get_default_ref(int) is None
    assert get_default_ref(Annotated[int, 'ref']) == 'ref'


def test_default_refs():
    '''test the default references defined by types'''

    from glass.types import (
        get_default_ref,
        # simulation
        RedshiftBins,
        NumberOfBins,
        Cosmology,
        ClsDict,
        # fields
        MatterField,
        ConvergenceField,
        ShearField,
    )

    assert get_default_ref(RedshiftBins) == 'zbins'
    assert get_default_ref(NumberOfBins) == 'nbins'
    assert get_default_ref(Cosmology) == 'cosmology'
    assert get_default_ref(ClsDict) == 'cls'

    assert get_default_ref(MatterField) == 'matter'
    assert get_default_ref(ConvergenceField) == 'convergence'
    assert get_default_ref(ShearField) == 'shear'
