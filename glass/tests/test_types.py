def test_get_default_ref():
    from glass.types import get_default_ref
    from typing import Annotated

    assert get_default_ref(int) is None
    assert get_default_ref(Annotated[int, 'ref']) == 'ref'


def test_default_refs():
    '''test the default references defined by types'''
    from glass.types import (
        get_default_ref,
        RedshiftBins,
        NumberOfBins,
        Cosmology,
        ClsDict,
    )

    assert get_default_ref(RedshiftBins) == 'zbins'
    assert get_default_ref(NumberOfBins) == 'nbins'
    assert get_default_ref(Cosmology) == 'cosmology'
    assert get_default_ref(ClsDict) == 'cls'
