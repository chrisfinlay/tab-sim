"""Setup comparisons cannot accept changed geometry or runtime versions."""
import copy
import pytest
from benchmarks.geometry_setup_sweep import compare_reports


def test_geometry_comparison_is_strict():
    row=dict(times=2,setup_s=1.,geometry_sample=[1.,2.],provenance=dict(
        python='3',host='host',device_kind='CPU',versions={},harness_sha256='same',environment={}))
    compare_reports(row,copy.deepcopy(row))
    for key,value in [('geometry_sample',[2.,3.]),('setup_s',float('nan'))]:
        other=copy.deepcopy(row);other[key]=value
        with pytest.raises((ValueError,AssertionError)):compare_reports(row,other)
    other=copy.deepcopy(row);other['provenance']['python']='different'
    with pytest.raises(ValueError):compare_reports(row,other)
