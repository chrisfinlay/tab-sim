"""Explicit authorization is required to time the all-array staged workload."""
from types import SimpleNamespace
import pytest
from benchmarks.staged_plugin import validate_mode


@pytest.mark.parametrize('capacity,warm,model,allowed',[
    (False,False,'staged',False),(True,False,'staged',True),
    (False,True,'staged',True),(True,True,'staged',False),
    (False,True,'conservative',False)])
def test_staged_mode_guard(capacity,warm,model,allowed):
    values={'--capacity':capacity,'--staged-warm':warm,'--memory-model':model}
    config=SimpleNamespace(getoption=lambda key,default=None:values.get(key,default))
    if allowed:validate_mode(config)
    else:
        with pytest.raises(pytest.UsageError):validate_mode(config)
