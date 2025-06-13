import tabsim

def test_version_format():
    assert isinstance(tabsim.__version__, str)
    assert len(tabsim.__version__) > 0
    assert tabsim.__version__.count('.') >= 1  # basic semantic versioning check
