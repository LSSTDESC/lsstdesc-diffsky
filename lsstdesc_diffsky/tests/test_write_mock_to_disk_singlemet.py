"""
"""
# flake8: noqa
import pytest


def test_write_mock_to_disk_imports():
    from .. import write_mock_to_disk_singlemet


@pytest.mark.xfail
def test_write_mock_to_disk_has_no_hard_coding_relics():
    from .. import write_mock_to_disk_singlemet

    pat = "write_mock_to_disk should not have a hard-coded variable {}"

    list_of_hard_coding_errors = ("Ntotal_synthetics",)
    for attr in list_of_hard_coding_errors:
        msg = pat.format(attr)
        assert not hasattr(write_mock_to_disk_singlemet, attr), msg
