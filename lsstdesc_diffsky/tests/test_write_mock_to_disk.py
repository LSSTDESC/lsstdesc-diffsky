"""
"""
import pytest

try:
    from .. import write_mock_to_disk

    WRITE_MOCK_TO_DISK_IMPORTS = True
except ImportError:
    WRITE_MOCK_TO_DISK_IMPORTS = False


def test_write_mock_to_disk_imports():
    assert WRITE_MOCK_TO_DISK_IMPORTS, "write_mock_to_disk module fails to import"


@pytest.mark.skipif(not WRITE_MOCK_TO_DISK_IMPORTS)
def test_write_mock_to_disk_has_no_hard_coding_relics():
    pat = "write_mock_to_disk should not have a hard-coded variable {}"

    list_of_hard_coding_errors = ("Ntotal_synthetics",)
    for attr in list_of_hard_coding_errors:
        msg = pat.format(attr)
        assert not hasattr(write_mock_to_disk, attr), msg
