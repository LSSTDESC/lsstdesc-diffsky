"""
"""
from ..infer_diffcode_versions import infer_software_versions


def test_infer_diffcode_versions():
    versions = infer_software_versions()
    assert set(list(versions.keys())) == set(
        ("diffmah", "diffstar", "dsps", "lsstdesc_diffsky")
    )
