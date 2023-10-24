"""
"""


def infer_software_versions():
    versions = dict()

    import diffmah
    import diffstar
    import dsps
    import lsstdesc_diffsky

    versions["diffmah"] = diffmah.__version__
    versions["diffstar"] = diffstar.__version__
    versions["dsps"] = dsps.__version__
    versions["lsstdesc_diffsky"] = lsstdesc_diffsky.__version__
    return versions
