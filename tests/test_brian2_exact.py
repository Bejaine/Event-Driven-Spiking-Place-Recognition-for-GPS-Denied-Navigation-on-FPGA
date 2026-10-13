"""pytest: snnsim float engine reproduces Brian2 E and I spike rasters exactly (3 queries)."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def test_brian2_raster_exact():
    import brian2_equivalence as BE
    assert BE.main(3, 0)
