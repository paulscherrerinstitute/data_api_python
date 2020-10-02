import os
import sys
import data_api3
import data_api3.h5
import h5py
import numpy as np

def test_api3_float_be_write_h5():
    ch = "SAROP11-CVME-PBPS1:Lnk9Ch3-DATA-CALIBRATED"
    fn = "tests/data/api3/floatbe.bin"
    with open(fn, "rb") as f1:
        data_api3.h5.read_buffered_stream(f1, filename="tmp.h5")
    f2 = h5py.File("tmp.h5", "r")
    f3 = h5py.File("tests/data/api3/floatbe.h5", "r")
    for n in ["timestamp", "pulse_id", "data"]:
        d1 = f2[ch][n]
        d2 = f3[ch][n]
        assert np.array_equal(d1, d2)
    f2.close()
    f3.close()
    os.unlink("tmp.h5")
