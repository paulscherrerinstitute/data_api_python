import os
import sys
import data_api3
import data_api3.h5
import h5py
import numpy as np

def test_api3_bool_write_h5():
    np.set_printoptions(linewidth=200)
    ch = "SAR-CVME-TIFALL4:EvtSet"
    fn = "tests/data/api3/bool_evtset.bin"
    with open(fn, "rb") as f1:
        data_api3.h5.read_buffered_stream(f1, filename="tmp.h5")
    with open(fn, "rb") as f1:
        data_api3.reader.read_buffered_stream(f1)
    f2 = h5py.File("tmp.h5", "r")
    ts = f2[ch]["timestamp"]
    pulse = f2[ch]["pulse_id"]
    val = f2[ch]["data"]
    print(ts[:4])
    print(pulse[:4])
    print(val.shape)
    print(val[:4, :20])
    assert False
