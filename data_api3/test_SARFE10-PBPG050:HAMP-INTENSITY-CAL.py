import os
import sys
import data_api3
import data_api3.h5
import h5py
import numpy as np

def test_api3_SARFE10_PBPG050_HAMP_INTENSITY_CAL_write_h5():
    np.set_printoptions(linewidth=240)
    ch = "SARFE10-PBPG050:HAMP-INTENSITY-CAL"
    fn = f"tests/data/api3/{ch}.bin"
    with open(fn, "rb") as f1:
        data_api3.h5.read_buffered_stream(f1, filename="tmp.h5")
    with open(fn, "rb") as f1:
        data2 = data_api3.reader.read_buffered_stream(f1)
    f2 = h5py.File("tmp.h5", "r")
    ts = f2[ch]["timestamp"]
    pulse = f2[ch]["pulse_id"]
    val = f2[ch]["data"]
    dtype2 = data2[ch][0][ch].dtype
    ts2 = np.array(list(map(lambda x: x["timestamp"], data2[ch])), dtype=np.int64)
    pulse2 = np.array(list(map(lambda x: x["pulse_id"], data2[ch])), dtype=np.int64)
    val2 = np.array(list(map(lambda x: x[ch], data2[ch])), dtype=dtype2)
    print(ts.dtype)
    print(ts.shape)
    print(val.dtype)
    print(val.shape)
    print(val2.dtype)
    print(val2.shape)
    print(val[:10])
    assert np.array_equal(ts, ts2)
    assert np.array_equal(pulse, pulse2)
    assert np.array_equal(val, val2)
    f3 = h5py.File(f"tests/data/api3/{ch}.h5", "r")
    for n in ["timestamp", "pulse_id", "data"]:
        d1 = f2[ch][n]
        d2 = f3[ch][n]
        assert np.array_equal(d1, d2)
    f2.close()
    f3.close()
    os.unlink("tmp.h5")
