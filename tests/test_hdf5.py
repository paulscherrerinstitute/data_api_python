import unittest
import os
import datetime

import data_api as api

import logging
logger = logging.getLogger("DataApiClient")
logger.setLevel(logging.WARNING)
logging.getLogger("requests").setLevel(logging.ERROR)


class HDF5ReadWrite(unittest.TestCase):
    
    def setUp(self):
        self.fname = "unittest.h5"

    def tearDown(self):
        os.remove(self.fname)

    def test(self):
        now = datetime.datetime.now()
        end = now
        start = end - datetime.timedelta(minutes=10)

        data = api.get_data(channels=["A", "B"], start=start, end=end,
                            base_url="http://localhost:8080/archivertestdata")

        api.to_hdf5(data, filename=self.fname, overwrite=False, compression="gzip", compression_opts=5, shuffle=True)
        data_readback = _from_hdf5(self.fname, index_field="globalSeconds")

        # This set the index to 'globalSeconds' - this will drop the previous index 'globalDate'
        data.set_index("globalSeconds", inplace=True)
        # Set the column order in readback data frame to the order of the data data frame
        data_readback = data_readback[data.columns.tolist()]

        print(data.head())
        print(data_readback.head())

        self.assertTrue((data_readback.dropna() == data.dropna()).all().all())


def _from_hdf5(filename, index_field="globalSeconds"):
    """ Utility function to read data back from hdf5 file """
    import h5py
    import pandas

    infile = h5py.File(filename, "r")
    data = pandas.DataFrame()
    for k in infile.keys():
        data[k] = infile[k][:]

    try:
        data.set_index(index_field, inplace=True)
    except:
        raise RuntimeError("Cannot set index on %s, possible values are: %s" % (index_field, str(list(infile.keys()))))

    return data

if __name__ == '__main__':
    unittest.main()
