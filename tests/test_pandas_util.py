import unittest
import os
import datetime

import data_api as api
import data_api.pandas_util as putil
import data_api.util as util

import logging
logger = logging.getLogger("DataApiClient")
logger.setLevel(logging.WARNING)
logging.getLogger("requests").setLevel(logging.ERROR)


class HDF5ReadWrite(unittest.TestCase):
    
    def setUp(self):
        self.fname = "unittest.h5"
        if os.path.isfile(self.fname):
            os.remove(self.fname)

    def test_local_to_hdf5(self):
        now = datetime.datetime.now()
        end = now
        start = end - datetime.timedelta(minutes=10)

        query = util.construct_data_query(channels=["A", "B"], start=start, end=end)
        data = api.get_data_json(query, base_url="http://localhost:8080/archivertestdata")

        data = putil.build_pandas_data_frame(data)
        putil.to_hdf5(data, filename=self.fname, overwrite=False, compression="gzip", compression_opts=5, shuffle=True)
        data_readback = putil.from_hdf5(self.fname, index_field="globalSeconds")

        # This set the index to 'globalSeconds' - this will drop the previous index 'globalDate'
        data.set_index("globalSeconds", inplace=True)
        # Set the column order in readback data frame to the order of the data data frame
        data_readback = data_readback[data.columns.tolist()]

        # print(data.head())
        # print(data_readback.head())

        self.assertTrue((data_readback.dropna() == data.dropna()).all().all())


if __name__ == '__main__':
    unittest.main()
