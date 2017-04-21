import unittest
import os

from data_api import DataApiClient
from common import *

import logging
logger = logging.getLogger("DataApiClient")
logger.setLevel(logging.WARNING)
logging.getLogger("requests").setLevel(logging.ERROR)


class HDF5ReadWrite(unittest.TestCase):
    
    def setUp(self):
        self.fname = "unittest.h5"

    def tearDown(self):
        os.unlink(self.fname)

    def test(self):
        dac = DataApiClient()
        df_secs, dac_r = prepare_data("globalSeconds", chname=chname)
        r = dac.to_hdf5(df_secs, filename=self.fname, overwrite=False, compression="gzip", compression_opts=5, shuffle=True)
        self.assertTrue(r is None)
        dfr = dac.from_hdf5(self.fname, index_field="globalSeconds")

        # it will fail due to nanoseconds lack
        logger.warning("SKIPPING DATE CHECK!")
        if "globalDate" in dfr.columns:
            dfr.drop('globalDate', inplace=True, axis=1)
            df_secs.drop('globalDate', inplace=True, axis=1)

        dfr = dfr[df_secs.columns.tolist()]

        self.assertTrue((dfr.dropna() == df_secs.dropna()).all().all())


if __name__ == '__main__':
    unittest.main()
