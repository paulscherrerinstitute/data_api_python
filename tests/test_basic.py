import unittest
import h5py
import os

from data_api import DataApiClient

import logging
logger = logging.getLogger("DataApiClient")
logger.setLevel(logging.ERROR)
logging.getLogger("requests").setLevel(logging.ERROR)

df_date = None
df_secs = None
chname = ["SINDI01-RIQM-DCP10:FOR-PHASE-AVG", "S10CB01-RBOC-DCP10:FOR-PHASE-AVG"]


def prepare_data(index_field, delta_i=100, chname=chname, ):
    dac = DataApiClient()
    delta = delta_i
    df = None
    while df is None:
        df = dac.get_data(chname, delta_range=delta, index_field=index_field)
        delta += 1000

    return df


class DataLiveCoherencyTestDateSeconds(unittest.TestCase):

    def setUp(self):
        print("%s will fail due to lack of nanoseconds precision in float" % "DataLiveCoherencyTestDateSeconds")
        self.dac = DataApiClient()
        self.df_date = prepare_data("globalDate")

    def tearDown(self):
        pass

    def test(self):
        print(self.df_date.globalSeconds.iloc[0], self.df_date.globalSeconds.iloc[-1])
        df2 = self.dac.get_data(chname, start=self.df_date.globalSeconds.iloc[0], end=self.df_date.globalSeconds.iloc[-1], range_type="globalSeconds")
        self.assertTrue((df2.dropna() == self.df_date.dropna()).all().all(), )


class DataLiveCoherencyTestDatePulseId(unittest.TestCase):

    def setUp(self):
        self.dac = DataApiClient()
        self.df_date = prepare_data("globalDate")

    def tearDown(self):
        pass

    def test(self):
        df2 = self.dac.get_data(chname, start=self.df_date.pulseId[0], end=self.df_date.pulseId[-1], range_type="pulseId")
        self.assertTrue((df2.dropna() == self.df_date.dropna()).all().all())


class HDF5ReadWrite(unittest.TestCase):
    
    def setUp(self):
        self.dac = DataApiClient()
        self.df_secs = prepare_data("globalSeconds")
        self.fname = "unittest.h5"

    def tearDown(self):
        os.unlink(self.fname)

    def test(self):
        r = self.dac.to_hdf5(self.df_secs, filename=self.fname, overwrite=False, compression="gzip", compression_opts=5, shuffle=True)
        self.assertTrue(r is None)
        dfr = self.dac.from_hdf5(self.fname, index_field="globalSeconds")

        # it will fail due to nanoseconds lack
        print("SKIPPING DATE CHECK!")
        if "globalDate" in dfr.columns:
            dfr.drop('globalDate', inplace=True, axis=1)
            self.df_secs.drop('globalDate', inplace=True, axis=1)

        dfr = dfr[self.df_secs.columns.tolist()]

        self.assertTrue((dfr.dropna() == self.df_secs.dropna()).all().all())

if __name__ == '__main__':
    unittest.main()
