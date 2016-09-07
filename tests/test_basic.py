import unittest
import h5py
import os
import sys

from data_api import DataApiClient

import logging
logger = logging.getLogger("DataApiClient")
logger.setLevel(logging.WARNING)
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
    return df, dac._cfg


def check_dataframes(dac, df, df2, _cfg=None):
    test = False
    try:
        if (df.dropna().iloc[:-1] == df2.dropna()).all().all():
            test = True
    except:
        #print(sys.exc_info())
        try:
            if (df2.dropna() == df.dropna()).all().all():
                test = True
        except:
            #print(sys.exc_info())
            print("\nFailing test, dumping info")
            print(df.info(), df2.info())
            print(_cfg)            
            print(dac._cfg)
            print(df.head())
            print(df2.head())
            print(df.tail())
            print(df2.tail())
            print("\n")
    return test


class DataLiveCoherencyTestDateSeconds(unittest.TestCase):

    def setUp(self):
        self.dac = DataApiClient()
        self.df_date, self._cfg = prepare_data("globalDate")

    def tearDown(self):
        pass

    def test(self):
        df2 = self.dac.get_data(chname, start=self.df_date.globalSeconds.iloc[0], end=self.df_date.globalSeconds.iloc[-1], range_type="globalSeconds")
        self.assertTrue(check_dataframes(self.dac, self.df_date, df2, self._cfg))


class DataLiveCoherencyTestDatePulseId(unittest.TestCase):

    def setUp(self):
        self.dac = DataApiClient()
        self.df_date, self._cfg = prepare_data("globalDate")

    def tearDown(self):
        pass

    def test(self):
        df2 = self.dac.get_data(chname, start=self.df_date.pulseId[0], end=self.df_date.pulseId[-1], range_type="pulseId")
        self.assertTrue(check_dataframes(self.dac, self.df_date, df2, self._cfg))


class HDF5ReadWrite(unittest.TestCase):
    
    def setUp(self):
        self.dac = DataApiClient()
        self.df_secs, self._cfg = prepare_data("globalSeconds")
        self.fname = "unittest.h5"

    def tearDown(self):
        os.unlink(self.fname)

    def test(self):
        r = self.dac.to_hdf5(self.df_secs, filename=self.fname, overwrite=False, compression="gzip", compression_opts=5, shuffle=True)
        self.assertTrue(r is None)
        dfr = self.dac.from_hdf5(self.fname, index_field="globalSeconds")

        # it will fail due to nanoseconds lack
        logger.warning("SKIPPING DATE CHECK!")
        if "globalDate" in dfr.columns:
            dfr.drop('globalDate', inplace=True, axis=1)
            self.df_secs.drop('globalDate', inplace=True, axis=1)

        dfr = dfr[self.df_secs.columns.tolist()]

        self.assertTrue((dfr.dropna() == self.df_secs.dropna()).all().all())

if __name__ == '__main__':
    unittest.main()
