import unittest
from data_api import DataApiClient

import logging
logger = logging.getLogger("DataApiClient")
logger.setLevel(logging.ERROR)
logging.getLogger("requests").setLevel(logging.ERROR)


class DataLiveCoherencyTest(unittest.TestCase):

    def setUp(self):
        self.dac = DataApiClient()

    def tearDown(self):
        pass

    def test(self):
        df = None
        chname = "SINDI01-RIQM-DCP10:FOR-PHASE-AVG"
        delta = 1000
        while df is None:
            df = self.dac.get_data(chname, delta_range=delta)
            delta += 1000
        df2 = self.dac.get_data(chname, start=df.globalSeconds[0], end=df.globalSeconds[-1], range_type="globalSeconds")
        self.assertTrue((df2[chname] == df[chname][1:-1]).all())


class DataLiveCoherencyTest2(unittest.TestCase):

    def setUp(self):
        self.dac = DataApiClient()

    def tearDown(self):
        pass

    def test(self):
        df = None
        chname = "SINDI01-RIQM-DCP10:FOR-PHASE-AVG"
        delta = 1000
        while df is None:
            df = self.dac.get_data(chname, delta_range=delta)
            delta += 1000
        df2 = self.dac.get_data(chname, start=df.pulseId[0], end=df.pulseId[-1], range_type="pulseId")
        print(df, df2)
        self.assertTrue((df2[chname] == df[chname][1:-1]).all())

if __name__ == '__main__':
    unittest.main()
