import unittest

from data_api import DataApiClient
from tests.common import prepare_data, check_dataframes, chname

import logging
logger = logging.getLogger("DataApiClient")
logger.setLevel(logging.WARNING)
logging.getLogger("requests").setLevel(logging.ERROR)


class DataLiveCoherencyTestDate(unittest.TestCase):

    def setUp(self):
        self.dac = DataApiClient()
        self.df_date, self.dac_r = prepare_data("globalDate", chname=chname)

    def tearDown(self):
        pass

    def test_seconds(self):
        df2 = self.dac.get_data(chname,
                                start=self.df_date.globalSeconds.iloc[0],
                                end=self.df_date.globalSeconds.iloc[-1],
                                range_type="globalSeconds")

        self.assertTrue(check_dataframes(self.dac, self.df_date, df2, self.dac_r._cfg))

    def test_pulseId(self):
        df2 = self.dac.get_data(chname,
                                start=self.df_date.pulseId[0],
                                end=self.df_date.pulseId[-1],
                                range_type="pulseId")
        self.assertTrue(check_dataframes(self.dac, self.df_date, df2, self.dac_r._cfg))


if __name__ == '__main__':
    unittest.main()
