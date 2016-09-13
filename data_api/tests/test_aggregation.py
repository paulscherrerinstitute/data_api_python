import unittest

from data_api import DataApiClient
from common import *

import logging
logger = logging.getLogger("DataApiClient")
logger.setLevel(logging.WARNING)
logging.getLogger("requests").setLevel(logging.ERROR)


class AggregationTest(unittest.TestCase):
    def setUp(self):
        pass
    
    def tearDown(self):
        pass

    def test(self):
        dac = DataApiClient()
        dac.set_aggregation(nr_of_bins=100)
        dfs = dac.get_data(chname, delta_range=10, index_field="pulseId")
        cfg = dac._cfg
        dac.__enable_server_reduction__(False)
        dfc = dac.get_data(cfg['channels'], start=cfg['range']["startDate"], end=cfg["range"]["endDate"], range_type="globalDate", index_field="pulseId")
        self.assertTrue((dfs.dropna() == dfc.dropna()).all().all())


if __name__ == '__main__':
    unittest.main()
