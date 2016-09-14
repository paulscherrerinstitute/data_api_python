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
        chname = ["SINEG01-RCIR-PUP10:SIG-AMPLT-AVG", "S10CB01-RBOC-DCP10:FOR-AMPLT-AVG"]
        dac = DataApiClient()
        dac.enable_server_aggregation(True)
        dac.set_aggregation(nr_of_bins=100)
        dfs = dac.get_data(chname, delta_range=100, index_field="pulseId")
        cfg = dac._cfg
        dac.enable_server_aggregation(False)
        dfc = dac.get_data(cfg['channels'], start=cfg['range']["startDate"], end=cfg["range"]["endDate"], range_type="globalDate", index_field="pulseId")
        result = check_dataframes(dac, dfs, dfc, cfg)
        self.assertTrue(result)


if __name__ == '__main__':
    unittest.main()
