import unittest

import math

import datetime
import data_api as api
from data_api import Aggregation

import pytz

import logging
logger = logging.getLogger()
logger.setLevel(logging.WARNING)
logging.getLogger("requests").setLevel(logging.ERROR)


class ClientTest(unittest.TestCase):

    def test_convert_date(self):
        import data_api.client
        data_api.client._convert_date(datetime.datetime.now())

        data_api.client._convert_date("2017-12-15 15:05:43.258077+02:00")

        # data_api.client._convert_date(pytz.timezone('Europe/Zurich').localize(datetime.datetime.now()))

        self.assertTrue(True)

    def test_retrieve(self):  # Only works if the testserver.py server is running
        now = datetime.datetime.now()
        end = now
        start = end - datetime.timedelta(minutes=10)

        data = api.get_data(channels=["A", "B"], start=start, end=end,
                            base_url="http://localhost:8080/archivertestdata")
        print(data)

        # Test function returns 10 datapoints with values from 0 to 9
        self.assertEqual(data.shape[0], 10)

        for i in range(10):
            self.assertEqual(data["A"][i], i)

        print(data["A"])

    def test_retrieve_merge(self):  # Only works if the testserver.py server is running
        now = datetime.datetime.now()
        end = now
        start = end - datetime.timedelta(minutes=10)

        data = api.get_data(channels=["A", "B"], start=start, end=end,
                            base_url="http://localhost:8080/archivertestdatamerge")
        print(data)

        # Test function returns 10 datapoints with values from 0 to 9
        self.assertEqual(data.shape[0], 20)

        counter = 0
        for i in range(20):
            if i % 2 == 0:
                self.assertEqual(data["A"][i], counter)
                counter += 1
            else:
                self.assertTrue(math.isnan(data["A"][i]))

        print(data["A"])

    def test_real_aggregation(self):
        data = api.get_data(["SINDI01-RIQM-DCP10:FOR-PHASE-AVG", "S10CB01-RBOC-DCP10:FOR-PHASE-AVG"],
                            delta_range=100, index_field="pulseId", aggregation=Aggregation(nr_of_bins=100))

        self.assertEqual(data.shape[0], 100)
        print(data)

    def test_real(self):  # Only works if archiver is accessible and data is available for used channel
        # Retrieve data from the archiver

        now = datetime.datetime.now()
        end = now - datetime.timedelta(minutes=1)
        start = end - datetime.timedelta(hours=12)

        data = api.get_data(channels=['sf-archiverappliance/S10CB02-CVME-ILK:CENTRAL-CORETEMP',
                                      'sf-archiverappliance/S10CB02-CVME-ILK:CENTRAL-CORETEMP2'], start=start, end=end)

        print(data)
        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
