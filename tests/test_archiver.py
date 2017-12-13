import unittest

import datetime
import data_api as api

import logging
logger = logging.getLogger()
logger.setLevel(logging.WARNING)
logging.getLogger("requests").setLevel(logging.ERROR)


class ArchiverTest(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_retrieve(self):

        # Retrieve data from the archiver
        now = datetime.datetime.now()
        end = now - datetime.timedelta(minutes=1)
        start = end - datetime.timedelta(hours=12)
        # print(now)
        # print(start)
        # print(end)
        # data = api.get_data(channels=['sf-archiverappliance/S10-CMON-TIM1432:FAN-SPEED'], start=start, end=end)
        data = api.get_data(channels=['sf-archiverappliance/S10CB02-CVME-ILK:CENTRAL-CORETEMP'], start=start, end=end)
        print(data)


if __name__ == '__main__':
    unittest.main()
