import unittest

import data_api.idread as iread
from data_api.h5 import Serializer

import logging
logger = logging.getLogger()
logger.setLevel(logging.WARNING)
logging.getLogger("requests").setLevel(logging.ERROR)


class ClientTest(unittest.TestCase):

    def test_decode(self):
        with open('../out.bin', mode='rb') as f:
            iread.decode(f)

        with open('../out_2.bin', mode='rb') as f:
            iread.decode(f)

        self.assertTrue(True)

    def test_request(self):

        import requests

        base_url = "https://data-api.psi.ch/sf"

        query = dict()
        query["channels"] = ["SARCL01-DSCR170:FPICTURE", "SINSB05-DBPM220:Q1"]
        query["fields"] = ["pulseId", "globalSeconds", "globalDate", "value", "eventCount"]
        query["range"] = {"startDate":"2018-04-17T10:00:00.000","endDate":"2018-04-17T10:00:01.000"}
        query["response"] = {"format":"rawevent"}

        serializer = Serializer()
        serializer.open('t.h5')

        with requests.post(base_url + '/query', json=query, stream=True) as response:
            iread.decode(response.raw, serializer=serializer)

        serializer.close()

        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
