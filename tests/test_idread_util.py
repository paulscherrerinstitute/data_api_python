import unittest
import requests
import data_api.idread_util as idread
from data_api import util
import datetime
from pathlib import Path

import logging
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logging.getLogger("requests").setLevel(logging.ERROR)


class PrintSerializer:
    def add_data(self, channel_name, value_name, value, dtype="f8", shape=[1, ]):
        logger.info(value)
        pass


class ClientTest(unittest.TestCase):
    data = Path(__file__).parent / 'data'

    @unittest.skip("Missing tmp.bin")
    def test_tmp(self):
        tmp = self.data / 'tmp.bin'
        with tmp.open('rb') as f:
            idread.decode(f)

        self.assertTrue(True)

    def test_decode(self):
        out = self.data / 'out.bin'
        with out.open(mode='rb') as f:
            idread.decode(f)

        out2 = self.data / 'out_2.bin'
        with out2.open('rb') as f:
            idread.decode(f)

        self.assertTrue(True)

    def test_request(self):

        base_url = "https://data-api.psi.ch/sf"

        end = datetime.datetime.now()
        start = end - datetime.timedelta(minutes=10)
        query = util.construct_data_query(channels=["SINEG01-RCIR-PUP10:SIG-AMPLT"], start=start, end=end,
                                          response=util.construct_response(format="rawevent"))

        with requests.post(base_url + '/query', json=query, stream=True) as response:
            idread.decode(response.raw)

        self.assertTrue(True)

    @unittest.skip("Missing tmp.bin")
    def test_decode_serializer(self):
        collector = idread.DictionaryCollector()
        tmp = self.data / 'tmp.bin'
        # filename = 'data/out_2.bin'
        with tmp.open('rb') as f:
            idread.decode(f, collector=collector.add_data)

        data = collector.get_data()
        print(len(data[0]["data"]))

        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
