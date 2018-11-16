import unittest

import data_api.idread as iread

import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logging.getLogger("requests").setLevel(logging.ERROR)


class PrintSerializer:

    def open(self, file_name):
        pass

    def close(self):
        pass

    def add_data(self, channel_name, value_name, value, dtype="f8", shape=[1, ]):
        logger.info(value)
        pass


class ClientTest(unittest.TestCase):

    def test_tmp(self):
        with open('t.bin', mode='rb') as f:
            iread.decode(f,iread.Collector())

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
        # query["channels"] = ["SARCL01-DSCR170:FPICTURE", "SINSB05-DBPM220:Q1"]
        query["channels"] = ["SINEG01-RCIR-PUP10:SIG-AMPLT"]
        query["fields"] = ["pulseId", "globalSeconds", "globalDate", "value", "eventCount"]
        query["range"] = {"startDate":"2018-04-17T10:00:00.000","endDate":"2018-04-17T10:50:00.000"}
        query["response"] = {"format":"rawevent"}

        persist = True

        if persist:
            serializer = PrintSerializer()
            serializer.open('t.h5')

            with requests.post(base_url + '/query', json=query, stream=True) as response:
                iread.decode(response.raw, collector=serializer)

            serializer.close()
        else:
            with requests.post(base_url + '/query', json=query, stream=True) as response:
                iread.decode(response.raw)

        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
