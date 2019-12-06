from unittest import TestCase
import data_api3.h5 as h5


class TestReader(TestCase):

    def test_read_http_hipa(self):
        query = {"channels": ["MHC1:IST:2"],
                 "range": {
                     "type": "date",
                     "startDate": "2019-11-15T10:50:00.000000000Z",
                     "endDate": "2019-11-15T10:51:00.000000000Z"
                 }
                 }

        h5.request(query, "my.h5", url="http://localhost:8080/api/v1/query")

    def test_read_images_swissfel(self):
        query = {
            "channels": [
                "SARES11-SPEC125-M1:FPICTURE"
            ],
            "range": {
                "type": "date",
                "startDate": "2019-12-05T10:00:00.000000000Z",
                "endDate": "2019-12-05T10:00:00.100000000Z"
            }
        }

        h5.request(query, "my.h5", url="http://sf-daq-5.psi.ch:8080/api/v1/query")

    def test_h5_reader(self):
        reader = h5.HDF5Reader('test2.h5')
        with open('iodata_test.bin', 'rb') as file:
            reader.read(file)
