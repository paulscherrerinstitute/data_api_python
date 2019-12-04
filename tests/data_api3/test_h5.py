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

        h5.request(query, url="http://localhost:8080/api/v1/query", filename="my.h5")
