from unittest import TestCase
import data_api3.reader as reader
import json
import io
import urllib3


class TestReader(TestCase):

    def test_resolve_numpy_dtype(self):
        dtype = reader.resolve_numpy_dtype({"name": "SLG-LSCP3-FNS:CH7:VAL_GET",
                                    "type": "float64",
                                    "compression": "0",
                                    "byteOrder": "BIG_ENDIAN",
                                    "shape": None})
        self.assertEqual(dtype, ">f8")

        dtype = reader.resolve_numpy_dtype({"name": "SLG-LSCP3-FNS:CH7:VAL_GET",
                                            "type": "float64",
                                            "compression": "0",
                                            "byteOrder": "LITTLE_ENDIAN",
                                            "shape": None})
        self.assertEqual(dtype, "f8")

        dtype = reader.resolve_numpy_dtype({"name": "SLG-LSCP3-FNS:CH7:VAL_GET",
                                            "type": "string",
                                            "compression": "0",
                                            "byteOrder": "LITTLE_ENDIAN",
                                            "shape": None})
        self.assertEqual(dtype, None)

    def test_read_file(self):
        data = reader.read("test.binary")
        print(data.keys())

        self.assertTrue(True)

# Doesn't work due to autoclose of request at the end!
    # def test_read_http(self):
    #     query = {"channels": ["SLG-LSCP3-FNS:CH7:VAL_GET"],
    #              "range": {"type": "date",
    #                        "startDate": "2019-10-02T10:00:43.667105315Z",
    #                        "endDate": "2019-10-03T10:00:43.667105315Z"}}
    #     print(query)
    #
    #     reader = Reader()
    #     # with requests.post("http://localhost:8080/api/v1/query", json=query, stream=True) as r:
    #     with requests.post("http://localhost:8080/api/v1/query", json=query, stream=True) as r:
    #         r.raise_for_status()
    #
    #         buffered_reader = io.BufferedReader(r.raw)
    #         try:
    #             reader.read(buffered_reader)
    #         except Exception as e:
    #             print(reader.messages_read)
    #             raise e
    #         buffered_reader.close()

    def test_read_http2(self):
        query = {"channels": ["SLG-LSCP3-FNS:CH7:VAL_GET"],
                 "range": {"type": "date",
                           "startDate": "2019-10-02T10:00:43.667105315Z",
                           "endDate": "2019-10-03T10:00:43.667105315Z"}}

        data = reader.request(query, url="http://localhost:8080/api/v1/query")
        print(data.keys())
