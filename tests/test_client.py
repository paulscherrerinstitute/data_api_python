import unittest

import math
import json

import datetime
import dateutil.tz
import data_api as api
from data_api import util, pandas_util

import logging
logger = logging.getLogger(__name__)

# All test_real_* functions will fail if no backend is available. Also these functions are dependent on certain
# channels that need to be accessible from the databuffer/archiver - eventually these channels need to be changed
# if tests are failing
test_offline_only = False
test_local_server = True


class ClientTest(unittest.TestCase):

    @unittest.skipIf(test_offline_only, "Offline only testing enabled")
    def test_real_search(self):

        # If test fails check if channels checked for are still recorded

        channels = api.search(".*BEAMOK$")
        logger.info(channels)
        self.assertIn("SIN-CVME-TIFGUN-EVR0:BEAMOK", channels["sf-databuffer"])

        channels = api.search("FOR-PHASE-AVG")
        logger.info(channels)
        self.assertIn("S10CB01-RBOC-DCP10:FOR-PHASE-AVG", channels["sf-databuffer"])

    @unittest.skipIf(test_offline_only, "Offline only testing enabled")
    def test_real_get_supported_backends(self):
        # If test fails maybe one of the checked backends are currently not online

        backends = api.get_supported_backends()
        logger.info("Returned backends: " + (" ".join(backends)))
        self.assertIn("sf-databuffer", backends)
        self.assertIn("sf-imagebuffer", backends)
        self.assertIn("sf-archiverappliance", backends)

    @unittest.skipIf(test_offline_only, "Offline only testing enabled")
    def test_real_get_global_date(self):
        # If test fails retrieve actual pulseid from the data ui and replace it here
        reference_pulse_id = 7083363958

        dates = api.get_global_date(reference_pulse_id)
        logger.info(dates)
        self.assertEqual(len(dates), 1)
        self.assertIsInstance(dates[0], datetime.datetime)

        dates = api.get_global_date([reference_pulse_id, reference_pulse_id+10, reference_pulse_id+20, reference_pulse_id+30])
        logger.info(dates)
        self.assertEqual(len(dates), 4)
        for value in dates:
            self.assertIsInstance(value, datetime.datetime)

    @unittest.skipIf(test_offline_only, "Offline only testing enabled")
    def test_real_get_pulse_id_from_timestamp(self):
        # If test fails check mapping channel as well as the timestamp (use the timestamp shown from the previous test)

        pulse_id = api.get_pulse_id_from_timestamp(datetime.datetime(2018, 11, 14, 13, 57, 7, 918363, tzinfo=dateutil.tz.tzoffset(None, 3600)))
        logger.info(pulse_id)
        self.assertEqual(pulse_id, 7083363958)

    @unittest.skipIf(test_offline_only, "Offline only testing enabled")
    def test_real_aggregation(self):
        # If test fails check whether channel currently has data

        now = datetime.datetime.now() - datetime.timedelta(hours=10)
        query = util.construct_data_query(["SIN-CVME-TIFGUN-EVR0:BEAMOK"], start=now, delta_range=100,
                                          aggregation=util.construct_aggregation(nr_of_bins=100))
        data = api.get_data_json(query)
        data = pandas_util.build_pandas_data_frame(data)

        logger.info(data)
        self.assertEqual(data.shape[0], 100)

    @unittest.skipIf(not test_local_server, "Testing against local test server not enabled")
    def test_local_get_data_json(self):  # Only works if the testserver.py server is running
        now = datetime.datetime.now()
        end = now
        start = end - datetime.timedelta(minutes=10)

        query = util.construct_data_query(channels=["A", "B"], start=start, end=end)
        data = api.get_data_json(query, base_url="http://localhost:8080/archivertestdata")
        data = pandas_util.build_pandas_data_frame(data)

        # Test function returns 10 datapoints with values from 0 to 9
        self.assertEqual(data.shape[0], 10)

        for i in range(10):
            self.assertEqual(data["A"][i], i)

    @unittest.skipIf(not test_local_server, "Testing against local test server not enabled")
    def test_local_get_data_json_pandas(self):
        # Only works if the testserver.py server is running
        now = datetime.datetime.now()
        end = now
        start = end - datetime.timedelta(minutes=10)

        query = util.construct_data_query(channels=["A", "B"], start=start, end=end)
        data = api.get_data_json(query, base_url="http://localhost:8080/archivertestdatamerge")
        print(data)
        data = pandas_util.build_pandas_data_frame(data)
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

    @unittest.skipIf(test_offline_only, "Offline only testing enabled")
    def test_real_get_data_json_long_timerange(self):
        # If this test fails check whether the used channels are currently available in the databuffer / archiver

        now = datetime.datetime.now()
        end = now - datetime.timedelta(minutes=1)
        start = end - datetime.timedelta(hours=12)

        query = util.construct_data_query(channels=['sf-archiverappliance/S10CB02-CVME-ILK:CENTRAL-CORETEMP',
                                                    'sf-archiverappliance/S10CB02-CVME-ILK:CENTRAL-CORETEMP2'],
                                          start=start, end=end)
        data = api.get_data_json(query)

        logger.info(data[0])
        self.assertTrue(True)

    @unittest.skipIf(test_offline_only, "Offline only testing enabled")
    def test_real_get_data_json_server_side_mapping(self):
        # If this test fails check whether the used channels are currently available in the databuffer / archiver

        now = datetime.datetime.now()
        end = now - datetime.timedelta(minutes=1)
        start = end - datetime.timedelta(minutes=1)

        query = util.construct_data_query(channels=[
                                      # 'sf-archiverappliance/S10CB02-CVME-ILK:CENTRAL-CORETEMP',
                                      # 'sf-archiverappliance/S10CB02-CVME-ILK:CENTRAL-CORETEMP2',
                                      # 'sf-databuffer/S10CB01-RKLY-DCP10:FOR-AMPLT-MAX',
                                      # 'sf-databuffer/S10CB01-RKLY-DCP10:REF-AMPLT-MAX',
                                      'SIN-CVME-TIFGUN-EVR0:BEAMOK',
                                      'sf-archiverappliance/S10CB01-CVME-ILK:P2020-CORETEMP'
                                     ],
                            start=start, end=end,
                            value_mapping=util.construct_value_mapping(incomplete="fill-null"))
        data = api.get_data_json(query)

        logger.info(data['data'][0])
        self.assertTrue(True)

    @unittest.skipIf(test_offline_only, "Offline only testing enabled")
    def test_real_get_data_idread_h5(self):
        now = datetime.datetime.now()
        end = now - datetime.timedelta(minutes=1)
        start = end - datetime.timedelta(minutes=1)

        query = util.construct_data_query(channels=['SIN-CVME-TIFGUN-EVR0:BEAMOK',
                                                    # 'sf-databuffer/SINEG01-RCIR-PUP10:SIG-AMPLT-MAX'
                                                    ],
                                          start=start, end=end, response=util.construct_response(format="rawevent"))
        data = api.get_data_idread(query)

        data = pandas_util.build_pandas_data_frame(data)
        pandas_util.to_hdf5(data, filename='test.h5')

        self.assertTrue(True)

    @unittest.skipIf(test_offline_only, "Offline only testing enabled")
    def test_get_data_idread_json_compare(self):
        """
        Test if get_data_json returns same data as get_data_idread
        """

        now = datetime.datetime.now()
        end = now - datetime.timedelta(minutes=1)
        start = end - datetime.timedelta(minutes=1)

        query = util.construct_data_query(channels=['SIN-CVME-TIFGUN-EVR0:BEAMOK',
                                                    # 'SINEG01-RCIR-PUP10:SIG-AMPLT',
                                                    # 'sf-databuffer/SINEG01-RCIR-PUP10:SIG-AMPLT-MAX'
                                                    ],
                                          start=start, end=end)

        json_data = api.get_data_json(query)
        print(json_data[0]["data"][:10])

        idread_data = api.get_data_idread(query)
        print(idread_data[0]["data"][:10])

        for i in range(len(json_data[0]["data"])):
            self.assertTrue(json_data[0]["data"][i]["pulseId"] == idread_data[0]["data"][i]["pulseId"])
            self.assertTrue(json_data[0]["data"][i]["value"] == idread_data[0]["data"][i]["value"])

    @unittest.skipIf(test_offline_only, "Offline only testing enabled")
    def test_get_data_iread(self):
        now = datetime.datetime.now()
        end = now - datetime.timedelta(minutes=1)
        start = end - datetime.timedelta(minutes=1)

        query = util.construct_data_query(channels=['SIN-CVME-TIFGUN-EVR0:BEAMOK',
                                                    # 'SINEG01-RCIR-PUP10:SIG-AMPLT',
                                                    # 'sf-databuffer/SINEG01-RCIR-PUP10:SIG-AMPLT-MAX'
                                                    ],
                                          start=start, end=end, response=util.construct_response(format="rawevent"))

        data = api.get_data_idread(query)
        print(data)

        self.assertTrue(True)

    @unittest.skipIf(test_offline_only, "Offline only testing enabled")
    def test_get_data_json(self):
        now = datetime.datetime.now()
        end = now - datetime.timedelta(minutes=1)
        start = end - datetime.timedelta(minutes=1)

        query = util.construct_data_query(channels=['SIN-CVME-TIFGUN-EVR0:BEAMOK',
                                                    # 'SINEG01-RCIR-PUP10:SIG-AMPLT',
                                                    # 'sf-databuffer/SINEG01-RCIR-PUP10:SIG-AMPLT-MAX'
                                                    ],
                                          start=start, end=end)

        data = api.get_data_json(query)
        print(len(data[0]["data"][0]))
        print(data[0]["data"][0])

        self.assertTrue(True)

    @unittest.skipIf(test_offline_only, "Offline only testing enabled")
    def test_get_data(self):
        now = datetime.datetime.now()
        end = now - datetime.timedelta(minutes=1)
        start = end - datetime.timedelta(minutes=1)

        data = api.get_data(channels=['SIN-CVME-TIFGUN-EVR0:BEAMOK', 'SINEG01-RCIR-PUP10:SIG-AMPLT'], start=start, end=end)

        print(data.head())
        print(len(data))

        self.assertTrue(True)

# test_compat_* is to ensure that the old api doesn't break

    @unittest.skipIf(not test_local_server, "Testing against local test server not enabled")
    def test_compat_retrieve(self):
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

    @unittest.skipIf(not test_local_server, "Testing against local test server not enabled")
    def test_compat_retrieve_merge(self):  # Only works if the testserver.py server is running
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

    @unittest.skipIf(test_offline_only, "Offline only testing enabled")
    def test_compat_real_aggregation(self):
        now = datetime.datetime.now() - datetime.timedelta(hours=10)
        data = api.get_data(["SINDI01-RIQM-DCP10:FOR-PHASE-AVG", "S10CB01-RBOC-DCP10:FOR-PHASE-AVG"],
                            start=now, delta_range=100, index_field="pulseId",
                            aggregation=util.construct_aggregation(nr_of_bins=100))

        print(data.shape)
        print(data)
        self.assertEqual(data.shape[0], 100)

    @unittest.skipIf(test_offline_only, "Offline only testing enabled")
    def test_compat_real(self):  # Only works if archiver is accessible and data is available for used channel
        # Retrieve data from the archiver

        now = datetime.datetime.now()
        end = now - datetime.timedelta(minutes=1)
        start = end - datetime.timedelta(hours=12)

        data = api.get_data(channels=['sf-archiverappliance/S10-CPCL-VM1MGC:LOAD',
                                      'sf-archiverappliance/S10-CPCL-VM1PDC:LOAD'], start=start, end=end)

        print(data)
        self.assertTrue(True)

    @unittest.skipIf(test_offline_only, "Offline only testing enabled")
    def test_compat_real_raw(self):  # Only works if archiver is accessible and data is available for used channel
        # Retrieve data from the archiver

        now = datetime.datetime.now()
        end = now - datetime.timedelta(minutes=1)
        start = end - datetime.timedelta(minutes=1)

        data = api.get_data(channels=[
                                      # 'sf-archiverappliance/S10CB02-CVME-ILK:CENTRAL-CORETEMP',
                                      # 'sf-archiverappliance/S10CB02-CVME-ILK:CENTRAL-CORETEMP2',
                                      'sf-databuffer/S10CB01-RKLY-DCP10:FOR-AMPLT-MAX',
                                      'sf-databuffer/S10CB01-RKLY-DCP10:REF-AMPLT-MAX',
                                      # 'sf-archiverappliance/S10CB01-CVME-ILK:P2020-CORETEMP'
                                     ],
                            start=start, end=end, mapping_function=lambda d, **kwargs: d,
                            server_side_mapping=True,
                            server_side_mapping_strategy="fill-null"
                            )

        print(data)
        self.assertTrue(True)


if __name__ == '__main__':
    logger.setLevel(logging.INFO)
    logging.getLogger("requests").setLevel(logging.ERROR)
    unittest.main()
