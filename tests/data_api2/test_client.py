import unittest

import datetime
import dateutil.tz
from data_api2 import util, client
import numpy

import logging
logger = logging.getLogger(__name__)

# All test_real_* functions will fail if no backend is available. Also these functions are dependent on certain
# channels that need to be accessible from the databuffer/archiver - eventually these channels need to be changed
# if tests are failing
test_offline_only = False
test_local_server = True


class ClientTest(unittest.TestCase):

    @unittest.skipIf(test_offline_only, "Offline only testing enabled")
    def test_search(self):

        # If test fails check if channels checked for are still recorded

        channels = client.search(".*BEAMOK$")
        logger.info(channels)
        self.assertIn("SIN-CVME-TIFGUN-EVR0:BEAMOK", channels["sf-databuffer"])

        channels = client.search("FOR-PHASE-AVG")
        logger.info(channels)
        self.assertIn("S10CB01-RBOC-DCP10:FOR-PHASE-AVG", channels["sf-databuffer"])

    @unittest.skipIf(test_offline_only, "Offline only testing enabled")
    def test_get_supported_backends(self):
        # If test fails maybe one of the checked backends are currently not online

        backends = client.get_supported_backends()
        logger.info("Returned backends: " + (" ".join(backends)))
        self.assertIn("sf-databuffer", backends)
        self.assertIn("sf-imagebuffer", backends)
        self.assertIn("sf-archiverappliance", backends)

    @unittest.skipIf(test_offline_only, "Offline only testing enabled")
    def test_get_timestamp_from_pulse_id(self):
        # If test fails retrieve actual pulseid from the data ui and replace it here
        reference_pulse_id = 7083363958

        dates = client.get_timestamp_from_pulse_id(reference_pulse_id)
        logger.info(dates)
        self.assertEqual(len(dates), 1)
        self.assertIsInstance(dates[0], datetime.datetime)

        dates = client.get_timestamp_from_pulse_id([reference_pulse_id, reference_pulse_id+10, reference_pulse_id+20, reference_pulse_id+30])
        logger.info(dates)
        self.assertEqual(len(dates), 4)
        for value in dates:
            self.assertIsInstance(value, datetime.datetime)

    @unittest.skipIf(test_offline_only, "Offline only testing enabled")
    def test_get_pulse_id_from_timestamp(self):
        # If test fails check mapping channel as well as the timestamp (use the timestamp shown from the previous test)

        pulse_id = client.get_pulse_id_from_timestamp(datetime.datetime(2018, 11, 14, 13, 57, 7, 918363, tzinfo=dateutil.tz.tzoffset(None, 3600)))
        logger.info(pulse_id)
        self.assertEqual(pulse_id, 7083363958)

    @unittest.skipIf(test_offline_only, "Offline only testing enabled")
    def test_get_data_aggregation(self):
        # If test fails check whether channel currently has data

        now = datetime.datetime.now() - datetime.timedelta(hours=10)
        query = util.construct_data_query(["SIN-CVME-TIFGUN-EVR0:BEAMOK"], start=now, delta_range=100,
                                          aggregation=util.construct_aggregation(nr_of_bins=100))
        data = client.get_data(query)

        logger.info(data)
        self.assertEqual(len(data[0]["data"]), 100)

    @unittest.skipIf(not test_local_server, "Testing against local test server not enabled")
    def test_local_get_data(self):  # Only works if the testserver.py server is running
        now = datetime.datetime.now()
        end = now
        start = end - datetime.timedelta(minutes=10)

        query = util.construct_data_query(channels=["A", "B"], start=start, end=end)
        data = client.get_data(query, base_url="http://localhost:8080/archivertestdata")

        # Test function returns 10 datapoints with values from 0 to 9
        self.assertEqual(len(data[0]["data"]), 10)

        for i in range(10):
            self.assertEqual(data[0]["data"][i]["value"], i)

    @unittest.skipIf(test_offline_only, "Offline only testing enabled")
    def test_get_data_long_timerange(self):
        # If this test fails check whether the used channels are currently available in the databuffer / archiver

        now = datetime.datetime.now()
        end = now - datetime.timedelta(minutes=1)
        start = end - datetime.timedelta(hours=12)

        query = util.construct_data_query(channels=['sf-archiverappliance/S10CB02-CVME-ILK:CENTRAL-CORETEMP',
                                                    'sf-archiverappliance/S10CB02-CVME-ILK:CENTRAL-CORETEMP2'],
                                          start=start, end=end)
        data = client.get_data_json(query)

        logger.info(data[0])
        self.assertTrue(True)

    @unittest.skipIf(test_offline_only, "Offline only testing enabled")
    def test_get_data_server_side_mapping(self):
        # When server side mapping is enabled the returned dictionary is different!
        # If this test fails check whether the used channels are currently available in the databuffer / archiver

        now = datetime.datetime.now()
        end = now - datetime.timedelta(minutes=1)
        start = end - datetime.timedelta(minutes=1)

        query = util.construct_data_query(channels=['SIN-CVME-TIFGUN-EVR0:BEAMOK',
                                                    'sf-archiverappliance/S10CB01-CVME-ILK:P2020-CORETEMP'],
                                          start=start, end=end,
                                          # value_mapping=util.construct_value_mapping(incomplete="fill-null"))
                            # value_mapping=util.construct_value_mapping(incomplete="drop"))
                            value_mapping=util.construct_value_mapping(incomplete="provide-as-is"))

        json_data = client.get_data_json(query)
        idread_data = client.get_data_idread(query)

        for i in range(len(json_data)):

            # logger.info(data['data'][i])
            json_value = json_data[i]
            idread_value = idread_data[i]

            # self.assertTrue(len(json_value) == len(idread_value))

            if json_value[0] is None:
                self.assertEqual(json_value[0], idread_value[0], "failed on index: %d" % i)
            else:
                self.assertTrue(json_value[0]["channel"] == "SIN-CVME-TIFGUN-EVR0:BEAMOK", "failed on index: %d" % i)
                self.assertTrue(idread_value[0]["channel"] == "SIN-CVME-TIFGUN-EVR0:BEAMOK", "failed on index: %d" % i)
                self.assertEqual(json_value[0]["value"], idread_value[0]["value"], "failed on index: %d" % i)

            if json_value[1] is None:
                self.assertEqual(json_value[1], idread_value[1], "failed on index: %d" % i)
            else:
                self.assertTrue(json_value[1]["channel"] == "S10CB01-CVME-ILK:P2020-CORETEMP", "failed on index: %d" % i)
                self.assertTrue(idread_value[1]["channel"] == "S10CB01-CVME-ILK:P2020-CORETEMP", "failed on index: %d" % i)
                self.assertEqual(json_value[1]["value"], idread_value[1]["value"], "failed on index: %d" % i)

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

        json_data = client.get_data_json(query)
        print(json_data[0]["data"][:10])

        idread_data = client.get_data_idread(query)
        print(idread_data[0]["data"][:10])

        for i in range(len(json_data[0]["data"])):
            json_value = json_data[0]["data"][i]
            idread_value = idread_data[0]["data"][i]
            self.assertEqual(json_value["pulseId"], idread_value["pulseId"])
            self.assertEqual(json_value["value"], idread_value["value"])

            # Due to rounding errors and limited precision of datetime to microseconds sometimes the parsing from the
            # long does not return the correct date object, therefore sometimes the compare here would fail
            # TODO try to reenable as soon as json retrieval offers globalTime
            # self.assertEqual(json_value["time"], idread_value["time"])

    @unittest.skipIf(test_offline_only, "Offline only testing enabled")
    def test_get_data_idread_json_compare_image(self):
        """
        Test if get_data_json returns same data as get_data_idread
        """

        query = util.construct_data_query(channels=['SF-IMAGEBUFFER/SLAAR21-LCAM-C511:FPICTURE',
                                                    # 'SINEG01-RCIR-PUP10:SIG-AMPLT',
                                                    # 'sf-databuffer/SINEG01-RCIR-PUP10:SIG-AMPLT-MAX'
                                                    ],
                                          start=7928427268, end=7928427268)

        json_data = client.get_data_json(query)
        # print(json_data[0]["data"][:10])

        idread_data = client.get_data_idread(query)
        # print(idread_data[0]["data"][:10])

        for i in range(len(json_data[0]["data"])):
            self.assertTrue(json_data[0]["data"][i]["pulseId"] == idread_data[0]["data"][i]["pulseId"])
            self.assertTrue(numpy.array_equal(json_data[0]["data"][i]["value"],idread_data[0]["data"][i]["value"]))

    @unittest.skipIf(test_offline_only, "Offline only testing enabled")
    def test_get_data_idread(self):
        now = datetime.datetime.now()
        end = now - datetime.timedelta(minutes=1)
        start = end - datetime.timedelta(minutes=1)

        query = util.construct_data_query(channels=['SIN-CVME-TIFGUN-EVR0:BEAMOK'],
                                          start=start, end=end)

        data = client.get_data_idread(query)
        logger.info(data[0]["data"][0])

        self.assertTrue(True)

    @unittest.skipIf(test_offline_only, "Offline only testing enabled")
    def test_get_data_json(self):
        now = datetime.datetime.now()
        end = now - datetime.timedelta(minutes=1)
        start = end - datetime.timedelta(minutes=1)

        query = util.construct_data_query(channels=['SIN-CVME-TIFGUN-EVR0:BEAMOK'], start=start, end=end)
        data = client.get_data_json(query)

        value = data[0]["data"][0]
        logger.info(value)

        self.assertTrue("value" in value)
        self.assertTrue("time" in value)
        self.assertTrue("pulseId" in value)

        query = util.construct_data_query(channels=['SIN-CVME-TIFGUN-EVR0:BEAMOK'], start=start, end=end,
                                          value_mapping=util.construct_value_mapping(incomplete="fill-null"))
        data = client.get_data_json(query)

        value = data[0][0]  # first row first column
        logger.info(value)

        self.assertTrue("value" in value)
        self.assertTrue("time" in value)
        self.assertTrue("pulseId" in value)


if __name__ == '__main__':
    logger.setLevel(logging.INFO)
    logging.getLogger("requests").setLevel(logging.ERROR)
    unittest.main()
