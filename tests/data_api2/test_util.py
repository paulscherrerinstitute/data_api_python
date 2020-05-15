import unittest

from data_api2 import util
import datetime

import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logging.getLogger("requests").setLevel(logging.ERROR)


class ClientTest(unittest.TestCase):

    def test_check_reachability_server(self):

        # Usually google should be reachable
        self.assertTrue(util.check_reachability_server("https://www.google.com"))
        self.assertFalse(util.check_reachability_server("https://www.google-nonexisting.com"))

    def test_convert_date(self):

        # Check if correct timezone information is attached
        date = util.convert_date("2017-12-15 15:05:43.258077+01:00")  # have offset as set
        self.assertEqual(date.utcoffset(), datetime.timedelta(hours=1))
        date = util.convert_date("2016-07-29 14:01")  # need to have +2 offset (summer time)
        self.assertEqual(date.utcoffset(), datetime.timedelta(hours=2))
        date = util.convert_date("2018-11-14 11:17:38.362582")  # need to have +1 offset (winter time)
        self.assertEqual(date.utcoffset(), datetime.timedelta(hours=1))

    def test_construct_channel_list_query(self):

        query = util.construct_channel_list_query(".*BUNCH-1-OK$")
        self.assertEqual(query["regex"], ".*BUNCH-1-OK$")
        self.assertNotIn("ordering", query)
        self.assertNotIn("backends", query)
        self.assertNotIn("reload", query)

    def test_parse_duration(self):

        # Month and year durations are not supported!
        raised = False
        try:
            util.parse_duration("P2Y")
        except RuntimeError:
            raised = True
        self.assertTrue(raised)

        # Check correct parsing
        delta = util.parse_duration("PT1H")
        self.assertEqual(delta, datetime.timedelta(hours=1))

        delta = util.parse_duration("PT1H70M")
        self.assertEqual(delta, datetime.timedelta(hours=1, minutes=70))

    def test_construct_data_query(self):

        raised = False
        try:
            util.construct_data_query("CHANNEL_A")
        except ValueError:
            raised = True
        self.assertTrue(raised)

        query = util.construct_data_query("CHANNEL_A", start=10)
        self.assertEqual(len(query["channels"]), 1)
        self.assertEqual(query["channels"][0]["name"], "CHANNEL_A")
        self.assertNotIn("backend", query["channels"][0])
        self.assertNotIn("ordering", query)
        logger.info(query)

        query = util.construct_data_query("backend01/CHANNEL_A", start=10)
        self.assertEqual(len(query["channels"]), 1)
        self.assertEqual(query["channels"][0]["name"], "CHANNEL_A")
        self.assertEqual(query["channels"][0]["backend"], "backend01")
        self.assertNotIn("ordering", query)
        logger.info(query)

        # Ordering checks
        query = util.construct_data_query("backend01/CHANNEL_A", start=10, ordering='asc')
        self.assertEqual(query["ordering"], "asc")

        raised = False
        try:
            util.construct_data_query("backend01/CHANNEL_A", start=10, ordering='bla')
        except ValueError:
            raised = True
        self.assertTrue(raised)

    def test_construct_range(self):
        raised = False
        try:
            util.construct_range()
        except ValueError:
            raised = True
        self.assertTrue(raised)

        query = util.construct_range(start=10)
        self.assertEqual(query["startPulseId"], 10)
        self.assertNotIn("startInclusive", query)
        self.assertNotIn("startExpansion", query)
        self.assertNotIn("endInclusive", query)
        self.assertNotIn("endExpansion", query)

        query = util.construct_range(start=10, start_inclusive=True)
        self.assertEqual(query["startPulseId"], 10)
        self.assertTrue(query["startInclusive"])
        self.assertNotIn("startExpansion", query)
        self.assertNotIn("endInclusive", query)
        self.assertNotIn("endExpansion", query)

        query = util.construct_range(start=10, start_inclusive=True, start_expansion=True, end_inclusive=True, end_expansion=True)
        self.assertEqual(query["startPulseId"], 10)
        self.assertTrue(query["startInclusive"])
        self.assertTrue(query["startExpansion"])
        self.assertTrue(query["endInclusive"])
        self.assertTrue(query["endExpansion"])

    def test_construct_value_mapping(self):

        incomplete_options = ["provide-as-is", "drop", "fill-null"]
        alignment_options = ["by-pulse", "by-time", "none"]
        aggregations_options = ["count", "min", "mean", "max"]

        for value in incomplete_options:
            mapping = util.construct_value_mapping(incomplete=value)
            self.assertEqual(mapping["incomplete"], value)
            self.assertNotIn("alignment", mapping)
            self.assertNotIn("aggregations", mapping)

        raised = False
        try:
            util.construct_value_mapping(incomplete="nonexistent")
        except ValueError:
            raised = True
        self.assertTrue(raised)

        for value in alignment_options:
            mapping = util.construct_value_mapping(alignment=value)
            self.assertEqual(mapping["alignment"], value)
            self.assertNotIn("incomplete", mapping)
            self.assertNotIn("aggregations", mapping)

        raised = False
        try:
            util.construct_value_mapping(alignment="nonexistent")
        except ValueError:
            raised = True
        self.assertTrue(raised)

        for value in aggregations_options:
            mapping = util.construct_value_mapping(aggregations=value)
            self.assertEqual(mapping["aggregations"], [value])
            self.assertNotIn("alignment", mapping)
            self.assertNotIn("incomplete", mapping)

        raised = False
        try:
            util.construct_value_mapping(aggregations=['count', 'something'])
        except ValueError:
            raised = True
        self.assertTrue(raised)


if __name__ == '__main__':
    unittest.main()
