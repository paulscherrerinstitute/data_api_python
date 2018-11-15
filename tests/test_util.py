import unittest

from data_api import util
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

        query = util.construct_channel_list_query(".*BEAM-OK$")
        self.assertEqual(query["regex"], ".*BEAM-OK$")
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

        # TODO create more test cases !
        query = util.construct_data_query("CHANNEL_A", start=10, range_type="pulseId")
        logger.info(query)

        query = util.construct_data_query("backend01/CHANNEL_A", start=10, range_type="pulseId")
        logger.info(query)


if __name__ == '__main__':
    unittest.main()
