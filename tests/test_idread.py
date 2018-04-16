import unittest

import data_api.idread as iread

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



if __name__ == '__main__':
    unittest.main()
