"""Defines CLI interface for data api 2"""
import sys
from datetime import datetime
from datetime import timedelta
import dateutil.parser
import argparse
import pprint

import pytz

import data_api2 as api

def search(args):
    """CLI Action search"""
    res = api.search(args.regex, backends=["sf-databuffer", "sf-archiverappliance"])
    pprint.pprint(res)
    return 0

def save(args):
    """CLI Action save"""
    pass

def _convert_date(date_string):
    if isinstance(date_string, str):
        date = dateutil.parser.parse(date_string)
    elif isinstance(date_string, datetime):
        date = date_string
    else:
        raise argparse.ArgumentTypeError("Unsupported date type: " + type(date_string))

    if date.tzinfo is None:  # localize time if necessary
        date = pytz.timezone('Europe/Zurich').localize(date)

    return date

def parse_args():
    """Parse cli arguments with argparse"""
    time_end = datetime.now()
    time_start = time_end - timedelta(minutes=1)

    parser = argparse.ArgumentParser(description='Command line interface for the Data API 2')

    subparsers = parser.add_subparsers(
        help='Action to be performed', metavar='action', dest='action')
    parser_search = subparsers.add_parser('search')
    parser_search.add_argument("regex", help="String to be searched")

    parser_save = subparsers.add_parser('save')
    parser_save.add_argument(
        "--from_time", help="Start time for the data query (default: now-1m)",
        default=time_start, metavar='TIMESTAMP', type=_convert_date)
    parser_save.add_argument(
        "--to_time", help="End time for the data query (default: now)", default=time_end,
        metavar='TIMESTAMP', type=_convert_date)
    parser_save.add_argument(
        "--from_pulse", help="Start pulseId for the data query", default=-1, metavar='PULSE_ID')
    parser_save.add_argument(
        "--to_pulse", help="End pulseId for the data query", default=-1, metavar='PULSE_ID')
    parser_save.add_argument(
        "--channels", help="Channels to be queried, comma-separated list", default="")
    parser_save.add_argument(
        "--filename", help="Name of the output file", default="")
    parser_save.add_argument(
        "--overwrite", action="store_true", help="Overwrite the output file", default="")
    parser_save.add_argument(
        "--split", help="Number of pulses or duration (ISO8601) per file", default="")
    parser_save.add_argument(
        "--print", help="Prints out the downloaded data. Output can be cut.", action="store_true")
    parser_save.add_argument(
        "--binary", help="Download as binary", action="store_true", default=False)

    args = parser.parse_args()
    if args.action is None:
        parser.print_help()
        sys.exit(-1)

    return args

def main():
    """Main function"""
    args = parse_args()

    if args.action == 'search':
        return search(args)
    if args.action == 'save':
        return save(args)

    return 0


if __name__ == '__main__':
    sys.exit(main())
