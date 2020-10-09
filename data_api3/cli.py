"""Defines CLI interface for data api 2"""
import argparse
from datetime import datetime
from datetime import timedelta
import sys
import logging
import dateutil.parser
import pytz
import requests
import data_api3
import data_api3 as api
import data_api3.h5 as h5


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


def search(args):
    """CLI Action search"""
    res = requests.get(f"{args.baseurl}/channels", {"regex": args.regex})
    res.raise_for_status()
    for channel in res.json():
        print(channel)
    return 0


def save(args):

    baseurl = args.baseurl
    filename = args.filename
    channels = args.channels
    start = args.start.astimezone(pytz.timezone('UTC')).strftime("%Y-%m-%dT%H:%M:%S.%fZ")  # isoformat()  # "2019-12-13T09:00:00.000000000Z"
    end = args.end.astimezone(pytz.timezone('UTC')).strftime("%Y-%m-%dT%H:%M:%S.%fZ")  # .isoformat()  # "2019-12-13T09:00:00.100000000Z"

    query = {
        "channels": channels,
        "range": {
            "type": "date",
            "startDate": start,
            "endDate": end
        }
    }

    h5.request(query, filename, url=f"{baseurl}/query")

    return 0


def parse_args():
    """Parse cli arguments with argparse"""
    time_end = datetime.now()
    time_start = time_end - timedelta(minutes=1)

    parser = argparse.ArgumentParser(description='Command line interface for the Data API-3 ' + data_api3.version())

    parser.add_argument(
        "--baseurl", help="Base url of the service.  Example: http://sf-daq-5.psi.ch:8371/api/1.0.1   Old default: http://sf-daq-5.psi.ch:8080/api/v1", default="http://sf-daq-5.psi.ch:8080/api/v1")

    subparsers = parser.add_subparsers(
        help='Action to be performed', metavar='action', dest='action')

    parser_search = subparsers.add_parser('search')
    parser_search.add_argument("regex", help="String to be searched", nargs='?', default=".*")

    parser_save = subparsers.add_parser('save')

    parser_save.add_argument(
        "filename", help="Name of the output file")

    parser_save.add_argument(
        "start", help="Start time for the data query (default: now-1m)",
        default=time_start, metavar='TIMESTAMP', type=_convert_date)
    parser_save.add_argument(
        "end", help="End time for the data query (default: now)", default=time_end,
        metavar='TIMESTAMP', type=_convert_date)

    parser_save.add_argument(
        "channels", help="Channels to be queried, space-separated list", nargs='+')

    parser_save.add_argument(
        "--overwrite", action="store_true", help="Overwrite the output file", default="")
    parser_save.add_argument(
        "--split", help="Number of pulses or duration (ISO8601) per file", default="")

    args = parser.parse_args()
    if args.action is None:
        parser.print_help()
        sys.exit(-1)

    return args


def main():
    """Main function"""
    args = parse_args()

    try:
        if args.action == 'search':
            return search(args)
        if args.action == 'save':
            return save(args)
    except RuntimeError as e:
        print(e)

    return 0


if __name__ == '__main__':
    logging.basicConfig(format="%(asctime)-15s  %(levelname)s  %(message)s")
    sys.exit(main())
