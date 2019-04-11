"""Defines CLI interface for data api 2"""
import argparse
import logging
from pathlib import Path
from datetime import datetime
from datetime import timedelta
import pprint
import sys

import dateutil.parser
import pytz
import h5py

import data_api2 as api
from data_api2 import util

logger = logging.getLogger("DataApiClient")
logger.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

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

def _to_hdf5(data, filename, overwrite=False, compression="gzip",
             compression_opts=5, shuffle=True):
    if not isinstance(filename, (str, Path)):
        raise RuntimeError("Filename must be str or Path")
    if isinstance(filename, str):
        filename = Path(filename)
    dataset_options = {'shuffle':shuffle}
    if compression != 'none':
        dataset_options['compression'] = compression
        if compression == "gzip":
            dataset_options['compression_opts'] = compression_opts
    if filename.exists():
        if overwrite:
            logger.info("Overwriting %s", filename.as_posix())
        else:
            raise RuntimeError("File %s exists, not overwriting by default." %
                               filename.as_posix())
    outfile = h5py.File(filename.as_posix(), "w")

    print(data)
    for channel in data:
        print(channel)
        group = outfile.create_group(channel['channel']['name'])
        values = []
        pulseids = []
        timestamps = []
        for datapoint in channel['data']:
            values.append(datapoint['value'])
            pulseids.append(datapoint['pulseId'])
            timestamps.append(datapoint['timeRaw'])

        print(values, pulseids, timestamps)
        group.create_dataset('values',
                data=values)
        group.create_dataset('pulseids',
                data=pulseids)
        group.create_dataset('timestamps',
                data=timestamps)

    #if data.index.name != "globalDate":
    #    outfile.create_dataset(data.index.name, data=data.index.tolist())

    #for dataset in data.columns:
    #    if dataset == "globalDate":
    #        continue

    #    outfile.create_dataset(dataset, data=data[dataset].tolist(),
    #            **dataset_options)

    outfile.close()



def search(args):
    """CLI Action search"""
    res = api.search(args.regex, backends=["sf-databuffer", "sf-archiverappliance"])
    pprint.pprint(res)
    return 0

def save(args):
    """CLI Action save"""
    # If range is pulse ids
    query = util.construct_data_query(
        channels=args.channels,
        start=args.from_pulse,
        end=args.to_pulse,
        range_type='pulseId',
        event_fields=["value", "time", "pulseId", "timeRaw"]
    )
    # IF range is timestamps
    # TODO
    res = api.get_data_idread(query)
    _to_hdf5(res, "test.h5", overwrite=args.overwrite)
    return 0

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
        "channels", help="Channels to be queried, comma-separated list", default="")
    parser_save.add_argument(
        "--filename", help="Name of the output file", default="")
    parser_save.add_argument(
        "--overwrite", action="store_true", help="Overwrite the output file", default="")
    parser_save.add_argument(
        "--split", help="Number of pulses or duration (ISO8601) per file", default="")
    parser_save.add_argument(
        "--print", help="Prints out the downloaded data. Output can be cut.", action="store_true")
    #parser_save.add_argument(
    #    "--binary", help="Download as binary", action="store_true", default=False)

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
