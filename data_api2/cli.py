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

# Do not modify global logging settings in a library!
# For the logger, the recommended Python style is to use the module name.
logger = logging.getLogger(__name__)


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

def to_hdf5(data, filename, overwrite=False, compression="gzip",
             compression_opts=5, shuffle=True):
    #pprint.pprint(data)
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

    for channel in data:
        group = outfile.create_group(channel['channel']['name'])
        values = []
        pulseids = []
        timestamps = []
        for datapoint in channel['data']:
            values.append(datapoint['value'])
            if 'pulseId' in datapoint:
                pulseids.append(datapoint['pulseId'])
            timestamps.append(datapoint['timeRaw'])

        group.create_dataset('values', data=values)
        if pulseids:
            group.create_dataset('pulseids', data=pulseids)
        group.create_dataset('timestamps', data=timestamps)

    outfile.close()

def from_hdf5(filename):
    if not isinstance(filename, (str, Path)):
        raise RuntimeError("Filename must be str or Path")
    if isinstance(filename, str):
        filename = Path(filename)
    infile = h5py.File(filename, "r")

    res = []

    for (channel_name, data) in infile.items():
        datapoints = []
        for i in range(len(data["values"])):
            if "pulseids" in data:
                datapoint = {
                    "value": data["values"][i],
                    "timeRaw": data["timestamps"][i],
                    "pulseId": data["pulseids"][i]}
            else:
                datapoint = {
                    "value": data["values"][i],
                    "timeRaw": data["timestamps"][i]}
            datapoints.append(datapoint)
        res.append(
            {
                "channel":{"name": channel_name, "backend": "hdf5"},
                "data":datapoints
            })

    return res

    infile.close()


def search(args):
    """CLI Action search"""
    res = api.search(args.regex, backends=["sf-databuffer", "sf-archiverappliance"])
    pprint.pprint(res)
    return 0

def save(args):
    """CLI Action save"""
    channels = args.channels.split(',')
    # Figure out wihch channels have pulse ids
    db_channels = []
    for channel in channels:
        res = api.search(channel, backends=["sf-databuffer"])
        if res['sf-databuffer']:
            db_channels.append(channel)

    aa_channels = []
    for channel in channels:
        if channel not in db_channels:
            aa_channels.append(channel)

    if args.from_pulse != -1 and args.to_pulse != -1:
        if aa_channels:
            logger.error(
                "Cannot search archiver appliance channels with pulse "
                "ids. The following channels were not found in data buffer: "
                "%s", aa_channels)
        # If range is pulse ids
        query = util.construct_data_query(
            channels=db_channels,
            start=args.from_pulse,
            end=args.to_pulse,
            range_type='pulseId',
            event_fields=["value", "pulseId", "timeRaw"]
        )
        res = api.get_data_idread(query)

    else:
        # If range is time
        query = util.construct_data_query(
            channels=aa_channels,
            start=args.from_time,
            end=args.to_time,
            event_fields=["value", "timeRaw"]
        )
        res = api.get_data_idread(query)

        # If range is time
        query = util.construct_data_query(
            channels=db_channels,
            start=args.from_time,
            end=args.to_time,
            event_fields=["value", "pulseId", "timeRaw"]
        )
        res += api.get_data_idread(query)
    to_hdf5(res, args.filename, overwrite=args.overwrite)
    return 0

def cli_open(args):
    """CLI Action open"""
    res = from_hdf5(args.filename)
    pprint.pprint(res)
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
        "filename", help="Name of the output file", default="")
    parser_save.add_argument(
        "channels", help="Channels to be queried, comma-separated list", default="")
    parser_save.add_argument(
        "--overwrite", action="store_true", help="Overwrite the output file", default="")
    parser_save.add_argument(
        "--split", help="Number of pulses or duration (ISO8601) per file", default="")
    parser_save.add_argument(
        "--print", help="Prints out the downloaded data. Output can be cut.", action="store_true")
    #parser_save.add_argument(
    #    "--binary", help="Download as binary", action="store_true", default=False)

    parser_open = subparsers.add_parser('open')
    parser_open.add_argument(
        "filename", help="Name of the output file", default="")

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
    if args.action == 'open':
        return cli_open(args)

    return 0


if __name__ == '__main__':
    logging.basicConfig(format="%(asctime)-15s  %(levelname)s  %(message)s")
    sys.exit(main())
