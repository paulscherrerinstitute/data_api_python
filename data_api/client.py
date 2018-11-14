from __future__ import print_function, division
from datetime import datetime, timedelta  # timezone

import requests
import os

import numpy as np
import pprint
import logging
import re

from data_api import util

logger = logging.getLogger("DataApiClient")
logger.setLevel(logging.INFO)

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')


# One time check at import time to set the default URL (if in SwissFEL network use Swissfel server)

default_base_url = "https://data-api.psi.ch/sf"

if util.check_reachability_server("https://sf-data-api.psi.ch"):
    default_base_url = "https://sf-data-api.psi.ch"
logger.info("Using enpoint %s" % default_base_url)





def _build_pandas_data_frame(data, **kwargs):
    import pandas
    # for nicer printing
    pandas.set_option('display.float_format', lambda x: '%.3f' % x)

    index_field = kwargs['index_field']

    data_frame = None

    # Same as query["fields"] except "value"
    metadata_fields = ["pulseId", "globalSeconds", "globalDate", "eventCount"]

    for channel_data in data:
        if not channel_data['data']:  # data_entry['data'] is empty, i.e. []
            # No data returned
            logger.warning("no data returned for channel %s" % channel_data['channel']['name'])
            # Create empty pandas data_frame
            tdf = pandas.DataFrame(columns=[index_field, channel_data['channel']['name']])
        else:
            if isinstance(channel_data['data'][0]['value'], dict):
                # Server side aggregation
                entry = []
                keys = sorted(channel_data['data'][0]['value'])

                for x in channel_data['data']:
                    entry.append([x[m] for m in metadata_fields] + [x['value'][k] for k in keys])
                columns = metadata_fields + [channel_data['channel']['name'] + ":" + k for k in keys]

            else:
                # No aggregation
                entry = []
                for data_entry in channel_data['data']:
                    entry.append([data_entry[m] for m in metadata_fields] + [data_entry['value']])
                # entry = [[x[m] for m in metadata_fields] + [x['value'], ] for x in data_entry['data']]
                columns = metadata_fields + [channel_data['channel']['name']]

            tdf = pandas.DataFrame(entry, columns=columns)
            tdf.drop_duplicates(index_field, inplace=True)

            # TODO check if necessary
            # because pd.to_numeric has not enough precision (only float 64, not enough for globalSeconds)
            # does 128 makes sense? do we need nanoseconds?
            conversions = {"pulseId": np.int64}
            for col in tdf.columns:
                if col in conversions:
                    tdf[col] = tdf[col].apply(conversions[col])

        if data_frame is not None:
            data_frame = pandas.merge(data_frame, tdf, how="outer")  # Missing values will be filled with NaN
        else:
            data_frame = tdf

    if data_frame.shape[0] > 0:
        # dataframe is not empty

        # Apply milliseconds rounding
        # this is a string manipulation !
        data_frame["globalNanoseconds"] = data_frame.globalSeconds.map(lambda x: int(x.split('.')[1][3:]))
        data_frame["globalSeconds"] = data_frame.globalSeconds.map(lambda x: float(x.split('.')[0] + "." + x.split('.')[1][:3]))
        # Fix pulseid to int64 - not sure whether this really works
        # data_frame["pulseId"] = data_frame["pulseId"].astype(np.int64)

        data_frame.set_index(index_field, inplace=True)
        data_frame.sort_index(inplace=True)

    return data_frame


class Aggregation(object):
    """ For more details see: https://git.psi.ch/sf_daq/ch.psi.daq.queryrest#data-aggregation """

    def __init__(self, aggregation_type="value", aggregations=["min", "mean", "max"], extrema=None, nr_of_bins=None,
                 duration_per_bin=None, pulses_per_bin=None):

        if (nr_of_bins is not None) + (duration_per_bin is not None) + (pulses_per_bin is not None) > 1:
            raise RuntimeError("Can specify only one of nr_of_bins, duration_per_bin or pulse_per_bin")

        self.aggregation_type = aggregation_type
        self.aggregations = aggregations
        self.extrema = extrema
        self.nr_of_bins = nr_of_bins
        self.duration_per_bin = duration_per_bin
        self.pulses_per_bin = pulses_per_bin

    def get_json(self):
        _aggregation = dict()
        _aggregation["aggregationType"] = self.aggregation_type
        _aggregation["aggregations"] = self.aggregations

        if self.extrema is not None:
            _aggregation["extrema"] = self.extrema

        if self.nr_of_bins is not None:
            _aggregation["nrOfBins"] = self.nr_of_bins
        elif self.duration_per_bin is not None:
            _aggregation["durationPerBin"] = self.duration_per_bin
        elif self.pulses_per_bin is not None:
            _aggregation["pulsesPerBin"] = self.pulses_per_bin

        return _aggregation


def get_data(channels, start=None, end= None, range_type="globalDate", delta_range=1, index_field="globalDate",
             include_nanoseconds=True, aggregation=None, base_url=None,
             server_side_mapping=False, server_side_mapping_strategy="provide-as-is",
             mapping_function=_build_pandas_data_frame):
    """
    Retrieve data from the Data API.

    Examples:
    df = dac.get_data(channels=['SINSB02-RIQM-DCP10:FOR-PHASE-AVG', 'SINSB02-RKLY-DCP10:FOR-PHASE-AVG', 'SINSB02-RIQM-DCP10:FOR-PHASE'], end="2016-07-28 08:05", range_type="globalDate", delta_range=100)
    df = dac.get_data(channels='SINSB02-RIQM-DCP10:FOR-PHASE-AVG', start=10000000, end=10000100, range_type="pulseId")

    Parameters:
    :param mapping_function:
        function to use to interpret returned data
    :param channels: string or list of strings
        string (or list of strings) containing the channel names
    :param start: string, int or float
        start of the range. It is a string in case of a date range, in the form of 'YYYY:MM:DD HH:MM[:SS]',
        an integer in case of pulseId, or a float in case of date range.
    :param end: string, int or float
        end of the range. See start for more details
    :param range_type: string
        range as 'globalDate' (default), 'globalSeconds', 'pulseId'
    :param delta_range: int
        when specifying only start or end, this parameter sets the other end of the range. It is pulses when pulseId
        range is used, seconds otherwise. When only start is defined, delta_range is added to that: conversely when
        only end is defined. You cannot define start, end and delta_range at the same time. If only delta_range is
        specified, then end is by default set to one minute ago, and start computed accordingly
    :param index_field: string
       you can decide whether data is indexed using globalSeconds, pulseId or globalDate.
    :param include_nanoseconds : bool
       NOT YET SUPPORTED! when returned in a DataFrame, globalSeconds are precise up to the microsecond level.
       If you need nanosecond information, put this option to True and a globalNanoseconds column will be created.
    :param base_url:
    :param aggregation:

    Returns:
    df : Pandas DataFrame
        Pandas DataFrame containing indexed data
    """

    if base_url is None:
        base_url = default_base_url

    # Check input parameters
    if range_type not in ["globalDate", "globalSeconds", "pulseId"]:
        RuntimeError("range_type must be 'globalDate', 'globalSeconds', or 'pulseId'")

    if index_field not in ["globalDate", "globalSeconds", "pulseId"]:
        RuntimeError("index_field must be 'globalDate', 'globalSeconds', or 'pulseId'")

    # Check if a single channel is passed instead of a list of channels
    if isinstance(channels, str):
        channels = [channels, ]

    # Build up channel list for the query
    channel_list = []
    for channel in channels:
        channel_name = channel.split("/")

        if len(channel_name) > 2:
            raise RuntimeError("%s is not a valid channel specification" % channel)
        elif len(channel_name) == 1:
            channel_list.append({"name": channel_name[0]})
        else:
            channel_list.append({"name": channel_name[1], "backend": channel_name[0]})

    logger.info("Querying channels: %s" % channels)
    logger.info("Querying on %s between %s and %s" % (range_type, start, end))

    query = dict()
    query["channels"] = channel_list
    query["fields"] = ["pulseId", "globalSeconds", "globalDate", "value", "eventCount"]

    # Set query ranges
    query["range"] = {}
    if range_type == "pulseId":
        _start, _end = util.calculate_range(start, end, delta_range)
        query["range"] = {"endPulseId": str(_end), "startPulseId": str(_start)}
    elif range_type == "globalSeconds":
        _start, _end = util.calculate_range(start, end, delta_range)
        query["range"] = {"startSeconds": "%.9f" % _start, "endSeconds": "%.9f" % _end}
    else:
        _start, _end = util.calculate_time_range(start, end, delta_range)
        query["range"] = {"startDate": datetime.isoformat(_start), "endDate": datetime.isoformat(_end)}

    # Set aggregation
    if aggregation is not None:
        query["aggregation"] = aggregation.get_json()

    if server_side_mapping:
        query["mapping"] = {"incomplete": server_side_mapping_strategy}

    # print(query)

    # Query server
    response = requests.post(base_url + '/query', json=query)

    # Check for successful return of data
    if response.status_code != 200:
        raise RuntimeError("Unable to retrieve data from server: ", response)

    data = response.json()

    # print(data)

    return mapping_function(data, index_field=index_field)


def get_data_iread(channels, start=None, end=None, range_type="globalDate", delta_range=1, index_field="globalDate",
             include_nanoseconds=True, aggregation=None, base_url=default_base_url,
             server_side_mapping=False, server_side_mapping_strategy="provide-as-is",
             mapping_function=_build_pandas_data_frame, filename=None):

    from data_api.h5 import Serializer
    import data_api.idread as iread

    # https://github.psi.ch/sf_daq/idread_specification#reference-implementation
    # https://github.psi.ch/sf_daq/ch.psi.daq.queryrest#rest-interface

    # Check input parameters
    if range_type not in ["globalDate", "globalSeconds", "pulseId"]:
        RuntimeError("range_type must be 'globalDate', 'globalSeconds', or 'pulseId'")

    if index_field not in ["globalDate", "globalSeconds", "pulseId"]:
        RuntimeError("index_field must be 'globalDate', 'globalSeconds', or 'pulseId'")

    # Check if a single channel is passed instead of a list of channels
    if isinstance(channels, str):
        channels = [channels, ]

    # Build up channel list for the query
    channel_list = []
    for channel in channels:
        channel_name = channel.split("/")

        if len(channel_name) > 2:
            raise RuntimeError("%s is not a valid channel specification" % channel)
        elif len(channel_name) == 1:
            channel_list.append({"name": channel_name[0], "backend": "sf-databuffer"})
        else:
            channel_list.append({"name": channel_name[1], "backend": channel_name[0]})

    logger.info("Querying channels: %s" % channels)
    logger.info("Querying on %s between %s and %s" % (range_type, start, end))

    query = dict()

    # Request iread packed data
    query["response"] = {"format": "rawevent"}

    query["channels"] = channel_list
    query["fields"] = ["pulseId", "globalSeconds", "globalDate", "value", "eventCount"]

    # Set query ranges
    query["range"] = {}
    if range_type == "pulseId":
        _start, _end = util.calculate_range(start, end, delta_range)
        query["range"] = {"endPulseId": str(_end), "startPulseId": str(_start)}
    elif range_type == "globalSeconds":
        _start, _end = util.calculate_range(start, end, delta_range)
        query["range"] = {"startSeconds": "%.9f" % _start, "endSeconds": "%.9f" % _end}
    else:
        _start, _end = util.calculate_time_range(start, end, delta_range)
        query["range"] = {"startDate": datetime.isoformat(_start), "endDate": datetime.isoformat(_end)}

    # Set aggregation
    if aggregation is not None:
        query["aggregation"] = aggregation.get_json()

    if server_side_mapping:
        query["mapping"] = {"incomplete": server_side_mapping_strategy}

    # print(query)

    import json
    logger.debug(json.dumps(query))
    logger.debug(base_url + '/query')

    serializer = Serializer()
    serializer.open(filename)

    with requests.post(base_url + '/query', json=query, stream=True) as response:
        iread.decode(response.raw, serializer=serializer)

    serializer.close()


def to_hdf5(data, filename, overwrite=False, compression="gzip", compression_opts=5, shuffle=True):
    import h5py

    dataset_options = {'shuffle': shuffle}
    if compression != 'none':
        dataset_options["compression"] = compression
        if compression == "gzip":
            dataset_options["compression"] = compression_opts

    if os.path.isfile(filename):
        if overwrite:
            logger.warning("Overwriting %s" % filename)
            os.remove(filename)
        else:
            raise RuntimeError("File %s exists, and overwrite flag is False, exiting" % filename)

    outfile = h5py.File(filename, "w")

    if data.index.name != "globalDate":  # Skip globalDate
        outfile.create_dataset(data.index.name, data=data.index.tolist())

    for dataset in data.columns:
        if dataset == "globalDate":  # Skip globalDate
            continue

        outfile.create_dataset(dataset, data=data[dataset].tolist(), **dataset_options)

    outfile.close()


def from_hdf5(filename, index_field="globalSeconds"):
    import h5py
    import pandas

    infile = h5py.File(filename, "r")
    data = pandas.DataFrame()
    for k in infile.keys():
        data[k] = infile[k][:]

    try:
        data.set_index(index_field, inplace=True)
    except:
        raise RuntimeError("Cannot set index on %s, possible values are: %s" % (index_field, str(list(infile.keys()))))

    return data


def search(regex, backends=None, base_url=None):
    """
    Search for channels
    :param regex:       Regular expression to match
    :param backends:    Data backends to search
    :param base_url:    Base URL of the data api
    :return:            List channels
    """

    if base_url is None:
        base_url = default_base_url

    cfg = {
        "regex": regex,
        # "backends": backends,
        "ordering": "asc",
        "reload": "true"
    }

    if backends is not None:
        if isinstance(backends, (list, tuple)):
            print(backends)
            cfg["backends"] = backends
        elif isinstance(backends, str):
            print("Using "+backends)
            cfg["backends"] = [backends]

    response = requests.post(base_url + '/channels', json=cfg)
    return response.json()


def get_global_date(pulse_ids, mapping_channel="SIN-CVME-TIFGUN-EVR0:BEAMOK", base_url=default_base_url):
    if not isinstance(pulse_ids, list):
        pulse_ids = [pulse_ids]

    dates = []
    for pulse_id in pulse_ids:
        # retrieve raw data - data object needs to contain one object for the channel with one data element
        data = get_data(mapping_channel, start=pulse_id, range_type="pulseId", mapping_function=lambda d, **kwargs: d,
                        base_url=base_url)
        if not pulse_id == data[0]["data"][0]["pulseId"]:
            raise RuntimeError('Unable to retrieve mapping')

        dates.append(util.convert_date(data[0]["data"][0]["globalDate"]))

    if len(pulse_ids) != len(dates):
        raise RuntimeError("Unable to retrieve mapping")

    return dates


def get_pulse_id_from_timestamp(global_timestamp=None, mapping_channel="SIN-CVME-TIFGUN-EVR0:BEAMOK",
                                base_url=default_base_url):

    if not global_timestamp:
        global_timestamp = datetime.now()

    start_date = global_timestamp - timedelta(seconds=30)

    # retrieve raw data - data object needs to contain one object for the channel with one data element
    data = get_data(mapping_channel, start=start_date, end=global_timestamp, mapping_function=lambda d, **kwargs: d,
                    base_url=base_url)

    if not data[0]["data"]:
          raise ValueError("Requested timestamp not in data buffer. Cannot determine pulse_id.")

    pulse_id = data[0]["data"][-1]["pulseId"]

    return pulse_id


def get_supported_backends(base_url=None):
    # Get the supported backend for the endpoint
    if base_url is None:
        base_url = default_base_url

    response = requests.get(base_url + '/params/backends')
    return response.json()


def parse_duration(duration_str):
    """https://en.wikipedia.org/wiki/ISO_8601"""

    match = re.match(
        r'P((?P<years>\d+)Y)?((?P<months>\d+)M)?((?P<weeks>\d+)W)?((?P<days>\d+)D)?(T((?P<hours>\d+)H)?((?P<minutes>\d+)M)?((?P<seconds>\d+)S)?)?',
        duration_str
    ).groupdict()

    print(match['years'], match['months'], match['weeks'], match['days'], match['hours'], match['minutes'], match['seconds'])

    if match['years'] is not None or match['months'] is not None:
        raise RuntimeError('year and month durations are not supported')

    delta = timedelta(hours=0 if match['hours'] is None else int(match['hours']),
                      minutes=0 if match['minutes'] is None else int(match['minutes']),
                      seconds=0 if match['seconds'] is None else int(match['seconds']),
                      days=0 if match['days'] is None else int(match['days']),
                      weeks=0 if match['weeks'] is None else int(match['weeks']))

    return delta


def cli():
    import argparse

    time_end = datetime.now()
    time_start = time_end - timedelta(minutes=1)

    parser = argparse.ArgumentParser(description='Command line interface for the Data API')
    parser.add_argument('action', type=str, default="",
                        help='Action to be performed. Possibilities: search, save')
    parser.add_argument("--regex", type=str, help="String to be searched", default="")
    parser.add_argument("--from_time", type=str, help="Start time for the data query", default=time_start)
    parser.add_argument("--to_time", type=str, help="End time for the data query", default=time_end)
    parser.add_argument("--from_pulse", type=str, help="Start pulseId for the data query", default=-1)
    parser.add_argument("--to_pulse", type=str, help="End pulseId for the data query", default=-1)
    parser.add_argument("--channels", type=str, help="Channels to be queried, comma-separated list", default="")
    parser.add_argument("--filename", type=str, help="Name of the output file", default="")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite the output file", default="")
    # parser.add_argument("--split", action="store_true", help="Split output file", default="")
    parser.add_argument("--split", type=str, help="Number of pulses or duration (ISO8601) per file", default="")
    parser.add_argument("--print", help="Prints out the downloaded data. Output can be cut.", action="store_true")
    parser.add_argument("--binary", help="Download as binary", action="store_true", default=False)

    args = parser.parse_args()

    split = args.split
    filename = args.filename
    binary_download = args.binary

    # Check if output files already exist
    if not args.overwrite and filename != "":
        import os.path
        if os.path.isfile(filename):
            logger.error("File %s already exists" % filename)
            return

        n_filename = "%s_%03d.h5" % (re.sub("\.h5$", "", filename), 0)
        if os.path.isfile(n_filename):
            logger.error("File %s already exists" % n_filename)
            return

    data = None
    if args.action == "search":
        if args.regex == "":
            logger.error("Please specify a regular expression with --regex\n")
            parser.print_help()
            return
        pprint.pprint(search(args.regex, backends=["sf-databuffer", "sf-archiverappliance"], base_url=default_base_url))
    elif args.action == "save":
        if args.filename == "" and not args.print:
            logger.warning("Please select either --print or --filename")
            parser.print_help()
            return
        if args.from_pulse != -1:
            if args.to_pulse == -1:
                logger.error("Please set a range limit with --to_pulse")
                return

            start_pulse = int(args.from_pulse)
            file_counter = 0

            while True:

                end_pulse = int(args.to_pulse)

                if start_pulse == end_pulse:
                    break

                if split != "" and filename != "" and (end_pulse-start_pulse) > int(split):
                    end_pulse = start_pulse+int(split)

                if filename != "":
                    if split != "":
                        new_filename = re.sub("\.h5$", "", filename)
                        new_filename = "%s_%03d.h5" % (new_filename, file_counter)
                    else:
                        new_filename = filename

                if binary_download:
                    get_data_iread(args.channels.split(","), start=start_pulse, end=end_pulse, range_type="pulseId",
                                   index_field="pulseId", filename=new_filename)

                else:
                    data = get_data(args.channels.split(","), start=start_pulse, end=end_pulse, range_type="pulseId", index_field="pulseId")

                    if data is not None:
                        if filename != "":
                            to_hdf5(data, filename=new_filename, overwrite=args.overwrite)
                        elif args.print:
                            print(data)
                        else:
                            logger.warning("Please select either --print or --filename")
                            parser.print_help()

                start_pulse = end_pulse
                file_counter += 1
        else:
            start_time = util.convert_date(args.from_time)
            file_counter = 0

            while True:

                end_time = util.convert_date(args.to_time)

                if start_time == end_time:
                    break

                if split != "" and filename != "" and (end_time-start_time) > parse_duration(split):
                    end_time = start_time+parse_duration(split)

                if filename != "":
                    if split != "":
                        new_filename = re.sub("\.h5$", "", filename)
                        new_filename = "%s_%03d.h5" % (new_filename, file_counter)
                    else:
                        new_filename = filename

                if binary_download:
                    get_data_iread(args.channels.split(","), start=start_time, end=end_time,
                                   range_type="globalDate",
                                   index_field="pulseId", filename=new_filename)

                else:
                    data = get_data(args.channels.split(","), start=start_time, end=end_time, range_type="globalDate", index_field="pulseId")

                    if data is not None:

                        if filename != "":
                            to_hdf5(data, filename=new_filename, overwrite=args.overwrite)
                        elif args.print:
                            print(data)
                        else:
                            logger.warning("Please select either --print or --filename")
                            parser.print_help()

                start_time = end_time
                file_counter += 1
    else:
        parser.print_help()
        return


if __name__ == "__main__":
    cli()
    # Testing:
    # --from_pulse 5166875100 --to_pulse 5166876100 --channels sf-databuffer/SINEG01-RCIR-PUP10:SIG-AMPLT --split 500 --filename testit_000.h5 save
    # --from_time "2018-04-05 09:00:00.000" --to_time "2018-04-05 10:00:00.000" --channels sf-databuffer/SINEG01-RCIR-PUP10:SIG-AMPLT --split PT30M --filename testit_000.h5 save
