from __future__ import print_function, division
from datetime import datetime, timedelta  # timezone

import requests
import os

import numpy as np
import pprint
import logging
import re
import json

from data_api import util, pandas_util

logger = logging.getLogger("DataApiClient")
logger.setLevel(logging.INFO)

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')


# One time check at import time to set the default URL (if in SwissFEL network use Swissfel server)

default_base_url = "https://data-api.psi.ch/sf"

if util.check_reachability_server("https://sf-data-api.psi.ch"):
    default_base_url = "https://sf-data-api.psi.ch"
logger.debug("Using endpoint %s" % default_base_url)


def get_data_json(query, base_url=None):

    if base_url is None:
        base_url = default_base_url

    logger.info("curl -H \"Content-Type: application/json\" -X POST -d '" + json.dumps(query) + "' " + base_url + "/query")
    response = requests.post(base_url + '/query', json=query)

    if response.status_code != 200:
        raise RuntimeError("Unable to retrieve data from server: ", response)

    return response.json()


def get_data_iread(query, base_url=None, filename=None):

    if base_url is None:
        base_url = default_base_url

    from data_api.h5 import Serializer
    import data_api.idread as iread

    # https://github.psi.ch/sf_daq/idread_specification#reference-implementation
    # https://github.psi.ch/sf_daq/ch.psi.daq.queryrest#rest-interface

    # curl command that can be used for debugging
    logger.info("curl -H \"Content-Type: application/json\" -X POST -d '"+json.dumps(query)+"' "+base_url + '/query')
    logger.debug(base_url + '/query')

    serializer = Serializer()
    serializer.open(filename)

    with requests.post(base_url + '/query', json=query, stream=True) as response:
        iread.decode(response.raw, serializer=serializer)

    serializer.close()


def search(regex, backends=None, ordering=None, reload=None, base_url=None):
    """
    Search for channels

    :param regex:       regex to search for
    :param backends:    query only specified backends
    :param ordering:    ordering of list [None, "asc", "desc"]
    :param reload:      force reload of cached channel names

    :param base_url:    Base URL of the data api
    :return:            dictionary of backends with its channels matching the regex string
                        example: [{"backend": "somebackend", "channels":["channel"]}, ...]
    """

    if base_url is None:
        base_url = default_base_url

    query = util.construct_channel_list_query(regex, backends=backends, ordering=ordering, reload=reload)

    # For debugging purposes print out curl command
    logger.info("curl -H \"Content-Type: application/json\" -X POST -d '" + json.dumps(query) + "' " + base_url + "/channels")

    response = requests.post(base_url + '/channels', json=query)

    if response.status_code != 200:
        raise RuntimeError("Unable to retrieve data from server: ", response)

    raw_results = response.json()

    # convert the return value to a dictionary
    results = dict()
    for value in raw_results:
        results[value["backend"]] = value["channels"]

    return results


def get_global_date(pulse_ids, mapping_channel="SIN-CVME-TIFGUN-EVR0:BEAMOK", base_url=None):
    """
    Get global data for a given pulse-id

    :param pulse_ids:           list of pulse-ids to retrieve global date for
    :param mapping_channel:     channel that is used to determine pulse-id<>timestamp mapping
    :param base_url:
    :return:                    list of corresponding global timestamps
    """
    if not isinstance(pulse_ids, list):
        pulse_ids = [pulse_ids]

    dates = []
    for pulse_id in pulse_ids:
        # retrieve raw data - data object needs to contain one object for the channel with one data element
        query = util.construct_data_query(mapping_channel, start=pulse_id, range_type="pulseId")
        data = get_data_json(query, base_url=base_url)

        if not pulse_id == data[0]["data"][0]["pulseId"]:
            raise RuntimeError('Unable to retrieve mapping')

        dates.append(util.convert_date(data[0]["data"][0]["globalDate"]))

    if len(pulse_ids) != len(dates):
        raise RuntimeError("Unable to retrieve mapping")

    return dates


def get_pulse_id_from_timestamp(global_timestamp=None, mapping_channel="SIN-CVME-TIFGUN-EVR0:BEAMOK",
                                base_url=default_base_url):
    """
    Retrieve pulse_id for given timestamp

    :param global_timestamp:    timestamp to retrieve pulseid for - if no timestamp is specified take current time
    :param mapping_channel:     Channel used to determine timestamp <> pulse-id mapping
    :param base_url:
    :return:                    pulse-id for timestamp
    """

    # Use current time if no timestamp is specified
    if not global_timestamp:
        global_timestamp = datetime.now()

    _start = global_timestamp - timedelta(seconds=1)
    _end = global_timestamp + timedelta(milliseconds=10)

    # retrieve raw data - data object needs to contain one object for the channel with one data element
    query = util.construct_data_query(mapping_channel, start=_start, end=_end)
    data = get_data_json(query, base_url=base_url)

    if not data[0]["data"]:
        raise ValueError("Requested timestamp not in data buffer. Cannot determine pulse_id.")

    pulse_id = data[0]["data"][-1]["pulseId"]
    # TODO Need to check whether actually does match (check pulse before/after)
    # pulse_id = data[0]["data"][-1]["globalDate"]

    return pulse_id


def get_supported_backends(base_url=None):
    """
    Get supported backend for the endpoint
    :param base_url:
    :return:
    """

    if base_url is None:
        base_url = default_base_url

    logger.info("curl " + base_url + "/params/backends")
    response = requests.get(base_url + '/params/backends')
    return response.json()


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

                query = util.construct_data_query(args.channels.split(","), start=start_pulse, end=end_pulse,
                                                  range_type="pulseId")

                if binary_download:
                    get_data_iread(query, filename=new_filename)

                else:
                    data = get_data_json(query)
                    data = pandas_util.build_pandas_data_frame(data, index_field="pulseId")

                    if data is not None:
                        if filename != "":
                            pandas_util.to_hdf5(data, filename=new_filename, overwrite=args.overwrite)
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

                if split != "" and filename != "" and (end_time-start_time) > util.parse_duration(split):
                    end_time = start_time+util.parse_duration(split)

                if filename != "":
                    if split != "":
                        new_filename = re.sub("\.h5$", "", filename)
                        new_filename = "%s_%03d.h5" % (new_filename, file_counter)
                    else:
                        new_filename = filename

                # construct query
                query = util.construct_data_query(args.channels.split(","), start=start_time, end=end_time,
                                                  range_type="globalDate")
                if binary_download:
                    get_data_iread(query, filename=new_filename)

                else:
                    data = get_data_json(query)
                    data = pandas_util.build_pandas_data_frame(data, index_field="pulseId")

                    if data is not None:

                        if filename != "":
                            pandas_util.to_hdf5(data, filename=new_filename, overwrite=args.overwrite)
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
