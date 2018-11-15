import pytz
import dateutil.parser
from datetime import datetime, timedelta
import re


def check_reachability_server(endpoint):
    """
    Check whether the enpoint server is reachable
    :param endpoint:    endpoint url
    :return:            True if enpoint is reachable, False otherwise
    """
    import socket
    import re

    m = re.match(r'^((http|https):\/\/)?([^\/]*).*$', endpoint)
    port = 80
    if m.group(2) == "https":
        port = 443
    hostname = m.group(3)

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(0.1)

    try:
        sock.connect((hostname, port))
    except socket.error:
        # Endpoint is not reachable
        return False
    finally:
        sock.close()

    # Endpoint is reachable
    return True


def convert_date(date_string):
    """
    Convert a date string to datetime (if not already datetime) and attach timezone (if not already attached)
    :param date_string:     Date string to convert
    :return:                datetime with correct timezone attached
    """

    if isinstance(date_string, str):
        date = dateutil.parser.parse(date_string)
    elif isinstance(date_string, datetime):
        date = date_string
    else:
        raise ValueError("Unsupported date type: " + type(date_string))

    if date.tzinfo is None:  # localize time if necessary
        date = pytz.timezone('Europe/Zurich').localize(date)

    return date


def calculate_range(start, end, delta):
    """
    Calculate start - end range based on given start, end and/or delta parameter
    :param start:
    :param end:
    :param delta:
    :return: start end tuple
    """
    if start is None and end is None:
        raise ValueError("Must select at least start or end")

    if start is not None and end is None:
        end = start + delta - 1
    elif start is None and end is not None:
        start = end - delta + 1

    return start, end


def calculate_time_range(start_date, end_date, delta_time):
    """
    Calculate start - end range base on given start, end and/or delta parameter. This method accepts strings and will
    return datetime objects

    :param start_date:  start date
    :param end_date:    end date
    :param delta_time:  delta time in seconds
    :return: start, end tuple
    """
    if start_date is None and end_date is None:
        raise ValueError("Must select at least start or end")

    if start_date is not None and end_date is not None:
        start = convert_date(start_date)
        end = convert_date(end_date)
    elif start_date is not None:
        start = convert_date(start_date)
        end = start + timedelta(seconds=delta_time)
    else:
        end = convert_date(end_date)
        start = end - timedelta(seconds=delta_time)

    return start, end


def construct_aggregation(aggregation_type="value", aggregations=["min", "mean", "max"],
                          extrema=None, nr_of_bins=None, duration_per_bin=None, pulses_per_bin=None):


    if (nr_of_bins is not None) + (duration_per_bin is not None) + (pulses_per_bin is not None) > 1:
        raise RuntimeError("Can specify only one of nr_of_bins, duration_per_bin or pulse_per_bin")

    aggregation = dict()
    aggregation["aggregationType"] = aggregation_type
    aggregation["aggregations"] = aggregations

    if extrema is not None:
        aggregation["extrema"] = extrema

    if nr_of_bins is not None:
        aggregation["nrOfBins"] = nr_of_bins
    elif duration_per_bin is not None:
        aggregation["durationPerBin"] = duration_per_bin
    elif pulses_per_bin is not None:
        aggregation["pulsesPerBin"] = pulses_per_bin

    return aggregation


def construct_data_query(channels, start=None, end=None, range_type="globalDate", delta_range=1,
                         server_side_mapping=False,
                         server_side_mapping_strategy="provide-as-is",
                         aggregation=None,
                         rawdata=False):

    # Implementation of the supported data queries defined at:
    # https://git.psi.ch/sf_daq/ch.psi.daq.queryrest

    # Check input parameters
    if range_type not in ["globalDate", "globalSeconds", "pulseId"]:
        RuntimeError("range_type must be 'globalDate', 'globalSeconds', or 'pulseId'")

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

    query = dict()
    query["channels"] = channel_list
    query["fields"] = ["pulseId", "globalSeconds", "globalDate", "value", "eventCount"]

    # request raw data instead of json encoded data
    if rawdata:
        query["response"] = {"format": "rawevent"}

    # Set query ranges
    query["range"] = {}
    if range_type == "pulseId":
        _start, _end = calculate_range(start, end, delta_range)
        query["range"] = {"endPulseId": str(_end), "startPulseId": str(_start)}
    elif range_type == "globalSeconds":
        _start, _end = calculate_range(start, end, delta_range)
        query["range"] = {"startSeconds": "%.9f" % _start, "endSeconds": "%.9f" % _end}
    else:
        _start, _end = calculate_time_range(start, end, delta_range)
        query["range"] = {"startDate": datetime.isoformat(_start), "endDate": datetime.isoformat(_end)}

    # Set aggregation
    if aggregation is not None:
        query["aggregation"] = aggregation

    if server_side_mapping:
        query["mapping"] = {"incomplete": server_side_mapping_strategy}

    return query


def construct_channel_list_query(regex, backends=None, ordering=None, reload=False):
    """
    Construct channel list query request as defined at:
    https://git.psi.ch/sf_daq/ch.psi.daq.queryrest#query-channel-names

    :param regex:       regex to search for
    :param backends:    query only specified backends
    :param ordering:    ordering of list ["none", "asc", "desc"]
    :param reload:      force reload of cached channel names
    :return:            query
    """

    ordering_options = ["none", "asc", "desc"]
    if ordering and ordering not in ordering_options:
        raise ValueError("Invalid order specified - supported orders: "+ordering_options)

    query = dict()

    query["regex"] = regex

    if ordering:
        query["ordering"] = ordering

    if reload:
        query["reload"] = reload

    if backends is not None:
        if isinstance(backends, str):
            backends = [backends, ]

        if isinstance(backends, (list, tuple)):
            query["backends"] = backends
        else:
            raise ValueError('backends are neither a string nor list')

    return query


def parse_duration(duration_str):
    """
    Parse ISO 8601 duration string as specified https://en.wikipedia.org/wiki/ISO_8601

    :param duration_str:
    :return: timedelta
    """

    match = re.match(
        r'P((?P<years>\d+)Y)?((?P<months>\d+)M)?((?P<weeks>\d+)W)?((?P<days>\d+)D)?(T((?P<hours>\d+)H)?((?P<minutes>\d+)M)?((?P<seconds>\d+)S)?)?',
        duration_str
    ).groupdict()

    # print(match['years'], match['months'], match['weeks'], match['days'], match['hours'], match['minutes'], match['seconds'])

    if match['years'] is not None or match['months'] is not None:
        raise RuntimeError('year and month durations are not supported')

    delta = timedelta(hours=0 if match['hours'] is None else int(match['hours']),
                      minutes=0 if match['minutes'] is None else int(match['minutes']),
                      seconds=0 if match['seconds'] is None else int(match['seconds']),
                      days=0 if match['days'] is None else int(match['days']),
                      weeks=0 if match['weeks'] is None else int(match['weeks']))

    return delta
