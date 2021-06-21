import pytz
import dateutil.parser
from datetime import datetime, timedelta
import re
import logging


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


def construct_range(start=None, end=None, delta_range=1, range_type=None,
                    start_inclusive=None, start_expansion=None,
                    end_inclusive=None, end_expansion=None):
    """
    Construct a range query

    :param start:
    :param end:
    :param delta_range:
    :param range_type:
    :param start_inclusive:
    :param start_expansion:
    :param end_inclusive:
    :param end_expansion:
    :return:
    """

    # Check input parameters
    if range_type is not None and range_type not in ["globalDate", "globalSeconds", "pulseId"]:
        RuntimeError("range_type must be 'globalDate', 'globalSeconds', or 'pulseId'")

    if range_type is None:
        # Determine range_type
        if (start is not None and isinstance(start, int)) or (end is not None and isinstance(end, int)):
            logging.info("Using range_type: pulseId")
            range_type = "pulseId"
        elif (start is not None and isinstance(start, float)) or (end is not None and isinstance(end, float)):
            range_type = "globalSeconds"
            logging.info("Using range_type: globalSeconds")
        else:
            range_type = "globalDate"
            logging.info("Using range_type: globalDate")

    query = dict()

    if range_type == "pulseId":
        _start, _end = calculate_range(start, end, delta_range)
        query["endPulseId"] = int(_end)
        query["startPulseId"] = int(_start)
    elif range_type == "globalSeconds":
        _start, _end = calculate_range(start, end, delta_range)
        query["startSeconds"] = "%.9f" % _start
        query["endSeconds"] = "%.9f" % _end
    else:
        _start, _end = calculate_time_range(start, end, delta_range)
        query["startDate"] = datetime.isoformat(_start)
        query["endDate"] = datetime.isoformat(_end)

    if start_inclusive:
        query["startInclusive"] = True

    if start_expansion:
        query["startExpansion"] = True

    if end_inclusive:
        query["endInclusive"] = True

    if end_expansion:
        query["endExpansion"] = True

    return query


def construct_value_mapping(incomplete=None, alignment=None, aggregations=None):
    """
    Create value mapping as specified at https://git.psi.ch/sf_daq/ch.psi.daq.queryrest#value-mapping

    :param incomplete:
    :param alignment:
    :param aggregations:
    :return:
    """

    incomplete_options = ["provide-as-is", "drop", "fill-null"]
    alignment_options = ["by-pulse", "by-time", "none"]
    aggregations_options = ["count", "min", "mean", "max"]

    mapping = dict()

    if incomplete is not None:
        if incomplete not in incomplete_options:
            raise ValueError("incomplete need to be in one of " + " ".join(incomplete_options))

        mapping["incomplete"] = incomplete

    if alignment is not None:
        if alignment not in alignment_options:
            raise ValueError("alignment need to be in one of " + " ".join(alignment_options))

        mapping["alignment"] = alignment

    if aggregations is not None:
        if isinstance(aggregations, str):
            aggregations = [aggregations]

        if not set(aggregations).issubset(aggregations_options):
            raise ValueError("Only following types of aggregation supported: " + " ".join(aggregations_options))

        mapping["aggregations"] = aggregations

    return mapping


def construct_response(format=None, compression=None):
    """
    Construct response - more details https://git.psi.ch/sf_daq/ch.psi.daq.queryrest#response-format
    :param format:
    :param compression:
    :return:
    """
    response = dict()

    if format is not None:
        if format not in ["json", "csv", "rawevent"]:
            raise ValueError("Invalid format")
        response["format"] = format
    if compression is not None:
        if compression not in ["gzip"]:
            raise ValueError("Invalid compression")
        response["compression"] = "gzip"

    return response


def construct_data_query(channels,

                         # range specific parameters
                         start=None, end=None, delta_range=1, range_type=None,
                         start_inclusive=None, start_expansion=None,
                         end_inclusive=None, end_expansion=None,

                         value_mapping=None,

                         ordering=None,

                         aggregation=None,
                         event_fields=None,

                         response=None
                         ):

    """
    Construct a data query

    :param channels:
    :param start:
    :param end:
    :param delta_range:
    :param range_type:
    :param start_inclusive:
    :param start_expansion:
    :param end_inclusive:
    :param end_expansion:
    :param value_mapping:
    :param ordering:
    :param aggregation:
    :param event_fields:    Supported event fields are: value, time, pulseId, severity, status
    :param response:
    :return:
    """

    # value_mapping - Setting this option activates a table like alignment of the response which differs from
    # the standard response format.

    # Implementation of the supported data queries defined at:
    # https://git.psi.ch/sf_daq/ch.psi.daq.queryrest

    if channels is None or range is None:
        raise ValueError("channels and range need to be defined")

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
    query["range"] = construct_range(start=start, end=end, delta_range=delta_range, range_type=range_type,
                                     start_inclusive=start_inclusive, start_expansion=start_expansion,
                                     end_inclusive=end_inclusive, end_expansion=end_expansion)

    if event_fields:
        if isinstance(event_fields, str):
            event_fields = [event_fields, ]
        query["eventFields"] = event_fields
    else:
        query["eventFields"] = ["value", "time", "pulseId"]
    # TODO Cleanup needed - shape must always be in fields !

    if ordering is not None:
        if ordering not in ["asc", "desc", "none"]:
            raise ValueError("Unsupported ordering" + ordering + " - supported values are: asc, desc, none")
        query["ordering"] = ordering

    if response is not None:
        query["response"] = response

    if aggregation is not None:
        query["aggregation"] = aggregation

    if value_mapping is not None:
        query["mapping"] = value_mapping

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
    """https://en.wikipedia.org/wiki/ISO_8601"""

    match = re.match(
        r'P((?P<years>\d+)Y)?((?P<months>\d+)M)?((?P<weeks>\d+)W)?((?P<days>\d+)D)?(T((?P<hours>\d+)H)?((?P<minutes>\d+)M)?((?P<seconds>\d+)S)?)?',
        duration_str
    )
    if match:
        match = match.groupdict()
    else:
        raise RuntimeError("Unable to parse time duration - check whether your duration is "
                           "https://en.wikipedia.org/wiki/ISO_8601 compliant - don't use fractions of units!")

    print(match['years'], match['months'], match['weeks'], match['days'], match['hours'], match['minutes'], match['seconds'])

    if match['years'] is not None or match['months'] is not None:
        raise RuntimeError('year and month durations are not supported')

    delta = timedelta(hours=0 if match['hours'] is None else int(match['hours']),
                      minutes=0 if match['minutes'] is None else int(match['minutes']),
                      seconds=0 if match['seconds'] is None else int(match['seconds']),
                      days=0 if match['days'] is None else int(match['days']),
                      weeks=0 if match['weeks'] is None else int(match['weeks']))

    return delta


def as_dict(data):

    class Wrapper:
        def __init__(self, data):
            self.data = data
            self.key_index = dict()
            self.short_key_index = dict()
            short_key_count = set()

            for index in range(len(data)):
                self.key_index[data[index]["channel"]["backend"]+"/"+data[index]["channel"]["name"]] = index
                self.short_key_index[data[index]["channel"]["name"]] = index

                if data[index]["channel"]["name"] in short_key_count:
                    # remove shortkey as the name is not unique
                    del self.short_key_index[data[index]["channel"]["name"]]

                short_key_count.add(data[index]["channel"]["name"])

        def __len__(self):
            return len(self.data)

        def __length_hint__(self):
            return len(self.data)

        def __getitem__(self, key):
            # take care that backend is all lowercase
            key_parts = key.split("/")
            if len(key_parts) == 2:
                key = key_parts[0].lower()+"/"+key_parts[1]
            else:
                # no backend specified - use short_key index
                return data[self.short_key_index[key]]["data"]

            return data[self.key_index[key]]["data"]

        def __setitem__(self, key, value):
            pass

        def __delitem__(self, key):
            pass

        def __missing__(self, key):
            pass

        def __iter__(self):
            return self.key_index.__iter__()

        def __reversed__(self):
            pass

        def __contains__(self, item):
            return item in self.key_index

    return Wrapper(data)
