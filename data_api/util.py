import pytz
import dateutil.parser
from datetime import datetime, timedelta


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