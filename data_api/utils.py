import socket
import re
import datetime
import dateutil
import pytz


def check_reachability_server(endpoint):
    """
    Check whether the enpoint server is reachable
    :param endpoint:    endpoint url
    :return:            True if enpoint is reachable, False otherwise
    """

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
        return False
    finally:
        sock.close()

    return True


def convert_date(date_string):
    """
    Convert a date string to datetime (if not already datetime) and attach timezone (if not already attached)
    :param date_string:     Date string to convert
    :return:                datetime with correct timezone attached
    """

    if isinstance(date_string, str):
        date = dateutil.parser.parse(date_string)
    elif isinstance(date_string, datetime.datetime):
        date = date_string
    else:
        raise ValueError("Unsupported date type: " + type(date_string))

    if date.tzinfo is None:  # localize time if necessary
        date = pytz.timezone('Europe/Zurich').localize(date)

    return date
