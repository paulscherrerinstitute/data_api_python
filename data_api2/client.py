from __future__ import print_function, division
from datetime import datetime, timedelta  # timezone

import requests
import logging
import json
import io
import numpy
import dateutil.parser

from data_api2 import util, idread_util

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')


# One time check at import time to set the default URL (if in SwissFEL network use Swissfel server)

default_base_url = "https://data-api.psi.ch/sf"

if util.check_reachability_server("https://sf-data-api.psi.ch"):
    default_base_url = "https://sf-data-api.psi.ch"
logger.debug("Using endpoint %s" % default_base_url)


def get_data(query, base_url=None, raw=False):
    """
    Get data from Data API
    :param query:
    :param base_url:
    :param raw:
    :return: data dictionary
    """
    if raw:
        return get_data_idread(query, base_url=base_url)
    else:
        return get_data_json(query, base_url=base_url)


def get_data_json(query, base_url=None):
    """
    Retrieve data in json format
    :param query:
    :param base_url:
    :return:            Usually the return format is like this
                        [{channel:{}, data:[{pulseId: , value: ...}]}, ]
                        However the format is depending on the kind of query
                        that is passed
    """

    # TODO enable this as soon as json backend supports it
    # supported_event_fields = ['value', 'time', 'pulseId', 'status', 'severity']
    supported_event_fields = ['value', 'time', 'pulseId']

    if "eventFields" in query:
        if not set(query["eventFields"]).issubset(supported_event_fields):
            raise ValueError("Requested event fields are not supported in raw mode. Supported event fields are: " +
                             " ".join(supported_event_fields))

        # convert eventFields into event fields the backend understands
        # supported event_fields by the backend are documented at
        # https://github.psi.ch/sf_daq/ch.psi.daq.domain/blob/master/src/main/java/ch/psi/daq/domain/query/operation/EventField.java
        requested_event_fields = query["eventFields"]
        backend_event_fields = []
        for field in requested_event_fields:
            # currently supported events: value, time, pulseId, severity, status
            if field == "value":
                backend_event_fields.append("value")
            elif field == "time":
                # TODO for more efficient download - change to globalTime once it is implemented by the backend !
                backend_event_fields.append("globalDate")
            elif field == "pulseId":
                backend_event_fields.append("pulseId")
            # TODO need to be uncommented as soon as severity status are supported by the json query !
            # elif field == "severity":
            #     backend_event_fields.append("severity")
            # elif field == "status":
            #     backend_event_fields.append("status")
            else:
                raise RuntimeError("event_field %s is not supported", field)

        # To be able to shape the data correctly we need shape in the eventFields
        backend_event_fields.append("shape")

        query = dict(query)  # copy the query dict so that the passed query can be reused
        query["eventFields"] = backend_event_fields

    if base_url is None:
        base_url = default_base_url

    logger.info("curl -H \"Content-Type: application/json\" -X POST -d '" + json.dumps(query) + "' " + base_url + "/query")
    response = requests.post(base_url + '/query', json=query)

    if response.status_code != 200:
        raise RuntimeError("Unable to retrieve data from server: ", response)

    data = response.json()

    # Post processing of the data
    # Convert multidimensional data to the correct shape
    # Convert global date to datetime
    if "mapping" in query:
        for entries in data["data"]:
            for value in entries:
                if value is not None and "shape" in value and len(value["shape"]) > 1:
                    value["value"] = numpy.asarray(value["value"]).reshape((value["shape"][::-1]))
                    del value["shape"]  # remove from result dictionary
                if value is not None and "globalDate" in value:
                    value["time"] = dateutil.parser.parse(value["globalDate"])
                    del value["globalDate"]  # remove globalDate from result dictionary
        # Just return array
        data = data["data"]

    else:
        for channel in data:
            for value in channel["data"]:
                if value is not None and "shape" in value and len(value["shape"]) > 1:
                    value["value"] = numpy.asarray(value["value"]).reshape((value["shape"][::-1]))
                if value is not None and "globalDate" in value:
                    value["time"] = dateutil.parser.parse(value["globalDate"])
                    # del value["globalDate"]
                # TODO globalSeconds - currently string

    return data


def get_data_idread(query, base_url=None):
    """
    Retrieve data in idread format
    :param query:
    :param base_url:
    :return:            The return format is like this
                        [{channel:{}, data:[{pulseId: , value: ...}]}, ]
    """

    # TODO remove and implement correct working
    # if "mapping" in query:
    #     raise RuntimeError("Server side mapping currently not supported with idread")

    supported_event_fields = ['value', 'time', 'timeRaw', 'pulseId', 'status', 'severity']
    # globalSeconds and iocSeconds need to be converted to string!
    # globalDate needs to be generated - remember to hard-code timezone Zurich!

    if "eventFields" in query:
        if not set(query["eventFields"]).issubset(supported_event_fields):
            raise ValueError("Requested event fields are not supported in raw mode. Supported event fields are: " +
                             " ".join(supported_event_fields))

        # Actually all this conversion is not needed as it is not supported by the backend anyway right now

        # convert eventFields into event fields the backend understands
        # supported event_fields by the backend are documented at
        # https://github.psi.ch/sf_daq/idread_specification
        requested_event_fields = query["eventFields"]
        backend_event_fields = []
        for field in requested_event_fields:
            if field == "time":  # need to convert
                backend_event_fields.append("globalDate")  # TODO need to change once supported by backend
            else:
                backend_event_fields.append(field)

        query = dict(query)  # copy the query dict so that the passed query can be reused
        query["eventFields"] = backend_event_fields

    if base_url is None:
        base_url = default_base_url

    # Ensure that we request raw events
    if "response" in query:
        # Overwrite whatever is in format
        query["response"]["format"] = "rawevent"
    else:
        query["response"] = util.construct_response(format="rawevent")

    # https://github.psi.ch/sf_daq/idread_specification#reference-implementation
    # https://github.psi.ch/sf_daq/ch.psi.daq.queryrest#rest-interface

    # curl command that can be used for debugging
    logger.info("curl -H \"Content-Type: application/json\" -X POST -d '"+json.dumps(query)+"' "+base_url + '/query')

    if "mapping" in query:
        collector = idread_util.MappingCollector(len(query["channels"]), event_fields=requested_event_fields)
    else:
        collector = idread_util.DictionaryCollector(event_fields=requested_event_fields)

    stream = False
    if stream:
        with requests.post(base_url + '/query', json=query, stream=stream) as response:
            idread_util.decode(response.raw, collector_function=collector.add_data)
    else:
        response = requests.post(base_url + '/query', json=query)
        idread_util.decode(io.BytesIO(response.content), collector_function=collector.add_data)

    return collector.get_data()


def save_data_iread(query, filename, base_url=None, collector=None):

    if base_url is None:
        base_url = default_base_url

    # Ensure that we request raw events
    # TODO TO BE REMOVED
    if "response" in query:
        # Overwrite whatever is in format
        query["response"]["format"] = "rawevent"
    else:
        query["response"] = util.construct_response(format="rawevent")

    # https://github.psi.ch/sf_daq/idread_specification#reference-implementation
    # https://github.psi.ch/sf_daq/ch.psi.daq.queryrest#rest-interface

    # curl command that can be used for debugging
    logger.info("curl -H \"Content-Type: application/json\" -X POST -d '"+json.dumps(query)+"' "+base_url + '/query')

    if collector is not None:
        serializer = collector
    else:
        serializer = idread_util.HDF5Collector()
        serializer.open(filename)

    stream = False
    if stream:
        with requests.post(base_url + '/query', json=query, stream=stream) as response:
            idread_util.decode(response.raw, collector_function=serializer.add_data)
    else:
        response = requests.post(base_url + '/query', json=query)
        idread_util.decode(io.BytesIO(response.content), collector_function=serializer.add_data)

    if collector is None:
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


def get_timestamp_from_pulse_id(pulse_ids, mapping_channel="SIN-CVME-TIFGUN-EVR0:BEAMOK", base_url=None):
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
        query = util.construct_data_query(mapping_channel, start=pulse_id, range_type="pulseId", event_fields=["pulseId", "time"])
        data = get_data_json(query, base_url=base_url)

        if not pulse_id == data[0]["data"][0]["pulseId"]:
            raise RuntimeError('Unable to retrieve mapping')

        dates.append(util.convert_date(data[0]["data"][0]["time"]))

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
