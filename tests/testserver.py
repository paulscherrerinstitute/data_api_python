from bottle import Bottle, run, request

import datetime
import json
import logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s')


# A request looks like this:
# {
#     "channels": [
#         {
#             "name": "S10CB02-CVME-ILK:CENTRAL-CORETEMP",
#             "backend": "sf-archiverappliance"
#         }
#     ],
#     "fields": [
#         "pulseId",
#         "globalSeconds",
#         "globalDate",
#         "value",
#         "eventCount"
#     ],
#     "range": {
#         "startDate": "2017-12-14T21:45:57.118088+01:00",
#         "endDate": "2017-12-15T09:45:57.118088+01:00"
#     }
# }

# Response looks like this:
# [
#     {
#         "channel": {
#             "name": "S10CB02-CVME-ILK:CENTRAL-CORETEMP",
#             "backend": "sf-archiverappliance"
#         },
#         "data": [
#             {
#                 "eventCount": 1,
#                 "pulseId": -1,
#                 "globalDate": "2017-12-14T21:46:01.797718484+01:00",
#                 "globalSeconds": "1513284361.797718484",
#                 "value": 46.263842773437545
#             },
#             {
#                 "eventCount": 1,
#                 "pulseId": -1,
#                 "globalDate": "2017-12-15T09:45:51.798119998+01:00",
#                 "globalSeconds": "1513327551.798119998",
#                 "value": 46.75600585937502
#             }
#         ]
#     }
# ]

# Response with server side mapping turned on:
# Order of the values are guaranteed and according to the sequence specified in the query.
# ATTENTION: A channel that does not exist currently does not get returned (by a null)
# {
#     "data": [
#         [
#             {
#                 "channel": "S10CB02-CVME-ILK:CENTRAL-CORETEMP",
#                 "backend": "sf-archiverappliance",
#                 "eventCount": 1,
#                 "pulseId": -1,
#                 "globalDate": "2017-12-18T09:44:31.797808901+01:00",
#                 "globalSeconds": "1513586671.797808901",
#                 "value": 46.263842773437545
#             },
#             null
#         ],
#         [
#             null,
#             {
#                 "channel": "S10CB01-CVME-ILK:P2020-CORETEMP",
#                 "backend": "sf-archiverappliance",
#                 "eventCount": 1,
#                 "pulseId": -1,
#                 "globalDate": "2017-12-18T09:44:31.871439399+01:00",
#                 "globalSeconds": "1513586671.871439399",
#                 "value": 52.40625
#             }
#         ]
#     ]
# }


def main():

    import argparse
    parser = argparse.ArgumentParser(description='Application configuration management utility')
    parser.add_argument('-n', '--name', help='Hostname to bind to', default="localhost")
    parser.add_argument('-p', '--port', help='Port to bind to', type=int, default=8080)

    arguments = parser.parse_args()

    # Read command line arguments
    hostname = arguments.name
    port = arguments.port

    # Setup and start server
    app = Bottle()

    @app.route('/archivertestdata/query', method='POST')
    def archiver_test_data():

        requested_channels = request.json["channels"]
        if "fields" not in request.json:
            requested_fields = ["pulseId", "globalSeconds", "globalDate", "value", "eventCount"]
        else:
            requested_fields = request.json["fields"]
        requested_range = request.json["range"]

        from dateutil.parser import parse

        start = parse(requested_range["startDate"])
        end = parse(requested_range["endDate"])

        response_data = []
        channel_value_increment = 0
        for channel in requested_channels:
            data = {"channel": channel}
            data_array = []

            time = start
            delta = (end - start)/10

            for i in range(10):
                data_point = {"value": i+channel_value_increment}
                if "pulseId" in requested_fields:
                    data_point["pulseId"] = -1
                if "globalSeconds" in requested_fields:
                    data_point["globalSeconds"] = str(time.timestamp())
                if "globalDate" in requested_fields:
                    data_point["globalDate"] = str(time)
                if "eventCount" in requested_fields:
                    data_point["eventCount"] = 1

                time += delta

                data_array.append(data_point)
            data["data"] = data_array
            response_data.append(data)

            channel_value_increment += 10

        return json.dumps(response_data)

    @app.route('/archivertestdatamerge/query', method='POST')
    def archiver_test_data_merge_needed():
        requested_channels = request.json["channels"]
        requested_range = request.json["range"]
        # TODO: Were fields renamed to eventFields?
        if "fields" not in request.json:
            requested_fields = ["pulseId", "globalSeconds", "globalDate", "value", "eventCount"]
        else:
            requested_fields = request.json["fields"]

        from dateutil.parser import parse

        start = parse(requested_range["startDate"])
        end = parse(requested_range["endDate"])

        response_data = []
        channel_value_increment = 0
        channel_counter = 0
        for channel in requested_channels:
            data = {"channel": channel}
            data_array = []

            time = start

            # For second channel shift time a little bit
            if channel_counter % 2 == 0:
                time -= datetime.timedelta(seconds=1)

            delta = (end - start) / 10

            for i in range(10):
                data_point = {"value": i + channel_value_increment}
                if "pulseId" in requested_fields:
                    data_point["pulseId"] = -1
                if "globalSeconds" in requested_fields:
                    data_point["globalSeconds"] = str(time.timestamp())
                if "globalDate" in requested_fields:
                    data_point["globalDate"] = str(time)
                if "eventCount" in requested_fields:
                    data_point["eventCount"] = 1

                time += delta

                data_array.append(data_point)
            data["data"] = data_array
            response_data.append(data)

            channel_value_increment += 10
            channel_counter += 1

        return json.dumps(response_data)

    run(app, host=hostname, port=port)


if __name__ == '__main__':
    main()
