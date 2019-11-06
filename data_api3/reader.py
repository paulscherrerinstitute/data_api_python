import struct
import json
import logging

import io
import urllib3


# def resolve_numpy_dtype(header: dict) -> str:
#
#     header_type = header["type"].lower()
#
#     if header_type == "float64":  # default
#         dtype = 'f8'
#     elif header_type == "uint8":
#         dtype = 'u1'
#     elif header_type == "int8":
#         dtype = 'i1'
#     elif header_type == "uint16":
#         dtype = 'u2'
#     elif header_type == "int16":
#         dtype = 'i2'
#     elif header_type == "uint32":
#         dtype = 'u4'
#     elif header_type == "int32":
#         dtype = 'i4'
#     elif header_type == "uint64":
#         dtype = 'u8'
#     elif header_type == "int64":
#         dtype = 'i8'
#     elif header_type == "float32":
#         dtype = 'f4'
#     else:
#         # Unsupported data types:
#         # STRING
#         # CHARACTER
#         # BOOL
#         # BOOL8
#         dtype = None
#
#     if dtype is not None and header["byteOrder"] == "BIG_ENDIAN":
#         dtype = ">" + dtype
#
#     return dtype


def resolve_struct_dtype(header: dict) -> str:

    header_type = header["type"].lower()

    if header_type == "float64":  # default
        dtype = 'd'
    elif header_type == "uint8":
        dtype = 'B'
    elif header_type == "int8":
        dtype = 'b'
    elif header_type == "uint16":
        dtype = 'H'
    elif header_type == "int16":
        dtype = 'h'
    elif header_type == "uint32":
        dtype = 'I'
    elif header_type == "int32":
        dtype = 'i'
    elif header_type == "uint64":
        dtype = 'Q'
    elif header_type == "int64":
        dtype = 'q'
    elif header_type == "float32":
        dtype = 'f'
    elif header_type == "bool8":
        dtype = '?'
    elif header_type == "bool":
        dtype = '?'
    elif header_type == "character":
        dtype = 'c'
    else:
        # Unsupported data types:
        # STRING
        dtype = None

    if dtype is not None and header["byteOrder"] == "BIG_ENDIAN":
        dtype = ">" + dtype

    return dtype


class Reader:
    def __init__(self):
        self.messages_read = 0
        self.data = {}

    def read(self, stream):
        length = 0
        length_check = 0

        current_data = []
        current_channel_name = None
        current_value_extractor = None

        while True:
            bytes_read = stream.read(4)
            if not bytes_read:  # handle the end of the file because we have more than one read statement in the loop
                break
            length = struct.unpack('>i', bytes_read)[0]

            bytes_read = stream.read(length)

            mtype = struct.unpack('b', bytes_read[:1])[0]

            if mtype == 1:  # data message - this one is more often thats why its on the top
                timestamp = struct.unpack('>q', bytes_read[1:9])[0]  # timestamp
                pulse_id = struct.unpack('>q', bytes_read[9:17])[0]  # pulseid

                value = current_value_extractor(bytes_read[17:])  # value

                current_data.append({"timestamp": timestamp, "pulse_id": pulse_id, current_channel_name: value})
                self.messages_read += 1
            elif mtype == 0:  # header message
                current_channel_info = json.loads(bytes_read[1:])
                # the header usually looks something like this:
                # {"name": "SLG-LSCP3-FNS:CH7:VAL_GET", "type":"float64", "compression":"0", "byteOrder":"BIG_ENDIAN",
                # "shape": null}
                logging.info(current_channel_info)

                current_channel_name = current_channel_info["name"]

                # Based on header use the correct value extractor
                # dtype = resolve_numpy_dtype(current_channel_info)
                dtype = resolve_struct_dtype(current_channel_info)
                if current_channel_info["compression"] == "0":
                    compression = None
                else:
                    # TODO need to support compression
                    raise RuntimeError("compression currently not supported")

                current_value_extractor = lambda b: b  # just return the bytes if no special extractor can be found
                if dtype is not None:
                    if compression is None:
                        if "shape" not in current_channel_info or \
                                current_channel_info["shape"] is None or \
                                current_channel_info["shape"] == [1]:

                            current_value_extractor = lambda b: struct.unpack(dtype, b)[0]
                        else:
                            # it is an x dimensional array
                            current_value_extractor = lambda b: struct.unpack(dtype, b)

                        # current_value_extractor = lambda b: numpy.frombuffer(b, dtype=dtype)
                        # TODO Take care of shape
                # TODO take care of compression

                # if "shape" in current_channel_info:
                #     current_shape = current_channel_info["shape"]
                # else:
                #     current_shape = None

                # value = None
                # if current_dtype is not None:
                #     if current_compression is None:
                #
                #         # value = numpy.frombuffer(bytes_read[17:], dtype=current_dtype)
                #         value = struct.unpack('>d', bytes_read[17:])[0]  # value
                #     #         if current_shape is not None:
                #     #             value.reshape(current_shape)
                #     else:
                #         # TODO Take care of compression
                #         value = bytes_read[17:]
                # else:
                #     # If type is not supported just save bytes
                #     if current_compression is None:
                #         value = bytes_read[17:]
                #     else:
                #         # TODO take care of compression
                #         value = bytes_read[17:]


                current_data = []
                self.data[current_channel_name] = current_data

            bytes_read = stream.read(4)
            #         length_check = int.from_bytes(bytes_read, byteorder='big')
            length_check = struct.unpack('>i', bytes_read)[0]
            if length_check != length:
                raise RuntimeError(f"corrupted file reading {length} {length_check}")

        print(f"{length}, {length_check}")


def request(query, url="http://localhost:8080/api/v1/query"):
    # IMPORTANT NOTE: the use of the requests library is not possible due to this issue:
    # https://github.com/urllib3/urllib3/issues/1305

    encoded_data = json.dumps(query).encode('utf-8')

    http = urllib3.PoolManager()
    response = http.request('POST', url,
                            body=encoded_data,
                            headers={'Content-Type': 'application/json'},
                            preload_content=False)

    # Were hitting this issue here:
    # https://github.com/urllib3/urllib3/issues/1305
    response._fp.isclosed = lambda: False  # monkey patch

    reader = Reader()
    buffered_response = io.BufferedReader(response)
    reader.read(buffered_response)

    return reader.data


def read(filename):
    reader = Reader()
    with open(filename, "rb") as stream:
        with io.BufferedReader(stream) as buffered_stream:
            reader.read(buffered_stream)
            buffered_stream.close()

    return reader.data
