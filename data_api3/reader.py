import struct
import json
import logging
import io
import urllib3
import bitshuffle
import numpy

# Do not modify global logging settings in a library!
# For the logger, the recommended Python style is to use the module name.
logger = logging.getLogger(__name__)


READER_BUILD = 1


class Compression:
    BITSHUFFLE_LZ4 = 1

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


def resolve_struct_dtype(data_type: str, byte_order: str) -> str:
    if data_type is None:
        return None
    data_type = data_type.lower()
    if data_type == "float64":
        dtype = 'd'
    elif data_type == "uint8":
        dtype = 'B'
    elif data_type == "int8":
        dtype = 'b'
    elif data_type == "uint16":
        dtype = 'H'
    elif data_type == "int16":
        dtype = 'h'
    elif data_type == "uint32":
        dtype = 'I'
    elif data_type == "int32":
        dtype = 'i'
    elif data_type == "uint64":
        dtype = 'Q'
    elif data_type == "int64":
        dtype = 'q'
    elif data_type == "float32":
        dtype = 'f'
    elif data_type == "bool8":
        dtype = '?'
    elif data_type == "bool":
        dtype = '?'
    elif data_type == "character":
        dtype = 'c'
    elif data_type == "string":
        dtype = 'string'
    else:
        # Unsupported data types:
        # STRING
        dtype = None

    if dtype is not None and byte_order == "BIG_ENDIAN":
        dtype = ">" + dtype

    return dtype


class Reader:
    def __init__(self):
        self.messages_read = 0
        self.data = {}
        self.in_channel = False

    def read(self, stream):
        length = 0
        length_check = 0

        current_data = []
        current_channel_name = None
        current_value_extractor = None
        current_compression = None
        header = None

        while True:
            bytes_read = stream.read(4)
            if len(bytes_read) != 4:
                break
            length = struct.unpack('>i', bytes_read)[0]
            bytes_read = stream.read(length)
            if len(bytes_read) != length:
                raise RuntimeError("unexpected EOF")
            mtype = struct.unpack('b', bytes_read[:1])[0]
            if mtype == 1 and self.in_channel:
                timestamp = struct.unpack('>q', bytes_read[1:9])[0]
                pulse_id = struct.unpack('>q', bytes_read[9:17])[0]

                raw_data_blob = bytes_read[17:]

                header.extractor_writer(timestamp, pulse_id, bytes_read[17:], current_data)

                if False:
                    if current_compression == 1:
                        c_length = struct.unpack(">q", raw_data_blob[:8])[0]
                        b_size = struct.unpack(">i", raw_data_blob[8:12])[0]

                        d_type = numpy.dtype(current_channel_info["type"])
                        d_type = d_type.newbyteorder('<' if current_channel_info["byteOrder"] == "LITTLE_ENDIAN" else ">")

                        d_shape = current_channel_info["shape"]
                        if d_shape is None or d_shape == []:
                            d_shape = (int(c_length / d_type.itemsize),)

                        value = bitshuffle.decompress_lz4(numpy.frombuffer(raw_data_blob[12:], dtype=numpy.uint8),
                                                         shape=d_shape,
                                                         dtype=d_type,
                                                         block_size=int(b_size / d_type.itemsize))
                        value[0] = 0

                    else:
                        value = current_value_extractor(raw_data_blob)  # value

                    current_data.append({"timestamp": timestamp, "pulse_id": pulse_id, current_channel_name: value})
                self.messages_read += 1

            # Channel header message
            # A json message that specifies among others data type, shape, compression flags.
            elif mtype == 0:
                self.in_channel = False
                msg = json.loads(bytes_read[1:])
                res = process_channel_header(msg)
                if res.error:
                    logger.error(f"Can not parse channel header message: {msg}")
                elif res.empty:
                    logger.debug(f"No data for channel {res.channel_name}")
                else:
                    if "type" not in msg:
                        raise RuntimeError()
                    self.in_channel = True
                    header = res
                    current_data = []
                    current_channel_info = res.channel_info
                    current_channel_name = res.channel_name
                    current_value_extractor = res.value_extractor
                    current_compression = res.compression
                    self.data[current_channel_name] = current_data

            bytes_read = stream.read(4)
            length_check = struct.unpack('>i', bytes_read)[0]
            if length_check != length:
                raise RuntimeError(f"corrupted file reading {length} {length_check}")


def debug_extractor_string_field(data):
    print("string field")
    raise RuntimeError()


class ProcessChannelHeaderResult:

    def __init__(self):
        self.error = False
        self.empty = False
        self.channel_info = None
        self.channel_name = None
        self.value_extractor = None
        self.extractor_writer = None
        self.compression = None
        self.shape = None


def extractor_do_uncompress(ts, pulse, buf, data, name, data_type, shape):
    c_length = struct.unpack(">q", buf[0:8])[0]
    b_size = struct.unpack(">i", buf[8:12])[0]
    nbuf = numpy.frombuffer(buf[12:], dtype=numpy.uint8)
    value = bitshuffle.decompress_lz4(nbuf, shape=shape, dtype=data_type, block_size=int(b_size / data_type.itemsize))
    data.append({"timestamp": ts, "pulse_id": pulse, name: value})


def extractor_basic_scalar(ts, pulse, buf, data, name, data_type, shape):
    value = numpy.frombuffer(buf, dtype=data_type)[0]
    data.append({"timestamp": ts, "pulse_id": pulse, name: value})


def extractor_basic_shaped(ts, pulse, buf, data, name, data_type, shape):
    value = numpy.reshape(numpy.frombuffer(buf, dtype=data_type), shape)
    data.append({"timestamp": ts, "pulse_id": pulse, name: value})


def extractor_writer_compressed_string_scalar(ts, pulse, buf, data, name, shape):
    clen = int(struct.unpack(">q", buf[0:8])[0])
    bsize = int(struct.unpack(">i", buf[8:12])[0])
    u8buf = numpy.frombuffer(buf[12:], dtype=numpy.uint8)
    bval = bitshuffle.decompress_lz4(u8buf, shape=(clen,), dtype=numpy.dtype(numpy.int8), block_size=bsize)
    value = bval.tobytes().decode()
    data.append({"timestamp": ts, "pulse_id": pulse, name: value})


def not_avail(msg):
    raise RuntimeError(msg)


def process_channel_header(msg):
    logger.info(msg)
    name = msg["name"]
    ty = msg.get("type")
    # If no data could be found for this channel, then there is no `type` key and we stop here:
    if ty is None:
        res = ProcessChannelHeaderResult()
        res.empty = True
        res.channel_name = name
        return res
    dtype = resolve_struct_dtype(ty, msg.get("byteOrder"))
    if dtype is None:
        raise RuntimeError("unsupported dtype {} for channel {}".format(dtype, name))
    shape = msg.get("shape", [])

    compression = msg.get("compression")
    # Older data api services indicate no-compression as `0` or even `"0"`
    # we handle these cases here
    if compression is not None:
        compression = int(compression)
    if compression == 0:
        compression = None
    if compression is None:
        if shape == [1]:
            # NOTE legacy compatibility: historically a shape [1] is treated as scalar
            # Which channels actually rely on this?
            shape = []
        if len(shape) == 0:
            if dtype == "string":
                extractor = debug_extractor_string_field
            else:
                data_type = numpy.dtype(msg.get("type")).newbyteorder('<' if msg.get("byteOrder") == "LITTLE_ENDIAN" else ">")
                extractor = lambda b: struct.unpack(dtype, b)[0]
                extractor_writer = lambda ts, pulse, b, data: extractor_basic_scalar(ts, pulse, b, data, name, data_type, shape)
        elif len(shape) > 0:
            if dtype == "string":
                raise RuntimeError("not yet supported, please report a channel that uses arrays of strings.")
            else:
                data_type = numpy.dtype(msg.get("type")).newbyteorder('<' if msg.get("byteOrder") == "LITTLE_ENDIAN" else ">")
                extractor = lambda b: numpy.reshape(numpy.frombuffer(b, dtype=dtype), shape)
                extractor_writer = lambda ts, pulse, b, data: extractor_basic_shaped(ts, pulse, b, data, name, data_type, shape)
        else:
            raise RuntimeError("unexpected shape: {shape}")
    elif compression == 1:
        if dtype == "string":
            if len(shape) == 0:
                extractor = lambda b: extractor_string(b)
                extractor_writer = lambda ts, pulse, b, data: extractor_writer_compressed_string_scalar(ts, pulse, b, data, name, shape)
            else:
                raise RuntimeError("arrays of strings not yet supported")
        else:
            if len(shape) == 0:
                raise RuntimeError("compression not supported on scalar numeric data")
            else:
                data_type = numpy.dtype(msg.get("type")).newbyteorder('<' if msg.get("byteOrder") == "LITTLE_ENDIAN" else ">")
                extractor = lambda b: not_avail("h5 does currently chunk-write in this case")
                extractor_writer = lambda ts, pulse, b, data: extractor_do_uncompress(ts, pulse, b, data, name, data_type, shape)
    else:
        raise RuntimeError(f"compression type {compression} is not yet supported")

    res = ProcessChannelHeaderResult()
    res.channel_info = msg
    res.channel_name = name
    res.value_extractor = extractor
    res.extractor_writer = extractor_writer
    res.compression = compression
    res.shape = shape
    return res


def http_data_query(query, url):
    # IMPORTANT NOTE: the use of the requests library is not possible due to this issue:
    # https://github.com/urllib3/urllib3/issues/1305
    http = urllib3.PoolManager(cert_reqs="CERT_NONE")
    urllib3.disable_warnings()
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/octet-stream",
    }
    body = json.dumps(query).encode()
    response = http.request("POST", url, body=body, headers=headers, preload_content=False)
    # Were hitting this issue here:
    # https://github.com/urllib3/urllib3/issues/1305
    response._fp.isclosed = lambda: False
    return response


def save_raw(query, url, fname):
    s = http_data_query(query, url)
    with open(fname, "wb") as f1:
        while True:
            buf = s.read()
            if buf is None:
                break
            if len(buf) < 0:
                raise RuntimeError()
            if len(buf) == 0:
                break
            f1.write(buf)


def request(query, url="http://localhost:8080/api/v1/query"):
    response = http_data_query(query, url)
    if response.status != 200:
        raise RuntimeError(f"Unable to retrieve data: {response.data}")
    reader = Reader()
    reader.read(io.BufferedReader(response))
    return reader.data


def read(filename):
    reader = Reader()
    with open(filename, "rb") as stream:
        with io.BufferedReader(stream) as buffered_stream:
            reader.read(buffered_stream)
            buffered_stream.close()
    return reader.data


def as_dataframe(data: dict):
    import pandas as pd

    dataframe = None

    for key in data:
        df = pd.DataFrame(data[key])
        df = df.drop(columns=["pulse_id"])  # were not interested in this
        df = df.set_index('timestamp')  # set timestamp as index

        if dataframe is None:
            dataframe = df
        else:
            dataframe = pd.merge(dataframe, df, how='outer', on='timestamp')

    dataframe.fillna(method='pad',
                     inplace=True)  # fill NaN with last known value (assuming recording system worked without error)

    return dataframe
