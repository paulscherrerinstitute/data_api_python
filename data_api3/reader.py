import sys
import struct
import json
import logging
import io
import urllib.parse
import ssl
import http
import http.client
import numpy
import bitshuffle
import urllib3
import re
import data_api3


# Do not modify global logging settings in a library!
# For the logger, the recommended Python style is to use the module name.
logger = logging.getLogger(__name__)


class Compression:
    BITSHUFFLE_LZ4 = 1


def resolve_struct_dtype(data_type: str, byte_order: str) -> str:
    if data_type is None:
        None
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
        dtype = "string"
    else:
        raise RuntimeError(f"unsupported dta type {data_type} {byte_order}")
    if byte_order is not None:
        x = byte_order.upper()
        if x not in ["LITTLE_ENDIAN", "BIG_ENDIAN"]:
            raise RuntimeError(f"unexpected byte order {byte_order}")
    if dtype != "string":
        if byte_order is not None and byte_order.upper() == "BIG_ENDIAN":
            dtype = ">" + dtype
        else:
            dtype = "<" + dtype
    return dtype


def resolve_numpy_dtype(data_type: str, byte_order: str) -> str:
    if data_type is None:
        return None
    if byte_order is not None and byte_order.upper() == "BIG_ENDIAN":
        endian = ">"
    else:
        endian = "<"
    data_type = data_type.lower()
    if data_type == "float32":
        dtype = endian + "f4"
    elif data_type == "float64":
        dtype = endian + "f8"
    elif data_type == "uint8":
        dtype = endian + "u1"
    elif data_type == "int8":
        dtype = endian + "i1"
    elif data_type == "uint16":
        dtype = endian + "u2"
    elif data_type == "int16":
        dtype = endian + "i2"
    elif data_type == "uint32":
        dtype = endian + "u4"
    elif data_type == "int32":
        dtype = endian + "i4"
    elif data_type == "uint64":
        dtype = endian + "u8"
    elif data_type == "int64":
        dtype = endian + "i8"
    elif data_type == "bool8":
        dtype = numpy.dtype(bool)
    elif data_type == "bool":
        dtype = numpy.dtype(bool)
    elif data_type == "string":
        dtype = numpy.dtype(str)
    else:
        dtype = None
    return dtype


class ProtocolError(RuntimeError):
    def __init__(self):
        super().__init__("ProtocolError")

class Reader:
    def __init__(self):
        self.messages_read = 0
        self.data = {}
        self.in_channel = False

    def read(self, stream):
        try:
            return self.read_throwing(stream)
        except http.client.IncompleteRead:
            logger.error("unexpected end of input")
            raise ProtocolError()

    def read_throwing(self, stream):
        length = 0
        length_check = 0

        current_data = []
        current_channel_name = None
        current_value_extractor = None
        current_compression = None
        current_channel_info = None
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
                self.messages_read += 1

            # Channel header message
            # A json message that specifies among others data type, shape, compression flags.
            elif mtype == 0:
                self.in_channel = False
                try:
                    msg = json.loads(bytes_read[1:])
                    res = process_channel_header(msg)
                except Exception as e:
                    raise RuntimeError("Can not process channel header") from e
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
    name = msg["name"]
    #logger.debug(f"Start with channel {name}")
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
    shape = list(reversed(msg.get("shape", [])))

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
            logger.warn(f"Received scalar-like shape, convert to true scalar  {name}")
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
            raise RuntimeError("unexpected  shape {shape}  channel {name}")
    elif compression == 1:
        if dtype == "string":
            if len(shape) == 0:
                extractor = lambda b: extractor_string(b)
                extractor_writer = lambda ts, pulse, b, data: extractor_writer_compressed_string_scalar(ts, pulse, b, data, name, shape)
            else:
                raise RuntimeError("arrays of strings not yet supported")
        else:
            if len(shape) == 0:
                raise RuntimeError(f"compression not supported on scalar numeric data {name}  shape {shape}  dtype {dtype}")
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


def create_http_conn(up):
    if up.scheme == "https":
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLSv1_2)
        ctx.check_hostname = False
        port = up.port
        if port is None:
            port = 443
        conn = http.client.HTTPSConnection(up.hostname, port, context=ctx)
    else:
        port = up.port
        if port is None:
            port = 80
        conn = http.client.HTTPConnection(up.hostname, port)
    return conn


def http_req(method, url):
    headers = {
        "X-PythonDataAPIPackageVersion": data_api3.version(),
        "X-PythonDataAPIModule": __name__,
        "X-PythonVersion": re.sub(r"[\t\n]", " ", str(sys.version)),
        "X-PythonVersionInfo": str(sys.version_info),
    }
    up = urllib.parse.urlparse(url)
    conn = create_http_conn(up)
    conn.request(method, up.path, None, headers)
    res = conn.getresponse()
    return res


def http_data_query(query, url):
    method = "POST"
    body = json.dumps(query)
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/octet-stream",
        "X-PythonDataAPIPackageVersion": data_api3.version(),
        "X-PythonDataAPIModule": __name__,
        "X-PythonVersion": re.sub(r"[\t\n]", " ", str(sys.version)),
        "X-PythonVersionInfo": str(sys.version_info),
    }
    up = urllib.parse.urlparse(url)
    conn = create_http_conn(up)
    conn.request(method, up.path, body, headers)
    res = conn.getresponse()
    return res


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


def get_request_status(url, reqid):
    url_status = re.sub(r"/[^/]+$", "/requestStatus/" + reqid, url)
    res = http_req("GET", url_status)
    errbody = res.read().decode()
    try:
        err = json.loads(errbody)
        return err
    except:
        logger.error(f"can not parse request status as json\n" + errbody)
        return errbody


def get_request_status_from_immediate_error(url, response):
    response_body = response.read(1024).decode()
    try:
        err = json.loads(response_body)
        reqid = err["requestId"]
        url_status = re.sub(r"/[^/]+$", "/requestStatus/" + reqid, url)
        res = http_req("GET", url_status)
        errbody = res.read().decode()
        try:
            err = json.loads(errbody)
            logger.error(err)
        except:
            logger.error(f"can not parse request status as json\n" + errbody)
    except:
        logger.error(f"can not parse error message as json:\n{response_body}")
        raise


def request(query, url=None, baseurl=None):
    if url is None:
        if baseurl is None:
            raise RuntimeError("need one of `url` or `baseurl`")
        url = baseurl + "/query"
    logger.info(f"data api 3 reader {data_api3.version()}")
    response = http_data_query(query, url)
    if response.status != 200:
        logger.error(f"Unable to retrieve data: {response.status}")
        status = get_request_status_from_immediate_error(url, response)
        raise RuntimeError(f"Unable to retrieve data  {str(status)}")
    reader = Reader()
    try:
        reader.read(io.BufferedReader(response))
        reqid = response.headers["x-daqbuffer-request-id"]
        stat = get_request_status(url, reqid)
        if stat.get("errors") is not None:
            raise RuntimeError("request error")
    except (ProtocolError, RuntimeError) as e:
        logger.error(f"error during request  {e}")
        reqid = response.headers["x-daqbuffer-request-id"]
        stat = get_request_status(url, reqid)
        logger.error(f"request status: {stat}")
        raise
    return reader.data


def read_buffered_stream(buffered_stream):
    reader = Reader()
    reader.read(buffered_stream)
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
