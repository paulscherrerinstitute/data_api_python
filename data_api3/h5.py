import struct
import json
import logging
import h5py
import io
import bitshuffle.h5
import data_api3.reader
import numpy
import urllib.parse
import http
import http.client

# Do not modify global logging settings in a library!
# For the logger, the recommended Python style is to use the module name.
logger = logging.getLogger(__name__)

PACKAGE_VERSION = "0.7.11"


class HDF5Reader:
    def __init__(self, filename: str):
        self.messages_read = 0
        self.nbytes_read = 0
        self.filename = filename
        self.in_channel = False

    def read(self, stream):
        length = 0
        length_check = 0

        current_channel_name = None
        current_value_extractor = None
        current_dtype = None
        current_shape = []
        current_h5shape = []
        channel_header = None

        ts_ds = None
        pulse_ds = None
        val_ds = None

        serializer = Serializer()
        serializer.open(self.filename)

        while True:
            bytes_read = stream.read(4)
            if len(bytes_read) != 4:
                break
            self.nbytes_read += len(bytes_read)
            length = struct.unpack('>i', bytes_read)[0]
            if length > 20 * 1024 * 1024:
                raise RuntimeError(f"unexpected large packet length {length}")
            bytes_read = stream.read(length)
            if len(bytes_read) != length:
                raise RuntimeError("unexpected EOF")
            self.nbytes_read += len(bytes_read)
            mtype = struct.unpack("b", bytes_read[:1])[0]
            if mtype == 1 and self.in_channel:
                try:
                    j = struct.unpack(">2q", bytes_read[1:17])
                    timestamp = j[0]
                    pulse_id = j[1]
                    raw_data_blob = bytes_read[17:]
                    ts_ds.append(timestamp)
                    pulse_ds.append(pulse_id)
                    if current_shape == [] and current_compression is None:
                        if current_dtype == "string":
                            logger.error("unexpected uncompressed scalar data")
                            raise RuntimeError("unexpected uncompressed scalar data")
                        else:
                            # Numeric scalar data is expected to be always uncompressed.
                            value = current_value_extractor(raw_data_blob)
                            val_ds.append(value)

                    elif current_shape == [] and current_compression is not None and current_dtype == "string":
                        # Strings seem to be always compressed in databuffer
                        dt = h5py.string_dtype()
                        string_value = decompress_string(raw_data_blob)
                        val_ds.append(value)

                    elif len(current_shape) > 0:
                        dt = current_channel_info["type"].lower()
                        sh = current_h5shape
                        if current_compression is None:
                            value = channel_header.value_extractor(raw_data_blob)
                            val_ds.append(value)
                        elif len(sh) == 1 and sh[0] < 0:
                            raise RuntimeError("todo")
                        else:
                            # Non-scalar data, compressed:
                            val_ds.append(raw_data_blob)

                    else:
                        raise RuntimeError(f"can not write  current_compression {current_compression}  current_shape {current_shape}")

                    self.messages_read += 1

                except Exception as e:
                    logger.error(f"ERROR in write for channel {current_channel_name}  {current_dtype}  {current_numpy_dtype}")
                    raise

            elif mtype == 0:
                if ts_ds is not None:
                    ts_ds.close()
                    pulse_ds.close()
                    val_ds.close()
                self.in_channel = False
                msg = json.loads(bytes_read[1:])
                res = data_api3.reader.process_channel_header(msg)
                if res.error:
                    logger.error("Can not parse channel header message: {}".format(msg))
                elif res.empty:
                    logger.debug("No data for channel {}".format(res.channel_name))
                else:
                    self.in_channel = True
                    channel_header = res
                    current_channel_info = res.channel_info
                    current_channel_name = res.channel_name
                    current_value_extractor = res.value_extractor
                    current_compression = res.compression
                    current_shape = res.shape
                    current_h5shape = res.shape[::-1]
                    current_dtype = data_api3.reader.resolve_struct_dtype(current_channel_info["type"], current_channel_info["byteOrder"])
                    current_numpy_dtype = data_api3.reader.resolve_numpy_dtype(current_channel_info["type"])
                    ts_ds = TsDataset(res.channel_name, serializer.file)
                    pulse_ds = PulseDataset(res.channel_name, serializer.file)
                    if len(res.shape) == 2 or (len(res.shape) == 1 and res.compression is not None):
                        # Images are assumed to be always direct chunk write
                        val_ds = DirectChunkwriteDataset(res.channel_name, "data", serializer.file, current_h5shape, current_numpy_dtype, res.compression)
                    elif current_dtype == "string":
                        val_ds = StringDataset(res.channel_name, "data", serializer.file)
                    else:
                        val_ds = NumericDataset(res.channel_name, "data", serializer.file, res.shape, current_numpy_dtype)

            bytes_read = stream.read(4)
            if len(bytes_read) != 4:
                raise RuntimeError("short read")
            length_check = struct.unpack('>i', bytes_read)[0]
            if length_check != length:
                raise RuntimeError(f"corrupted file reading {length} {length_check}")
            self.nbytes_read += len(bytes_read)

        if ts_ds is not None:
            ts_ds.close()
            pulse_ds.close()
            val_ds.close()
        serializer.close()


class NumericDataset:
    def __init__(self, channel, field, h5file, shape, dtype):
        self.channel = channel
        self.h5file = h5file
        shape = tuple(shape)
        self.shape = shape
        self.dtype = dtype
        chunks = (2<<13,)
        if len(shape) == 0:
            pass
        elif len(shape) == 1:
            n = 8 * 1024 // shape[0]
            if n < 2:
                n = 2
            chunks = (n,) + shape
        else:
            raise RuntimeError("unsupported")
        self.chunks = chunks
        self.dataset = self.h5file.create_dataset(f"/{channel}/{field}", (0,) + shape, maxshape=(None,) + shape, dtype=dtype, chunks=chunks)
        self.buf = numpy.zeros(shape=chunks, dtype=dtype)
        self.nbuf = 0
        self.nwritten = 0
    def append(self, v):
        if self.nbuf >= len(self.buf):
            self.flush()
        self.buf[self.nbuf] = v
        self.nbuf += 1
    def flush(self):
        nn = self.nwritten + self.nbuf
        self.dataset.resize((nn,)+self.shape)
        self.dataset[self.nwritten:nn] = self.buf[:self.nbuf]
        self.nwritten = nn
        self.nbuf = 0
    def close(self):
        self.flush()


class StringDataset:
    def __init__(self, channel, field, h5file):
        dtype = h5py.string_dtype()
        self.channel = channel
        self.h5file = h5file
        shape = tuple()
        self.shape = shape
        chunks = [8 * 1024]
        if len(shape) == 0:
            pass
        elif len(shape) == 1:
            n = 8 * 1024 // shape[0]
            if n < 2:
                n = 2
            chunks = (n,) + shape
        else:
            raise RuntimeError("unsupported")
        chunks = tuple(chunks)
        self.chunks = chunks
        self.dataset = self.h5file.create_dataset(f"/{channel}/{field}", (0,) + shape, maxshape=(None,) + shape, dtype=dtype, chunks=chunks)
        self.buf = numpy.zeros(shape=chunks, dtype=dtype)
        self.nbuf = 0
        self.nwritten = 0
    def append(self, v):
        if self.nbuf >= len(self.buf):
            self.flush()
        self.buf[self.nbuf] = v
        self.nbuf += 1
    def flush(self):
        nn = self.nwritten + self.nbuf
        self.dataset.resize((nn,)+self.shape)
        self.dataset[self.nwritten:nn] = self.buf[:self.nbuf]
        self.nwritten = nn
        self.nbuf = 0
    def close(self):
        self.flush()


class ScalarI8Dataset:
    def __init__(self, channel, field, h5file):
        self.channel = channel
        self.h5file = h5file
        self.dataset = self.h5file.create_dataset(f"/{channel}/{field}", (0,), maxshape=(None,), dtype="i8", chunks=(8*1024,))
        self.buf = numpy.zeros(shape=(8 * 1024,), dtype="i8")
        self.nbuf = 0
        self.nwritten = 0
    def append(self, v):
        if self.nbuf >= len(self.buf):
            self.flush()
        self.buf[self.nbuf] = v
        self.nbuf += 1
    def flush(self):
        nn = self.nwritten + self.nbuf
        self.dataset.resize((nn,))
        self.dataset[self.nwritten:nn] = self.buf[:self.nbuf]
        self.nwritten = nn
        self.nbuf = 0
    def close(self):
        self.flush()


class TsDataset(ScalarI8Dataset):
    def __init__(self, channel, h5file):
        return super().__init__(channel, "timestamp", h5file)


class PulseDataset(ScalarI8Dataset):
    def __init__(self, channel, h5file):
        return super().__init__(channel, "pulse_id", h5file)


class DirectChunkwriteDataset:
    def __init__(self, channel, field, h5file, shape, dtype, compression):
        self.channel = channel
        self.field = field
        self.h5file = h5file
        shape = tuple(shape)
        self.shape = shape
        self.dtype = dtype
        if compression != data_api3.reader.Compression.BITSHUFFLE_LZ4:
            raise RuntimeError(f"unsupported compression {compression}")
        self.compression = bitshuffle.h5.H5FILTER
        self.dataset = self.h5file.create_dataset(f"/{channel}/{field}",
            (0,)+shape, maxshape=(None,)+shape,
            compression=self.compression,
            compression_opts=(2<<13, bitshuffle.h5.H5_COMPRESS_LZ4),
            chunks=(1,)+shape,
            dtype=dtype,
        )
        self.nwritten = 0
    def append(self, buf):
        nr = self.nwritten + 1
        self.dataset.resize((nr,)+self.shape)
        k = struct.unpack(">qi", buf[:12])
        uncompressed_size = k[0]
        block_size = k[1]
        if False:
            # is there some `trace` log level?
            logger.debug(f"uncompressed_size {uncompressed_size}  block_size {block_size}")
        self.compression_opts = (block_size, bitshuffle.h5.H5_COMPRESS_LZ4)
        off = (self.nwritten,) + (0,) * len(self.shape)
        self.dataset.id.write_direct_chunk(off, buf)
        self.nwritten += 1
    def close(self):
        pass


class Dataset:
    def __init__(self, name, reference):
        self.name = name
        self.reference = reference
        self.count = 0
    def close(self):
        pass


class Serializer:

    def __init__(self):
        self.file = None
        self.datasets = dict()
        self.datasets_chunkwrite = dict()

    def open(self, file_name):
        if self.file:
            logger.info('File '+self.file.name+' is currently open - will close it')
            self.close_file()
        self.file = h5py.File(file_name, "w")

    def close(self):
        self.compact_data()
        self.file.close()

    def compact_data(self):
        # Compact datasets, i.e. shrink them to actual size

        for key, dataset in self.datasets.items():
            if dataset.count < dataset.reference.shape[0]:
                logger.info('Compact data for dataset ' + dataset.name + ' from ' + str(dataset.reference.shape[0]) + ' to ' + str(dataset.count))
                dataset.reference.resize(dataset.count, axis=0)


    def append_dataset(self, dataset_name, value, dtype="f8", shape=[], compress=False):
        if value is None:
            raise RuntimeError("attempt to write None value")
        if dataset_name not in self.datasets:
            dataset_options = {}
            if compress:
                compression = "gzip"
                compression_opts = 5
                shuffle = True
                dataset_options = {'shuffle': shuffle}
                if compression != 'none':
                    dataset_options["compression"] = compression
                    if compression == "gzip":
                        dataset_options["compression"] = compression_opts

            reference = self.file.require_dataset(dataset_name, [1024]+shape, dtype=dtype, maxshape=[None,]+shape, **dataset_options)
            self.datasets[dataset_name] = Dataset(dataset_name, reference)

        dataset = self.datasets[dataset_name]
        # Check if dataset has required size, if not extend it
        if dataset.reference.shape[0] <= dataset.count:
            dataset.reference.resize(dataset.count + 1024, axis=0)
        dataset.reference[dataset.count] = value
        dataset.count += 1


DTYPE_BYTE = numpy.dtype("b")

def decompress_string(buf):
    global DTYPE_BYTE
    c_length = struct.unpack(">q", buf[0:8])[0]
    b_size = struct.unpack(">i", buf[8:12])[0]
    if c_length < 1 or c_length > 4 * 1024:
        raise RuntimeError(f"unexpected string size: {c_length}")
    if b_size < 512 or b_size > 16 * 1024:
        raise RuntimeError(f"unexpected block size: {b_size}")
    nbuf = numpy.frombuffer(buf[12:], dtype=numpy.uint8)
    dtype = DTYPE_BYTE
    value = bitshuffle.decompress_lz4(nbuf, shape=[c_length], dtype=dtype, block_size=int(b_size / dtype.itemsize))
    s1 = value.tobytes().decode()
    return s1


class Stream:
    def __init__(self, inner):
        self.inner = inner
        self.left = None
        self.total_read = 0
        self.print_interval = 10 * 1024 * 1024
        self.print_next = self.print_interval
    def read(self, n):
        if n is None:
            raise RuntimeError("NOT SUPPORTED")
            if self.left is None:
                a = self.inner.read1()
                if len(a) == 0:
                    print("EOF")
                self.add_total_read(len(a))
                return a
            else:
                ret = self.left
                self.left = None
                return ret
        elif n <= 0:
            raise RuntimeError("NOT SUPPORTED")
        elif self.left is not None and len(self.left) >= n:
            ret = self.left[:n]
            self.left = self.left[n:]
            if len(self.left) == 0:
                self.left = None
            return ret
        else:
            while True:
                a = bytes()
                try:
                    a = self.inner.read1()
                except http.client.IncompleteRead as e:
                    pass
                self.add_total_read(len(a))
                if len(a) == 0:
                    if self.left is None:
                        return bytes()
                    else:
                        ret = self.left
                        self.left = None
                        return ret
                if self.left is None:
                    self.left = a
                else:
                    self.left += a
                if len(self.left) >= n:
                    ret = self.left[:n]
                    self.left = self.left[n:]
                    if len(self.left) == 0:
                        self.left = None
                    return ret
    def add_total_read(self, n):
        self.total_read += n
        if self.total_read >= self.print_next:
            self.print_next = ((self.total_read // self.print_interval) + 1) * self.print_interval
            logger.debug(f"Total bytes read so far: {self.total_read / 1024 / 1024 : .3f} MB")



class RequestResult:
    def __init__(self):
        pass


def request(query: dict, filename: str, url: str):
    method = "POST"
    body = json.dumps(query)
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/octet-stream",
        "X-PythonDataAPIPackageVersion": PACKAGE_VERSION,
        "X-PythonDataAPIModule": __name__,
    }
    up = urllib.parse.urlparse(url)
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
    conn.request(method, up.path, body, headers)
    res = conn.getresponse()
    if res.status != 200:
        raise RuntimeError(f"Unable to retrieve data  {res.status}")
    hdf5reader = HDF5Reader(filename=filename)
    buffered_response = Stream(res)
    hdf5reader.read(buffered_response)
    ret = RequestResult()
    ret.nbytes_read = hdf5reader.nbytes_read
    return ret
