import struct
import json
import logging
import h5py
import io
import urllib3
import bitshuffle.h5
import data_api3.reader
import numpy

# Do not modify global logging settings in a library!
# For the logger, the recommended Python style is to use the module name.
logger = logging.getLogger(__name__)


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

        serializer = Serializer()
        serializer.open(self.filename)

        while True:
            bytes_read = stream.read(4)
            if len(bytes_read) != 4:
                break
            self.nbytes_read += len(bytes_read)
            length = struct.unpack('>i', bytes_read)[0]
            bytes_read = stream.read(length)
            if len(bytes_read) != length:
                raise RuntimeError("unexpected EOF")
            self.nbytes_read += len(bytes_read)
            mtype = struct.unpack('b', bytes_read[:1])[0]
            if mtype == 1 and self.in_channel:
                timestamp = struct.unpack('>q', bytes_read[1:9])[0]
                pulse_id = struct.unpack('>q', bytes_read[9:17])[0]

                raw_data_blob = bytes_read[17:]

                serializer.append_dataset('/' + current_channel_name + '/pulse_id', pulse_id, dtype='i8')
                serializer.append_dataset('/' + current_channel_name + '/timestamp', timestamp, dtype='i8')

                if current_shape == [] and current_compression is None:
                    if current_dtype == "string":
                        logger.error("unexpected uncompressed scalar data")
                        raise RuntimeError("unexpected uncompressed scalar data")
                    else:
                        # Numeric scalar data is expected to be always uncompressed.
                        serializer.append_dataset('/' + current_channel_name + '/data', current_value_extractor(raw_data_blob),
                                                  dtype=current_dtype,
                                                  shape=current_h5shape, compress=False)

                elif current_shape == [] and current_compression is not None and current_dtype == "string":
                    # Strings seem to be always compressed in databuffer
                    string_value = decompress_string(raw_data_blob)
                    serializer.append_dataset('/' + current_channel_name + '/data', string_value,
                        dtype=h5py.string_dtype(),
                        shape=[],
                    )

                elif len(current_shape) > 0:
                    if current_compression is None:
                        value = channel_header.value_extractor(raw_data_blob)
                        serializer.append_dataset('/' + current_channel_name + '/data', value,
                                                             dtype=current_channel_info["type"].lower(),
                                                             shape=current_h5shape)
                    else:
                        # Non-scalar data, compressed:
                        serializer.append_dataset_chunkwrite('/' + current_channel_name + '/data', raw_data_blob,
                                                             dtype=current_channel_info["type"].lower(),
                                                             shape=current_h5shape,
                                                             compression=current_compression)

                else:
                    raise RuntimeError(f"can not write  current_compression {current_compression}  current_shape {current_shape}")

                self.messages_read += 1

            elif mtype == 0:
                self.in_channel = False
                msg = json.loads(bytes_read[1:])
                res = data_api3.reader.process_channel_header(msg)
                if res.error:
                    logger.error("Can not parse channel header message: {}".format(msg))
                elif res.empty:
                    logger.info("No data for channel {}".format(res.channel_name))
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

            bytes_read = stream.read(4)
            length_check = struct.unpack('>i', bytes_read)[0]
            if length_check != length:
                raise RuntimeError(f"corrupted file reading {length} {length_check}")
            self.nbytes_read += len(bytes_read)

        serializer.close()


class Dataset:
    def __init__(self, name, reference):
        self.name = name
        self.reference = reference
        self.count = 0


class Serializer:

    def __init__(self):
        self.file = None
        self.datasets = dict()
        self.datasets_chunkwrite = dict()

    def open(self, file_name):

        if self.file:
            logger.info('File '+self.file.name+' is currently open - will close it')
            self.close_file()

        logger.info('Open file '+file_name)
        self.file = h5py.File(file_name, "w")

    def close(self):
        self.compact_data()
        self.compact_data_chunkwrite()

        logger.info('Close file '+self.file.name)
        self.file.close()

    def compact_data(self):
        # Compact datasets, i.e. shrink them to actual size

        for key, dataset in self.datasets.items():
            if dataset.count < dataset.reference.shape[0]:
                logger.info('Compact data for dataset ' + dataset.name + ' from ' + str(dataset.reference.shape[0]) + ' to ' + str(dataset.count))
                dataset.reference.resize(dataset.count, axis=0)

    def compact_data_chunkwrite(self):
        # Compact datasets, i.e. shrink them to actual size

        for key, dataset in self.datasets_chunkwrite.items():
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


    def append_dataset_chunkwrite(self, dataset_name, value, dtype, shape, compression):
        if compression != data_api3.reader.Compression.BITSHUFFLE_LZ4:
            # We currently accept no other compression method for direct chunk write
            raise RuntimeError(f"unsupported compression {compression}")

        # the first 8 bytes hold the uncompressed byte size
        # uncompressed_size = struct.unpack('>q', value[:8])[0]
        # the next 4 bytes hold the blocksize
        block_size = struct.unpack('>i', value[8:12])[0]
        # print(f"blocksize: {block_size}")

        # Create dataset if not existing
        if dataset_name not in self.datasets_chunkwrite:

            reference = self.file.create_dataset(dataset_name, tuple([1024]+shape), maxshape=tuple([None]+shape),
                                                 compression=bitshuffle.h5.H5FILTER,
                                                 compression_opts=(block_size, bitshuffle.h5.H5_COMPRESS_LZ4),
                                                 chunks=tuple([1]+shape), dtype=dtype)

            self.datasets_chunkwrite[dataset_name] = Dataset(dataset_name, reference)

        dataset = self.datasets_chunkwrite[dataset_name]

        if dataset.reference.shape[0] <= dataset.count:
            dataset.reference.resize(dataset.count + 1024, axis=0)

        if value is not None:
            x_shape = (dataset.count,) + (0,) * len(shape)
            dataset.reference.id.write_direct_chunk(x_shape, value)
        else:
            raise RuntimeError(f"unexpected value type {type(value)}")

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


class RequestResult:
    def __init__(self):
        pass


def request(query: dict, filename: str, url="http://localhost:8080/api/v1/query"):
    # IMPORTANT NOTE: the use of the requests library is not possible due to this issue:
    # https://github.com/urllib3/urllib3/issues/1305

    encoded_data = json.dumps(query).encode('utf-8')

    http = urllib3.PoolManager(cert_reqs='CERT_NONE')
    urllib3.disable_warnings()
    response = http.request('POST', url,
                            body=encoded_data,
                            headers={'Content-Type': "application/json", "Accept": "application/octet-stream"},
                            preload_content=False)
    if response.status != 200:
        raise RuntimeError(f"Unable to retrieve data: {response.data}")

    # Were hitting this issue here:
    # https://github.com/urllib3/urllib3/issues/1305
    response._fp.isclosed = lambda: False  # monkey patch

    hdf5reader = HDF5Reader(filename=filename)
    buffered_response = io.BufferedReader(response)
    hdf5reader.read(buffered_response)
    ret = RequestResult()
    ret.nbytes_read = hdf5reader.nbytes_read
    return ret
