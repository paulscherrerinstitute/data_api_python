import struct
import json
import logging
import h5py
logger = logging.getLogger()

import io
import urllib3
import bitshuffle.h5


def resolve_struct_dtype(header_type: str, header_byte_order: str) -> str:

    header_type = header_type.lower()

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

    if dtype is not None and header_byte_order == "BIG_ENDIAN":
        dtype = ">" + dtype

    return dtype


class HDF5Reader:
    def __init__(self, filename: str):
        self.messages_read = 0
        self.data = {}
        self.filename = filename

    def read(self, stream):
        length = 0
        length_check = 0

        current_data = []
        current_channel_name = None
        current_value_extractor = None
        current_dtype = None
        current_shape = [1]

        serializer = Serializer()
        serializer.open(self.filename)

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
                value = bytes_read[17:]  # valu

                serializer.append_dataset('/' + current_channel_name + '/pulse_id', pulse_id, dtype='i8')
                serializer.append_dataset('/' + current_channel_name + '/timestamp', timestamp, dtype='i8')

                # Decide whether to write data as chunks or not
                # if data is compressed already always chunked writing
                # its kind of indicating that data is either waveform or image
                if current_compression is None and (current_shape is None or current_shape == [1]):

                    serializer.append_dataset('/' + current_channel_name + '/data', value,
                                              dtype=current_dtype,
                                              shape=current_shape, compress=False)
                else:
                    serializer.append_dataset_chunkwrite('/' + current_channel_name + '/data', value,
                                                         dtype=current_channel_info["type"].lower(),
                                                         shape=current_shape,
                                                         compression=current_compression)

                self.messages_read += 1
            elif mtype == 0:  # header message
                current_channel_info = json.loads(bytes_read[1:])
                # the header usually looks something like this:
                # {"name": "SLG-LSCP3-FNS:CH7:VAL_GET", "type":"float64", "compression":"0", "byteOrder":"BIG_ENDIAN",
                # "shape": null}
                logger.info(current_channel_info)

                current_channel_name = current_channel_info["name"]

                # Based on header use the correct value extractor
                # dtype = resolve_numpy_dtype(current_channel_info)
                current_dtype = resolve_struct_dtype(current_channel_info["type"], current_channel_info["byteOrder"])

                if current_channel_info["compression"] != "0": # TODO this needs to be fixed on the server side
                    if current_channel_info["compression"] == "1":
                        current_compression = 'bitshuffle_lz4'
                    else:
                        raise RuntimeError("Unsupported compression")  # TODO need to decide whether we completely abort or whether we just warn and skip to next channel
                else:
                    current_compression = None

                if current_channel_info['shape'] is not None:
                    # The API serves the shape in [width,height]
                    # numpy and hdf5 need the shape in the opposite order [height, width] therefore switching order
                    current_shape = current_channel_info['shape'][::-1]
                else:
                    current_shape = None

                logger.info(f"{current_channel_name} - type: {current_dtype} compression: {current_compression} shape: {current_shape}")

            bytes_read = stream.read(4)
            #         length_check = int.from_bytes(bytes_read, byteorder='big')
            length_check = struct.unpack('>i', bytes_read)[0]
            if length_check != length:
                raise RuntimeError(f"corrupted file reading {length} {length_check}")

        # print(f"{length}, {length_check}")

        serializer.close()  # closing will take care of compaction as well


class Dataset:
    def __init__(self, name, reference, count=0):
        self.name = name
        self.count = count
        self.reference = reference


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

    def append_dataset(self, dataset_name, value, dtype="f8", shape=[1,], compress=False):
        # print(dataset_name, dtype, shape, compress)

        # Create dataset if not existing
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

            reference = self.file.require_dataset(dataset_name, [1]+shape, dtype=dtype, maxshape=[None,]+shape, **dataset_options)
            self.datasets[dataset_name] = Dataset(dataset_name, reference)

        dataset = self.datasets[dataset_name]
        # Check if dataset has required size, if not extend it
        if dataset.reference.shape[0] < dataset.count + 1:
            dataset.reference.resize(dataset.count + 1000, axis=0)

        # TODO need to add an None check - i.e. for different frequencies
        if value is not None:
            dataset.reference[dataset.count] = value

        dataset.count += 1


    def append_dataset_chunkwrite(self, dataset_name, value, dtype="float32", shape=[1,], compression=None):
        # We currently accept only bitshuffle lz4 for direct chunk write
        if compression != "bitshuffle_lz4":
            raise RuntimeError(f"unsupported compression {compression}")

        # the first 8 bytes hold the uncompressed byte size
        # uncompressed_size = struct.unpack('>q', value[:8])[0]
        # the next 4 bytes hold the blocksize
        block_size = struct.unpack('>i', value[8:12])[0]
        # print(f"blocksize: {block_size}")

        # Create dataset if not existing
        if dataset_name not in self.datasets_chunkwrite:

            reference = self.file.create_dataset(dataset_name, tuple([1000]+shape), maxshape=tuple([None]+shape),
                                                 compression=bitshuffle.h5.H5FILTER,
                                                 compression_opts=(block_size, bitshuffle.h5.H5_COMPRESS_LZ4),
                                                 chunks=tuple([1]+shape), dtype=dtype)

            self.datasets_chunkwrite[dataset_name] = Dataset(dataset_name, reference)

        dataset = self.datasets_chunkwrite[dataset_name]

        if dataset.reference.shape[0] < dataset.count + 1:
            dataset.reference.resize(dataset.count + 1000, axis=0)

        # TODO need to add an None check - i.e. for different frequencies
        if value is not None:
            x_shape = (dataset.count,) + (0,) * len(shape)
            dataset.reference.id.write_direct_chunk(x_shape, value)

        dataset.count += 1


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

    reader = HDF5Reader(filename=filename)
    buffered_response = io.BufferedReader(response)
    reader.read(buffered_response)

