    # The specification of idread can be found here: https://github.psi.ch/sf_daq/idread_specification

# https://docs.scipy.org/doc/numpy-1.13.0/reference/arrays.dtypes.html

import numpy
import bitshuffle
import json
import struct

import logging

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


def decode(bytes, serializer=None):

    channels = None

    while True:
        # read size
        b = bytes.read(8)
        if b == b'':
            print('end of file')
            break
        size = numpy.frombuffer(b, dtype='>i8')

        # read id
        id = numpy.frombuffer(bytes.read(2), dtype='>i2')

        if id == 1:  # Read Header
            header = _read_header(bytes, size)
            logging.debug(header)

            channels = []
            for channel in header['channels']:
                n_channel = {}
                if "type" not in channel or channel["type"] == "float64" or channel["type"] == "float":  # default
                    n_channel = {'size': 8, 'dtype': 'f8'}
                elif channel["type"] == "uint8":
                    n_channel = {'size': 1, 'dtype': 'u1'}
                elif channel["type"] == "int8":
                    n_channel = {'size': 1, 'dtype': 'i1'}
                elif channel["type"] == "uint16":
                    n_channel = {'size': 2, 'dtype': 'u2'}
                elif channel["type"] == "int16":
                    n_channel = {'size': 2, 'dtype': 'i2'}
                elif channel["type"] == "uint32":
                    n_channel = {'size': 4, 'dtype': 'u4'}
                elif channel["type"] == "int32":
                    n_channel = {'size': 4, 'dtype': 'i4'}
                elif channel["type"] == "uint64":
                    n_channel = {'size': 8, 'dtype': 'u8'}
                elif channel["type"] == "int64" or channel["type"] == "int":
                    n_channel = {'size': 8, 'dtype': 'i8'}
                elif channel["type"] == "float32":
                    n_channel = {'size': 4, 'dtype': 'f4'}
                else:
                    # Raise exception for others (including strings)
                    raise RuntimeError('Unsupported data type')

                # need to fix dtype with encoding
                n_channel['encoding'] = '>' if 'encoding' in channel and channel["encoding"] == "big" else ''
                # n_channel['dtype'] = n_channel['encoding']+n_channel['dtype']

                n_channel['compression'] = channel['compression'] if 'compression' in channel else None
                # Numpy is slowest dimension first, but bsread is fastest dimension first.
                n_channel['shape'] = channel['shape'][::-1] if 'shape' in channel else [1]

                n_channel['name'] = channel['name']
                channels.append(n_channel)

            print(channels)

        elif id == 0:  # Read Values

            if channels is None or channel == []:  # Header was not yet received
                bytes.read(int(size - 2))
                logging.warning('No channels specified, cannot deserialize - drop remaining bytes')

            else:
                size_counter = 0
                for channel in channels:

                    # eventSize - int32
                    # iocTime - int64
                    # pulseId - int64
                    # globalTime - int64
                    # status - int8
                    # severity - int8
                    # value - dtype

                    event_size = numpy.frombuffer(bytes.read(4), dtype=channel['encoding']+'i4')
                    ioc_time = numpy.frombuffer(bytes.read(8), dtype=channel['encoding']+'i8')
                    pulse_id = numpy.frombuffer(bytes.read(8), dtype=channel['encoding']+'i8')
                    global_time = numpy.frombuffer(bytes.read(8), dtype=channel['encoding']+'i8')
                    status = numpy.frombuffer(bytes.read(1), dtype=channel['encoding']+'i1')
                    severity = numpy.frombuffer(bytes.read(1), dtype=channel['encoding']+'i1')

                    # number of bytes to subtract from event_size = 8 - 8 - 8 - 1 - 1 = 26
                    raw_bytes = bytes.read(int(event_size-26))

                    if channel['compression'] is not None:

                        # TODO need to check for compression type -
                        # Ideally this is done while header parsing, and here I would get the decode function
                        length = struct.unpack(">q", raw_bytes[:8])[0]
                        b_size = struct.unpack(">i", raw_bytes[8:12])[0]

                        data = bitshuffle.decompress_lz4(numpy.frombuffer(raw_bytes[12:], dtype=numpy.uint8),
                                                         shape=(channel['shape']),
                                                         dtype=numpy.dtype(n_channel['encoding']+channel["dtype"]),
                                                         block_size=b_size/channel['size'])

                    else:
                        data = numpy.frombuffer(raw_bytes, dtype=n_channel['encoding']+channel["dtype"])

                    # reshape the array
                    if channel['shape'] is not None and channel['shape'] != [1]:
                        data = data.reshape(channel['shape'])

                    size_counter += (2 + 4 + event_size)  # 2 for id, 4 for event_size

                    if serializer is not None:
                        serializer.append_dataset('/' + channel['name'] + '/data', data, dtype=channel['dtype'], shape=channel['shape'])
                        serializer.append_dataset('/' + channel['name'] + '/pulse_id', pulse_id, dtype='i8')
                        serializer.append_dataset('/' + channel['name'] + '/timestamp', global_time, dtype='i8')
                        serializer.append_dataset('/' + channel['name'] + '/ioc_timestamp', ioc_time, dtype='i8')
                        serializer.append_dataset('/' + channel['name'] + '/status', status, dtype='i1')
                        serializer.append_dataset('/' + channel['name'] + '/severity', severity, dtype='i1')

                remaining_bytes = size-size_counter
                if remaining_bytes > 0:
                    logger.warning("Remaining bytes - %d - drop remaining bytes" % remaining_bytes)
                    bytes.read(remaining_bytes)

        else:

            logging.warning("id %i not supported - drop remaining bytes" % id)
            bytes.read(int(size-2))


def _read_header(bytes, size):
    hash = numpy.frombuffer(bytes.read(8), dtype='>i8')
    compression = numpy.frombuffer(bytes.read(1), dtype='>i1')

    raw_data = bytes.read(int(size - 2 - 8 - 1))

    # print(size)
    # print(id)
    # print(hash)
    # print(compression)

    if compression == 0:
        data = raw_data.decode()
    elif compression == 1:
        length = struct.unpack(">q", raw_data[:8])[0]
        b_size = struct.unpack(">i", raw_data[8:12])[0]
        # print(length)
        # print(b_size)

        byte_array = bitshuffle.decompress_lz4(numpy.frombuffer(raw_data[12:], dtype=numpy.uint8),
                                               shape=(length,),
                                               dtype=numpy.dtype('uint8'),
                                               block_size=b_size)
        data = byte_array.tobytes().decode()
    else:
        raise RuntimeError('Compression not supported')

    return json.loads(data)
