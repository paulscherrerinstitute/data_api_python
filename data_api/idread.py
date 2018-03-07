# The specification of idread can be found here: https://github.psi.ch/sf_daq/idread_specification

# https://docs.scipy.org/doc/numpy-1.13.0/reference/arrays.dtypes.html

import numpy
import bitshuffle
import json


channels = None


def decode(bytes):
    # print('decode')
    # print(raw_data)

    while True:
        # read size
        b = bytes.read(8)
        if b == b'':
            print('end of file')
            break
        size = numpy.frombuffer(b, dtype='>i8')

        # read id
        id = numpy.frombuffer(bytes.read(2), dtype='>i2')

        if id == 1:
            data = read_header(bytes, size)
            print(data)

            channels = []
            for channel in data['channels']:
                if channel["type"] == "uint32" and channel["encoding"] == "big":
                    channels.append({'size': 4, 'dtype': '>u4'})
                else:
                    pass

        elif id == 0:
            if channels is not None:
                for channel in channels:
                    # eventSize - int32
                    # iocTime - int64
                    # pulseId - int64
                    # globalTime - int64
                    # status - int8
                    # severity - int8
                    # value - dtype
                    numpy.frombuffer(bytes.read(4), dtype='>i4')
                    numpy.frombuffer(bytes.read(8), dtype='>i8')
                    numpy.frombuffer(bytes.read(8), dtype='>i8')
                    numpy.frombuffer(bytes.read(8), dtype='>i8')
                    numpy.frombuffer(bytes.read(1), dtype='>i1')
                    numpy.frombuffer(bytes.read(1), dtype='>i1')

                    # number of bytes to read = size - 2 - 4 - 8 - 8 - 8 - 1 - 1 = size - 32
                    # first 2 is from the id, the rest from the reads above
                    data = numpy.frombuffer(bytes.read(int(size-32)), dtype=channel["dtype"])
                    print(data)

                print('do')
            else:
                bytes.read(int(size - 2))
                print('cannot deserialize')

        else:

            print("id %i not supported" % id)
            bytes.read(int(size-2))


def read_header(bytes, size):
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
        import struct
        length = struct.unpack(">q", raw_data[:8])[0]
        b_size = struct.unpack(">i", raw_data[8:12])[0]
        # print(length)
        # print(b_size)

        byte_array = bitshuffle.decompress_lz4(numpy.frombuffer(raw_data[12:], dtype=numpy.uint8), shape=(length,),
                                               dtype=numpy.dtype('uint8'))
        data = byte_array.tobytes().decode()
    else:
        raise RuntimeError('Compression not supported')

    return json.loads(data)

