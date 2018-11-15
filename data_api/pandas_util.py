import logging
import numpy
import os


def build_pandas_data_frame(data, index_field="globalDate"):
    import pandas
    # for nicer printing
    pandas.set_option('display.float_format', lambda x: '%.3f' % x)

    if index_field not in ["globalDate", "globalSeconds", "pulseId"]:
        RuntimeError("index_field must be 'globalDate', 'globalSeconds', or 'pulseId'")

    data_frame = None

    # Same as query["fields"] except "value"
    metadata_fields = ["pulseId", "globalSeconds", "globalDate", "eventCount"]

    for channel_data in data:
        if not channel_data['data']:  # data_entry['data'] is empty, i.e. []
            # No data returned
            logging.warning("no data returned for channel %s" % channel_data['channel']['name'])
            # Create empty pandas data_frame
            tdf = pandas.DataFrame(columns=[index_field, channel_data['channel']['name']])
        else:
            if isinstance(channel_data['data'][0]['value'], dict):
                # Server side aggregation
                entry = []
                keys = sorted(channel_data['data'][0]['value'])

                for x in channel_data['data']:
                    entry.append([x[m] for m in metadata_fields] + [x['value'][k] for k in keys])
                columns = metadata_fields + [channel_data['channel']['name'] + ":" + k for k in keys]

            else:
                # No aggregation
                entry = []
                for data_entry in channel_data['data']:
                    entry.append([data_entry[m] for m in metadata_fields] + [data_entry['value']])
                # entry = [[x[m] for m in metadata_fields] + [x['value'], ] for x in data_entry['data']]
                columns = metadata_fields + [channel_data['channel']['name']]

            tdf = pandas.DataFrame(entry, columns=columns)
            tdf.drop_duplicates(index_field, inplace=True)

            # TODO check if necessary
            # because pd.to_numeric has not enough precision (only float 64, not enough for globalSeconds)
            # does 128 makes sense? do we need nanoseconds?
            conversions = {"pulseId": numpy.int64}
            for col in tdf.columns:
                if col in conversions:
                    tdf[col] = tdf[col].apply(conversions[col])

        if data_frame is not None:
            data_frame = pandas.merge(data_frame, tdf, how="outer")  # Missing values will be filled with NaN
        else:
            data_frame = tdf

    if data_frame.shape[0] > 0:
        # dataframe is not empty

        # Apply milliseconds rounding
        # this is a string manipulation !
        data_frame["globalNanoseconds"] = data_frame.globalSeconds.map(lambda x: int(x.split('.')[1][3:]))
        data_frame["globalSeconds"] = data_frame.globalSeconds.map(lambda x: float(x.split('.')[0] + "." + x.split('.')[1][:3]))
        # Fix pulseid to int64 - not sure whether this really works
        # data_frame["pulseId"] = data_frame["pulseId"].astype(np.int64)

        data_frame.set_index(index_field, inplace=True)
        data_frame.sort_index(inplace=True)

    return data_frame


def to_hdf5(data, filename, overwrite=False, compression="gzip", compression_opts=5, shuffle=True):
    # Write dataframe to hdf5
    import h5py

    dataset_options = {'shuffle': shuffle}
    if compression != 'none':
        dataset_options["compression"] = compression
        if compression == "gzip":
            dataset_options["compression"] = compression_opts

    if os.path.isfile(filename):
        if overwrite:
            logging.warning("Overwriting %s" % filename)
            os.remove(filename)
        else:
            raise RuntimeError("File %s exists, and overwrite flag is False, exiting" % filename)

    outfile = h5py.File(filename, "w")

    if data.index.name != "globalDate":  # Skip globalDate
        outfile.create_dataset(data.index.name, data=data.index.tolist())

    for dataset in data.columns:
        if dataset == "globalDate":  # Skip globalDate
            continue

        outfile.create_dataset(dataset, data=data[dataset].tolist(), **dataset_options)

    outfile.close()
