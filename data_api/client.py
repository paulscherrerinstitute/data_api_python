from __future__ import print_function, division
from datetime import datetime, timedelta  # timezone
import pytz
import requests
import os
import pandas as pd
import sys
import numpy as np
import pprint
import logging

# for nicer printing
pd.set_option('display.float_format', lambda x: '%.3f' % x)

logger = logging.getLogger("DataApiClient")
logger.setLevel(logging.INFO)

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

# because pd.to_numeric has not enough precision (only float 64, not enough for globalSeconds)
# does 128 makes sense? do we need nanoseconds?
conversions = {"pulseId": np.int64}

default_base_url = "https://data-api.psi.ch/"


def _convert_date(date_string):
    """
    Convert a date string in isoformat
    
    Parameters
    ----------
    date_string : string
        Date string in ("%Y-%m-%d %H:%M" or "%Y-%m-%d %H:%M:%S") format
    
    Returns
    -------
    st : 
        isoformat version of the string
    """
    try:
        st = pd.to_datetime(date_string, ).tz_localize(pytz.timezone('Europe/Zurich'))
    except ValueError:
        raise RuntimeError("Cannot convert date " + date_string + ", please check")

    return st, datetime.isoformat(st)


def _set_pulseid_range(start, end, delta):
    if start == "" and end == "":
        raise RuntimeError("Must select at least start or end")
    if start != "":
        if end == "":
            end = start + delta - 1
    else:
        start = end - delta + 1
    return {"endPulseId": str(end), "startPulseId": str(start)}


def _set_seconds_range(start, end, delta):
    if start == "" and end == "":
        raise RuntimeError("Must select at least start or end")
    if start != "":
        if end == "":
            end = start + delta - 1
    else:
        start = end - delta + 1
    return {"startSeconds": "%.9f" % start, "endSeconds": "%.9f" % end}


def _set_time_range(start_date, end_date, delta_time):
    d = {}
    if start_date == "" and end_date == "":
        _, d["startDate"] = _convert_date(datetime.isoformat(datetime.now() - timedelta(seconds=delta_time + 60)))
        _, d["endDate"] = _convert_date(datetime.isoformat(datetime.now() - timedelta(seconds=60)))
    else:
        if start_date != "" and end_date != "":
            st_dt, st_iso = _convert_date(start_date)
            d["startDate"] = st_iso
            st_dt, st_iso = _convert_date(end_date)
            d["endDate"] = st_iso
        elif start_date != "":
            st_dt, st_iso = _convert_date(start_date)
            d["startDate"] = st_iso
            d["endDate"] = datetime.isoformat(st_dt + timedelta(seconds=delta_time))
        else:
            st_dt, st_iso = _convert_date(end_date)
            d["endDate"] = st_iso
            d["startDate"] = datetime.isoformat(st_dt - timedelta(seconds=delta_time))
    return d


class DataApiClient(object):

    def __init__(self, source_name=default_base_url):
        self._aggregation = {}
        self._server_aggregation = True
        self.source_name = source_name

    @property
    def server_aggregation(self):
        """
        Enables / disables server-side aggregation (default is: enabled). If set to True enables it, to False disables (it enables client side reduction, more limited and resource intensive, just for debug / edge cases)
        """
        return self._server_aggregation

    @server_aggregation.setter
    def server_aggregation(self, value=True):
        self._server_aggregation = value
        logger.info("Server side aggregation set to %s" % self._server_aggregation)
            
    def set_aggregation(self, aggregation_type="value", aggregations=["min", "mean", "max"], extrema=[],
                        nr_of_bins=None, duration_per_bin=None, pulses_per_bin=None, ):
        """
        Configure data aggregation (reduction). It follows the API description detailed here: https://github.psi.ch/sf_daq/ch.psi.daq.queryrest#data-aggregation. Binning is performed dividing the interval in a number of equally-spaced bins: the bin length can be set up in seconds, pulses, or setting up the total number of bins. If reduction is performed server-side, then also the number of events in each bin (eventCount) is returned.
        
        Parameters
        ----------
        aggregation_type : string
            How to aggregate data. It can be 'value', 'index', 'extrema'. Default: 'value'. See https://github.psi.ch/sf_daq/ch.psi.daq.domain/blob/master/src/main/java/ch/psi/daq/domain/query/operation/AggregationType.java

        aggregations : list of strings
            what kind of aggregation is required. Possible values are: min, max, mean, sum, count, variance, stddev, kurtosis, skewness. Default: ["min", "mean", "max"].  See https://github.psi.ch/sf_daq/ch.psi.daq.domain/blob/master/src/main/java/ch/psi/daq/domain/query/operation/Aggregation.java

        extrema : list of strings
            (NOT SUPPORTED ATM) returns in addition to data global extrema. Possible values are: minValue, maxValue. Default: []

        nr_of_bins : int
            Number of bins used to aggregate data. Mutually exclusive with duration_per_bin and pulses_per_bin. Default: None

        duration_per_bin : int
            Number of seconds to be used per each aggregation bin. Mutually exclusive with nr_of_bins and pulse_per_bin. Default: None

        pulses_per_bin : int
            Number of pulses to be used per each aggregation bin. Mutually exclusive with nr_of_bins and duration_per_bin. Default: None

        Returns
        -------
        None
        """
        if not self._server_aggregation:
            logger.warning("Client-side aggregation on waveforms is not supported, sorry. Please fill a request ticket in case you would need it, or switch to server-side aggregation using enable_server_aggregation.")
        
        # keep state?
        
        self._aggregation["aggregationType"] = aggregation_type
        self._aggregation["aggregations"] = aggregations

        if extrema != []:
            self._aggregation["extrema"] = []

        if (nr_of_bins is not None) + (duration_per_bin is not None) + (pulses_per_bin is not None) > 1:
            logger.error("Can specify only one of nr_of_bins, duration_per_bin or pulse_per_bin")
            return
        
        if nr_of_bins is not None:
            self._aggregation["nrOfBins"] = nr_of_bins
            for k in ["durationPerBin", "pulsesPerBin"]:
                if k in self._aggregation:
                    self._aggregation.pop(k)
        elif duration_per_bin is not None:
            logger.error("durationPerBin aggregation not supported yet client-side, doing nothing")
        elif pulses_per_bin is not None:
            self._aggregation["pulsesPerBin"] = pulses_per_bin
            for k in ["durationPerBin", "nrOfBins"]:
                if k in self._aggregation:
                    self._aggregation.pop(k)

        return
    
    def get_aggregation(self, ):
        return self._aggregation

    def clear(self, ):
        """
        Resets all stored configurations, excluding source_name
        """ 

        self._aggregation = {}

    def get_data(self, channels, start="", end="", range_type="globalDate", delta_range=1, index_field=None,
                 include_nanoseconds=True):
        """
           Retrieve data from the Data API. You can define different ranges, as 'globalDate', 'globalSeconds', 'pulseId' (the start, end and delta_range parameters will be checked accordingly). At the moment, globalSeconds are returned up to the millisecond (truncated).

           Examples:
           df = dac.get_data(channels=['SINSB02-RIQM-DCP10:FOR-PHASE-AVG', 'SINSB02-RKLY-DCP10:FOR-PHASE-AVG', 'SINSB02-RIQM-DCP10:FOR-PHASE'], end="2016-07-28 08:05", range_type="globalDate", delta_range=100)
           df = dac.get_data(channels='SINSB02-RIQM-DCP10:FOR-PHASE-AVG', start=10000000, end=10000100, range_type="pulseId")
           
           Parameters
           ----------
           channels: string or list of strings
               string (or list of strings) containing the channel names
           start: string, int or float
               start of the range. It is a string in case of a date range, in the form of 'YYYY:MM:DD HH:MM[:SS]', an integer in case of pulseId, or a float in case of date range.
           end: string, int or float
               end of the range. See start for more details
           delta_range: int
               when specifying only start or end, this parameter sets the other end of the range. It is pulses when pulseId range is used, seconds otherwise. When only start is defined, delta_range is added to that: conversely when only end is defined. You cannot define start, end and delta_range at the same time. If only delta_range is specified, then end is by default set to one minute ago, and start computed accordingly
           index_field : string
               you can decide whether data is indexed using globalSeconds, pulseId or globalDate.
           include_nanoseconds : bool
               NOT YET SUPPORTED! when returned in a DataFrame, globalSeconds are precise up to the microsecond level. If you need nanosecond information, put this option to True and a globalNanoseconds column will be created. 

           Returns
           -------
           df : Pandas DataFrame
               Pandas DataFrame containing indexed data
        """

        # Check input parameters
        if range_type not in ["globalDate", "globalSeconds", "pulseId"]:
            RuntimeError("range_type must be 'globalDate', 'globalSeconds', or 'pulseId'")

        if index_field is None:
            logger.info("indexing will be done on %s" % range_type)
            index_field = "globalDate"
            
        if index_field not in ["globalDate", "globalSeconds", "pulseId"]:
            RuntimeError("index_field must be 'globalDate', 'globalSeconds', or 'pulseId'")

        # Check if a single channel is passed instead of a list of channels
        if isinstance(channels, str):
            channels = [channels, ]

        # Build up channel list for the query
        channel_list = []
        for channel in channels:
            channel_name = channel.split("/")

            if len(channel_name) > 2:
                raise RuntimeError("%s is not a valid channel specification" % channel)
            elif len(channel_name) == 1:
                channel_list.append({"name": channel_name[0], "backend": "sf-databuffer"})
            else:
                channel_list.append({"name": channel_name[1], "backend": channel_name[0]})

        logger.info("Querying channels: %s" % channels)
        logger.info("Querying on %s between %s and %s" % (range_type, start, end))

        query = dict()
        query["channels"] = channel_list
        query["fields"] = ["pulseId", "globalSeconds", "globalDate", "value", ]

        query["range"] = {}
        if range_type == "pulseId":
            query["range"] = _set_pulseid_range(start, end, delta_range)
        elif range_type == "globalSeconds":
            query["range"] = _set_seconds_range(start, end, delta_range)
        else:
            query["range"] = _set_time_range(start, end, delta_range)

        if self._aggregation != {} and self._server_aggregation:
            query["aggregation"] = self._aggregation

        metadata_fields = ["pulseId", "globalSeconds", "globalDate"]
        if self._server_aggregation:
            query["fields"].append("eventCount")
            metadata_fields.append("eventCount")

        # Query server
        response = requests.post(self.source_name + '/sf/query', json=query)

        # Check for successful return of data
        if response.status_code != 200:
            raise RuntimeError("Unable to retrieve data from server: ", response)

        data = response.json()
        print(data)

        df = None
        for d in data:
            if d['data'] == []:
                logger.warning("no data returned for channel %s" % d['channel']['name'])
                continue

            if isinstance(d['data'][0]['value'], dict):
                entry = []
                keys = sorted(d['data'][0]['value'])
                for x in d['data']:
                    entry.append([x[m] for m in metadata_fields] + [x['value'][k] for k in keys])
                columns = metadata_fields + [d['channel']['name'] + ":" + k for k in keys]

            else:
                entry = [[x[m] for m in metadata_fields] + [x['value'], ] for x in d['data']]
                columns = metadata_fields + [d['channel']['name'], ]

            tdf = pd.DataFrame(entry, columns=columns)
            tdf.drop_duplicates(index_field, inplace=True)
            for col in tdf.columns:
                if col in conversions:
                    tdf[col] = tdf[col].apply(conversions[col])

            if df is not None:
                if self._server_aggregation:
                    try:
                        if not (tdf.index == df.index).all():
                            logger.warning("It can be that server-side reduction returned results with different indexing. You can check this enabling client-side reduction with enable_server_aggregation(False), perform the query again and compare the results")
                    except ValueError:
                        logger.info("Got two lists of different length. Missing values will be filled with NaN")
                df = pd.merge(df, tdf, how="inner")
            else:
                df = tdf

        # milliseconds rounding
        df["globalNanoseconds"] = df.globalSeconds.map(lambda x: int(x.split('.')[1][3:]))
        df["globalSeconds"] = df.globalSeconds.map(lambda x: float(x.split('.')[0] + "." + x.split('.')[1][:3]))

        df["pulseId"] = df["pulseId"].astype(np.int64)
        df.set_index(index_field, inplace=True)

        if df is None:
            logger.warning("no data returned overall")
            return df
            
        if not self._server_aggregation and self._aggregation != {}:
            df = self._clientside_aggregation(df)

        return df

    def _clientside_aggregation(self, df):
        #self.df = df
        if df is None:
            return df

        logger.info("Client-side aggregation")

        if self._aggregation["aggregationType"] != "value":
            logger.error("Only value based aggregation is supported, doing nothing")
            return df
        df_aggr = None

        #durationPerBin, pulsesPerBin
        if "pulsesPerBin" in self._aggregation:
            bin_mask = np.array([i // self._aggregation["pulsesPerBin"] for i in range(len(df.index))])
            bins = df.index[[0, ] + (1 + np.where((bin_mask[1:] - bin_mask[0:-1]) == 1)[0]).tolist()]
            groups = df.groupby(bin_mask)
        elif "nrOfBins" in self._aggregation:
            bins = np.linspace(df.index[0], 1 + df.index[-1], self._aggregation["nrOfBins"], endpoint=False).astype(int)
            groups = df.groupby(np.digitize(df.index, bins))

        for aggr in self._aggregation["aggregations"]:
            if aggr not in groups.describe().index.levels[1].values:
                logger.error("%s aggregation not supported, skipping" % aggr)
            if df_aggr is None:
                df_aggr = groups.describe().xs(aggr, level=1)
                orig_columns = df_aggr.columns
                df_aggr.columns = [x + ":" + aggr for x in orig_columns]
            else:
                for c in orig_columns:
                    df_aggr[c + ":" + aggr] = groups.describe().xs(aggr, level=1)[c]
        df_aggr.set_index(bins, inplace=True)

        for i in ['globalSeconds', 'globalDate', 'globalNanoseconds', 'pulseId']:
            if i + ":min" in df_aggr.columns:
                df_aggr[i] = df_aggr[i + ":min"]
                df_aggr.drop(i + ":min", axis=1, inplace=True)
            for j in [x for x in df_aggr.columns if x.find(i) != -1 and x.find(":") != -1]:
                df_aggr.drop(j, axis=1, inplace=True)

        # add here also date reindexing
        return df_aggr


def to_hdf5(data, filename="data_api_output.h5", overwrite=False, compression="gzip", compression_opts=5, shuffle=True):
    """
    Dumps DataFrame from DataApi as a HDF5 file. It assumes that the index is either pulseId or globalSeconds: 
    in case it is date, it will convert it to globalSeconds.

    Example:
    dac.to_hdf5(df, filename="test.h5", overwrite=True)
    [INFO] File test.h5 written

    
    Parameters
    ----------
    data: Data to write to file
    filename : string
        Name of the output file. Defaults to data_api_output.h5
    overwrite: bool
        Flag to overwrite existing files. False by default.
    compression: string
        Valid values are 'gzip', 'lzf', 'none'
    compression_opts: int
        Compression settings.  This is an integer for gzip, not used for lzf.
    shuffle:        Use bitshuffle

    Returns
    -------
    r : 
        None if successful, otherwise -1
    """

    import h5py

    dset_opts = {'shuffle': shuffle}
    if compression != 'none':
        dset_opts["compression"] = compression
        if compression == "gzip":
            dset_opts["compression"] = compression_opts

    if os.path.isfile(filename):
        if overwrite:
            logger.warning("Overwriting %s" % filename)
        else:
            logger.error("File %s exists, and overwrite flag is False, exiting" % filename)
            sys.exit(-1)

    # this part is very convoluted, rewrite...
    try:
        index_name = data.index.name
        if index_name != "globalDate":
        #    if "globalDate" not in data.columns:
        #        print(data)
        #        index_list = pd.to_datetime(data.index).to_series().apply(lambda x: x.replace(tzinfo=pytz.utc).timestamp()).tolist()
        #        #index_list = pd.to_datetime(data.index).to_series().apply(lambda x: x.replace(tzinfo=pytz.utc).value).tolist()
        #    else:
        #        index_name = None
        #else:
            index_list = data.index.tolist()
    except:
        logger.error(sys.exc_info()[1])
        return -1

    outfile = h5py.File(filename, "w")
    if index_name is not None:
        try:
            if index_name != "globalDate":
                outfile.create_dataset(index_name, data=data.index.tolist())
        except:
            logger.error("error in creating %s dataset, %s" % (index_name, sys.exc_info()))
    for dataset in data.columns:
        if dataset == "globalDate":
            logger.info("Skipping globalDate (it will be recreated upon reading). Saving it as globalSeconds")
            if "globalSeconds" in data.columns:
                outfile.create_dataset(dataset, data=data["globalSeconds"], **dset_opts)
            elif data.index.name == "globalSeconds":
                outfile.create_dataset(dataset, data=data.index, **dset_opts)
            else:
                logger.warn("globalSeconds not available, globalDate information will be dropped")
        else:
            #print(data[dataset].tolist())
            outfile.create_dataset(dataset, data=data[dataset].tolist(), **dset_opts)

    outfile.close()
    logger.info("File %s written" % filename)
    return


def from_hdf5(filename, index_field="globalSeconds", recreate_date=True):
    """
    Loads DataFrame from HDF5 file. It assumes that file has been produced by DataApiClient.to_hdf5 routine.
    
    :param filename: 
    :param index_field:     Field to be used as index. Can be "globalSeconds" (default), "globalDate" or "pulseId".
    :param recreate_date: 
    :return:                Dataframe or None
    """

    import h5py

    try:
        infile = h5py.File(filename, "r")
    except:
        logger.error(sys.exc_info()[1])
        return None

    data = pd.DataFrame()

    for k in infile.keys():
        if k == "globalDate":
            data["globalDate"] = infile["globalDate"][:].astype(np.double)
            data["globalDate"] = data["globalDate"].apply(datetime.fromtimestamp)
            data["globalDate"] = data["globalDate"].apply(lambda t: t.strftime("%Y-%m-%dT%H:%M:%S.%f"))
        else:
            data[k] = infile[k][:]

    try:
        data.set_index(index_field, inplace=True)
    except:
        logger.error("Cannot set index on %s, possible values are:" % index_field, list(infile.keys()))

    return data


def get_data(channels, start="", end="", range_type="globalDate", delta_range=1, index_field=None,
             include_nanoseconds=True):

    return DataApiClient().get_data(channels,
                                    start=start, end=end, range_type=range_type, delta_range=delta_range,
                                    index_field=index_field, include_nanoseconds=include_nanoseconds)


def search(regex, backends=["sf-databuffer", "sf-archiverappliance"], base_url=default_base_url):
    """
    Search for channels
    :param regex:       Regular expression to match
    :param backends:    Data backends to search
    :param base_url:    Base URL of the data api
    :return:            List channels
    """

    cfg = {
        "regex": regex,
        "backends": backends,
        "ordering": "asc",
        "reload": "true"
    }

    response = requests.post(base_url + '/sf/channels', json=cfg)
    return response.json()


def cli():
    import argparse

    time_end = datetime.now()
    time_start = time_end - timedelta(minutes=1)

    parser = argparse.ArgumentParser(description='Command line interface for the Data API')
    parser.add_argument('action', type=str, default="",
                        help='Action to be performed. Possibilities: search, save')
    parser.add_argument("--regex", type=str, help="String to be searched", default="")
    parser.add_argument("--from_time", type=str, help="Start time for the data query", default=time_start)
    parser.add_argument("--to_time", type=str, help="End time for the data query", default=time_end)
    parser.add_argument("--from_pulse", type=str, help="Start pulseId for the data query", default=-1)
    parser.add_argument("--to_pulse", type=str, help="End pulseId for the data query", default=-1)
    parser.add_argument("--channels", type=str, help="Channels to be queried, comma-separated list", default="")
    parser.add_argument("--filename", type=str, help="Name of the output file", default="")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite the output file", default="")
    parser.add_argument("--print", help="Prints out the downloaded data. Output can be cut.", action="store_true")

    args = parser.parse_args()

    data = None
    if args.action == "search":
        if args.regex == "":
            logger.error("Please specify a regular expression with --regex\n")
            parser.print_help()
            return
        pprint.pprint(search(args.regex, backends=["sf-databuffer", "sf-archiverappliance"], base_url=default_base_url))
    elif args.action == "save":
        if args.filename == "" and not args.print:
            logger.warning("Please select either --print or --filename")
            parser.print_help()
            return
        if args.from_pulse != -1:
            if args.to_pulse == -1:
                logger.error("Please set a range limit with --to_pulse")
                return
            data = get_data(args.channels.split(","), start=args.from_pulse, end=args.to_pulse, range_type="pulseId", index_field=None)
        else:
            data = get_data(args.channels.split(","), start=args.from_time, end=args.to_time, range_type="globalDate", index_field=None)
    else:
        parser.print_help()
        return

    if data is not None:
        if args.filename != "":
            to_hdf5(data, filename=args.filename, overwrite=args.overwrite)
        elif args.print:
            print(data)
        else:
            logger.warning("Please select either --print or --filename")
            parser.print_help()


if __name__ == "__main__":
    cli()
