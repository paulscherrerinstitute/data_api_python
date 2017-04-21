from __future__ import print_function, division
from datetime import datetime, timedelta  # timezone
import pytz
import requests
import os
import pandas as pd
import json
import sys
import numpy as np

# for nicer printing
pd.set_option('display.float_format', lambda x: '%.3f' % x)

import logging
logger = logging.getLogger("DataApiClient")
logger.setLevel(logging.INFO)

#formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logging.basicConfig(level=logging.DEBUG, format='[%(levelname)s] %(message)s')

# because pd.to_numeric has not enough precision (only float 64, not enough for globalSeconds)
# does 128 makes sense? do we need nanoseconds?
conversions = {}
#conversions["globalSeconds"] = np.float128
conversions["pulseId"] = np.int64


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
    except:
        logger.error("Cannot convert date " + date_string + ", please check")
        raise RuntimeError
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


def configure(source_name="https://data-api.psi.ch/", ):
    """
    Factory method to create a DataApiClient instance.
    
    Parameters
    ----------
    source_name : string
        Name of the Data source. Can be a Data API server, or a JSON file dumped from a Data API server
    
    Returns
    -------
    dac : 
        DataApiClient instance
    """
    logger.warn("This method will be deprecated in next minor release, please use DataApiClient to create a client instance")
    return DataApiClient(source_name=source_name)


class DataApiClient(object):

    def __init__(self, source_name="https://data-api.psi.ch/", debug=False):
        self._aggregation = {}
        self.debug = debug
        self._server_aggregation = True
        self.source_name = source_name
        self.is_local = False

        if os.path.isfile(source_name):
            self.is_local = True

    @property
    def server_aggregation(self):
        """
        Enables / disables server-side aggregation (default is: enabled). If set to True enables it, to False disables (it enables client side reduction, more limited and resource intensive, just for debug / edge cases)
        """
        print(self._server_aggregation)
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
        self._cfg = {}
        self.data = None
        self.dfs = []

    def get_data(self, channels, start="", end="", range_type="globalDate", delta_range=1, index_field=None, include_nanoseconds=True):
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
        # do I want to modify cfg manually???
        df = None
        cfg = {}

        if range_type not in ["globalDate", "globalSeconds", "pulseId"]:
            logger.error("range_type must be 'globalDate', 'globalSeconds', or 'pulseId'")
            return -1

        if index_field is None:
            logger.info("indexing will be done on %s" % range_type)
            index_field = "globalDate"
            
        if index_field not in ["globalDate", "globalSeconds", "pulseId"]:
            logger.error("index_field must be 'globalDate', 'globalSeconds', or 'pulseId'")
            return -1

        if isinstance(channels, str):
            channels = [channels, ]
            
        # add aggregation cfg
        if self._aggregation != {} and self._server_aggregation:
            cfg["aggregation"] = self._aggregation
        # add option for date range and pulse_id range, with different indexes
        cfg["channels"] = channels
        cfg["range"] = {}
        cfg["fields"] = ["pulseId", "globalSeconds", "globalDate", "value", ]
        # this part is still to be improved
        metadata_fields = ["pulseId", "globalSeconds", "globalDate"]

        if self._server_aggregation:
            cfg["fields"].append("eventCount")
            metadata_fields.append("eventCount")
            
        if range_type == "pulseId":
            cfg["range"] = _set_pulseid_range(start, end, delta_range)
        elif range_type == "globalSeconds":
            cfg["range"] = _set_seconds_range(start, end, delta_range)
        else:
            cfg["range"] = _set_time_range(start, end, delta_range)

        self._cfg = cfg
        if not self.is_local:
            response = requests.post(self.source_name + '/sf/query', json=cfg)
            data = response.json()
            if isinstance(data, dict):
                logger.error(data["error"])
                try:
                    logger.error(data["errors"])
                except:
                    pass
                print(data)
                raise RuntimeError(data["message"])
        else:
            data = json.load(open(self.source_name))

        if self.debug:
            self.data = data
            self.dfs = []

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
            if self.debug:
                self.dfs.append(tdf)
            if df is not None:
                if self._server_aggregation:
                    if not (tdf.index == df.index).all():
                        print((tdf.index == df.index))
                        logger.warning("It can be that server-side reduction returned results with different indexing. You can check this enabling client-side reduction with enable_server_aggregation(False), perform the query again and compare the results")
                df = pd.merge(df, tdf, how="outer")
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
        
        if self.is_local:
            # if default values, do not filter
            if not (start == "" and end == "" and delta_range == 1):
                start_s = [x for x in cfg["range"].keys() if x.find("start") != -1][0]
                end_s = [x for x in cfg["range"].keys() if x.find("end") != -1][0]
                df = df[self._cfg["range"][start_s]:self._cfg["range"][end_s]]
            
        if (self.is_local or not self._server_aggregation) and self._aggregation != {}:
            df = self._clientside_aggregation(df)
        return df

    def search_channel(self, regex, backends=["sf-databuffer", "sf-archiverappliance"], ):
        """
        Search a channel names in the DataAPI. Regular expressions are supported
        Example:

             dac.search_channel("SIN-CVME.*STATUS")

        Parameters
        ----------
        regex : string
            Search string
        
        backends : list of strings
            Backends where to search. Possible values are "sf-databuffer", "sf-archiverappliance". Default: both.

        Returns
        -------
        response : list
            list of dictionaries
        """
        cfg = {
            "regex": regex,
            "backends": backends,
            "ordering": "asc",
            "reload": "true"
        }

        response = requests.post(self.source_name + '/sf/channels', json=cfg)
        return response.json()

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

    @staticmethod
    def to_hdf5(df, filename="data_api_output.h5", overwrite=False, compression="gzip", compression_opts=5, shuffle=True):
        """
        Dumps DataFrame from DataApi as a HDF5 file. It assumes that the index is either pulseId or globalSeconds: in case it is date, it will convert it to globalSeconds.

        Example:
        dac.to_hdf5(df, filename="test.h5", overwrite=True)
        [INFO] File test.h5 written

        
        Parameters
        ----------
        filename : string
            Name of the output file. Defaults to data_api_output.h5
        overwrite: bool
            Flag to overwrite existing files. False by default.
        compression: string
            Valid values are 'gzip', 'lzf', 'none'
        compression_opts: int
            Compression settings.  This is an integer for gzip, not used for lzf.

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
                logger.warn("Overwriting %s" % filename)
            else:
                logger.error("File %s exists, and overwrite flag is False, exiting" % filename)
                sys.exit(-1)

        # this part is very convoluted, rewrite...
        try:
            index_name = df.index.name
            if index_name == "globalDate":
                if "globalDate" not in df.columns:
                    #index_list = pd.to_datetime(df.index).to_series().apply(lambda x: x.replace(tzinfo=timezone.utc).timestamp()).tolist()
                    index_list = pd.to_datetime(df.index).to_series().apply(lambda x: x.replace(tzinfo=pytz.utc).timestamp()).tolist()
                # why?
                else:
                    index_name = None
            else:
                index_list = df.index.tolist()
        except:
            logger.error(sys.exc_info()[1])
            return -1

        outfile = h5py.File(filename, "w")
        if index_name is not None:
            outfile.create_dataset(index_name, data=index_list)

        for dataset in df.columns:
            if dataset == "globalDate":
                logger.info("Skipping globalDate (it will be recreated upon reading). Saving it as globalSeconds")
                if "globalSeconds" in df.columns:
                    outfile.create_dataset(dataset, data=df["globalSeconds"], **dset_opts)
                elif df.index.name == "globalSeconds":
                    outfile.create_dataset(dataset, data=df.index, **dset_opts)
                else:
                    logger.warn("globalSeconds not available, globalDate information will be dropped")
            else:
                outfile.create_dataset(dataset, data=df[dataset], **dset_opts)

        outfile.close()
        logger.info("File %s written" % filename)
        return

    @staticmethod
    def from_hdf5(filename, index_field="globalSeconds", recreate_date=True):
        """
        Loads DataFrame from HDF5 file. It assumes that file has been produced by DataApiClient.to_hdf5 routine.

        Example:
        import data_api
        dac = data_api.DataApiClient()
        df = dac.from_hdf5("test.h5")
        
        Parameters
        ----------
        filename : string
            Name of the HDF5 file.
        index_field: string
            Field to be used as index. Can be "globalSeconds", "globalDate" or "puldeId". Defaults to "globalSeconds"

        Returns
        -------
        df : DataFrame or None
        """
        import h5py

        try:
            infile = h5py.File(filename, "r")
        except:
            logger.error(sys.exc_info()[1])
            return None

        df = pd.DataFrame()

        for k in infile.keys():
            if k == "globalDate":
                df["globalDate"] = infile["globalDate"][:].astype(np.double)
                df["globalDate"] = df["globalDate"].apply(datetime.fromtimestamp)
                df["globalDate"] = df["globalDate"].apply(lambda t: (t).strftime("%Y-%m-%dT%H:%M:%S.%f"))
            else:
                df[k] = infile[k][:]

        try:
            df.set_index(index_field, inplace=True)
        except:
            logger.error("Cannot set index on %s, possible values are:" % index_field, list(infile.keys()))

        return df
