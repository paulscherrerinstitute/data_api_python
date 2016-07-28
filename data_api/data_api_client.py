from datetime import datetime, timedelta
import requests
import os
import pandas as pd
import json
import sys
import numpy as np

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

#formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logging.basicConfig(level=logging.DEBUG, format='[%(levelname)s] %(message)s')

ENABLE_SERVER_REDUCTION = False


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
        st = datetime.strptime(date_string, "%Y-%m-%d %H:%M")
    except:
        try:
            st = datetime.strptime(date_string, "%Y-%m-%d %H:%M:%S")
        except:
            # improve herror handling
            logger.error("Cannot convert date " + date_string + ", please check")
            raise RuntimeError
    return st, datetime.isoformat(st)


def _set_pulseid_range(start_pulseid, end_pulseid, delta):
    d = {}
    if start_pulseid == "" and end_pulseid == "":
        raise RuntimeError("Must select at least start_pulseid or end_pulseid")
    if start_pulseid != "":
        d["startPulseId"] = start_pulseid
        if end_pulseid != "":
            d["endPulseId"] = end_pulseid
        else:
            d["endPulseId"] = start_pulseid + delta - 1
    else:
        d["endPulseId"] = end_pulseid
        d["startPulseId"] = end_pulseid - delta + 1

    return d


def _set_time_range(start_date, end_date, delta_time):
    d = {}
    if start_date == "" and end_date == "":
        d["startDate"] = datetime.isoformat(datetime.now() - timedelta(seconds=delta_time))
        d["endDate"] = datetime.isoformat(datetime.now())
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


def configure(source_name="http://data-api.psi.ch/", ):
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
    return DataApiClient(source_name=source_name)


class DataApiClient(object):

    source_name = None
    is_local = False
    _aggregation = {}
    
    def __init__(self, source_name="http://data-api.psi.ch/"):
        self.enable_server_reduction = ENABLE_SERVER_REDUCTION
        self.source_name = source_name
        if os.path.isfile(source_name):
            self.is_local = True

    def __enable_server_reduction__(self, value=True):
        self.enable_server_reduction = value
            
    def set_aggregation(self, aggregation_type="value", aggregations=["min", "mean", "max"], extrema=[],
                        nr_of_bins=None, duration_per_bin=None, pulses_per_bin=None):
        if not self.enable_server_reduction:
            logger.warning("Server-wise aggregation still not fully supported. Until this is fixed, reduction will be done on the client after data is got.")
            logger.warning("Aggregation on waveforms is not supported at all client-side, sorry. Please fill a request ticket in case you would need it.")
        
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
            #self._aggregation["durationPerBin"] = duration_per_bin
            #for k in ["nrOfBins", "pulsesPerBin"]:
            #    if k in self._aggregation:
            #        self._aggregation.pop(k)
        elif pulses_per_bin is not None:
            self._aggregation["pulsesPerBin"] = pulses_per_bin
            for k in ["durationPerBin", "nrOfBins"]:
                if k in self._aggregation:
                    self._aggregation.pop(k)
            
    def get_aggregation(self, ):
        return self._aggregation

    def clear(self, ):
        """
        Resets all stored configurations, excluding source_name
        """ 

        self._aggregation = {}

    def get_data(self, channels, start="", end="", range_type="date", delta_range=1, index_field="globalSeconds", dump_other_index=True):
        # do I want to modify cfg manually???
        df = None
        cfg = {}

        if range_type not in ["date", "globalSeconds", "pulseId"]:
            logger.error("range_type must be 'date', 'globalSeconds', or 'pulseId'")
            return -1
        
        # add aggregation cfg
        if self._aggregation != {} and self.enable_server_reduction:
            cfg["aggregation"] = self._aggregation
        # add option for date range and pulse_id range, with different indexes
        cfg["channels"] = channels
        cfg["range"] = {}
                
        if range_type == "pulseId":
            cfg["range"] = _set_pulseid_range(start, end, delta_range)
            print(start, end, delta_range)
        elif range_type == "globalSeconds":
            cfg["range"] = {"startSeconds": str(start), "endSeconds": str(end)}
        else:
            cfg["range"] = _set_time_range(start, end, delta_range)

        self._cfg = cfg
        if not self.is_local:
            response = requests.post(self.source_name + '/sf/query', json=cfg)
            data = response.json()
            self.data = data
            if isinstance(data, dict):
                print("[ERROR]", data["error"])
                try:
                    print("[ERROR]", data["errors"])
                except:
                    pass
                raise RuntimeError(data["message"])
        else:
            data = json.load(open(self.source_name))

        not_index_field = "globalSeconds"
        number_conversion = int
        if index_field == "globalSeconds":
            not_index_field = "pulseId"
            number_conversion = float

        first_data = True
        for d in data:
            if d['data'] == []:
                logger.warning("no data returned for channel %s" % d['channel']['name'])
                continue
            if dump_other_index and first_data:
                if isinstance(d['data'][0]['value'], dict):
                    entry = []
                    keys = sorted(d['data'][0]['value'])
                    for x in d['data']:
                        # workaround
                        entry.append([x[index_field], ] + [x['value'][k] for k in keys])
                    #entry = [[x[index_field], x['value']] for x in d['data']]
                    columns = [index_field, ] + [d['channel']['name'] + ":" + k for k in keys]
                    
                else:
                    entry = [[x[index_field], x['value']] for x in d['data']]
                    columns = [index_field, d['channel']['name']]
            else:
                if isinstance(d['data'][0]['value'], dict):
                    entry = []
                    keys = sorted(d['data'][0]['value'])
                    for x in d['data']:
                        entry.append([x[index_field], x[not_index_field]] + [x['value'][k] for k in keys])
                    #entry = [[x[index_field], x['value']] for x in d['data']]
                    columns = [index_field, not_index_field] + [d['channel']['name'] + ":" + k for k in keys]
                    
                else:
                    entry = [[x[index_field], x[not_index_field], x['value']] for x in d['data']]
                    columns = [index_field, not_index_field, d['channel']['name']]

            if df is not None:
                df2 = pd.DataFrame(entry, columns=columns)
                df2[index_field] = df2[index_field].apply(number_conversion)
                df2.set_index(index_field, inplace=True)
                df = pd.concat([df, df2], axis=1)
            else:
                df = pd.DataFrame(entry, columns=columns)
                print(df.columns)
                df[index_field] = df[index_field].apply(number_conversion)
                df.set_index(index_field, inplace=True)

        if self.is_local:
            # do the pulse_id, time filtering
            logger.info("Here I am")
            
        if (self.is_local or not ENABLE_SERVER_REDUCTION) and self._aggregation != {}:
            self.df = df
            if df is None:
                return df

            logger.info("Client-side aggregation")

            if self._aggregation["aggregationType"] != "value":
                logger.error("Only value based aggregation is supported, doing nothing")
                return df
            df_aggr = None

            #durationPerBin, pulsesPerBin
            if "pulsesPerBin" in self._aggregation:
                bin_mask = np.array([i // self._aggregation["pulsesPerBin"] for i in range(df.count().values[0])])
                bins = df.index[[0, ] + (1 + np.where((bin_mask[1:] - bin_mask[0:-1]) == 1)[0]).tolist()]
                groups = df.groupby(bin_mask)
            elif "nrOfBins" in self._aggregation:
                bins = np.linspace(df.index[0], 1 + df.index[-1], self._aggregation["nrOfBins"], endpoint=False).astype(int)
                groups = df.groupby(np.digitize(df.index, bins))
            
            for aggr in self._aggregation["aggregations"]:
                print(aggr)
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
            return df_aggr
        
        return df

    def search_channel(self, regex, backends=["sf-databuffer", "sf-archiverappliance"]):
        cfg = {
            "regex": regex,
            "backends": backends,
            "ordering": "asc",
            "reload": "true"
        }

        response = requests.post(self.source_name + '/sf/channels', json=cfg)
        return response.json()
