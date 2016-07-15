from datetime import datetime, timedelta
import requests
import os
import pandas as pd
import json


"""
time_base = datetime.datetime.now()
time_start = time_base - datetime.timedelta(minutes=60)
time_end = time_base
start = datetime.datetime.isoformat(time_start)
end = datetime.datetime.isoformat(time_end)

#channel_names = ['SINSB01-RIQM-DCP10:FOR-PHASE']
channel_names = ['SINSB02-RIQM-DCP10:FOR-PHASE-AVG']
data={"channels": channel_names, "range":{"startDate": start,  "endDate":end}}

response = requests.post('http://data-api.psi.ch/sf/query', json=data)
print(response.json())
"""


def convert_date(date_string):
    try:
        st = datetime.strptime(date_string, "%Y-%m-%d %H:%M")
    except:
        try:
            st = datetime.strptime(date_string, "%Y-%m-%d %H:%M:%S")
        except:
            print("[ERROR]")
    return st, datetime.isoformat(st)


def configure(source_name="http://data-api.psi.ch/", ):
    return DataBufferClient(source_name=source_name)


class DataBufferClient(object):

    source_name = None
    cfg = {
        'channels': [],
        "range": {"startDate": "", "endDate": ""},
    }

    is_local = False
    
    def __init__(self, source_name="http://data-api.psi.ch/"):
        self.source_name = source_name
        if os.path.isfile(source_name):
            self.is_local = True

    #def get_configuration(self, ):
    #    pass

    #def configure(self, channels=[], start_date="", end_date="", delta_time=1, index="time"):
    #    pass

    def get_data(self, channels=[], start_date="", end_date="", delta_time=1, index="time"):
        df = None
        # add option for date range and pulse_id range, with different indexes
        if channels != []:
            self.cfg["channels"] = channels
        if start_date == "" and self.cfg["range"]["startDate"] == "":
            self.cfg["range"]["startDate"] = datetime.isoformat(datetime.now() - timedelta(seconds=delta_time))
        elif start_date != "":
            st_dt, st_iso = convert_date(start_date)
            self.cfg["range"]["startDate"] = st_iso

        if end_date == "" and self.cfg["range"]["endDate"] == "":
            if self.cfg["range"]["startDate"] == "":
                self.cfg["range"]["endDate"] = datetime.isoformat(datetime.now())
            else:
                self.cfg["range"]["endDate"] = datetime.isoformat(st_dt + timedelta(seconds=delta_time))
        elif end_date != "":
            st_dt, st_iso = convert_date(end_date)
            self.cfg["range"]["endDate"] = st_iso
        
        if not self.is_local:
            response = requests.post(self.source_name + '/sf/query', json=self.cfg)
            data = response.json()
            if isinstance(data, dict):
                raise RuntimeError(data["message"])

        else:
            data = json.load(open(self.source_name))

        self.data = data
        for d in data:
            if df is not None:
                entry = [x['value'] for x in d['data']]
                pid = [x['pulseId'] for x in d['data']]
                g_secs = [x['globalSeconds'] for x in d['data']]
                #df[d['channel']['name']] = pd.Series(entry, pid)
                if index == "time":
                    df2 = pd.DataFrame({d['channel']['name']: entry, 'global_seconds': g_secs})
                    df2.set_index('global_seconds', inplace=True)
                    df = pd.concat([df, df2], axis=1)
                    
                else:
                    df = pd.concat([df, pd.DataFrame([g_secs, entry]).set_index('pid', inplace=True)])
            else:
                if index == "time":
                    # not sure which option is the fastest, whether to set index later, or from the creation (impliyng a new loop)
                    entry = [[x['pulseId'], x['globalSeconds'], x['value']] for x in d['data']]
                    #print(entry[0])
                    df = pd.DataFrame(entry, columns=["pulse_id", "global_seconds", d['channel']['name']])
                    df.set_index("global_seconds", inplace=True)
                    
                else:
                    entry = [[x['globalSeconds'], x['value']] for x in d['data']]
                    pid = [x['pulseId'] for x in d['data']]
                    df = pd.DataFrame(entry, columns=["global_seconds", d['channel']['name']], index=pid)

        if self.is_local:
            # do the pulse_id, time filtering
            pass

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
