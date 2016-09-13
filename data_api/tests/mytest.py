import unittest
import h5py
import os

from data_api import DataApiClient

df_date = None
df_secs = None
chname = ["SINDI01-RIQM-DCP10:FOR-PHASE-AVG", "S10CB01-RBOC-DCP10:FOR-PHASE-AVG"]


def prepare_data(index_field, delta_i=100, chname=chname, ):
    dac = DataApiClient(debug=True)
    delta = delta_i
    df = None
    while df is None:
        df = dac.get_data(chname, delta_range=delta, index_field=index_field)
        delta += 1000
    return df, dac


def check_dataframes(dac, df, df2, _cfg=None):
    test = False
    try:
        if (df2.iloc[:-1].dropna() == df.dropna()).all().all():
            test = True
    except:
        try:
            if (df2.iloc.dropna() == df.dropna()).all().all():
                test = True
        except:
            print("\nFailing test, dumping info")
            print(df.info(), df2.info())
            print(_cfg)            
            print(dac._cfg)
            print("\n")
    return test


def test():
    dac = DataApiClient(debug=True)
    df_date, dac0 = prepare_data("globalDate")

    df2 = dac.get_data(chname, start=df_date.globalSeconds.iloc[0], end=df_date.globalSeconds.iloc[-1], range_type="globalSeconds")
    print(check_dataframes(dac, df_date, df2, dac0._cfg))
    return dac0, dac, df_date, df2

