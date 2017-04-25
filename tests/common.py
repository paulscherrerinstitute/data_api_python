from data_api import DataApiClient

import logging
logger = logging.getLogger("DataApiClient")
logger.setLevel(logging.WARNING)
logging.getLogger("requests").setLevel(logging.ERROR)

chname = ["SINSB02-RIQM-DCP10:FOR-PHASE-MAX", "SINSB02-RIQM-DCP10:FOR-PHASE-AVG"]


def prepare_data(index_field, delta_i=100, chname=None, ):
    dac = DataApiClient()
    delta = delta_i
    df = None
    while df is None:
        df = dac.get_data(chname, delta_range=delta, index_field=index_field)
        delta += 1000
    return df, dac


def check_dataframes(dac, df, df2, _cfg=None):
    test = False
    try:
        if (df.dropna().iloc[:-1] == df2.dropna()).all().all():
            test = True
    except:
        #print(sys.exc_info())
        try:
            if (df2.dropna() == df.dropna()).all().all():
                test = True
        except:
            #print(sys.exc_info())
            print("\nFailing test, dumping info")
            print((df == df2).all())
            print(df.info(), df2.info())
            print(_cfg)
            print(dac._cfg)
            print(df.head(5))
            print(df2.head(5))
            print(df.tail(5))
            print(df2.tail(5))
            print("\n")
    return test
