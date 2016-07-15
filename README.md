## Examples

Create a client instance:

```python
In [1]: import data_api_client
Out [1]: dbc = data_api_client.configure()
```

Search channels:

```python
In [5]: dbc.search_channel("SINSB02-RIQM-DCP10:FOR-PHASE")
Out[5]: 
[{'backend': 'sf-databuffer',
  'channels': ['SINSB02-RIQM-DCP10:FOR-PHASE',
   'SINSB02-RIQM-DCP10:FOR-PHASE-AVG']},
 {'backend': 'sf-archiverappliance',
  'channels': ['SINSB02-RIQM-DCP10:FOR-PHASE-AVG-P2P',
   'SINSB02-RIQM-DCP10:FOR-PHASE-JIT-P2P',
   'SINSB02-RIQM-DCP10:FOR-PHASE-STDEV']}]
```

Get data:

```python
df = dbc.get_data(channels=['SINSB02-RIQM-DCP10:FOR-PHASE-AVG', 'SINSB02-RKLY-DCP10:FOR-PHASE-AVG', 'SINSB02-RIQM-DCP10:FOR-PHASE'], start_date="2016-07-14 08:05", delta_time=1)

In [9]: df.head()
Out[9]: 
                       pulse_id  SINSB02-RIQM-DCP10:FOR-PHASE-AVG  \
1468476300.007550977  137550977                        -177.14816   
1468476300.017550978  137550978                        -177.14882   
1468476300.027550979  137550979                        -177.14958   
1468476300.037550980  137550980                        -177.15004   
1468476300.047550981  137550981                        -177.14587   

                      SINSB02-RKLY-DCP10:FOR-PHASE-AVG  \
1468476300.007550977                        -174.75343   
1468476300.017550978                        -174.75346   
1468476300.027550979                        -174.75249   
1468476300.037550980                        -174.75266   
1468476300.047550981                        -174.74506   

                     SINSB02-RIQM-DCP10:FOR-PHASE  
1468476300.007550977                          NaN  
1468476300.017550978                          NaN  
1468476300.027550979                          NaN  
1468476300.037550980                          NaN  
1468476300.047550981                          NaN 

In [10]: df.loc["1468476300.047550981"]
Out[10]: 
pulse_id                            137550981
SINSB02-RIQM-DCP10:FOR-PHASE-AVG     -177.146
SINSB02-RKLY-DCP10:FOR-PHASE-AVG     -174.745
SINSB02-RIQM-DCP10:FOR-PHASE              NaN
Name: 1468476300.047550981, dtype: object

```

