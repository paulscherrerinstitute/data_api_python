# The Python Client for the Data API

This simple client retrieves data from the Data API (see http://ui-data-api.psi.ch/latest/) and loads it into a nice Pandas DataFrame (see: http://pandas.pydata.org/).
What a Pandas DataFrame is? Think about it as a nice and big table, where all your values are indexed using either the `global timestamp` or the `pulse_id`. This allows you to execute statistical operations, correlations, filtering etc in a very easy and efficient way.

More about Pandas:
* http://pandas.pydata.org/index.html
* http://pandas.pydata.org/pandas-docs/stable/10min.html

## Installation

You can install through our Anaconda repository:

```
conda install -c paulscherrerinstitute data_api
```

otherwise, you can download the source code, and install it locally, either through `conda`:

```
git clone https://git.psi.ch/sf_daq/data-api_python.git
cd data-api_python
conda build conda-recipe
conda install <package_location>
```

or through `setup.py`
```
git clone https://git.psi.ch/sf_daq/data-api_python.git
cd data-api_python
python setup.py install
```


## Examples

If you want to run our Jupyter Notebook examples, please clone this repository locally, then:
```
cd examples
ipython notebook
```


Create a client instance:

```python
In [1]: import data_api
Out [1]: dac = data_api.configure()
```

Search channels:

```python
In [5]: dac.search_channel("SINSB02-RIQM-DCP10:FOR-PHASE")
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
df = dac.get_data(channels=['SINSB02-RIQM-DCP10:FOR-PHASE-AVG', 'SINSB02-RKLY-DCP10:FOR-PHASE-AVG', 'SINSB02-RIQM-DCP10:FOR-PHASE'], start="2016-07-14 08:05", range_type="date", delta_range=1)

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

Plot data:
```python
import matplotlib.pyplot as plt
df.plot.scatter("SINSB02-RIQM-DCP10:FOR-PHASE-AVG", "SINSB02-RKLY-DCP10:FOR-PHASE-AVG")
plt.show()
```

![alt text](docs/scatter_plot.png)

```python
import matplotlib.pyplot as plt
df[['SINSB02-RIQM-DCP10:FOR-PHASE-AVG', ]].plot.box()
plt.show()
```

![alt text](docs/box_plot.png)


Plot waveforms:
```python

# find where you do have data:
In [10]: df[df['SINSB02-RIQM-DCP10:FOR-PHASE'].notnull()]
Out[10]: 
                       pulse_id  SINSB02-RIQM-DCP10:FOR-PHASE-AVG  \
1468476300.237551000  137551000                        -177.14268   

                      SINSB02-RKLY-DCP10:FOR-PHASE-AVG  \
1468476300.237551000                        -174.74382   

                                           SINSB02-RIQM-DCP10:FOR-PHASE  
1468476300.237551000  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ...  

# plot it
plt.plot(df['SINSB02-RIQM-DCP10:FOR-PHASE']['1468476300.237551000'])
plt.show()
```

![alt text](docs/waveform_plot.png)

Save data

```python
# to CSV
df.to_csv("test.csv")

# to HDF5
df.to_hdf("test.h5", "/dataset")

# etc...
```
