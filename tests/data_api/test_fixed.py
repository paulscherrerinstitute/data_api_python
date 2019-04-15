import data_api as api
import datetime
import pandas as pd

start='2018-12-10 00:00:00.000'
end  ='2018-12-18 00:00:00.000'
channels = ['ABK1:IST:2','MHC1:IST:2']

# standard method
data = api.get_data(channels=channels, start=start, end=end, base_url='https://data-api.psi.ch/hipa')

# fixed time 1 day interval, with padding
data10 = api.get_data(channels=channels, start=start, end=end, base_url='https://data-api.psi.ch/hipa', fixed_time=True, fixed_time_interval='1 D')

# same with backpadding
data10back = api.get_data(channels=channels, start=start, end=end, base_url='https://data-api.psi.ch/hipa', fixed_time=True, fixed_time_interval='1 D', interpolation_method='previous')

# same with linear interpolation
data10lin = api.get_data(channels=channels, start=start, end=end, base_url='https://data-api.psi.ch/hipa', fixed_time=True, fixed_time_interval='1 D', interpolation_method='linear')

# same with nearest neighbour
data10nn = api.get_data(channels=channels, start=start, end=end, base_url='https://data-api.psi.ch/hipa', fixed_time=True, fixed_time_interval='1 D', interpolation_method='nearest')

