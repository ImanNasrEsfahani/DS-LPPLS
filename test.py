from joblib import dump, load
from pprint import pprint
import pandas as pd
import numpy as np

filename = r".\models\BTC-USD_2020-01-01_2024-11-10\BTC_USD_2020-01-01_2024-11-10_filtered.sav"
data = load(filename)
data = pd.DataFrame(data)

pprint(data)
pprint(len(data))
pprint(data.info())
pprint(data.describe())
pprint(data.columns)
# print(data["confident_positive_bubble"].describe(), " info")
# print(data[["bet", "ome", "phi", "A", "B", "C", "tc", "t2", 'confident_positive_bubble', 'confident_negative_bubble', 'trust_positive_bubble', 'trust_negative_bubble']].describe(), " info")
pprint(data.iloc[-1])
print(data['t2_date'])
# data['t1_date'] = pd.to_datetime(data['t1_date'])
data['t2_date'] = pd.to_datetime(data['t2_date'])
#
# # Remove timezone information
# data['t1_date'] = data['t1_date'].dt.tz_localize(None)
data['t2_date'] = data['t2_date'].dt.tz_localize(None)
# pprint(data.tail())
# pprint(data.iloc[-1])
#
data.to_excel('output-filter.xlsx', index=False)
# data.to_excel('output-filtered.xlsx', index=False)

# # dump(data, filename)