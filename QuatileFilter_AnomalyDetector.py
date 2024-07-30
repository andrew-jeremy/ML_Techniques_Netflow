'''
anomaly_detector

An anomaly detector.

q

Type → float

The quantile level above which to classify an anomaly score as anomalous.

protect_anomaly_detector

Default → True

Indicates whether or not the anomaly detector should be updated when the 
anomaly score is anomalous. If the data contains sporadic anomalies, then 
the anomaly detector should likely not be updated. Indeed, if it learns 
the anomaly score, then it will slowly start to consider anomalous anomaly 
scores as normal. This might be desirable, for instance in the case of drift.

Andrew Kiruluta, Netography, Inc. 2023
'''

from river import anomaly
from river import compose
from river import datasets
from river import metrics
from river import preprocessing
import pandas as pd

# Assuming `df` is your DataFrame containing the extracted packet features
df = pd.read_csv('data/beaconing/2021-05-13-Hancitor-traffic-with-Ficker-Stealer-and-Cobalt-Strike.csv')

# Feature Engineering: Calculate the time interval between successive packets to the same destination
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df = df.sort_values(by='Timestamp')
df['Time Interval'] = df.groupby(['Source IP', 'Destination IP'])['Timestamp'].diff().dt.total_seconds().fillna(0)

# Convert IP addresses and port numbers to categorical types and numerical codes
df['Source IP Code'] = df['Source IP'].astype(str).astype('category').cat.codes
df['Destination IP Code'] = df['Destination IP'].astype(str).astype('category').cat.codes
df['Source Port Code'] = df['Source Port'].astype(str).astype('category').cat.codes
df['Destination Port Code'] = df['Destination Port'].astype(str).astype('category').cat.codes

# Features for the model
#X = df[['Source IP Code', 'Destination IP Code', 'Source Port Code', 'Destination Port Code', 'Length', 'TTL', 'Time Interval']].fillna(0)
X = df[['Source Port Code', 'Destination Port Code', 'Length', 'TTL', 'Time Interval']].fillna(0)

model = compose.Pipeline(
    preprocessing.MinMaxScaler(),
    anomaly.QuantileFilter(
        anomaly.HalfSpaceTrees(n_trees=5,height=3,window_size=3,seed=42),
        q=0.995,
        protect_anomaly_detector=True))

timestamps = []
scores = []
for i in range(df.shape[0]):
    x = X.iloc[i].to_dict()
    score = model.score_one(x)
    is_anomaly = model['QuantileFilter'].classify(score)
    model = model.learn_one(x)
    if is_anomaly:
        print(is_anomaly)
        scores.append(score)
        timestamps.append(list(df.iloc[i]))
        print(score)
        print(len(scores))
    

