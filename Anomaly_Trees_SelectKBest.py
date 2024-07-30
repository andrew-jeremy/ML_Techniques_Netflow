'''
LSTM Autoencoder Architecture by Srivastava et al. 2016 
(https://arxiv.org/abs/1502.04681). Decoding is performed in 
reverse order to introduce short term dependencies between inputs 
and outputs. Additional to the encoding, the decoder gets fed the 
time-shifted original inputs.

latest netflow anomaly detection with LSTM autoencoder architectures
Andrew Kiruluta, Netography April 2023
'''
from river import anomaly as anomaly
from river import compose
from river import preprocessing
from river import feature_selection,tree
from river import feature_extraction
from river import compose, preprocessing, stats, metrics
from river.preprocessing import OneHotEncoder
from river.preprocessing import StandardScaler

import random
import pandas as pd
import numpy as np
import boto3
import argparse
from s3fs import S3FileSystem
from smart_open import open
import json
from pprint import pprint
from torch import nn, manual_seed
import argparse
import numbers 
from datetime import datetime
from packet_calculations import *

# Initialize a one-hot encoder for categorical features
one_hot_encoder = OneHotEncoder()

def get_pipeline(model):
    #mean = stats.Mean()
    #mode = stats.Mode()
    cat = (
        compose.SelectType(str)
        | preprocessing.StatImputer()
        | preprocessing.OneHotEncoder(sparse=True)
    )
    num = compose.SelectType(numbers.Number) | preprocessing.StatImputer() | preprocessing.StandardScaler(stats.Mean())
    processor = num + cat
    return processor | model


categorical_columns = ['Protocol','bogondst','bogonsrc','srcinternal','dstinternal','tcpflagsint','Source IP','Source Port','Destination IP','Destination Port']
numerical_columns = ['Length','flowbrate','packets','duration','bitsxrate','Inter-Arrival Time','Packet Size Variance','flow_duration','total_bytes','total_packets',\
                    'bytes_per_second','packets_per_second','unique_dst_ips','unique_dst_ports']


# feature importance selection
selector_num = feature_selection.SelectKBest(
    similarity=stats.PearsonCorr(),
    k=5  # The number of features to keep, defaults to 10
)

# ------------->
def netflow_to_text(netflow_record):
    # Convert NetFlow record to a space-separated string of 'key=value' pairs
    return ' '.join(f'{k}={v}' for k, v in netflow_record.items() if isinstance(v, str))


# Pipeline for categorical features (LDA)
categorical_pipeline = compose.Pipeline(
    ('to_text', compose.FuncTransformer(netflow_to_text)),
    ('bag_of_words', feature_extraction.BagOfWords(lowercase=False)),
    ('lda', preprocessing.LDA(n_components=2, number_of_documents=60, seed=42))
)

# Correct pass-through for numerical features
pass_through = compose.SelectType(float, int)

# Combine categorical and numerical feature processing
combined_features = compose.TransformerUnion(categorical_pipeline, pass_through)


# Define the base model for anomaly detection
base_model = anomaly.HalfSpaceTrees()

# Define the metric to evaluate the model
# Here, we use Accuracy, but you might want to use a different metric more suited for anomaly detection
metric = metrics.Accuracy()

class ShufflingImportance:
    def __init__(self, base_model, shuffle_times=5):
        self.base_model = base_model.clone()
        self.shuffle_times = shuffle_times
        self.importances = {}

    def update(self, x, y=None):
        if not isinstance(x, dict):
            raise ValueError(f"Expected a dictionary of features, but received: {type(x)}")

        model_clone = self.base_model.clone()
        base_score = model_clone.score_one(x)

        # Iterate over each feature in the dictionary
        for feature in x.keys():
            # Check if the feature value is iterable and not a string
            if hasattr(x[feature], '__iter__') and not isinstance(x[feature], str):
                shuffled_scores = []
                for _ in range(self.shuffle_times):
                    x_shuffled = x.copy()
                    shuffled_feature = list(x_shuffled[feature])
                    random.shuffle(shuffled_feature)
                    x_shuffled[feature] = shuffled_feature

                    shuffled_score = model_clone.score_one(x_shuffled)
                    shuffled_scores.append(shuffled_score)

                avg_shuffled_score = sum(shuffled_scores) / self.shuffle_times
                importance = abs(base_score - avg_shuffled_score)
                self.importances[feature] = importance
            else:
                # Handle non-iterable or string features differently if needed
                # For now, we can assign a default low importance
                self.importances[feature] = 0

        return self

    def score(self, feature):
        return self.importances.get(feature, 0)



# Anomaly detection model
anomaly_detector = anomaly.HalfSpaceTrees()

# Initialize the feature selector with shuffling importance
feature_selector = feature_selection.SelectKBest(
    similarity=ShufflingImportance(base_model=base_model, shuffle_times=5),
    k=10  # Select top 10 features
)

# Complete pipeline with combined features, shuffling importance, and anomaly detection
pipeline = combined_features | feature_selector | anomaly_detector

#------------->

def shuffle_feature(data, feature):
    # Extract the values of the specified feature
    feature_values = [d[feature] for d in data]
    
    # Shuffle the extracted feature values
    np.random.shuffle(feature_values)
    
    # Assign the shuffled values back to the dictionaries
    shuffled_data = [{**d, feature: feature_values[i]} for i, d in enumerate(data)]
    
    return shuffled_data

if __name__ == "__main__":
    # Testing/Inference
    parser = argparse.ArgumentParser(description='online training')
    parser.add_argument('--encoder', type=int, default=1,help="1:HalfSpaceTrees,\
        2: , 3: ")
    parser.add_argument('--pretrained', type=bool, default=False,help='load pretrained model,\
         True: for Yes')
    args = parser.parse_args()


    if args.encoder == 1:
        model = anomaly.HalfSpaceTrees(n_trees=25,window_size=100,seed=42)
    elif args.encoder == 2:   # TBD
        pass
    else:                     # TBD
        pass

    pipeline_trees = get_pipeline(model)
    
    # open s3  netflow bucket data
    s3 = boto3.resource('s3')
    s3_file = S3FileSystem()
    s3_client = boto3.client("s3")
    bucket = s3.Bucket('ml-flow-dump')
    scores = []
    data = []
    counter = 0
    warmup = 10000
    stream_data = [] 
    score_threshold = 0.98
    i = 0
    
    for obj in bucket.objects.filter(Prefix="flow"):
        source_url = 's3://ml-flow-dump/' + obj.key
        print(source_url)
        if i == 10:
                break
        for i,json_line in enumerate(open(source_url, transport_params={"client": s3_client})):
            my_json = json.loads(json_line)
            df = pd.json_normalize(my_json)  # one line at a time 
            df.fillna(0, inplace=True)
            record = my_json 
            
            # Extract basic features from the packet
            timestamp = datetime.fromtimestamp(record["timestamp"] / 1e3)   # timestamp
            
            # categorical features
            src_ip = record["srcip"]       # srcip
            dst_ip = record["dstip"]       # dstip
            src_port = record["srcport"]   # srcport
            dst_port = record["dstport"]   # dstport
            srcinternal = record["srcinternal"] # srcinternal
            dstinternal = record["dstinternal"] # dstinternal
            bogondst = record["bogondst"] # bogondst
            bogonsrc = record["bogonsrc"] # bogonsrc
            protocol = record["protocolint"]            # protocolint
            dstinternal = record["dstinternal"] # dstinternal
            tcpflagsint = record["tcpflagsint"]
            
            # numerical features
            flowbrate = record["flowbrate"] # flowbrate
            length  =  int(record["bits"]/8) 
            packets = record["packets"] # packets
            duration = record["duration"] # duration
            bitsxrate = record["bitsxrate"] # bitsxrate
        
            state = beacon_states[src_ip]

            # Update the state with the current packet
            updated_state = update_state_with_packet_json(state, record, timestamp)

            # Calculate the additional features based on the updated state
            additional_features = calculate_additional_features(updated_state)

            # Calculate inter-arrival time and packet size variance
            inter_arrival_time = calculate_inter_arrival_time(state, timestamp)
            packet_size_variance = calculate_packet_size_variance(state, length)
            
            # Prepare the packet features for the model
            packet_features = {
                'Timestamp': timestamp,
                'Protocol': protocol,
                'Length': length,
                'flowbrate': flowbrate,
                'packets': packets,
                'duration': duration,
                'bitsxrate': bitsxrate,
                'bogondst': bogondst,
                'bogonsrc': bogonsrc,
                'srcinternal': srcinternal,
                'dstinternal': dstinternal,
                'tcpflagsint': tcpflagsint,
                'Source IP': src_ip,
                'Source Port': src_port,
                'Destination IP': dst_ip,
                'Destination Port': dst_port,
                'Inter-Arrival Time': inter_arrival_time,
                'Packet Size Variance': packet_size_variance,
                **additional_features,
                # Add other features that might be relevant
            }
            #packet_data.append(packet_features)
            df_features = pd.DataFrame([packet_features])
            
            df_features.drop(['Timestamp'], axis=1, inplace=True)
            df_features = df_features.fillna(0)
            # Get the anomaly score from the model
            x = df_features.iloc[0].to_dict()
            df_features[categorical_columns] = df_features[categorical_columns].astype(str)
            df_features[numerical_columns] = df_features[numerical_columns].astype(float)

            # Get the anomaly score from the model
            categorical_features = df_features[categorical_columns].to_dict(orient='records')[0]
            numeric_features = df_features[numerical_columns].to_dict(orient='records')[0]
            
            # Convert categorical values to strings
            for key in categorical_columns:
                categorical_features[key] = str(categorical_features[key])
            
            # Combine the encoded categorical and numeric features
            x_encoded = {**categorical_features, **numeric_features}
            
            # Get the anomaly score from the model
            #anomaly_score = pipeline_trees.score_one(x_encoded)
            #anomaly_score = pipeline.score_one(x_encoded)
            #print(f"Anomaly score: {anomaly_score}")
      
            #pipeline_trees.learn_one(x=x_encoded)
            pipeline = pipeline.learn_one(x_encoded,{})  # LDA does not use labels
            counter +=1
            
            # append to stream_data and check if the list length exceeded N  and check for 
            # feature importance with isolation forest.
            #stream_data = stream_data.append(pd.json_normalize(x_encoded))
            
            is_anomaly = anomaly_score > score_threshold  # Label for classifier (1 for anomaly, 0 for normal)
            
           # Periodically update feature importance
            if counter > warmup  and anomaly_score > score_threshold:
                
                # Sort and display the feature importance
                print(f"Anomaly detected from {src_ip} to {dst_ip} at {timestamp} with feature importance:\n")
                # Print the top K features
                for feature in feature_selection_step.top_k:
                    print(f"{feature}: {feature_selection_step.similarity.score(feature)}")
                print("\n")
                