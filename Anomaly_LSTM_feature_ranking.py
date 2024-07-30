'''
latest netflow anomaly detection with LSTM autoencoder architectures
Andrew Kiruluta, Netography April 2023

'''
from deep_river import anomaly
from river import metrics
from river import compose
from river import preprocessing
from river import feature_extraction
from river import feature_selection
from torch import nn
import pandas as pd
from datetime import datetime
import ipaddress
import pickle
import boto3
import argparse
import statistics
from packet_calculations import *
from s3fs import S3FileSystem
from smart_open import open
import json
from pprint import pprint
import uuid
from river import compose, preprocessing, metrics, stats
from river.compose import Pipeline  
from torch import nn, manual_seed
import torch
from tqdm import tqdm
import argparse
import numbers 
from feat_preprocessor import feat_eng, feat_select

'''
LSTM Autoencoder Architecture by Srivastava et al. 2016 
(https://arxiv.org/abs/1502.04681). Decoding is performed in 
reverse order to introduce short term dependencies between inputs 
and outputs. Additional to the encoding, the decoder gets fed the 
time-shifted original inputs.
'''
import random

def Generate(): #function generates a random 6 digit number
    code = ''
    for i in range(6):
        code += str(random.randint(0,9))
    return code

class LSTMAutoencoderSrivastava(nn.Module):
    def __init__(self, n_features, hidden_size=90, n_layers=10, batch_first=True):
        super().__init__()
        self.n_features = n_features  # does not matter, picked up automatically from x_encoded.size()
        print(f'Number of self.n_features: {self.n_features}')
        print(f'n_features: {n_features}')
        print(f'Hidden size: {hidden_size}')
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.batch_first = batch_first
        self.time_axis = 1 if batch_first else 0
        self.encoder = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=batch_first,
        )
        self.decoder = nn.LSTM(
            input_size=hidden_size,
            hidden_size=n_features,
            num_layers=n_layers,
            batch_first=batch_first,
        )

    def forward(self, x):
        _, (h, _) = self.encoder(x)
        h = h[-1].view(1, 1, -1)
        x_flipped = torch.flip(x[0:], dims=[self.time_axis])
        input = torch.cat((h.squeeze(0), x_flipped), dim=1)
        x_hat, _ = self.decoder(h)
        x_hat = x_hat.squeeze(1)
        x_hat = torch.flip(x_hat, dims=[self.time_axis])

        return x_hat
    
'''
Architecture inspired by Cho et al. 2014 (https://arxiv.org/abs/1406.1078).
 Decoding occurs in natural order and the decoder is only provided with the 
 encoding at every timestep.
'''
class LSTMAutoencoderCho(nn.Module):
    def __init__(self, n_features, hidden_size=90, n_layers=1, batch_first=False):
        super().__init__()
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.batch_first = batch_first
        print(f'Number of self.n_features: {self.n_features}')
        print(f'n_features: {n_features}')
        self.encoder = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=batch_first,
        )
        self.decoder = nn.LSTM(
            input_size=hidden_size,
            hidden_size=n_features,
            num_layers=n_layers,
            batch_first=batch_first,
        )

    def forward(self, x):
        _, (h, _) = self.encoder(x)
        target_shape = (
            (-1, x.shape[0], -1) if self.batch_first else (x.shape[0], -1, -1)
        )
        #h = h[-1].expand(target_shape)
        #h = h[-1]
        x_hat, _ = self.decoder(h)
        return x_hat

'''
LSTM Encoder-Decoder architecture by Sutskever et al. 2014 (https://arxiv.org/abs/1409.3215). 
The decoder only gets access to its own prediction of the previous timestep. Decoding also 
takes performed backwards.
'''
class LSTMDecoder(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        sequence_length=None,
        predict_backward=True,
        num_layers=1,
    ):
        super().__init__()

        self.cell = nn.LSTMCell(input_size, hidden_size)
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.predict_backward = predict_backward
        self.sequence_length = sequence_length
        self.num_layers = num_layers
        self.lstm = (
            None
            if num_layers <= 1
            else nn.LSTM(
                input_size=hidden_size,
                hidden_size=hidden_size,
                num_layers=num_layers - 1,
            )
        )
        self.linear = (
            None if input_size == hidden_size else nn.Linear(hidden_size, input_size)
        )

    def forward(self, h, sequence_length=None):
        """Computes the forward pass.

        Parameters
        ----------
        x:
            Input of shape (batch_size, input_size)

        Returns
        -------
        Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
            Decoder outputs (output, (h, c)) where output has the shape (sequence_length, batch_size, input_size).
        """

        if sequence_length is None:
            sequence_length = self.sequence_length
        x_hat = torch.empty(sequence_length, h.shape[0], self.hidden_size)
        for t in range(sequence_length):
            if t == 0:
                h, c = self.cell(h)
            else:
                input = h if self.linear is None else self.linear(h)
                h, c = self.cell(input, (h, c))
            t_predicted = -t if self.predict_backward else t
            x_hat[t_predicted] = h

        if self.lstm is not None:
            x_hat = self.lstm(x_hat)

        return x_hat, (h, c)


class LSTMAutoencoderSutskever(nn.Module):
    def __init__(self, n_features, hidden_size=90, n_layers=1):
        super().__init__()
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.encoder = nn.LSTM(
            input_size=n_features, hidden_size=hidden_size, num_layers=n_layers
        )
        self.decoder = LSTMDecoder(
            input_size=hidden_size, hidden_size=n_features, predict_backward=True
        )

    def forward(self, x):
        _, (h, _) = self.encoder(x)
        x_hat, _ = self.decoder(h[-1], x.shape[0])
        return x_hat

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

# feature importance selection
selector_num = feature_selection.SelectKBest(
    similarity=stats.PearsonCorr(),
    k=3  # The number of features to keep, defaults to 10
)

selector_cat = feature_selection.VarianceThreshold(10)

categorical_columns = ['Protocol','bogondst','bogonsrc','srcinternal','dstinternal','tcpflagsint','Source IP','Source Port','Destination IP','Destination Port']
numerical_columns = ['Length','flowbrate','packets','duration','bitsxrate','Inter-Arrival Time','Packet Size Variance','flow_duration','total_bytes','total_packets',\
                    'bytes_per_second','packets_per_second','unique_dst_ips','unique_dst_ports']


def shuffle_feature(data, feature):
    # Extract the values of the specified feature
    feature_values = [d[feature] for d in data]
    
    # Shuffle the extracted feature values
    np.random.shuffle(feature_values)
    
    # Assign the shuffled values back to the dictionaries
    shuffled_data = [{**d, feature: feature_values[i]} for i, d in enumerate(data)]
    
    return shuffled_data

def normalize_score(score, min_score, max_score):
    if (max_score - min_score) != 0:
        normalized_score = (score - min_score) / (max_score - min_score)
    else:
        normalized_score = 0
    return min(np.max([normalized_score, 0]), 1)  # Ensure the score is within [0, 1]


if __name__ == "__main__":
    # Testing/Inference
    parser = argparse.ArgumentParser(description='online training')
    parser.add_argument('--encoder', type=int, default=2,help="1:LSTMAutoencoderSrivastava,\
        2: LSTMAutoencoderCho, 3: LSTMAutoencoderSutskever")
    parser.add_argument('--pretrained', type=bool, default=False,help='load pretrained model,\
         True: for Yes')
    args = parser.parse_args()

    '''
    1-LSTM Autoencoder Architecture by Srivastava et al. 2016 
    (https://arxiv.org/abs/1502.04681). Decoding is performed in 
    reverse order to introduce short term dependencies between 
    inputs and outputs. Additional to the encoding, the decoder 
    gets fed the time-shifted original inputs.

    2-Architecture inspired by Cho et al. 2014 (https://arxiv.org/abs/1406.1078).
     Decoding occurs in natural order and the decoder is only provided with 
     the encoding at every timestep.
    
    3-LSTM Encoder-Decoder architecture by Sutskever et al. 2014 
    (https://arxiv.org/abs/1409.3215). The decoder only gets access to its own 
    prediction of the previous timestep. Decoding also takes performed backwards.
    '''

    if args.encoder == 1:
        #model = anomaly.Autoencoder(module= LSTMAutoencoderSrivastava, lr=0.005)
        model = anomaly.RollingAutoencoder(module= LSTMAutoencoderSrivastava, lr=0.005)
        tag = 'Sriva'
    elif args.encoder == 2:
        #model = anomaly.Autoencoder(module= LSTMAutoencoderCho, lr=0.005)
        model = anomaly.RollingAutoencoder(module= LSTMAutoencoderCho, lr=0.005)
        tag = 'Cho'
    else:
        model = anomaly.Autoencoder(module= LSTMAutoencoderSutskever, lr=0.005)
        tag ='Sut'

    pipeline = get_pipeline(model)
    lda = compose.Pipeline(feature_extraction.BagOfWords(),preprocessing.LDA(n_components=20,number_of_documents=60000,seed=42))

    with open('data/anomalies_{}.txt'.format(tag), 'w') as document: pass  # initially create empty file
    fm = open('data/model_{}_2.pkl'.format(tag), 'w+b')
    fa =  open('data/anomalies_{}.txt'.format(tag), 'a')

    if args.pretrained:
        # start training from previous pretrained model
        model = pickle.load(open('data/model_ae_{}_bak.pkl'.format(tag), 'rb'))
    else:
        fm = open('data/model_{}_2.pkl'.format(tag), 'w+b')

    pipeline_lstm = get_pipeline(model)

     # open s3  netflow bucket data
    s3 = boto3.resource('s3')
    s3_file = S3FileSystem()
    s3_client = boto3.client("s3")
    bucket = s3.Bucket('ml-flow-dump')
    scores = []
    anomalies = 0
    data = []
    counter = 0
    warmup = 10 #10000
    stream_data = [] 
    score_threshold = 0.2 #0.98
    
    min_score = float('inf')
    max_score = float('-inf')

    i = 0
    max = 0
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
            #print(f'record size: {len(record)}')
            
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
            
            # encode the categorical features
            y = str(list(categorical_features.values())).replace("[","") # convert all values to a string to embed
            y = y.replace("]","")
            y = y.replace("'", '')
           
            lda = lda.learn_one(y)
            categorical_features = lda.transform_one(y)   # transform the categorical features
            
            # Combine the encoded categorical and numeric features
            x_encoded = {**categorical_features, **numeric_features}
            
            # Get the anomaly score from the model
          
            anomaly_score = pipeline_lstm.score_one(x_encoded)
            #print(f"Anomaly score: {anomaly_score}")

            # Update min and max scores for normalization
            min_score = min(min_score, anomaly_score)
            max_score = np.max([max_score, anomaly_score])
            normalized_score = normalize_score(anomaly_score, min_score, max_score)   # Normalize the score

            pipeline_lstm.learn_one(x_encoded)
            counter +=1
            
            # append to stream_data and check if the list length exceeded N  and check for 
            # feature importance with isolation forest.
            #stream_data = stream_data.append(pd.json_normalize(x_encoded))
            stream_data.append(x_encoded)
            scores.append(anomaly_score)
            
            if len(stream_data) > warmup:    # keep the list length at N
                #rf.fit(stream_data, scores)
                
                # Remove the oldest element from the stream data and scores
                stream_data.pop(0)
                scores.pop(0)
                
            #print(f"Anomaly score max: {np.max(scores)}\n")
            #print(f"Anomaly score mean: {np.mean(scores)}")
            #print(f"Anomaly score std: {np.std(scores)}")
            #print(f"Anomaly score min: {np.min(scores)}")
            
            
           # Periodically update feature importance
            if counter > warmup  and normalized_score > score_threshold:
                # Permutation feature importance
                feature_importance = {}
                for feature in x_encoded.keys():
                    # Shuffle the feature and get new scores
                    shuffled_data = shuffle_feature(stream_data, feature)
                    shuffled_scores = [pipeline_lstm.score_one(x) for x in shuffled_data]

                    # Original scores
                    original_scores = scores

                    # Calculate the importance as the difference in mean scores
                    feature_importance[feature] = np.mean(original_scores) - np.mean(shuffled_scores)

                # Sort and display the feature importance
                sorted_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
                message = f"Anomaly detected from {src_ip} to {dst_ip} at {timestamp} with feature importance:\n"

                feat_dict = {}
                for i, (feature,importance) in enumerate(sorted_importance):
                    print(f"{feature}: {importance}")
                    feat_dict[feature]=importance
                    if i > 5:
                        break
                feat = message + str(feat_dict) + '\n'
                print(feat)
                stream_data = []  # reset stream data after feature importance calculation
                print("\n")
                