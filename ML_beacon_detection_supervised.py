from collections import defaultdict
from datetime import datetime
import config
import pandas as pd
import numpy as np
import pickle
import csv
import os
import argparse
import pyshark
from river import anomaly
from river import compose
from statistics import variance, mean, median
from river.preprocessing import StandardScaler
from river.preprocessing import OneHotEncoder
from river import naive_bayes
from river import metrics

# This will store the state for each source IP
beacon_states = defaultdict(lambda: {
    'last_seen': None,
    'intervals': [],
    'packet_sizes': [],
    'total_bytes': 0,
    'total_packets': 0,
    'flow_start': None,
    'flow_end': None,
    'unique_dst_ips': set(),
    'unique_dst_ports': set(),
})

def update_state_with_packet(state, packet, timestamp):
    """
    Update the state dictionary with the new packet's information.
    """
    state['total_bytes'] += int(packet.length)
    state['total_packets'] += 1
    state['unique_dst_ips'].add(packet.ip.dst)
    state['unique_dst_ports'].add(packet[packet.transport_layer].dstport)
    if state['flow_start'] is None or timestamp < state['flow_start']:
        state['flow_start'] = timestamp
    state['flow_end'] = timestamp

    return state

            
def calculate_additional_features(state, packet, timestamp):
    """
    Calculate additional features for beacon detection based on the updated state.
    """
    flow_duration = (state['flow_end'] - state['flow_start']).total_seconds() if state['flow_start'] else 0
    bytes_per_second = state['total_bytes'] / flow_duration if flow_duration > 0 else 0
    packets_per_second = state['total_packets'] / flow_duration if flow_duration > 0 else 0

    packet_features = {
        'flow_duration': flow_duration,
        'total_bytes': state['total_bytes'],
        'total_packets': state['total_packets'],
        'bytes_per_second': bytes_per_second,
        'packets_per_second': packets_per_second,
        'unique_dst_ips': len(state['unique_dst_ips']),
        'unique_dst_ports': len(state['unique_dst_ports']),
    }
    
    return packet_features
def calculate_inter_arrival_time(state, timestamp):
    current_time = timestamp.timestamp()
    if state['last_seen'] is not None:
        inter_arrival_time = current_time - state['last_seen']
    else:
        inter_arrival_time = 0 #8888888  # some large value
    state['last_seen'] = current_time
    return inter_arrival_time

def calculate_packet_size_variance(state, packet_size):
    state['packet_sizes'].append(packet_size)
    if len(state['packet_sizes']) > 100:  # keep only the last 100 packet sizes
        state['packet_sizes'].pop(0)
    return mean(state['packet_sizes']),variance(state['packet_sizes']) if len(state['packet_sizes']) > 1 else 0 #8888888 # some large value

def beacon_activity(packet_features, score, threshold=0.8):
    """
    Check for beacon-like activity based on additional packet features.
    
    Arguments:
    packet_features -- dictionary of packet features including timing and size characteristics.
    score -- anomaly score from the model for the current packet.
    threshold -- threshold for considering an activity as beaconing (can be tuned).
    
    Returns:
    Boolean indicating if the current packet is considered part of beaconing activity.
    """

    # The regularity of inter-arrival times and the variance of packet sizes
    # are key indicators of beaconing activity. Low variance in packet sizes
    # and regular inter-arrival times are suggestive of automated C2 communication.
    
    # Define thresholds for detection criteria - these may need to be tuned based on validation data
    interval_threshold = config.interval_threshold  # seconds (this is an example and may need to be adjusted)
    size_variance_threshold = config.size_variance_threshold  # bytes^2 (example value, adjust based on your data)
    unique_dst_ips_threshold = config.unique_dst_ips_threshold # Number of unique destination IPs indicative of scanning or distributed C2
    packet_size_threshold = config.packet_size_threshold 
    
    # Assess interval regularity and size variance
    is_regular_intervals = packet_features['Inter-Arrival Time'] < interval_threshold
    is_low_variance = packet_features['Packet Size Variance'] < size_variance_threshold
    is_low_packet_size = packet_features['Mean Packet Size'] < packet_size_threshold
    is_focused_traffic = packet_features['unique_dst_ips'] <= unique_dst_ips_threshold

    # Score from the model is also considered
    is_anomalous = score > threshold

    # Combine the criteria to determine beaconing activity
    if is_regular_intervals and is_low_variance and is_focused_traffic and is_anomalous: # and is_low_packet_size:
        print(f"packet_features['Inter-Arrival Time']:{packet_features['Inter-Arrival Time']}")
        print(f"packet_features['Packet Size Variance']:{packet_features['Packet Size Variance']}")
        print(f"packet_features['Mean Packet Size']:{packet_features['Mean Packet Size']}")
        print(f"packet_features['unique_dst_ips']:{packet_features['unique_dst_ips']}")
        print(f"packet_features['unique_dst_ports']:{packet_features['unique_dst_ports']}")
        print(f"packet_features['total_bytes']:{packet_features['total_bytes']}")
        print(f"packet_features['total_packets']:{packet_features['total_packets']}")
        print(f"packet_features['bytes_per_second']:{packet_features['bytes_per_second']}")
        print(f"packet_features['packets_per_second']:{packet_features['packets_per_second']}")
        print(f"packet_features['flow_duration']:{packet_features['flow_duration']}")
        print(f"packet_features['Mean Packet Size']:{packet_features['Mean Packet Size']}")
        print(f"score:{score}")
            
        
        print('\n')
        return True
    
    return False

# Initialize a one-hot encoder for categorical features
one_hot_encoder = OneHotEncoder()

# Initialize the anomaly detection model
anomaly_model = anomaly.HalfSpaceTrees(n_trees=25, window_size=100, seed=42)

# Create a streaming pipeline
pipeline = compose.Pipeline(
    #('features', compose.Select('feature_1', 'feature_2', ...)),  # Select relevant features
    ('scaler', StandardScaler()),  # Standardize features (if needed)
    ('model', anomaly_model)  # Anomaly detection model
)

# Initialize the classifier for semi-supervised learning
pipeline_1 = compose.Pipeline(
    ('scaler', StandardScaler()),  # Standardize features (if needed)
    ('anomaly_score', anomaly_model))

def update_with_anomaly_score(packet_features):
    anomaly_score = pipeline_1.score_one(packet_features)
    anomaly_model.learn_one(packet_features)
    packet_features['Anomaly Score'] = anomaly_score
    return packet_features

# Create a pipeline
pipeline_2 = compose.Pipeline(
    ('add_anomaly_score', compose.FuncTransformer(update_with_anomaly_score)),
    ('classifier', naive_bayes.GaussianNB())
)

# Initialize a metric to track the performance of the model
metric = metrics.Accuracy()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Streaming ML model')
    parser.add_argument('--method', type=int, default = 1, help='pcap file, 1:IcedID, 2:Hancitor')
    
    args = parser.parse_args()
    

   # Parse the pcap file using pyshark and apply the beaconing detection algorithm
    if args.method == 1:
        pcap_file_path = 'data/beaconing/2021-04-12-IcedID-infection-part-1-of-2.pcap'
        filename_packets = open('results/beaconing_events_IcedID.txt', 'w', newline='')
    elif args.method == 2:
        pcap_file_path = "data/beaconing/2021-05-13-Hancitor-traffic-with-Ficker-Stealer-and-Cobalt-Strike.pcap"
        filename_packets = open('results/beaconing_events_Hancitor.txt', 'w', newline='')
    
    capture = pyshark.FileCapture(pcap_file_path)
    beacons = []
    count = 0
    for packet in capture:
        try:
            # Extract basic features from the packet
            timestamp = datetime.fromtimestamp(float(packet.sniff_timestamp))
            src_ip = packet.ip.src
            dst_ip = packet.ip.dst
            src_port = packet[packet.transport_layer].srcport
            dst_port = packet[packet.transport_layer].dstport
            protocol = packet.transport_layer
            length = int(packet.length)

            state = beacon_states[src_ip]

            # Update the state with the current packet
            updated_state = update_state_with_packet(state, packet, timestamp)

            # Calculate the additional features based on the updated state
            additional_features = calculate_additional_features(updated_state, packet, timestamp)

            # Calculate inter-arrival time and packet size variance
            inter_arrival_time = calculate_inter_arrival_time(state, timestamp)
            packet_size, packet_size_variance = calculate_packet_size_variance(state, length)
            
            # Prepare the packet features for the model
            packet_features = {
                'Timestamp': timestamp,
                'Protocol': protocol,
                'Length': length,
                'Source IP': src_ip,
                'Source Port': src_port,
                'Destination IP': dst_ip,
                'Destination Port': dst_port,
                'Inter-Arrival Time': inter_arrival_time,
                'Packet Size Variance': packet_size_variance,
                'Mean Packet Size': packet_size,
                **additional_features,
                # Add other features that might be relevant
            }
            #packet_data.append(packet_features)
            df_features = pd.DataFrame([packet_features])
            #X_packets = df_features[['Protocol','Source Port','Destination Port','Length', 'Inter-Arrival Time', 'Packet Size Variance']].fillna(0)
            df_features.drop(['Timestamp'], axis=1, inplace=True)
            
            #df_features.drop(['Source IP'], axis=1, inplace=True)
            #df_features.drop(['Destination IP'], axis=1, inplace=True)
            
            X_packets = df_features.fillna(0)
            #X_packets[X_packets.select_dtypes(['object']).columns] = X_packets.select_dtypes(['object']).apply(lambda x: x.astype(str).astype('category').cat.codes)
            X_packets.replace([np.inf, -np.inf], 0, inplace=True)   # replace any inf and -inf with zero

                    
            # Get the anomaly score from the model
            x = X_packets.iloc[0].to_dict()
            
            # process categorical features and separate out categorical and numeric features
            categorical_features = {}
            numeric_features = {}
            for key, value in x.items():
                if isinstance(value, str):  # Assuming strings are categorical
                    categorical_features[key] = value
                else:
                    numeric_features[key] = value
            # Apply one-hot encoding to the categorical features
            encoded_categorical_features = one_hot_encoder.transform_one(categorical_features)

            # Combine the encoded categorical and numeric features
            x_encoded = {**encoded_categorical_features, **numeric_features}

            # semi-supervision for known beaconing activity
            if (src_ip == '10.4.12.101' and dst_ip == '83.97.20.176') or (src_ip=='83.97.20.176' and dst_ip=='10.4.12.101'): 
                label = 1
                pipeline_2.learn_one(x_encoded, label)
                metric.update(label, pipeline_2.predict_one(packet_features))
                print("Model accuracy (labeled data):", metric)
            else:
                label = 0
                # Predict using the classifier for unlabeled data
                prediction = pipeline.predict_one(x_encoded)
                print(prediction)
                
            #if beacon_activity(packet_features, score,config.score_threshold):
            #    #print('packet_features:', packet_features)
            #    beacons.append([f"Beaconing detected from {src_ip} to {dst_ip} at {timestamp}"])
            #    print(f"Beaconing detected from {src_ip} to {dst_ip} at {timestamp}")
            count += 1
        except AttributeError as e:
            # This handles packets that do not have the expected attributes
            pass
            #print(f"Error parsing packet attributes: {e}")
    print(f"total packets processed: {count}")
    with filename_packets as file:
        writer = csv.writer(file)
        for row in beacons:
                writer.writerow(row)
    filename_packets.close()
