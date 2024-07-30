from collections import defaultdict
from datetime import datetime
import config
import pandas as pd
import numpy as np
import boto3
import csv
import json
import argparse
import pyshark
from river import anomaly
from river import cluster
from statistics import variance, mean
from ipaddress import ip_address


#------ botnet detection class for ML--------------------->
class FeatureExtractor:
    def __init__(self, window_size=1000):
        self.window_size = window_size
        self.packet_sizes = defaultdict(list)
        self.connection_counts = defaultdict(int)
        self.last_n_dest_ips = defaultdict(list)

    def extract_features(self, record):
        src_ip = record['Source IP']
        dst_ip = record['Destination IP']
        
        # Feature 1: Unusual Volume of Traffic (average packet size)
        self.packet_sizes[src_ip].append(record['total_packets'])
        if len(self.packet_sizes[src_ip]) > self.window_size:
            self.packet_sizes[src_ip].pop(0)
        avg_packet_size = sum(self.packet_sizes[src_ip]) / len(self.packet_sizes[src_ip])

        # Feature 2: Repetitive Connections to Certain IPs
        self.connection_counts[dst_ip] += 1

        # Feature 3: Irregular Communication Patterns (Change in Destination IPs)
        if dst_ip not in self.last_n_dest_ips[src_ip]:
            self.last_n_dest_ips[src_ip].append(dst_ip)
        if len(self.last_n_dest_ips[src_ip]) > self.window_size:
            self.last_n_dest_ips[src_ip].pop(0)
        unique_dst_ips = len(set(self.last_n_dest_ips[src_ip]))

        return {
            "avg_packet_size": avg_packet_size,
            "connection_count_to_dst_ip": self.connection_counts[dst_ip],
            "unique_dst_ips": unique_dst_ips
        }
#--------------------------------------------------------->

# Initialize the HalfSpaceTrees anomaly detector
model = anomaly.HalfSpaceTrees(
    n_trees=25,
    window_size=100,
    seed=42  # Random seed for reproducibility
)

dbstream = cluster.DBSTREAM(
    clustering_threshold=1.5,
    fading_factor=0.05,
    cleanup_interval=4,
    intersection_factor=0.5,
    minimum_weight=1
)
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

def update_state_with_packet_json(state, record, timestamp):
    """
    Update the state dictionary with the new packet's information.
    """
    state['total_bytes'] += int(record["bits"]/8)
    state['total_packets'] += 1
    state['unique_dst_ips'].add(record["dstip"])
    state['unique_dst_ports'].add(record["dstport"])
    if state['flow_start'] is None or timestamp < state['flow_start']:
        state['flow_start'] = timestamp
    state['flow_end'] = timestamp

    return state

def calculate_additional_features(state):
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
        inter_arrival_time = 0
    state['last_seen'] = current_time
    return inter_arrival_time

def calculate_packet_size_variance(state, packet_size):
    state['packet_sizes'].append(packet_size)
    if len(state['packet_sizes']) > 100:  # keep only the last 100 packet sizes
        state['packet_sizes'].pop(0)
    return variance(state['packet_sizes']) if len(state['packet_sizes']) > 1 else 0

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
    
    # Assess interval regularity and size variance
    is_regular_intervals = packet_features['Inter-Arrival Time'] < interval_threshold
    is_low_variance = packet_features['Packet Size Variance'] < size_variance_threshold
    is_focused_traffic = packet_features['unique_dst_ips'] <= unique_dst_ips_threshold

    # Score from the model is also considered
    is_anomalous = score > threshold

    # Combine the criteria to determine beaconing activity
    if is_regular_intervals and is_low_variance and is_focused_traffic and is_anomalous:
        return True
    return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Streaming ML model')
    parser.add_argument('--method', type=int, default = 1, help='read json file, 1:read from s3 bucket, 2: stream from kafka')
    
    args = parser.parse_args()
    
   # Parse the pcap file using pyshark and apply the beaconing detection algorithm
    if args.method == 1:
        netflow_json_data = 'data/data.json'
        filename_packets_json = open('results/beaconing_events.txt', 'w', newline='')
    elif args.method == 2:
        # stream from kafka logic goes here
        netflow_json_data = 'data/data.json'
    elif args.method == 3:
        # open s3  netflow bucket data
        s3 = boto3.resource('s3')
        s3_client = boto3.client("s3")
        bucket = s3.Bucket('ml-flow-dump')
        
    l = []
    beacons = []
    count = 0
    warmup = 10
    with open(netflow_json_data, 'r') as file:   
        for line in file:
            record = json.loads(line)
            #print(line)
            try:
                # Extract basic features from the packet
                timestamp = datetime.fromtimestamp(record["timestamp"] / 1e3)   # timestamp
                src_ip = record["srcip"]       # srcip
                dst_ip = record["dstip"]       # dstip
                src_port = record["srcport"]   # srcport
                dst_port = record["dstport"]   # dstport
                protocol = record["protocolint"]                   # protocolint
                length = int(record["bits"]/8)                     # bits/8

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
                #X_packets = df_features[['Protocol','Source Port','Destination Port','Length', 'Inter-Arrival Time', 'Packet Size Variance']].fillna(0)
                df_features.drop(['Timestamp'], axis=1, inplace=True)
                df_features[['Protocol','Source Port','Destination Port']] = df_features[['Protocol','Source Port','Destination Port']].astype('object')
                X_packets = df_features.fillna(0)
                X_packets[X_packets.select_dtypes(['object']).columns] = X_packets.select_dtypes(['object']).apply(lambda x: x.astype(str).astype('category').cat.codes)
                X_packets.replace([np.inf, -np.inf], 0, inplace=True)   # replace any inf and -inf with zero
        
                # Get the anomaly score from the model
                x = X_packets.iloc[0].to_dict()
                
                #'''
                #--------------> botnet feature extraction ---------->
                feature_extractor = FeatureExtractor(window_size=1000)
                features = feature_extractor.extract_features(x)
                
                model = anomaly.HalfSpaceTrees()
                # Get anomaly score and update the model
                score = model.score_one(features)
                model.learn_one(features)
                x.update(features)  # combine botnet features with packet features
                #---------------------------------------------------->
                #'''
                
                # Get the anomaly score from the model
                score = model.score_one(x)
                dbstream = dbstream.learn_one(x)
                
                score_threshold = 0.1 #config.score_threshold   # threshold for considering an anomaly score as anomalous
                l.append(dst_ip)
                if score > score_threshold and count > int(warmup):
                    cluster = dbstream.predict_one(x)
                    print(f"Anomaly score: {score}")
                    print(f"Number of clusters: {dbstream.n_clusters}")
                    #pass

                # Update the model with the new data
                model.learn_one(x)

                # Check for beaconing activity
                if beacon_activity(packet_features, score,score_threshold):
                    print(f"Beaconing detected from {src_ip} to {dst_ip} at {timestamp}")
                    #beacons.append([src_ip, dst_ip, timestamp])
                    #beacons.append([f"Beaconing detected from {src_ip} to {dst_ip} at {timestamp}"])

            except AttributeError as e:
                # This handles packets that do not have the expected attributes
                pass
                #print(f"Error parsing packet attributes: {e}")
        with filename_packets_json as file:
            writer = csv.writer(file)
            for row in beacons:
                    writer.writerow(row)
        filename_packets_json.close()
        
        #print(set(l))
        #print(len(set(l)))
