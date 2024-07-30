from collections import defaultdict
from datetime import datetime
import argparse
import pandas as pd
import numpy as np
import pyshark
from river import anomaly
from river import compose
from river import preprocessing
from statistics import variance


# This will store the state for each source IP
beacon_states = defaultdict(lambda: {
    'last_seen': None,
    'intervals': [],
    'packet_sizes': [],
})

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

def is_beacon_activity(packet_features, score, interval_threshold=10, size_variance_threshold=8, score_threshold=0.3):
    """
    Check for beacon-like activity based on time intervals and packet size variance.
    """
    state = beacon_states[packet_features['Source IP']]
    inter_arrival_time = packet_features['Inter-Arrival Time']
    packet_size_variance = packet_features['Packet Size Variance']

    # Check if intervals are regular
    if inter_arrival_time < interval_threshold:
        # Check if packet sizes have low variance
        if packet_size_variance < size_variance_threshold:
            # Check if recent anomaly score is high
            if score > score_threshold:
                return True
    return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Streaming ML model')
    parser.add_argument('--method', type=int, default = 1, help='thresholding method, 1:percentile, 2:running mean, 3: fixed threshold')
    args = parser.parse_args()
    
    # Parse the pcap file using pyshark and apply the beaconing detection algorithm
    if args.method == 1:
        pcap_file_path = 'data/beaconing/2021-04-12-IcedID-infection-part-1-of-2.pcap'
    elif args.method == 2:
        pcap_file_path = "data/beaconing/2021-05-13-Hancitor-traffic-with-Ficker-Stealer-and-Cobalt-Strike.pcap"

    capture = pyshark.FileCapture(pcap_file_path)

   
    # Initialize the Hoeffding Tree Regressor model pipeline
    model_packets = compose.Pipeline(preprocessing.MinMaxScaler(),anomaly.HalfSpaceTrees(n_trees=5,height=3,window_size=3,seed=42))
    
    packet_data = []
    for packet in capture:
        try:
            # Extract packet features
            timestamp = datetime.fromtimestamp(float(packet.sniff_timestamp))
            src_ip = packet.ip.src
            dst_ip = packet.ip.dst
            src_port = packet[packet.transport_layer].srcport
            dst_port = packet[packet.transport_layer].dstport
            protocol = packet.transport_layer
            length = int(packet.length)

            state = beacon_states[src_ip]

            # Calculate inter-arrival time and packet size variance
            inter_arrival_time = calculate_inter_arrival_time(state, timestamp)
            packet_size_variance = calculate_packet_size_variance(state, length)

            # Prepare the features for the model
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
            }
            #packet_data.append(packet_features)
            df_features = pd.DataFrame([packet_features])
            X_packets = df_features[['Protocol','Source Port','Destination Port','Length', 'Inter-Arrival Time', 'Packet Size Variance']].fillna(0)
            X_packets[X_packets.select_dtypes(['object']).columns] = X_packets.select_dtypes(['object']).apply(lambda x: x.astype(str).astype('category').cat.codes)
            X_packets.replace([np.inf, -np.inf], 0, inplace=True)   # replace any inf and -inf with zero
    
            # Get the anomaly score from the model
            x = X_packets.iloc[0].to_dict()
            score = model_packets.score_one(x)
            #print("Anomaly score: ", score)

            # Update the model with the new data
            model_packets.learn_one(x)

            # Check for beaconing activity
            if is_beacon_activity(packet_features, score):
                print(f"Beaconing detected from {src_ip} to {dst_ip} at {timestamp}")

        except AttributeError as e:
            # This handles packets that do not have the expected attributes
            pass
            #print(f"Error parsing packet attributes: {e}")

