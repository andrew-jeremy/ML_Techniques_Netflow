from river import anomaly
from river import stream
from datetime import datetime
import numpy as np
import datetime
import argparse

'''
parse_network_data() function, you need to extract features from the network data that
are relevant to detecting beaconing activity. These features might include:
-Timestamps to calculate inter-arrival times
-Source and destination IP addresses
-Source and destination ports
-Protocol type (TCP/UDP/ICMP, etc.)
-Packet sizes for calculating the variance
-Packet counts from a source over a window of time
-Periodicity and regularity in communication patterns
'''
def parse_network_data(packet):
    """
    Parse network data packet to extract relevant features for beacon detection.
    
    :param packet: A dictionary representing a network data packet.
    :return: Parsed features as a dictionary.
    """
    parsed_data = {
        'timestamp': datetime.datetime.strptime(packet['timestamp'], "%Y-%m-%d %H:%M:%S"),
        'source_ip': packet['src_ip'],
        'destination_ip': packet['dst_ip'],
        'source_port': int(packet['src_port']),
        'destination_port': int(packet['dst_port']),
        'protocol': packet['protocol'],
        'packet_size': int(packet['size']),
        # You can add additional features that you deem relevant
        # For example, 'payload': packet['payload']  # if you have payload data
    }
    return parsed_data

# Define a function to parse incoming network data
def parse_network_data(data):
    # Placeholder for actual parsing logic
    # Should extract timestamp, source IP, destination IP, packet size, etc.
    return {
        'timestamp': datetime.datetime.now(),
        'source_ip': '192.168.1.1',
        'destination_ip': '8.8.8.8',
        'packet_size': len(data),
        # Add other relevant features here
    }

from collections import defaultdict
from statistics import variance

# This will store the state for each source IP
beacon_states = defaultdict(lambda: {
    'last_seen': None,
    'intervals': [],
    'packet_sizes': [],
    'score_history': []
})

'''
To write a is_beacon_activity() function, we need to look for specific indicators of beaconing 
behavior. One common approach is to analyze the timing of communications to see if there's a 
consistent interval between messages, which can suggest automated check-ins to a command and 
control server. We might also want to look for a low variance in the size of the packets, as 
beaconing often involves sending and receiving messages of a similar size.
'''
def is_beacon_activity(source_ip, timestamp, packet_size, score, interval_threshold=10, size_variance_threshold=8, score_threshold=0.3):
    """
    Check for beacon-like activity based on time intervals and packet size variance.
    
    :param source_ip: Source IP address of the packet
    :param timestamp: Timestamp of the packet arrival
    :param packet_size: Size of the packet
    :param score: Anomaly score from the detector
    :param interval_threshold: Threshold for the minimum interval regularity (in seconds)
    :param size_variance_threshold: Threshold for the maximum allowed variance in packet sizes
    :param score_threshold: Threshold for the anomaly score to consider for beacon detection
    :return: True if beacon-like activity is detected, False otherwise
    """
    state = beacon_states[source_ip]
    current_time = timestamp.timestamp()

    # Update intervals if we have seen this IP before
    if state['last_seen']:
        interval = current_time - state['last_seen']
        state['intervals'].append(interval)

        # Maintain a fixed window size for intervals
        if len(state['intervals']) > 100:
            state['intervals'].pop(0)

    # Update packet sizes
    state['packet_sizes'].append(packet_size)
    
    # Maintain a fixed window size for packet sizes
    if len(state['packet_sizes']) > 100:
        state['packet_sizes'].pop(0)

    # Update score history
    state['score_history'].append(score)
    if len(state['score_history']) > 100:
        state['score_history'].pop(0)

    state['last_seen'] = current_time

    # Check if intervals are regular
    if len(state['intervals']) > 2 and max(state['intervals']) - min(state['intervals']) < interval_threshold:
        # Check if packet sizes have low variance
        if len(state['packet_sizes']) > 2 and variance(state['packet_sizes']) < size_variance_threshold:
            # Check if recent anomaly scores are consistently high
            if all(s > score_threshold for s in state['score_history'][-5:]):
                return True

    return False


# Initialize HalfSpaceTrees anomaly detector
detector = anomaly.HalfSpaceTrees(
    n_trees=25,
    height=15,
    window_size=100,
    seed=42
)

# Placeholder for a function that reads from your data source
def get_streaming_data():
    # This should be replaced with actual streaming data logic
    while True:
        # Simulate network packet arrival
        simulated_packet = np.random.bytes(np.random.randint(1, 1500))
        yield simulated_packet
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Streaming ML model')
    parser.add_argument('--method', type=int, default = 2, help='thresholding method, 1:percentile, 2:running mean, 3: fixed threshold')
    args = parser.parse_args()
    
    # Process the streaming data
    for raw_data in get_streaming_data():
        features = parse_network_data(raw_data)
        features_without_timestamp = {k: v for k, v in features.items() if k != 'timestamp'}
        score = detector.score_one(features_without_timestamp)
        detector.learn_one(features_without_timestamp)
        
        # Define a custom function to check for beacon-like activity
        # This could check for regularity in timestamps, periodicity in communication with certain IPs, etc.
        if is_beacon_activity(features, score):
            print(f"Potential beacon event detected: {features}")

    # You need to define the is_beacon_activity function based on domain knowledge
    # For example, it could analyze the periodicity and regularity of events coming from the same source IP
    def is_beacon_activity(features, score):
        # Implement your logic for detecting beacon events
        # This could involve checking for:
        # - Regular time intervals
        # - Repeated communication with known malicious IPs
        # - Other indicators of C2 activity
        # Return True if beacon-like activity is detected, False otherwise
        pass
