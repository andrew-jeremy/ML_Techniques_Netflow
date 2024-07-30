
from collections import defaultdict
import numpy as np

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
    return np.var(state['packet_sizes']) if len(state['packet_sizes']) > 1 else 0

def beacon_activity(packet_features, cluster,score, interval_threshold, size_variance_threshold,unique_dst_ips_threshold,threshold=0.8):
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
    #interval_threshold = config.interval_threshold  # seconds (this is an example and may need to be adjusted)
    #size_variance_threshold = config.size_variance_threshold  # bytes^2 (example value, adjust based on your data)
    #unique_dst_ips_threshold = config.unique_dst_ips_threshold # Number of unique destination IPs indicative of scanning or distributed C2
    
    # Assess interval regularity and size variance
    is_regular_intervals = packet_features['Inter-Arrival Time'] < interval_threshold
    is_low_variance = packet_features['Packet Size Variance'] < size_variance_threshold
    is_focused_traffic = packet_features['unique_dst_ips'] <= unique_dst_ips_threshold

    # Score from the model is also considered
    is_anomalous = score > threshold

    # Combine the criteria to determine beaconing activity - NEED TO ADD IN CLUSTER INFO...
    if is_regular_intervals and is_low_variance and is_focused_traffic and is_anomalous:
        return True
    return False
