'''
pyshark simplifies the process of reading pcap files and provides a 
high-level API for working with packet data. You can access various 
attributes of each packet, such as protocol layers and field values, 
easily using the packet object.

Beaconing detection generally requires analyzing network traffic to identify regular, 
consistent communication patterns, which are characteristic of beaconing activity. To 
do this, you might want to extract various pieces of information from the IP layer and 
higher-level protocols like TCP or UDP.

Here are some additional IP layer and TCP/UDP layer features that could be useful for 
beaconing detection:
-Time Stamps: The time at which each packet was captured can be crucial for identifying 
regular communication patterns.
-Packet Size: The size of the packets can sometimes be a giveaway, especially if the 
beaconing malware sends packets of a consistent size.
-Protocol: The protocol being used (TCP, UDP, etc.) can also be an important feature.
-Time-to-Live (TTL): This field can sometimes be used to identify packets that have 
traveled a certain number of hops.
-Flags: In the case of TCP packets, flags can indicate the start or end of a connection, 
which might be useful in identifying beaconing patterns.
-Window Size: In TCP, this can provide information about the network conditions and the 
sender's view of the network.
'''
import pandas as pd
import pyshark
import datetime

# Specify the path to your pcap file
#pcap_file = "data/beaconing/2021-04-12-IcedID-infection-part-1-of-2.pcap"
pcap_file = "data/beaconing/2021-05-13-Hancitor-traffic-with-Ficker-Stealer-and-Cobalt-Strike.pcap"

# Read the PCAP file using pyshark
cap = pyshark.FileCapture(pcap_file)

# Initialize a list to store packet data
packet_data = []

# Loop through the packets and extract information
for packet in cap:
    packet_info = {
        'Timestamp': datetime.datetime.fromtimestamp(float(packet.sniff_timestamp)),
        'Length': int(packet.length)
    }
    
    # Extract IP layer information if available
    if 'IP' in packet:
        packet_info.update({
            'Source IP': packet.ip.src,
            'Destination IP': packet.ip.dst,
            'TTL': int(packet.ip.ttl),
            'Protocol': packet.ip.proto
        })
    
    # Extract TCP/UDP information if available
    if 'TCP' in packet or 'UDP' in packet:
        packet_info.update({
            'Source Port': int(packet[packet.transport_layer].srcport),
            'Destination Port': int(packet[packet.transport_layer].dstport),
            'Window Size': int(packet.tcp.window) if 'TCP' in packet and hasattr(packet.tcp, 'window') else None,
            'Flags': packet.tcp.flags if 'TCP' in packet else None
        })
    
    packet_data.append(packet_info)

# Create a DataFrame from the extracted data
df = pd.DataFrame(packet_data)

# Show the DataFrame
df.head(5)
df.to_csv('data/beaconing/2021-05-13-Hancitor-traffic-with-Ficker-Stealer-and-Cobalt-Strike.csv')

# how to stream from a csv file
'''
from river import stream
X_y = stream.iter_csv('data/beaconing/2021-05-13-Hancitor-traffic-with-Ficker-Stealer-and-Cobalt-Strike.csv')
x, y = next(X_y)
'''