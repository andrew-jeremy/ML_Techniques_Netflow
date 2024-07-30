from kafka import KafkaConsumer
import json

def read_netflow_events_from_kafka(topic, bootstrap_servers):
    """
    Read NetFlow events from a Kafka topic that streams JSON records.

    :param topic: Kafka topic to subscribe to.
    :param bootstrap_servers: List of bootstrap servers for the Kafka cluster.
    """
    # Create a Kafka consumer
    consumer = KafkaConsumer(
        topic,
        bootstrap_servers=bootstrap_servers,
        auto_offset_reset='latest',  # Start from only the latest messages
        value_deserializer=lambda m: json.loads(m.decode('utf-8'))  # Deserialize messages from JSON
    )

    # Read messages from Kafka
    for message in consumer:
        netflow_event = message.value  # This is now a Python dictionary
        yield netflow_event

# Usage
kafka_topic = 'netflow_topic'  # Replace with your Kafka topic
kafka_bootstrap_servers = ['localhost:9092']  # Replace with your Kafka cluster servers

# Process the stream
for netflow_event in read_netflow_events_from_kafka(kafka_topic, kafka_bootstrap_servers):
    print(netflow_event)
    # You can now process each NetFlow event as it arrives
