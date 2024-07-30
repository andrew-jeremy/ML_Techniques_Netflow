from sklearn.ensemble import IsolationForest
import pandas as pd

# Assuming `df` is your DataFrame containing the extracted packet features
df = pd.read_csv('data/beaconing/2021-05-13-Hancitor-traffic-with-Ficker-Stealer-and-Cobalt-Strike.csv')

# Feature Engineering: Calculate the time interval between successive packets to the same destination
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df = df.sort_values(by='Timestamp')
df['Time Interval'] = df.groupby(['Source IP', 'Destination IP'])['Timestamp'].diff().dt.total_seconds().fillna(0)

# Convert IP addresses to categorical type
df['Source IP'] = df['Source IP'].astype('category')
df['Destination IP'] = df['Destination IP'].astype('category')

# Convert port numbers to categorical type
df['Source Port'] = df['Source Port'].astype('category')
df['Destination Port'] = df['Destination Port'].astype('category')

# Convert categorical variables to numerical codes
df['Source IP Code'] = df['Source IP'].cat.codes
df['Destination IP Code'] = df['Destination IP'].cat.codes
df['Source Port Code'] = df['Source Port'].cat.codes
df['Destination Port Code'] = df['Destination Port'].cat.codes

# Features
X = df[['Length', 'TTL', 'Source Port', 'Destination Port', 'Window Size', 'Time Interval']].fillna(0)

# Initialize the Isolation Forest model
model = IsolationForest()

# Train the model on the data
model.fit(X)

# Make predictions on new data
new_data = pd.DataFrame({
    'Length': [100],
    'TTL': [64],
    'Source Port': [12345],
    'Destination Port': [80],
    'Window Size': [65535],
    'Time Interval': [10]
})

prediction = model.predict(new_data)

# In Isolation Forest, -1 indicates an anomaly and 1 indicates a normal data point
print("Beaconing Prediction (1 is normal, -1 is anomaly):", prediction)
