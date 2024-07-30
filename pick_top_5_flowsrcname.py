# pick top 5 flowsrcname 
# https://stackoverflow.com/questions/30787391/sorting-entire-csv-by-frequency-of-occurence-in-one-column

from heapq import nlargest
import pandas as pd
tag = "AMFC"
df = pd.read_csv("os_classification_AMFClassifier.csv")
df['Frequency'] = df.groupby('flowsrcname')['flowsrcname'].transform('count')
df.sort_values('Frequency', inplace=True, ascending=False)
l = set(df.Frequency.to_list())
l1 = nlargest(5, l)  # top 5 flowsrcname IPs
df2 = df[df['Frequency'].isin(l1)]
del df2['Frequency']
file_name = "os_classification_Top5_{}.csv".format(tag)
df2.to_csv(file_name,index=False)