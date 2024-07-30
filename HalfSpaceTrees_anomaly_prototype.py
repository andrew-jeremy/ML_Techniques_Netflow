from river import anomaly
import statistics

import numpy as np
import scipy.stats

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h

X = [0.5, 0.45, 0.43, 0.44, 0.445, 0.45, 0.0]
hst = anomaly.HalfSpaceTrees(n_trees=5,height=3,window_size=3,seed=42)

for x in X[:3]:
    hst = hst.learn_one({'x': x})  # Warming up
scores = []
for x in X:
    features = {'x': x}
    hst = hst.learn_one(features)
    scores.append(hst.score_one(features))
    print(f'Anomaly score for x={x:.3f}: {hst.score_one(features):.3f}')
print("mean score: {:.2f}".format(statistics.mean(scores)))
print("std score: {:.2f}".format(statistics.stdev(scores)))
print("median score: {:.2f}".format(statistics.median(scores)))
print("confidence interval (mean, low, high): %.2f,%.2f,%.2f" %mean_confidence_interval(scores))
# anomalies have significatly higher scores.