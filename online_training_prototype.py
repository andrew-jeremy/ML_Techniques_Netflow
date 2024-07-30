#import os
#os.environ['KMP_DUPLICATE_LIB_OK']='True'
from river_torch.anomaly import Autoencoder
from river import metrics
from river.datasets import CreditCard
from torch import nn
import torch
import math
import pandas as pd
import pickle
from river.compose import Pipeline
from river.preprocessing import MinMaxScaler


dataset = CreditCard().take(5000)

metric = metrics.ROCAUC(n_thresholds=50)

class MyAutoEncoder(nn.Module):
    def __init__(self, n_features, latent_dim=3):
        super(MyAutoEncoder, self).__init__()
        self.linear1 = nn.Linear(n_features, latent_dim)
        self.nonlin = nn.LeakyReLU()
        self.linear2 = nn.Linear(latent_dim, n_features)
        self.sigmoid = nn.Sigmoid()

    def forward(self, X, **kwargs):
         X = self.linear1(X)
         X = self.nonlin(X)
         X = self.linear2(X)
         return self.sigmoid(X)

ae = Autoencoder(module=MyAutoEncoder, lr=0.005)
scaler = MinMaxScaler()
model = Pipeline(scaler, ae)

# original code
for x, y in dataset:
    score = model.score_one(x)
    model = model.learn_one(x=x)
    #metric = metric.update(y, score)
    #print(f"ROCAUC: {metric.get():.4f}")
    print(score)
pickle.dump(model, open('model.pkl', 'wb'))

# now load save model and make predictions with it
pickled_model = pickle.load(open('model.pkl', 'rb'))
score = pickled_model.score_one(x)
#print(score)

