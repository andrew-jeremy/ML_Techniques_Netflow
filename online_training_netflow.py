'''
online training of netflow single layer autoencoder based anomaly 
detector.
Andrew Kiruluta
Copyright: Netography Oct 2022
'''
#import os
#os.environ['KMP_DUPLICATE_LIB_OK']='True'
from river_torch import anomaly
from river import metrics
from river import compose
from river import preprocessing
from torch import nn
import math
import pandas as pd
import pickle
import boto3
import argparse
import statistics

from river.compose import Pipeline
from river.preprocessing import MinMaxScaler

from smart_open import open
import json
import uuid

metric = metrics.ROCAUC(n_thresholds=50)

#'''
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
#'''

#---->
# Creating a PyTorch class
class AE(nn.Module):
	def __init__(self, n_features, latent_dim=18):
		super().__init__()
		
		# Building an linear encoder with Linear
		self.encoder = nn.Sequential(
			nn.Linear(n_features, 1024),
			nn.ReLU(),
			nn.Linear(1024, 512),
			nn.ReLU(),
			nn.Linear(512, 128),
			nn.ReLU(),
			nn.Linear(128, latent_dim),
		)
		
		# Building an linear decoder with Linear
		self.decoder = nn.Sequential(
			nn.Linear(18, 128),
			nn.ReLU(),
			nn.Linear(128, 512),
			nn.ReLU(),
			nn.Linear(512, 1024),
			nn.ReLU(),
			nn.Linear(1024, n_features),
			nn.Sigmoid()
		)

	def forward(self, x):
		encoded = self.encoder(x)
		decoded = self.decoder(encoded)
		return decoded
#---->
def replace_boolean(data):
    for col in data:
        data[col].replace(True, 1, inplace=True)
        data[col].replace(False, 0, inplace=True)
    return df

def netflow_preprocess(df_out):
    l = l1 + l2 + l3
    df_out = df_out.drop(l,axis=1)
    df_out = replace_boolean(df_out)

    # featurize all ip address features (['dstip', 'flowsrcip', 'nexthop', 'srcip']) into 4 numbers each
    df[['dstip', 'flowsrcip', 'nexthop', 'srcip']] = df[['dstip', 'flowsrcip', 'nexthop', 'srcip']].fillna('0.0.0.0')
    df_out[['dstip1','dstip2','dstip3','dstip4']] =  df_out.dstip.str.split(".", expand=True)
    df_out.drop('dstip', axis=1, inplace=True)

    df_out[['flowsrcip1','flowsrcip2','flowsrcip3','flowsrcip4']] =  df_out.flowsrcip.str.split(".", expand=True)
    df_out.drop('flowsrcip', axis=1, inplace=True)

    df_out[['nexthop1','nexthop2','nexthop3','nexthop4']] =  df_out.nexthop.str.split(".", expand=True)
    df_out.drop('nexthop', axis=1, inplace=True)

    df_out[['srcip1','srcip2','srcip3','srcip4']] =  df_out.srcip.str.split(".", expand=True)
    df_out.drop('srcip', axis=1, inplace=True)

    cols = df_out.columns
    df_out[cols] = df_out[cols].apply(pd.to_numeric, errors='coerce')
    df_out = df_out.fillna(0)

    return df_out.to_dict('records')[0] # in form to be used by online river autoencoder anomaly detector

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='online training')
    parser.add_argument('--encoder', type=int, default=1,help='use encoder latent space,\
         1==YES, 0: for HalfSpaceTree')
    args = parser.parse_args()
    
    # https://stackoverflow.com/questions/12761991/how-to-use-append-with-pickle-in-python
    if args.encoder == 1:
        tag = 'ae'
    else:
        tag ='hst'
    fm = open('data/model_{}.pkl'.format(tag), 'w+b')
    fs =  open('data/scores_{}.txt'.format(tag), 'a')
    fa =  open('data/anomalies_{}.pkl'.format(tag), 'w+b')
  


    # features to drop
    l1 = ['flowsrcname','site','flowtype','inputname','tags','action','customer','dstas.org',
        'dstiprep.categories', 'dstgeo.continentcode', 'dstgeo.countrycode',
        'dstgeo.subdiso', 'srcas.org', 'srciprep.categories',
        'srcgeo.continentcode', 'srcgeo.countrycode', 'srcgeo.subdiso']
        
    l2 = ['dstvlan','inputalias','inputclasses','outputalias','outputclasses',
    'outputname','payload','dstowneras.org','srcvlan','dstowneras.number','srcas.number','srcowneras.number','srcowneras.org','dstas.number']

    l3 =['flowrtime','end', 'bitsxrate', 'packetsxrate','flowversion','ipversion','samplerate']


    # open s3  netflow bucket data
    s3 = boto3.resource('s3')
    s3_client = boto3.client("s3")
    bucket = s3.Bucket('ml-flow-dump')
    scores = []
    anomalies = []

    count = 1
    # select model to use
    if args.encoder == 1:
        ae = anomaly.Autoencoder(module=MyAutoEncoder, lr=0.005)
        scaler = MinMaxScaler()
        model = Pipeline(scaler, ae)
    else:
        from river import anomaly
        hst = anomaly.HalfSpaceTrees(n_trees=5,height=3,window_size=3,seed=42)
        model = compose.Pipeline(preprocessing.MinMaxScaler(),hst)

    for obj in bucket.objects.filter(Prefix="flow"):
        source_url = 's3://ml-flow-dump/' + obj.key
        print(source_url)
        for i,json_line in enumerate(open(source_url, transport_params={"client": s3_client})):
            my_json = json.loads(json_line)
            df = pd.json_normalize(my_json)  # one line at a time
            x = netflow_preprocess(df)   
            score = model.score_one(x)
            model = model.learn_one(x=x)
            #auc = auc.update(y, score)      # once you have labels y
            scores.append(score)
            fs.write(str(score) + ',')
            print(score)
            #print(count)
            if score > 0.15 and count > 100000  and args.encoder == 1: # only for Autoencoder algo
                print(count)
                print("median score: {:.2f}".format(statistics.median(scores)))
                #print("score: {:.2f}".format(score))
                anomalies.append(my_json)
                filename = "data/anomaly/vae_1/anomaly_" + str(uuid.uuid4()) + ".json"
                with open(filename, 'w') as f:
                    json.dump(my_json, f, ensure_ascii=False)
                #pickle.dump(anomalies,fa)
            if score > 0.1 and count > 10000  and args.encoder == 0: # only for HalfSpaceTrees algo
                print("median score: {:.2f}".format(statistics.median(scores)))
                print("score: {:.2f}".format(score))
                anomalies.append(my_json)
            if count % 20000 == 0:  # save new model every 10k batch
                print("score: {:.2f}".format(score))
                pickle.dump(model, fm)   
                #pickle.dump(scores,fs)
                #pickle.dump(anomalies,fa)
            count +=1
        #break
    #print(score)
    '''
    if args.encoder == 1:
        pickle.dump(model, open('data/model_ae.pkl', 'wb'))
        pickle.dump(scores, open('data/scores_ae.pkl', 'wb'))
        pickle.dump(anomalies, open('data/anomalies_ae.pkl', 'wb'))
    else:
        pickle.dump(model, open('data/model_hst.pkl', 'wb'))
        pickle.dump(scores, open('data/scores_hst.pkl', 'wb'))
        pickle.dump(anomalies, open('data/anomalies_hst.pkl', 'wb'))
    # write trained model to disk
    '''
    '''
    # to read scores into a list
    with open("scores_ae.txt", "r") as f:
         l=[i for line in f for i in line.split(',')]
         k=[float(i) for i in l[:-1]]
    '''