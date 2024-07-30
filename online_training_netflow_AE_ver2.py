'''
online training of netflow 4 layer autoencoder based anomaly 
detector.
Andrew Kiruluta
Copyright: Netography Oct 2022
'''
'''
s3_client = boto3.client("s3", 
                  region_name='us-east-1', 
                  aws_access_key_id='AKIARYZ6O3NJXVUGY455', 
                  aws_secret_access_key='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')
'''
#from river_torch import anomaly
from deep_river import anomaly
from river import anomaly as anomaly2
from river import metrics
from river import compose
from river import preprocessing
from torch import nn
import pandas as pd
import pickle
import boto3
import argparse
import statistics
import time
from s3fs import S3FileSystem
from feat_preprocessor import feat_eng

#from river.compose import Pipeline
#from river.preprocessing import MinMaxScaler
from river import (
    stream,
    compose,
    preprocessing,
    evaluate,
    metrics,
    tree,
    imblearn,
    stats,
)
import numbers 
import numpy as np

from smart_open import open
import json
import uuid
#from LSTMClass import  LSTMAutoencoderSrivastava, LSTMAutoencoderCho,LSTMDecoder, LSTMAutoencoderSutskever
#from river_torch.anomaly import RollingAutoencoder

metric = metrics.ROCAUC(n_thresholds=50)

# Creating a PyTorch class
class AE(nn.Module):
	def __init__(self, n_features, latent_dim=18):
		super().__init__()
		self.n_features = n_features
		# Building an linear encoder with Linear
		self.encoder = nn.Sequential(
			nn.Linear(self.n_features, 1024),
			nn.LeakyReLU(),
			nn.Linear(1024, 512),
			nn.LeakyReLU(),
			nn.Linear(512, 128),
			nn.LeakyReLU(),
			nn.Linear(128, latent_dim),
		)
		
		# Building an linear decoder with Linear
		self.decoder = nn.Sequential(
			nn.Linear(latent_dim, 128),
			nn.LeakyReLU(),
			nn.Linear(128, 512),
			nn.LeakyReLU(),
			nn.Linear(512, 1024),
			nn.LeakyReLU(),
			nn.Linear(1024, self.n_features),
			nn.Sigmoid()
		)

	def forward(self, x):
		encoded = self.encoder(x)
		decoded = self.decoder(encoded)
		return decoded

def replace_boolean(data):
    for col in data:
        data[col].replace(True, 1, inplace=True)
        data[col].replace(False, 0, inplace=True)
    return data

def get_pipeline(model):
    #mean = stats.Mean()
    #mode = stats.Mode()
    cat = (
        compose.SelectType(str)
        | preprocessing.StatImputer()
        | preprocessing.OneHotEncoder(sparse=True)
    )
    num = compose.SelectType(numbers.Number) | preprocessing.StandardScaler(stats.Mean())
    processor = num + cat
    return processor | model

def feat_eng2(df_out):
    # features to drop
    l1 = ['flowsrcname','site','flowtype','inputname','tags','action','customer','dstas.org',
            'dstiprep.categories', 'dstgeo.continentcode', 'dstgeo.countrycode',
            'dstgeo.subdiso', 'srcas.org', 'srciprep.categories',
            'srcgeo.continentcode', 'srcgeo.countrycode', 'srcgeo.subdiso', 'dstiprep.count','tos',
            'bogondst','bogonsrc','dstinternal','input','output','srciprep.count','srcinternal']
            
    l2 = ['dstvlan','inputalias','inputclasses','outputalias','outputclasses',
        'outputname','payload','dstowneras.org','srcvlan','dstowneras.number','srcas.number','srcowneras.number','srcowneras.org','dstas.number']

    l3 = ['flowrtime','end','flowversion','ipversion','samplerate']

    tcp = [col for col in df_out.columns if 'tcp' in col]
    icm = [col for col in df_out.columns if 'icm' in col]

    l = l1 + l2 + l3 + tcp + icm
    df_out = df_out.drop(l,axis=1)
    df_out = replace_boolean(df_out)
    df_out[['dstip', 'flowsrcip', 'nexthop', 'srcip']] = df_out[['dstip', 'flowsrcip', 'nexthop', 'srcip']].astype(str).fillna('0.0.0.0')

    # convert to categorical type 
    # ['srcport','dstport','protocolint','dstgeo.location.lat',
    # 'dstgeo.location.lon','srcgeo.location.lat','srcgeo.location.lon'] 
    df_out[['srcport','dstport','protocolint','dstgeo.location.lat','dstgeo.location.lon','srcgeo.location.lat','srcgeo.location.lon']]\
    = df_out[['srcport','dstport','protocolint','dstgeo.location.lat','dstgeo.location.lon','srcgeo.location.lat','srcgeo.location.lon']].astype(str)
    df_out['timeDelta'] = df_out["timestamp"] - df_out["start"]
    df_out.drop(["timestamp","start"],axis=1, inplace=True)

    #df_out['transferred_ratio'] = (df_out['bits'] / 8 / df_out['packets']) / (df['duration'] + 1)
    #df_out.drop(columns = ['duration', 'packets', 'bits'], axis =1, inplace=True)

    return df_out.to_dict('records')[0]  # in form to be used by online river autoencoder anomaly detector

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='online training')
    parser.add_argument('--encoder', type=int, default=1,help='use encoder latent space,\
         1==ae, 2: prob_ae, 3: hst, 4:svm')
    parser.add_argument('--pretrained', type=bool, default=False,help='load pretrained model,\
         True: for Yes')
    args = parser.parse_args()

    # open s3  netflow bucket data
    s3 = boto3.resource('s3')
    s3_file = S3FileSystem()
    s3_client = boto3.client("s3")
    bucket = s3.Bucket('ml-flow-dump')
    scores = []
    anomalies = 0
    data = []

    count = 1
    # select model to use
    if args.encoder == 1:
        tag = 'ae'
        model = anomaly.Autoencoder(module=AE, lr=0.005)
    elif args.encoder == 2:
        tag = 'prob_ae'
        model = anomaly.ProbabilityWeightedAutoencoder(module=AE, lr=0.05)
    elif args.encoder == 3:
        tag = 'hst'
        model = anomaly2.HalfSpaceTrees(n_trees=5,height=3,window_size=3,seed=42)
    else:
        tag = 'svm'
        model = anomaly2.OneClassSVM(nu=0.2)
    
    with open('data/anomalies_{}.txt'.format(tag), 'w') as document: pass  # initially create empty file
    fm = open('data/model_{}_2.pkl'.format(tag), 'w+b')
    fa =  open('data/anomalies_{}.txt'.format(tag), 'a')

    if args.pretrained:
        # start training from previous pretrained model
        model = pickle.load(open('data/model_ae_2_bak.pkl', 'rb'))
    else:
        # now build pipeline
        pipeline = get_pipeline(model)

        fm = open('data/model_{}_2.pkl'.format(tag), 'w+b')  # for model save later
    
    filename = "data/anomaly/vae_2/anomaly_{}".format(tag) +  ".json"
    #filename = "data/anomaly/vae_2/test" +  ".json"   # test point
    filename = "/Users/kiruluta/Dropbox/Docker/ml-kafka/output.json"
  
    for obj in bucket.objects.filter(Prefix="flow"):
        source_url = 's3://ml-flow-dump/' + obj.key
        print(source_url)
        for i,json_line in enumerate(open(source_url, transport_params={"client": s3_client})):
            start_time = time.time()
            my_json = json.loads(json_line)
            df = pd.json_normalize(my_json)  # one line at a time
            df.fillna(0, inplace=True)
            x = feat_eng(df)   
            #x = pipeline.transform_one(x)
            
            #print(x.values())
            #score = 0

            score = pipeline.score_one(x)
            pipeline.learn_one(x=x)
            #print("--- %s seconds ---" % (time.time() - start_time))

            #auc = auc.update(y, score)      # once you have labels y
            scores.append(score)
            count +=1
            #print(score)
            if count % 1000 == 0:
               print(score)
               #print(count)
            if score > 50 and count > 100000: # 10k training warmup
                anomalies +=1
                print(count)
                print("median score: {:.2f}".format(statistics.median(scores)))
                print("score: {:.2f}".format(score))
                #filename = "data/anomaly/vae_2/anomaly_{}".format(tag) +  ".json"
                path_to_s3_object = 's3://ml-flow-dump/' + filename
                if anomalies == 1:
                    #with s3_file.open(path_to_s3_object, 'w') as file:
                    #    json.dump(my_json, file)
                    content = json.dumps(my_json)
                    s3.Object('ml-flow-dump', filename).put(Body=content)
                else:
                    resp=s3_client.get_object(Bucket="ml-flow-dump", Key=filename)
                    #data=resp.get('Body')
                    json_data = resp.get('Body').read().decode("utf-8")

                    # append new anomalies to s3 bucket
                    #json_data = json.load(data)
                    #content = json.dumps(json_data)+'\n'+ json.dumps(my_json)+'\n'
                    content = json_data+'\n'+ json.dumps(my_json)
                    s3.Object('ml-flow-dump', filename).put(Body=content)
   
