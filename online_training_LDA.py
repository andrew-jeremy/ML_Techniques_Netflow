'''
netflow anomaly detection with LSTM autoencoder architectures
Andrew Kiruluta, Netography
'''
#from river_torch import anomaly
from river import metrics
from river import compose
from river import preprocessing
from river import feature_extraction
from torch import nn
import pandas as pd
import pickle
import boto3
import argparse
import statistics
from s3fs import S3FileSystem
from smart_open import open
import json
import uuid
from river import compose, preprocessing, metrics, datasets

#from river_torch.anomaly import RollingAutoencoder
from deep_river.anomaly import RollingAutoencoder
from river import feature_extraction
from torch import nn, manual_seed
import torch
from tqdm import tqdm
import argparse
import random
from feat_preprocessor import feat_eng

def Generate(): #function generates a random 6 digit number
    code = ''
    for i in range(6):
        code += str(random.randint(0,9))
    return code

'''
LSTM Autoencoder Architecture by Srivastava et al. 2016 
(https://arxiv.org/abs/1502.04681). Decoding is performed in 
reverse order to introduce short term dependencies between inputs 
and outputs. Additional to the encoding, the decoder gets fed the 
time-shifted original inputs.
'''
class LSTMAutoencoderSrivastava(nn.Module):
    def __init__(self, n_features, hidden_size=200, n_layers=10, batch_first=False):
        super().__init__()
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.batch_first = batch_first
        self.time_axis = 1 if batch_first else 0
        self.encoder = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=batch_first,
        )
        self.decoder = nn.LSTM(
            input_size=hidden_size,
            hidden_size=n_features,
            num_layers=n_layers,
            batch_first=batch_first,
        )

    def forward(self, x):
        _, (h, _) = self.encoder(x)
        h = h[-1].view(1, 1, -1)
        x_flipped = torch.flip(x[1:], dims=[self.time_axis])
        input = torch.cat((h, x_flipped), dim=self.time_axis)
        x_hat, _ = self.decoder(input)
        x_hat = torch.flip(x_hat, dims=[self.time_axis])

        return x_hat

'''
Architecture inspired by Cho et al. 2014 (https://arxiv.org/abs/1406.1078).
 Decoding occurs in natural order and the decoder is only provided with the 
 encoding at every timestep.
'''
class LSTMAutoencoderCho(nn.Module):
    def __init__(self, n_features, hidden_size=90, n_layers=1, batch_first=False):
        super().__init__()
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.batch_first = batch_first
        self.encoder = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=batch_first,
        )
        self.decoder = nn.LSTM(
            input_size=hidden_size,
            hidden_size=n_features,
            num_layers=n_layers,
            batch_first=batch_first,
        )

    def forward(self, x):
        _, (h, _) = self.encoder(x)
        target_shape = (
            (-1, x.shape[0], -1) if self.batch_first else (x.shape[0], -1, -1)
        )
        h = h[-1].expand(target_shape)
        x_hat, _ = self.decoder(h)
        return x_hat

'''
LSTM Encoder-Decoder architecture by Sutskever et al. 2014 (https://arxiv.org/abs/1409.3215). 
The decoder only gets access to its own prediction of the previous timestep. Decoding also 
takes performed backwards.
'''
class LSTMDecoder(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        sequence_length=None,
        predict_backward=True,
        num_layers=1,
    ):
        super().__init__()

        self.cell = nn.LSTMCell(input_size, hidden_size)
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.predict_backward = predict_backward
        self.sequence_length = sequence_length
        self.num_layers = num_layers
        self.lstm = (
            None
            if num_layers <= 1
            else nn.LSTM(
                input_size=hidden_size,
                hidden_size=hidden_size,
                num_layers=num_layers - 1,
            )
        )
        self.linear = (
            None if input_size == hidden_size else nn.Linear(hidden_size, input_size)
        )

    def forward(self, h, sequence_length=None):
        """Computes the forward pass.

        Parameters
        ----------
        x:
            Input of shape (batch_size, input_size)

        Returns
        -------
        Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
            Decoder outputs (output, (h, c)) where output has the shape (sequence_length, batch_size, input_size).
        """

        if sequence_length is None:
            sequence_length = self.sequence_length
        x_hat = torch.empty(sequence_length, h.shape[0], self.hidden_size)
        for t in range(sequence_length):
            if t == 0:
                h, c = self.cell(h)
            else:
                input = h if self.linear is None else self.linear(h)
                h, c = self.cell(input, (h, c))
            t_predicted = -t if self.predict_backward else t
            x_hat[t_predicted] = h

        if self.lstm is not None:
            x_hat = self.lstm(x_hat)

        return x_hat, (h, c)


class LSTMAutoencoderSutskever(nn.Module):
    def __init__(self, n_features, hidden_size=90, n_layers=1):
        super().__init__()
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.encoder = nn.LSTM(
            input_size=n_features, hidden_size=hidden_size, num_layers=n_layers
        )
        self.decoder = LSTMDecoder(
            input_size=hidden_size, hidden_size=n_features, predict_backward=True
        )

    def forward(self, x):
        _, (h, _) = self.encoder(x)
        x_hat, _ = self.decoder(h[-1], x.shape[0])
        return x_hat

def replace_boolean(data):
    for col in data:
        data[col].replace(True, 1, inplace=True)
        data[col].replace(False, 0, inplace=True)
    return data

if __name__ == "__main__":
    # Testing/Inference
    parser = argparse.ArgumentParser(description='online training')
    parser.add_argument('--encoder', type=int, default=1,help="1:LSTMAutoencoderSrivastava,\
        2: LSTMAutoencoderCho, 3: LSTMAutoencoderSutskever")
    parser.add_argument('--pretrained', type=bool, default=False,help='load pretrained model,\
         True: for Yes')
    parser.add_argument('--vect', type=int, default=1,help='vectorizer to use,\
         1: LDA, 2:TFDIF')
    args = parser.parse_args()

    '''
    1-LSTM Autoencoder Architecture by Srivastava et al. 2016 
    (https://arxiv.org/abs/1502.04681). Decoding is performed in 
    reverse order to introduce short term dependencies between 
    inputs and outputs. Additional to the encoding, the decoder 
    gets fed the time-shifted original inputs.

    2-Architecture inspired by Cho et al. 2014 (https://arxiv.org/abs/1406.1078).
     Decoding occurs in natural order and the decoder is only provided with 
     the encoding at every timestep.
    
    3-LSTM Encoder-Decoder architecture by Sutskever et al. 2014 
    (https://arxiv.org/abs/1409.3215). The decoder only gets access to its own 
    prediction of the previous timestep. Decoding also takes performed backwards.
    '''

    if args.vect == 1:
        lda = compose.Pipeline(feature_extraction.BagOfWords(),preprocessing.LDA(n_components=200,number_of_documents=60000,seed=42))
        vect = 'lda'   
    else:
        tfidf = feature_extraction.TFIDF()
        vect = 'tfidf'
    
    if args.encoder == 1:
        model = RollingAutoencoder(module= LSTMAutoencoderSrivastava, lr=0.05)
        tag = 'Sriva_' + vect # threshold set at 0.5
    elif args.encoder == 2:
        model = RollingAutoencoder(module= LSTMAutoencoderCho, lr=0.05)
        tag = 'Cho_' + vect 
    else:
        model = RollingAutoencoder(module=  LSTMAutoencoderSutskever, lr=0.05)
        tag = 'Sut_' + vect


    with open('data/anomalies_{}.txt'.format(tag), 'w') as document: pass  # initially create empty file
    fm = open('data/model_{}_2.pkl'.format(tag), 'w+b')
    fa =  open('data/anomalies_{}.txt'.format(tag), 'a')

    if args.pretrained:
        # start training from previous pretrained model
        model = pickle.load(open('data/model_ae_{}_bak.pkl'.format(tag), 'rb'))
    else:
        fm = open('data/model_{}_2.pkl'.format(tag), 'w+b')

    # open s3  netflow bucket data
    s3 = boto3.resource('s3')
    s3_file = S3FileSystem()
    s3_client = boto3.client("s3")
    bucket = s3.Bucket('ml-flow-dump')
    scores = []
    anomalies = 0
    data = []
    count = 1
    filename = "data/anomaly/vae_2/anomaly_{}".format(tag) +  ".json"
    for obj in bucket.objects.filter(Prefix="flow"):
        source_url = 's3://ml-flow-dump/' + obj.key
        print(source_url)
        for i,json_line in enumerate(open(source_url, transport_params={"client": s3_client})):
            my_json = json.loads(json_line)
            df = pd.json_normalize(my_json)  # one line at a time 
            x = feat_eng(df)

            #x = scaler.learn_one(x).transform_one(x)
            y = str(list(x.values())).replace("[","") # convert all values to a string to embed
            y = y.replace("]","")
            y = y.replace("'", '')
            if args.vect == 1:              # lda
                lda = lda.learn_one(y)
                x = lda.transform_one(y) 
            else:
                tfidf = tfidf.learn_one(y)  # https://riverml.xyz/0.14.0/api/feature-extraction/TFIDF/
                x = tfidf.transform_one(y)  
                print(len(y))  

            score = model.score_one(x)
            model.learn_one(x=x, y=None)

            #auc = auc.update(y, score)     # once you have labels y
            scores.append(score)
            count +=1
            if count % 1 == 0:
               print(score)
               #print(count)
            if score > 0.5 and count > 20000: # 10k training warmup
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
                    if anomalies % 1000 == 0:
                        filename = "data/anomaly/vae_2/anomaly_{}".format(tag) + Generate() + ".json"
                        content = json.dumps(my_json)
                        s3.Object('ml-flow-dump', filename).put(Body=content)
                    else:
                        resp=s3_client.get_object(Bucket="ml-flow-dump", Key=filename)
                        #data=resp.get('Body')
                        json_data = resp.get('Body').read().decode("utf-8")

                        # append new anomalies to s3 bucket
                        content = json_data+'\n'+ json.dumps(my_json)
                        s3.Object('ml-flow-dump', filename).put(Body=content)