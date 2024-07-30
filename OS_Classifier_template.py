'''
# Tree Based Classifier for Netflow Application Classification - Online Learning Version
Andrew Kiruluta, Netography 2022
'''
from river.datasets import synth
from river import evaluate
from river import metrics
from river import tree
from river import forest
from river import compose
import pandas as pd
import numpy as np
from sklearn import preprocessing as preprocess
from river import compose, preprocessing, metrics, stats
import argparse
import numbers
import pickle
from statistics import mean
from sklearn.model_selection import train_test_split
import joblib
import torch

def get_pipeline(model):
    #mean = stats.Mean()
    #mode = stats.Mode()
    cat = (
        compose.SelectType(str)
        | preprocessing.StatImputer()
        | preprocessing.OneHotEncoder(sparse=True)
    )
    num = compose.SelectType(numbers.Number) | preprocessing.StatImputer() | preprocessing.StandardScaler(stats.Mean())
    processor = num + cat
    return processor | model

if __name__ == "__main__":
    # Testing/Inference
    parser = argparse.ArgumentParser(description='online training for os classification')
    parser.add_argument('--train', type=bool, default=True, help='train model,\
         True: for Yes') 
    parser.add_argument('--model', type=int, default=5,help="1:ExtremelyFastDecisionTreeClassifier,\
        2: HoeffdingTreeClassifier, 3: HoeffdingAdaptiveTreeRegressor,\
        4:forest.ARFClassifier,5:forest.AMFClassifier")
    parser.add_argument('--pretrained', type=bool, default=False,help='load pretrained model,\
         True: for Yes') 
    args = parser.parse_args()

    if args.model == 1:
        tag = "ExtremelyFastDecisionTreeClassifier"
        model = tree.ExtremelyFastDecisionTreeClassifier(grace_period=100,delta=1e-5, \
        nominal_attributes=['Source.Port','Destination.Port','Protocol','L7Protocol'],\
        remove_poor_attrs = True,
        max_depth = 10,
        min_samples_reevaluate=100)
    elif args.model == 2:
        tag = "HoeffdingTreeClassifier"
        model = tree.HoeffdingTreeClassifier(\
                grace_period=100,delta=1e-5,\
                nominal_attributes=['Source.Port','Destination.Port','Protocol','L7Protocol'])
    elif args.model == 3:
        tag = "HoeffdingAdaptiveTreeRegressor"
        model = tree.HoeffdingAdaptiveTreeRegressor(\
                grace_period=50,model_selector_decay=0.3,seed=0,\
                nominal_attributes=['Source.Port','Destination.Port','Protocol','L7Protocol'])
    elif args.model == 4:
        tag = "ARFClassifier"
        model = forest.ARFClassifier(seed=8, leaf_prediction="mc")
    else:
        tag = "AMFClassifier"
        model = forest.AMFClassifier(n_estimators=10,use_aggregation=True,\
        dirichlet=0.5,seed=1)

    pipeline = get_pipeline(model)
    le = preprocess.LabelEncoder()
    metric_acc = metrics.Accuracy()
    metric_f1 = metrics.F1()

    if args.train:
        dataset = pd.read_csv("./KaggleImbalanced.csv")
        dataset['target'] = le.fit_transform(dataset['ProtocolName'])

        # -----> test point ,----
        # extract small balanced dataset for debugging
        #dataset = dataset.groupby('target').sample(n=20)
        #------>

        # train-test split
        #X = dataset.drop(['ProtocolName','Source.IP','Destination.IP'],axis=1)
        X_train, X_test, y_train, y_test = train_test_split(dataset, dataset.target, test_size=0.1,  random_state=42)
        
        # sort data to group target since online training learns only one sample at a time
        X_train = X_train.sort_values('target')
        y_train = X_train.target
        
        # without src & dst ip addresses:
        X_train = X_train.drop(['ProtocolName','Source.IP','Destination.IP','target'],axis=1).to_dict('records')
        X_test = X_test.drop(['ProtocolName','Source.IP','Destination.IP','target'],axis=1).to_dict('records')
        
        # with src & dst ip addresses:
        #X_train = X_train.drop(['ProtocolName','target'],axis=1).to_dict('records')
        #X_test = X_test.drop(['ProtocolName','target'],axis=1).to_dict('records')
    
    #if args.train:
        print("==>model training<==")
        pred = []
        tru = []
        acc = []
        f1 = []
        count = 1

        for count, (x,y) in enumerate(zip(X_train,y_train)):
            y_pred = pipeline.predict_one(x)
            acc.append(metric_acc.update(y, y_pred).get())  # update the metric
            pipeline.learn_one(x,y)
        
            if count > 0: 
                tru.append(y)
                pred.append(y_pred)
                if count % 100 == 0:
                    print(y, y_pred)
                    y_prob = pipeline.predict_proba_one(x)
                    print(round(y_prob[y_pred],2))
                    print(count)
            count +=1
        print(f'train accuracy: {mean(acc)}')

        f1 = []
        for yt, yp in zip(y_train, pred):
            f1.append(metric_f1.update(yt, yp).get())
        print(f'train F1 score: {mean(f1)}\n')
        

        #====> test on held out data <====
        pred = []
        tru = []
        f1 = []
        acc = []
        predictions = {}
        for i, (x, y) in enumerate(zip(X_test,y_test)):
            y_pred = pipeline.predict_one(x)
            y_prob = pipeline.predict_proba_one(x)
            tru.append(y)
            pred.append(y_pred)
            predictions[i] = round(y_prob[y_pred],2)
            print(round(y_prob[y_pred],2))
            if y_prob[y_pred] > 0.95:
                pipeline.learn_one(x,y) # update pipeline

            print(round(y_prob[y_pred],2))
            f1.append(metric_f1.update(y, y_pred).get())
            acc.append(metric_acc.update(y, y_pred).get())
        print(f'test F1 score: {mean(f1)}')
        print(f'test accuracy: {mean(acc)}\n')
