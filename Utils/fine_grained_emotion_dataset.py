import csv
import numpy as np
import sys, os
from Utils.Representations import *
from Utils.WordVecs import *
sys.path.append('/home/jeremy/NS/Keep/Permanent/Tools/twitter_nlp/python/')
from twokenize import *

def rem_mentions_urls(tokens):
    final = []
    for t in tokens:
        if not t.startswith(r'@') and not t.startswith('http'):
            final.append(t)
    return final

def words(sentence, model):
    return rem_mentions_urls(tokenize(sentence.lower()))

class Fine_Grained_Emotion_Dataset():

    def __init__(self, DIR, model, one_hot=True,
                 dtype=np.float32, rep=ave_vecs):

        self.rep = rep
        self.one_hot = one_hot

        Xtrain, Xdev, Xtest, ytrain, ydev, ytest = self.open_data(DIR, model, rep)

        self._Xtrain = Xtrain
        self._ytrain = ytrain
        self._Xdev = Xdev
        self._ydev = ydev
        self._Xtest = Xtest
        self._ytest = ytest

    def open_data(self, DIR, model, rep):

        i = 0

        train = {}
        with open(os.path.join(DIR,'train.csv')) as csv_file:
            reader = csv.reader(csv_file, delimiter=';')
            for line in reader:
                try:
                    (tweet, sent, stance, target, op_towards, all_ann, voted_ann, num_ann) = line
                    train[i] = {'tweet':tweet,
                       'sentiment': sent,
                       'stance': stance,
                       'target': target,
                       'opinion_towards': op_towards,
                       'all_annotations': np.fromstring(all_ann.replace(r'[', '').replace(']','').replace(' ',''), dtype='int32', sep=';'),
                       'voted_annotation': np.fromstring(voted_ann.replace('None', '0').replace(r'[', '').replace(']','').replace(' ',''), dtype='int32', sep=';'),    
                       'number_of_annotators': num_ann
                       }
                    i += 1
                except:
                     pass

        i = 0

        test = {}
        with open(os.path.join(DIR,'test.csv')) as csv_file:
            reader = csv.reader(csv_file, delimiter=';')
            for line in reader:
                try:
                    (tweet, sent, stance, target, op_towards, all_ann, voted_ann, num_ann) = line
                    test[i] = {'tweet':tweet,
                       'sentiment': sent,
                       'stance': stance,
                       'target': target,
                       'opinion_towards': op_towards,
                       'all_annotations': np.fromstring(all_ann.replace(r'[', '').replace(']','').replace(' ',''), dtype='int32', sep=';'),
                       'voted_annotation': np.fromstring(voted_ann.replace('None', '0').replace(r'[', '').replace(']','').replace(' ',''), dtype='int32', sep=';'),    
                       'number_of_annotators': num_ann
                       }
                    i += 1
                except:
                     pass

        Xtrain = [tweet['tweet'] for tweet in train.values()]
        Xtrain  = [rep(sent, model) for sent in Xtrain]
        ytrain = [tweet['voted_annotation'] for tweet in train.values()]

        dev_idx = int(len(Xtrain) * .1)
        Xdev = Xtrain[:dev_idx]
        ydev = ytrain[:dev_idx]

        Xtrain = Xtrain[dev_idx:]
        ytrain = ytrain[dev_idx:]

        Xtest = [tweet['tweet'] for tweet in test.values()]
        Xtest  = [rep(sent, model) for sent in Xtest]
        ytest = [tweet['voted_annotation'] for tweet in test.values()]


        if self.rep is not words:
            Xtrain = np.array(Xtrain)
            Xdev = np.array(Xdev)
            Xtest = np.array(Xtest)

        ytrain = np.array(ytrain)
        ydev = np.array(ydev)
        ytest = np.array(ytest)
        
        return Xtrain, Xdev, Xtest, ytrain, ydev, ytest

