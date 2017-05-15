import csv
import numpy as np
import sys, os
from Utils.Representations import *
from Utils.WordVecs import *
sys.path.append('/home/jeremy/NS/Keep/Permanent/Tools/twitter_nlp/python/')
from twokenize import *


def words(sentence, model):
    return rem_mentions_urls(tokenize(sentence.lower()))

def rem_mentions_urls(tokens):
    final = []
    for t in tokens:
        if not t.startswith(r'@') and not t.startswith('http'):
            final.append(t)
    return final

class Twitter_Dataset():

    def __init__(self, DIR, model, one_hot=True,
                 dtype=np.float32, rep=ave_vecs):

        self.rep = rep
        self.one_hot = one_hot

        Xtrain, Xtest, ytrain,  ytest = self.open_data(DIR, model, rep)

        self._Xtrain = Xtrain
        self._ytrain = ytrain
        self._Xtest = Xtest
        self._ytest = ytest

    def to_array(self, y, N):
        '''
        converts an integer-based class into a one-hot array
        y = the class integer
        N = the number of classes
        '''
        return np.eye(N)[y]

    def convert_ys(self, y):
        dic = {'negative':0,
               'positive':1,
               'neutral' :2,
               }
        return dic[y]

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
        ytrain = [self.convert_ys(tweet['sentiment']) for tweet in train.values()]

        Xtest = [tweet['tweet'] for tweet in test.values()]
        Xtest  = [rep(sent, model) for sent in Xtest]
        ytest = [self.convert_ys(tweet['sentiment']) for tweet in test.values()]

        if self.one_hot:
            ytrain = [self.to_array(y, 3) for y in ytrain]
            ytet = [self.to_array(y,3) for y in ytest]


        if self.rep is not words:
            Xtrain = np.array(Xtrain)
            Xtest = np.array(Xtest)

        ytrain = np.array(ytrain)
        ytest = np.array(ytest)
        
        return Xtrain, Xtest, ytrain, ytest
