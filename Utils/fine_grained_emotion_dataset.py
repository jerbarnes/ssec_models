import csv
import numpy as np
import sys, os
from Utils.Representations import *
from Utils.WordVecs import *
from Utils.twokenize import *

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
                 dtype=np.float32, rep=ave_vecs, threshold=0.0):

        self.rep = rep
        self.one_hot = one_hot
        self.threshold = threshold

        Xtrain, Xdev, Xtest, ytrain, ydev, ytest = self.open_data(DIR, model, rep)

        self._Xtrain = Xtrain
        self._ytrain = ytrain
        self._Xdev = Xdev
        self._ydev = ydev
        self._Xtest = Xtest
        self._ytest = ytest

    def open_data(self, DIR, model, rep):

        Xtrain = []
        with open(os.path.join(DIR,'train.csv')) as csv_file:
            reader = csv.reader(csv_file, delimiter='\t')
            for i, line in enumerate(reader):
              # don't get header
              if i != 0:
                Xtrain.append(line[0])
                  

        Xtest = []
        with open(os.path.join(DIR,'test.csv')) as csv_file:
            reader = csv.reader(csv_file, delimiter='\t')
            for i, line in enumerate(reader):
                # don't get header
                if i != 0:
                  Xtest.append(line[0])

        # get Ys
        Y = []
        try:
          for line in open(os.path.join(DIR, 'vote{0}'.format(self.threshold))):
            Y.append(np.array(line.split('\t'), dtype=int))
        except FileNotFoundError:
          print('Annotation File not found')

        Y = np.array(Y)
            


        Xtrain  = [rep(sent, model) for sent in Xtrain]
        ytrain = Y[:len(Xtrain)] #
        ytest = Y[len(Xtrain):]

        dev_idx = int(len(Xtrain) * .1)
        Xdev = Xtrain[:dev_idx]
        ydev = ytrain[:dev_idx]

        Xtrain = Xtrain[dev_idx:]
        ytrain = ytrain[dev_idx:]

        Xtest  = [rep(sent, model) for sent in Xtest]
        

        assert len(Xtrain) == len(ytrain)
        assert len(Xdev) == len(Xdev)
        assert len(Xtest) == len(ytest)



        if self.rep is not words:
            Xtrain = np.array(Xtrain)
            Xdev = np.array(Xdev)
            Xtest = np.array(Xtest)

        ytrain = np.array(ytrain)
        ydev = np.array(ydev)
        ytest = np.array(ytest)
        
        return Xtrain, Xdev, Xtest, ytrain, ydev, ytest




