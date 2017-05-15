from keras.models import Sequential, Model
from keras.layers import LSTM, Dropout, Dense, Embedding, Bidirectional, Convolution1D, MaxPooling1D, Flatten, Merge, Input
from keras.preprocessing.sequence import pad_sequences
import sys
import argparse
from stance_dataset import *
from Utils.MyMetrics import *
from Utils.Representations import *
from Utils.WordVecs import *

def plot_val(h):
    pass

def get_idx_from_sent(sent, word_idx_map, max_l=50, k=50, filter_h=5):
    """
    Transforms sentence into a list of indices, padded with zeros.
    """
    x = []
    pad = filter_h-1
    for i in range(pad):
        x.append(0)
    words = sent.split()
    for word in words:
        if word in word_idx_map:
            x.append(word_idx_map[word])
    while len(x) < max_l + 2 * pad:
        x.append(0)
    return x
    

def add_unknown_words(wordvecs, vocab, min_df=1, dim=50):
    """
    For words that occur at least min_df, create a separate word vector
    0.25 is chosen so the unk vectors have approximately the same variance
    as pretrained ones
    """
    for word in vocab:
        if word not in wordvecs and vocab[word] >= min_df:
            wordvecs[word] = np.random.uniform(-0.25, 0.25, dim)


def get_W(wordvecs, dim=300):
    """
    Get word matrix. W[i] is the vector for word indexed by i
    """
    vocab_size = len(wordvecs)
    word_idx_map = dict()
    W = np.zeros(shape=(vocab_size+1, dim), dtype='float32')
    W[0] = np.zeros(dim, dtype='float32')
    i = 1
    for word in wordvecs:
        W[i] = wordvecs[word]
        word_idx_map[word] = i
        i += 1
    return W, word_idx_map

def create_deep(input_dim, num_hidden_layers=3, dim=300,
                output_dim=2, dropout=.5):
    model = Sequential()
    model.add(Dense(input_dim=input_dim, output_dim=dim))
    model.add(Dropout(dropout))
    for i in range(num_hidden_layers):
        model.add(Dense(dim))
        model.add(Dropout(dropout))
    model.add(Dense(output_dim, activation='sigmoid'))
    model.compile('adam', 'binary_crossentropy',
                  metrics=['accuracy'])
    return model


def create_LSTM(wordvecs, lstm_dim=300, output_dim=2, dropout=.5,
                weights=None, train=True):
    model = Sequential()
    model.add(Embedding(W.shape[0], W.shape[1], weights=[W], trainable=True))
    model.add(Dropout(.3))
    model.add(LSTM(100))
    model.add(Dropout(.3))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(8, activation='sigmoid'))
    model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])
    return model

def create_BiLSTM(wordvecs, lstm_dim=300, output_dim=2, dropout=.5,
                weights=None, train=True):
    model = Sequential()
    model.add(Embedding(W.shape[0], W.shape[1], weights=[W], trainable=True))
    model.add(Dropout(.3))
    model.add(Bidirectional(LSTM(100)))
    model.add(Dropout(.3))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(8, activation='sigmoid'))
    model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])
    return model

def create_cnn(W, max_length):

    # Convolutional model
    filter_sizes=(2,3,4)
    num_filters = 3
    dropout_prob=(.25,.5)
    hidden_dim=100
    graph_in = Input(shape=(max_length,vecs.vector_size))
    convs = []
    for fsz in filter_sizes:
        conv = Convolution1D(nb_filter=num_filters,
			     filter_length=fsz,
			     border_mode='valid',
			     activation='relu',
			     subsample_length=1)(graph_in)
        pool = MaxPooling1D(pool_length=2)(conv)
        flatten = Flatten()(pool)
        convs.append(flatten)
        
    out = Merge(mode='concat')(convs)
    graph = Model(input=graph_in, output=out)

    # Full model
    model = Sequential()
    model.add(Embedding(output_dim=W.shape[1],
                        input_dim=W.shape[0],
                        input_length=max_length, weights=[W],
                        trainable=True))
    model.add(Dropout(0.25))
    model.add(graph)
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(.3))
    model.add(Dense(8, activation='sigmoid'))
    model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])

    return model

def cutoff(x, c=.5):
    """
    Decide where the cutoff for probability is for predicting a positive
    class.
    """
    z = []
    for i in x:
        if i > c:
            z.append(1)
        else:
            z.append(0)
    return np.array(z)

def pad_dataset(dataset, maxlen=50):
    dataset._Xtrain = pad_sequences(dataset._Xtrain, maxlen)
    dataset._Xdev = pad_sequences(dataset._Xdev, maxlen)
    dataset._Xtest = pad_sequences(dataset._Xtest, maxlen)

def write_vecs(matrix, w2idx, outfile):
    vocab = sorted(w2idx.keys())
    with open(outfile, 'w') as out:
        for w in vocab:
            try:
                out.write(w + ' ')
                v = matrix[w2idx[w]]
                for j in v:
                    out.write('{0:.7}'.format(j))
                out.write('\n')
            except UnicodeEncodeError:
                pass


def lstm_dev_experiment(Xdev, ydev, params=[]):
	pass


if __name__ == '__main__':

    names = ['DAN', 'LSTM', 'Bi-LSTM']
    emotions = ["anger", "anticipation", "disgust",
            "fear", "joy", "sadness",
            "surprise", "trust"]
    
    print('Importing vectors...')
    vecs = WordVecs('../sentiment_retrofitting/embeddings/wikipedia-sg-100-window10-sample1e-4-negative5.txt')

    # Deep Averaging Network
     
    dataset = Stance_Dataset('data', vecs, rep=ave_vecs)

    print('Basic statistics')
    for i, emo in enumerate(emotions):
        train = dataset._ytrain[:,i].sum()
        test = dataset._ytest[:,i].sum()
        print('{0} train:{1} test:{2}'.format(emo, train, test))
    print()
	
    print('Training DAN')
    dan = create_deep(input_dim=vecs.vector_size,
                      num_hidden_layers=3,
                      dim=300,
                      output_dim=dataset._ytrain.shape[1],
                      dropout=.3)

    h = dan.fit(dataset._Xtrain, dataset._ytrain, validation_split=.1, verbose=0)
    pred = dan.predict(dataset._Xtest)
    pred = np.array([cutoff(x) for x in pred])
    y = dataset._ytest

    results = []
    for i in range(len(emotions)):
        emo_y = y[:,i]
        emo_pred = pred[:,i]
        mm = MyMetrics(emo_y, emo_pred, one_hot=False, average='binary')
        acc = mm.accuracy()
        precision, recall, f1 = mm.get_scores()
        results.append([acc, precision, recall, f1])
        
    for emo, result in zip(emotions, results):
        a,p,r,f = result
        print('{0}: {1:.3f}'.format(emo, f))
    r = np.array(results)
    ave_acc = r[:,0].mean()
    ave_prec = r[:,1].mean()
    ave_rec = r[:,2].mean()
    ave_f1 = r[:,3].mean()
    print('acc: {0:.3f} prec:{1:.3f} rec:{2:.3f} f1:{3:.3f}'.format(ave_acc,
                                                     ave_prec,
                                                     ave_rec,
                                                     ave_f1))
    print()

    
    

    # LSTM
    print('Preparing LSTM data...')
    dataset = Stance_Dataset('data', vecs._w2idx, rep=words)

    dim = vecs.vector_size

    max_length = 0
    vocab = {}
    for sent in list(dataset._Xtrain) + list(dataset._Xtest):
        if len(sent) > max_length:
            max_length = len(sent)
        for w in sent:
            if w not in vocab:
                vocab[w] = 1
            else:
                vocab[w] += 1

    wordvecs = {}
    for w in vecs._w2idx.keys():
        if w in vocab:
            wordvecs[w] = vecs[w]

    add_unknown_words(wordvecs, vocab, min_df=1, dim=dim)
    W, word_idx_map = get_W(wordvecs, dim=dim)

    print('Converting dataset to LSTM format...')

    dataset._Xtrain = np.array([get_idx_from_sent(' '.join(sent), word_idx_map, max_l=max_length, k=dim)
                                    for sent in dataset._Xtrain])
    dataset._Xtest = np.array([get_idx_from_sent(' '.join(sent), word_idx_map, max_l=max_length, k=dim)
                                    for sent in dataset._Xtest])

    
    print('Training LSTM...')
    lstm = create_LSTM(wordvecs, lstm_dim=200, output_dim=8, dropout=.3,
                      weights=W, train=True)
    h = lstm.fit(dataset._Xtrain, dataset._ytrain, validation_split=.1,
                 nb_epoch=50)
    pred = lstm.predict(dataset._Xtest)
    pred = np.array([cutoff(x) for x in pred])
    y = dataset._ytest

    results = []
    for i in range(len(emotions)):
        emo_y = y[:,i]
        emo_pred = pred[:,i]
        mm = MyMetrics(emo_y, emo_pred, one_hot=False, average='binary')
        acc = mm.accuracy()
        precision, recall, f1 = mm.get_scores()
        results.append([acc, precision, recall, f1])
        
    for emo, result in zip(emotions, results):
        a,p,r,f = result
        print('{0}: {1:.3f}'.format(emo, f))
    r = np.array(results)
    ave_acc = r[:,0].mean()
    ave_prec = r[:,1].mean()
    ave_rec = r[:,2].mean()
    ave_f1 = r[:,3].mean()
    print('acc: {0:.3f} prec:{1:.3f} rec:{2:.3f} f1:{3:.3f}'.format(ave_acc,
                                                     ave_prec,
                                                     ave_rec,
                                                     ave_f1))
    print()

    # Bi-LSTM

    print('Training bi-lstm...')
    bilstm = create_BiLSTM(wordvecs, lstm_dim=200, output_dim=8, dropout=.3,
                        weights=W, train=True)
    h = bilstm.fit(dataset._Xtrain, dataset._ytrain, validation_split=.1,
                   nb_epoch=50)
    pred = bilstm.predict(dataset._Xtest)
    pred = np.array([cutoff(x) for x in pred])
    y = dataset._ytest

    results = []
    for i in range(len(emotions)):
        emo_y = y[:,i]
        emo_pred = pred[:,i]
        mm = MyMetrics(emo_y, emo_pred, one_hot=False, average='binary')
        acc = mm.accuracy()
        precision, recall, f1 = mm.get_scores()
        results.append([acc, precision, recall, f1])
        
    for emo, result in zip(emotions, results):
        a,p,r,f = result
        print('{0}: {1:.3f}'.format(emo, f))
    r = np.array(results)
    ave_acc = r[:,0].mean()
    ave_prec = r[:,1].mean()
    ave_rec = r[:,2].mean()
    ave_f1 = r[:,3].mean()
    print('acc: {0:.3f} prec:{1:.3f} rec:{2:.3f} f1:{3:.3f}'.format(ave_acc,
                                                     ave_prec,
                                                     ave_rec,
                                                     ave_f1))
    
    # CNN

    cnn = create_cnn(W, dataset._Xtrain.shape[1])
    h = cnn.fit(dataset._Xtrain, dataset._ytrain, validation_split=.1,
                   nb_epoch=50)
    pred = cnn.predict(dataset._Xtest)
    pred = np.array([cutoff(x) for x in pred])
    y = dataset._ytest

    results = []
    for i in range(len(emotions)):
        emo_y = y[:,i]
        emo_pred = pred[:,i]
        mm = MyMetrics(emo_y, emo_pred, one_hot=False, average='binary')
        acc = mm.accuracy()
        precision, recall, f1 = mm.get_scores()
        results.append([acc, precision, recall, f1])
        
    for emo, result in zip(emotions, results):
        a,p,r,f = result
        print('{0}: {1:.3f}'.format(emo, f))
    r = np.array(results)
    ave_acc = r[:,0].mean()
    ave_prec = r[:,1].mean()
    ave_rec = r[:,2].mean()
    ave_f1 = r[:,3].mean()
    print('acc: {0:.3f} prec:{1:.3f} rec:{2:.3f} f1:{3:.3f}'.format(ave_acc,
                                                     ave_prec,
                                                     ave_rec,
                                                     ave_f1))

    
    
    
