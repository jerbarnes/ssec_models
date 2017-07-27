from keras.models import Sequential, Model
from keras.layers import LSTM, Dropout, Dense, Embedding, Bidirectional, Convolution1D, MaxPooling1D, Flatten, Merge, Input
from keras.regularizers import l2
from keras.preprocessing.sequence import pad_sequences
import sys
import tabulate
import argparse
import json
from Utils.fine_grained_emotion_dataset import *
from Utils.MyMetrics import *
from Utils.Representations import *
from Utils.WordVecs import *


def get_dev_params(model_name, outfile,
                   Xtrain, ytrain, Xdev, ydev, wordvecs, W):
    
    # If you have already run the dev experiment, just get results
    if os.path.isfile(outfile):
        with open(outfile) as out:
            dev_results = json.load(out)
            if model_name in dev_results:
                f1 = dev_results[model_name]['f1']
                dim = dev_results[model_name]['dim']
                dropout = dev_results[model_name]['dropout']
                epoch = dev_results[model_name]['epoch']
                return dim, dropout, epoch, f1



    # Otherwise, run a test on the dev set to get the best parameters
    best_f1 = 0
    best_dim = 0
    best_dropout = 0
    best_epoch = 0

    output_dim = ytrain.shape[1]
    labels = sorted(set(ytrain.argmax(1)))

    dims = np.arange(50, 300, 25)
    dropouts = np.arange(0.1, 0.6, 0.1)
    epochs = np.arange(3, 25)

    # Do a random search over the parameters
    for i in range(10):

        dim = int(dims[np.random.randint(0, len(dims))])
        dropout = float(dropouts[np.random.randint(0, len(dropouts))])
        epoch = int(epochs[np.random.randint(0, len(epochs))])

        if model_name == 'DAN':
            clf = create_deep(input_dim=len(W[0]),
                                    num_hidden_layers=3,
                                    dim=dim,
                                    output_dim=ytrain.shape[1],
                                    dropout=dropout)
        elif model_name == 'LSTM':
            clf = create_LSTM(wordvecs, dim=dim, output_dim=8, dropout=dropout,
                          weights=W, train=True)
        elif model_name == 'BiLSTM':
            clf = create_BiLSTM(wordvecs, dim=dim, output_dim=8, dropout=dropout,
                            weights=W, train=True)
        elif model_name == 'CNN':
            clf = create_cnn(W, Xtrain.shape[1], dim=dim, dropout=dropout)

        h = clf.fit(Xtrain, ytrain, nb_epoch=epoch, verbose=0)
        pred = clf.predict(Xdev, verbose=0)
        if len(labels) == 2:
            mm = MyMetrics(ydev, pred, one_hot=True, labels=labels, average='binary')
            _, _, dev_f1 = mm.get_scores()
        else:
            mm = MyMetrics(ydev, pred, one_hot=True, labels=labels, average='micro')
            _, _, dev_f1 = mm.get_scores()
        if dev_f1 > best_f1:
            best_f1 = dev_f1
            best_dim = dim
            best_dropout = dropout
            best_epoch = epoch
        print('new best f1: {0:.3f} dim:{1} dropout:{2} epochs:{3}'.format(best_f1, dim, dropout, epoch))

        if os.path.isfile(outfile):
            with open(outfile) as out:
                dev_results = json.load(out)
                dev_results[model_name] = {'f1': best_f1,
                         'dim': best_dim,
                         'dropout': best_dropout,
                         'epoch': best_epoch}
            with open(outfile, 'w') as out:
                json.dump(dev_results, out)

        else:
            dev_results = {}
            dev_results[model_name] = {'f1': best_f1,
                         'dim': best_dim,
                         'dropout': best_dropout,
                         'epoch': best_epoch}
            with open(outfile, 'w') as out:
                json.dump(dev_results, out)

    return best_dim, best_dropout, best_epoch, best_f1

def macro_f1(y, pred):
    precisions, recalls = [], []
    num_labels = y.shape[1]
    for emo in range(num_labels):
        tp = 0
        fp = 0
        fn = 0
        for i, j in enumerate(y[:,emo]):
            if j == 1 and pred[:,emo][i] == 1:
                tp += 1
            elif j == 1 and pred[:,emo][i] == 0:
                fn += 1
            elif j == 0 and pred[:,emo][i] == 1:
                fp += 1
        try:
            pr = tp / (tp + fp)
        except ZeroDivisionError:
            pr = 0
        try:
            rc = tp / (tp + fn)
        except ZeroDivisionError:
            rc = 0
        precisions.append(pr)
        recalls.append(rc)
    precisions = np.array(precisions)
    recalls = np.array(precisions)
    macro_precision = precisions.mean()
    macro_recall = recalls.mean()
    macro_f1 = 2 * ((macro_precision * macro_recall) / (macro_precision + macro_recall))
    return macro_f1

def micro_f1(y, pred):
    true_pos, false_pos, false_neg = [], [], []
    num_labels = y.shape[1]
    for emo in range(num_labels):
        tp = 0
        fp = 0
        fn = 0
        for i, j in enumerate(y[:,emo]):
            if j == 1 and pred[:,emo][i] == 1:
                tp += 1
            elif j == 1 and pred[:,emo][i] == 0:
                fn += 1
            elif j == 0 and pred[:,emo][i] == 1:
                fp += 1
        true_pos.append(tp)
        false_pos.append(fp)
        false_neg.append(fn)
        
    true_pos = np.array(true_pos)
    false_pos = np.array(false_pos)
    false_neg = np.array(false_neg)
    micro_precision = true_pos.sum() / (true_pos.sum() + false_pos.sum())
    micro_recall = true_pos.sum() / (true_pos.sum() + false_neg.sum())
    micro_f1 = 2 * ((micro_precision * micro_recall) / (micro_precision + micro_recall))
    return micro_precision, micro_recall, micro_f1    

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


def create_LSTM(wordvecs, dim=300, output_dim=8, dropout=.5,
                weights=None, train=True):
    model = Sequential()
    model.add(Embedding(weights.shape[0], weights.shape[1], weights=[weights], trainable=train))
    model.add(Dropout(dropout))
    model.add(LSTM(dim))
    #model.add(LSTM(dim, W_regularizer=l2()))
    model.add(Dropout(dropout))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(output_dim, activation='sigmoid'))
    model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])
    return model

def create_BiLSTM(wordvecs, dim=300, output_dim=8, dropout=.5,
                weights=None, train=True):
    model = Sequential()
    model.add(Embedding(weights.shape[0], weights.shape[1], weights=[weights], trainable=train))
    model.add(Dropout(dropout))
    model.add(Bidirectional(LSTM(dim)))
    #model.add(Bidirectional(LSTM(dim, regularizer=l2())))
    model.add(Dropout(dropout))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(output_dim, activation='sigmoid'))
    model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])
    return model

def create_cnn(W, max_length, dim=300,
               dropout=.5, output_dim=8):

    # Convolutional model
    filter_sizes=(2,3,4)
    num_filters = 3
    hidden_dim=100
    graph_in = Input(shape=(max_length, len(W[0])))
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
    model.add(Dropout(dropout))
    model.add(graph)
    model.add(Dense(dim, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(output_dim, activation='sigmoid'))
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


def test_embeddings(file, threshold, file_type):
    emotions = ["anger", "anticipation", "disgust",
            "fear", "joy", "sadness",
            "surprise", "trust"]
    
    
    
    # Import dataset where each test example is the words in the tweet
    dataset = Fine_Grained_Emotion_Dataset('data', None, rep=words, threshold=threshold)
    

    print('Basic statistics')
    table = []
    for i, emo in enumerate(emotions):
        train = dataset._ytrain[:,i].sum()
        test = dataset._ytest[:,i].sum()
        table.append((emo, train, test))
    print(tabulate.tabulate(table, headers=['emotion', '#train', '#test']))

#### Get Parameters ####
    max_length = 0
    vocab = {}
    for sent in list(dataset._Xtrain) + list(dataset._Xdev) + list(dataset._Xtest):
        if len(sent) > max_length:
            max_length = len(sent)
        for w in sent:
            if w not in vocab:
                vocab[w] = 1
            else:
                vocab[w] += 1

    wordvecs = {}

    print('Importing vectors')
    for line in open(file):
        try:
            split = line.split()
            word = split[0]
            vec = np.array(split[1:], dtype='float32')
            if word in vocab:
                wordvecs[word] = vec
        except ValueError:
            pass

    dim = len(vec)

    oov = len(vocab) - len(wordvecs)
    print('OOV: {0}'.format(oov))
    

    # Add vectors for <unk>
    add_unknown_words(wordvecs, vocab, min_df=1, dim=dim)
    W, word_idx_map = get_W(wordvecs, dim=dim)

    # TODO: change this so I don't have to import vectors I don't need
    vecs = WordVecs(file)
    vecs._matrix = W
    vecs._w2idx = word_idx_map
    vecs.vocab_length, vecs.vector_size = W.shape

    ave_dataset = Fine_Grained_Emotion_Dataset('data', vecs, rep=ave_vecs)

    # Get padded word indexes for all X
    Xtrain = np.array([get_idx_from_sent(' '.join(sent), word_idx_map, max_l=max_length, k=dim)
                                    for sent in dataset._Xtrain])
    Xdev = np.array([get_idx_from_sent(' '.join(sent), word_idx_map, max_l=max_length, k=dim)
                                    for sent in dataset._Xdev])
    Xtest = np.array([get_idx_from_sent(' '.join(sent), word_idx_map, max_l=max_length, k=dim)
                                    for sent in dataset._Xtest])

#### Test Models ####

    names = ['LSTM', 'BiLSTM', 'CNN']


    # Keep all mean and standard deviations of each emotion over datasets here
    all_emo_results = []
    all_emo_std_devs = []

    # Keep all mean and standard deviations of the averaged emotions here
    averaged_results = []
    averaged_std_devs = []

    # TEST EACH MODEL
    for name in names:

        print('Getting best parameters')

        dev_params_file = 'dev_params/'+str(W.shape[1])+'_params.txt'
        best_dim, best_dropout, best_epoch, best_f1 = get_dev_params(name, dev_params_file,
                                                    Xtrain, dataset._ytrain, Xdev, dataset._ydev, wordvecs, W)
        
        print('Testing {0}'.format(name))

        # Keep the results for the 5 runs over the dataset
        model_results = []
        model_average_results = []

        # 5 runs to get average and standard deviation
        for i, it in enumerate(range(5)):
            print('Run: {0}'.format(i + 1))
            
            # create and train a new classifier for each iteration
            if name == 'LSTM':
                model = create_LSTM(wordvecs, dim=best_dim, output_dim=8, dropout=best_dropout,
                          weights=W, train=True)
            elif name == 'BiLSTM':
                model = create_BiLSTM(wordvecs, dim=best_dim, output_dim=8, dropout=best_dropout,
                            weights=W, train=True)
            elif name == 'CNN':
                model = create_cnn(W, Xtrain.shape[1])

            h = model.fit(Xtrain, dataset._ytrain,
                              validation_data=[Xdev, dataset._ydev],
                              nb_epoch=best_epoch,
                              verbose=0)
            pred = model.predict(Xtest)
                    
            pred = np.array([cutoff(x) for x in pred])
            y = dataset._ytest

            emo_results = []
            for j in range(len(emotions)):
                emo_y = y[:,j]
                emo_pred = pred[:,j]
                mm = MyMetrics(emo_y, emo_pred, one_hot=False, average='binary')
                acc = mm.accuracy()
                precision, recall, f1 = mm.get_scores()
                emo_results.append([acc, precision, recall, f1])

            emo_results = np.array(emo_results)
            model_results.append(emo_results)

            # print('F1 scores')
            # for emo, result in zip(emotions, emo_results):
            #    a, p, r, f = result
            #    print('{0}: {1:.3f}'.format(emo, f))
            ave_acc, ave_prec, ave_rec, mac_f1 = emo_results.mean(axis=0)
            mic_prec, mic_rec, mic_f1 = micro_f1(dataset._ytest, pred)
            model_average_results.append((ave_acc, mic_prec, mic_rec, mic_f1))
            
            print('acc: {0:.3f} micro-prec:{1:.3f} micro-rec:{2:.3f} micro-f1:{3:.3f}'.format(ave_acc,
                                                             mic_prec,
                                                             mic_rec,
                                                             mic_f1))
            print()
    
        model_results = np.array(model_results)
        model_average_results = np.array(model_average_results)
        average_model_results = model_results.mean(axis=0)
        model_std_dev_results = model_results.std(axis=0)
        overall_avg = model_average_results.mean(axis=0)
        overall_std = model_average_results.std(axis=0)

        all_emo_results.append(average_model_results)
        all_emo_std_devs.append(model_std_dev_results)

        averaged_results.append(overall_avg)
        averaged_std_devs.append(overall_std)


    return names, all_emo_results, all_emo_std_devs, averaged_results, averaged_std_devs, dim

def print_results(file, out_file, threshold, file_type):

    names, all_emo_results, all_emo_std_devs, averaged_results, averaged_std_devs, dim = test_embeddings(file, threshold, file_type)

    emotions = ["anger", "anticipation", "disgust",
            "fear", "joy", "sadness",
            "surprise", "trust", "micro-averaged"]
    
    
    if out_file:
        with open(out_file, 'a') as f:
            for name, results, std_devs, ave_results, ave_std_dev in zip(names, all_emo_results, all_emo_std_devs, averaged_results, averaged_std_devs):
                rr = [[u'{0:.3f} \u00B1{1:.3f}'.format(r, s) for r, s in zip(result, std_dev)] for result, std_dev in zip(results, std_devs)]
                av = [u'{0:.3f} \u00B1{1:.3f}'.format(r, s) for r, s in zip(ave_results, ave_std_dev)]
                rr += [av]
                table = [[emo] + r for emo, r in zip(emotions, rr)]
                f.write('+++{0}+++\n'.format(name))
                f.write(tabulate.tabulate(table, headers=['acc', 'prec', 'rec', 'f1']))
                f.write('\n')
    else:
        for name, results, std_devs, ave_results, ave_std_dev in zip(names, all_emo_results, all_emo_std_devs, averaged_results, averaged_std_devs):
            rr = [[u'{0:.3f} \u00B1{1:.3f}'.format(r, s) for r, s in zip(result, std_dev)] for result, std_dev in zip(results, std_devs)]
            av = [u'{0:.3f} \u00B1{1:.3f}'.format(r, s) for r, s in zip(ave_results, ave_std_dev)]
            rr += [av]
            table = [[emo] + r for emo, r in zip(emotions, rr)]
            print(name)
            print(tabulate.tabulate(table, headers=['acc', 'prec', 'rec', 'f1']))
            print()


def main(args):
    parser = argparse.ArgumentParser(
        description='test embeddings on a suite of datasets')
    parser.add_argument('-bi', default=False, type=bool)
    parser.add_argument('-emb', help='location of embeddings', 
        default='../comparison_of_sentiment_methods/embeddings/amazon-sg-50-window10-sample1e-4-negative5.txt')
    parser.add_argument('-threshold', default='0.0', type=str)
    parser.add_argument('-file_type', help='glove style embeddings or word2vec style: default is w2v',
        default='word2vec')
    parser.add_argument('-output', help='output file for results', default='./results.txt')
    parser.add_argument('-printout', help='instead of printing to file, print to sysout',
                        type=bool, default=False)

    args = vars(parser.parse_args())
    embedding_file = args['emb']
    threshold = args['threshold']
    file_type = args['file_type']
    output = args['output']
    printout = args['printout']

    print('testing on %s' % embedding_file)

    if printout:
        print_results(embedding_file, None, threshold, file_type)
    else:
        print_results(embedding_file, output, threshold, file_type)

if __name__ == '__main__':

    args = sys.argv
    main(args)
