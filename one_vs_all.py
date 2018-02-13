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
from multi_class import *


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
    
    input_dim = W.shape[1]
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
            clf = one_vs_all_classifier('DAN', input_dim=input_dim, dim=dim,
                                              out_dim=8, dropout=best_dropout, wordvecs=wordvecs,
                                              max_length=Xtrain.shape[1], weights=W, train=True)
        elif model_name == 'LSTM':
            clf = one_vs_all_classifier('LSTM', input_dim=input_dim, dim=dim,
                                              out_dim=8, dropout=best_dropout, wordvecs=wordvecs,
                                              max_length=Xtrain.shape[1], weights=W, train=True)
        elif model_name == 'BiLSTM':
            clf = one_vs_all_classifier('BiLSTM', input_dim=input_dim, dim=dim,
                                              out_dim=8, dropout=best_dropout, wordvecs=wordvecs,
                                              max_length=Xtrain.shape[1], weights=W, train=True)
        elif model_name == 'CNN':
            clf = one_vs_all_classifier('CNN', input_dim=input_dim, dim=dim,
                                              out_dim=8, dropout=best_dropout, wordvecs=wordvecs,
                                              max_length=Xtrain.shape[1], weights=W, train=True)
        h = clf.fit(Xtrain, ytrain, nb_epoch=epoch, verbose=0)
        pred = clf.predict(Xdev)
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



def get_ave_std(pred, y):
    emo_results = []
    for j in range(len(emotions)):
        emo_y = y[:,j]
        emo_pred = pred[:,j]
        mm = MyMetrics(emo_y, emo_pred, one_hot=False, average='binary')
        acc = mm.accuracy()
        precision, recall, f1 = mm.get_scores()
        emo_results.append([acc, precision, recall, f1])
    emo_results = np.array(emo_results)
    return emo_results.mean(axis=0), emo_results.std(axis=0)

class one_vs_all_classifier():
    """
    Creates one classifier per class label and
    trains to correctly predict that class.
    """
    def __init__(self, model_name, input_dim, dim=100,
             out_dim=8, dropout=.5, wordvecs=None,
             max_length=30, weights=None, train=True):
        self.model_name = model_name
        self.input_dim = input_dim
        self.dim = dim
        self.out_dim = out_dim
        self.dropout = dropout
        self.wordvecs = wordvecs
        self.max_length = max_length
        self.weights = weights
        self.train = train
        
        self.classifiers = self._setup()
        
    def _setup(self):
        """
        create as many classifiers as labels
        """
        classifiers = []
        for i in range(self.out_dim):
            if self.model_name == 'DAN':
                clf = create_deep(self.input_dim,
                              dim=self.dim,
                              output_dim=1,
                              dropout=self.dropout)
            elif self.model_name == 'LSTM':
                clf = create_LSTM(self.wordvecs,
                                  self.dim,
                                  output_dim=1,
                                  dropout=self.dropout,
                                  weights = self.weights,
                                  train=self.train)
            elif self.model_name == 'BiLSTM':
                clf = create_BiLSTM(self.wordvecs,
                                    self.dim,
                                    output_dim=1,
                                    dropout=self.dropout,
                                    weights=self.weights,
                                    train=self.train)
            else:
                clf = create_cnn(self.weights, self.max_length,
                                 self.dim, self.dropout,
                                 output_dim=1)
            
            classifiers.append(clf)
        return classifiers
    
    def fit(self, Xtrain, ytrain, validation_data=None,
        nb_epoch=10, verbose=1):
        for i, classifier in enumerate(self.classifiers):
            h = classifier.fit(Xtrain, ytrain[:,i],
                     validation_data=validation_data,
                     nb_epoch=nb_epoch,
                     verbose=verbose)

    def predict(self, Xtest):
        predictions = []
        for classifier in self.classifiers:
            pred = classifier.predict(Xtest)
            predictions.append(pred)
        predictions = np.array(predictions)
        return predictions.reshape((Xtest.shape[0], len(predictions)))
    
            
def test_embeddings(file, file_type):
    names = ['DAN', 'LSTM', 'Bi-LSTM']
    emotions = ["anger", "anticipation", "disgust",
            "fear", "joy", "sadness",
            "surprise", "trust"]
    
    
    
    # Import dataset where each test example is the words in the tweet
    dataset = Fine_Grained_Emotion_Dataset('data', None, rep=words)
    

    print('Basic statistics')
    table = []
    for i, emo in enumerate(emotions):
        train = dataset._ytrain[:,i].sum()
        test = dataset._ytest[:,i].sum()
        table.append((emo, train, test))
    print(tabulate.tabulate(table, headers=['emotion', '#train', '#test']))
    print()

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

    input_dim = W.shape[1]

    # TODO: change this so I don't have to import vectors I don't need
    vecs = WordVecs('../comparison_of_sentiment_methods/embeddings/sswe-u-50.txt')
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

    names = ['DAN', 'LSTM', 'BiLSTM', 'CNN']


    # Keep all mean and standard deviations of each emotion over datasets here
    all_emo_results = []
    all_emo_std_devs = []

    # Keep all mean and standard deviations of the averaged emotions here
    averaged_results = []
    averaged_std_devs = []

    # TEST EACH MODEL
    for name in names:

        print('Getting best parameters')

        dev_params_file = 'dev_params/all_vs_one_best_params.txt'
        if name == 'DAN':
            best_dim, best_dropout, best_epoch, best_f1 = get_dev_params(name, dev_params_file,
                                                           ave_dataset._Xtrain, ave_dataset._ytrain,
                                                           ave_dataset._Xdev, ave_dataset._ydev, wordvecs, W)
        else:
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
            if name == 'DAN':
                model = one_vs_all_classifier('DAN', input_dim=input_dim, dim=best_dim,
                                              out_dim=8, dropout=best_dropout, wordvecs=wordvecs,
                                              max_length=None, weights=W, train=True)
                h = model.fit(ave_dataset._Xtrain, ave_dataset._ytrain,
                         nb_epoch=best_epoch, verbose=0)
                pred = model.predict(ave_dataset._Xtest)

            # DAN uses different representations than the other models
            else:
                if name == 'LSTM':
                    model = one_vs_all_classifier('LSTM', input_dim=input_dim, dim=best_dim,
                                              out_dim=8, dropout=best_dropout, wordvecs=wordvecs,
                                              max_length=Xtrain.shape[1], weights=W, train=True)
                elif name == 'BiLSTM':
                    model = one_vs_all_classifier('BiLSTM', input_dim=input_dim, dim=best_dim,
                                              out_dim=8, dropout=best_dropout, wordvecs=wordvecs,
                                              max_length=Xtrain.shape[1], weights=W, train=True)
                elif name == 'CNN':
                    model = one_vs_all_classifier('CNN', input_dim=input_dim, dim=best_dim,
                                              out_dim=8, dropout=best_dropout, wordvecs=wordvecs,
                                              max_length=Xtrain.shape[1], weights=W, train=True)

                h = model.fit(Xtrain, dataset._ytrain,
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
            mic_f1 = micro_f1(dataset._ytest, pred)
            model_average_results.append((ave_acc,ave_prec, ave_rec, mac_f1, mic_f1))
            
            print('acc: {0:.3f} prec:{1:.3f} rec:{2:.3f} macro-f1:{3:.3f} micro-f1:{4:.3f}'.format(ave_acc,
                                                             ave_prec,
                                                             ave_rec,
                                                             mac_f1,
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

def print_results(file, out_file, file_type):

    names, all_emo_results, all_emo_std_devs, averaged_results, averaged_std_devs, dim = test_embeddings(file, file_type)

    emotions = ["anger", "anticipation", "disgust",
            "fear", "joy", "sadness",
            "surprise", "trust", "averaged"]
    
    
    if out_file:
        with open(out_file, 'a') as f:
            for name, results, std_devs, ave_results, ave_std_dev in zip(names, all_emo_results, all_emo_std_devs, averaged_results, averaged_std_devs):
                rr = [[u'{0:.3f} \u00B1{1:.3f}'.format(r, s) for r, s in zip(result, std_dev)] for result, std_dev in zip(results, std_devs)]
                av = [u'{0:.3f} \u00B1{1:.3f}'.format(r, s) for r, s in zip(ave_results, ave_std_dev)]
                for r in rr:
                    r += ['-']
                rr += [av]
                table = [[emo] + r for emo, r in zip(emotions, rr)]
                f.write('+++{0}+++\n'.format(name))
                f.write(tabulate.tabulate(table, headers=['acc', 'prec', 'rec', 'macro-f1', 'micro-f1']))
                f.write('\n')
    else:
        for name, results, std_devs, ave_results, ave_std_dev in zip(names, all_emo_results, all_emo_std_devs, averaged_results, averaged_std_devs):
            rr = [[u'{0:.3f} \u00B1{1:.3f}'.format(r, s) for r, s in zip(result, std_dev)] for result, std_dev in zip(results, std_devs)]
            av = [u'{0:.3f} \u00B1{1:.3f}'.format(r, s) for r, s in zip(ave_results, ave_std_dev)]
            for r in rr:
                r += ['-']
            rr += [av]
            table = [[emo] + r for emo, r in zip(emotions, rr)]
            print(name)
            print(tabulate.tabulate(table, headers=['acc', 'prec', 'rec', 'macro-f1', 'micro-f1']))
            print()


def main(args):
    parser = argparse.ArgumentParser(
        description='test embeddings on a suite of datasets')
    parser.add_argument('-bi', default=False, type=bool)
    parser.add_argument('-emb', help='location of embeddings', 
        default='../comparison_of_sentiment_methods/embeddings/amazon-sg-50-window10-sample1e-4-negative5.txt')
    parser.add_argument('-file_type', help='glove style embeddings or word2vec style: default is w2v',
        default='word2vec')
    parser.add_argument('-output', help='output file for results', default='./results.txt')
    parser.add_argument('-printout', help='instead of printing to file, print to sysout',
                        type=bool, default=False)

    args = vars(parser.parse_args())
    embedding_file = args['emb']
    file_type = args['file_type']
    output = args['output']
    printout = args['printout']

    print('testing on %s' % embedding_file)

    if printout:
        print_results(embedding_file, None, file_type)
    else:
        print_results(embedding_file, output, file_type)

if __name__ == '__main__':

    args = sys.argv
    main(args)

   
