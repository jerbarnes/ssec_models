from multi_class import *
from sklearn.svm import LinearSVC
import tabulate

def dev_parameters(Xtrain, ytrain, Xdev, ydev):

    best_dev_f1 = 0
    best_C = 0
    C = [0.001, 0.003, 0.005, 0.007, 0.009,
         0.01, 0.03, 0.05, 0.07, 0.09,
         0.1, 0.3, 0.5, 0.7, 0.9,
         1, 1.5, 1.7, 2]

    for c in C:
        ova = one_vs_all(8, C=c)
        ova.fit(Xtrain, ytrain)
        pred = ova.predict(Xdev)
        mic_prec, mic_rec, mic_f1 = micro_f1(ydev, pred)
        if mic_f1 > best_dev_f1:
            sys.stdout.write('\rbest dev f1: {0:.1f}'.format(mic_f1 * 100))
            best_dev_f1 = mic_f1
            best_C = c
    return best_dev_f1, best_C
    


class one_vs_all():
    """
    One vs all linear SVM classifiers.
    """
    def __init__(self, output_dim, C=1):
        self.classifiers = self._setup(output_dim, C=C)
        
    def _setup(self, output_dim, C=1):
        """
        Create a separate SVM for each y label.
        """
        classifiers = []
        for i in range(output_dim):
            classifiers.append(LinearSVC(C=C))
        return classifiers
    
    def fit(self, Xtrain, ytrain):
        """
        input - Xtrain - n x d matrix, n = number of examples, d = feature dimension
              - ytrain - n x l matrix, l = number of labels

        the class labels can have more than one positive label for
        each instance.

        i.e. [1,0,1,0,1] is acceptable
        """
        for i,classifier in enumerate(self.classifiers):
            classifier.fit(Xtrain, ytrain[:,i])
            
    def predict(self, X):
        prediction = []
        for classifier in self.classifiers:
            prediction.append(classifier.predict(X))
        return np.array(prediction).T


def main():

    emotions = ["anger", "anticipation", "disgust",
            "fear", "joy", "sadness",
            "surprise", "trust"]

    vecs = WordVecs('embeddings/twitter_embeddings.txt')
    dataset = Fine_Grained_Emotion_Dataset('data', None, rep=words)
    vocab = {}
    for sent in list(dataset._Xtrain) + list(dataset._Xdev) + list(dataset._Xtest):
        for w in sent:
            if w not in vocab:
                vocab[w] = 1
            else:
                vocab[w] += 1

                
    wordvecs = {}
    for line in open('embeddings/twitter_embeddings.txt'):
        try:
            split = line.split()
            word = split[0]
            vec = np.array(split[1:], dtype='float32')
            if word in vocab:
                wordvecs[word] = vec
        except ValueError:
            pass

    add_unknown_words(wordvecs, vocab, dim=300)
    W, word_idx_map = get_W(wordvecs)
    vecs._matrix = W

    for threshold in ['0.0', '0.33', '0.5, 0.66', '0.99']:
        dataset = Fine_Grained_Emotion_Dataset('data', vecs,
                                               rep=ave_vecs, threshold=threshold)

        best_dev_f1, best_C = dev_parameters(dataset._Xtrain, dataset._ytrain,
                                             dataset._Xdev, dataset._ydev)


        ova = one_vs_all(8, C=best_C)
        ova.fit(dataset._Xtrain, dataset._ytrain)
        pred = ova.predict(dataset._Xtest)
        y = dataset._ytest

        emo_results = []
        average_results  = []

        for j in range(len(emotions)):
            emo_y = y[:,j]
            emo_pred = pred[:,j]
            mm = MyMetrics(emo_y, emo_pred, one_hot=False, average='binary')
            acc = mm.accuracy()
            precision, recall, f1 = mm.get_scores()
            emo_results.append([acc, precision, recall, f1])
        emo_results = np.array(emo_results)
        ave_acc, ave_prec, ave_rec, mac_f1 = emo_results.mean(axis=0)
        mic_prec, mic_rec, mic_f1 = micro_f1(dataset._ytest, pred)
        average_results.append([ave_acc, mic_prec, mic_rec, mic_f1])
        average_results  = np.array(average_results)

        emo_results *= 100
        average_results *= 100
        average_results = list(average_results[0])

        rr = list([list(x) for x in emo_results])
        rr.append(average_results)
        emotions.append('Micro Avg.')

        table = [[emo] + r for emo, r in zip(emotions, rr)]
        print()
        print(tabulate.tabulate(table, headers=['emotion', 'acc', 'prec', 'rec', 'f1'],
                                floatfmt='.0f'))


if __name__ == '__main__':
    main()
