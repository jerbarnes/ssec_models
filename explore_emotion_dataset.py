import nltk
from stance_dataset import *

def prob_word(word, vocab):
    keys = list(vocab.keys())
    count = 0
    for k in keys:
        count += vocab[k][word]
    return max(0.0001, count)

def prob_cat(cat, vocab):
    all_cats = 0
    for v in vocab.values():
        all_cats += len(v)
    return len(vocab[cat]) / all_cats

def pmi(word, cat, vocab):
    return np.log((vocab[cat][word]) / (prob_word(word, vocab) * prob_cat(cat, vocab)))

vecs = WordVecs('sentiment_retrofitting/embeddings/amazon-sg-100-window10-sample1e-4-negative5.txt')
dataset = Stance_Dataset('.', vecs._w2idx, rep=words)

emotions = ["anger", "anticipation", "disgust",
            "fear", "joy", "sadness",
            "surprise", "trust"]

anger = []
anticipation = []
disgust = []
fear = []
joy = []
sadness = []
surprise = []
trust = []

for y,x in zip(dataset._ytrain, dataset._Xtrain):
    if y[0] == 1:
        anger.append(x)
    if y[1] == 1:
        anticipation.append(x)
    if y[2] == 1:
        disgust.append(x)
    if y[3] == 1:
        fear.append(x)
    if y[4] == 1:
        joy.append(x)
    if y[5] == 1:
        sadness.append(x)
    if y[6] == 1:
        surprise.append(x)
    if y[7] == 1:
        trust.append(x)

categories = [anger, anticipation, disgust,
        fear, joy, sadness, surprise,
        trust]

vocab = {}
full_vocab = []

for name, cat in zip(emotions, categories):
    v = nltk.FreqDist()
    for line in cat:
        v.update(line)
        for w in line:
            if w not in full_vocab:
                full_vocab.append(w)
    vocab[name] = v

