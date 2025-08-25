import nltk
from collections import Counter
import re

class Vocabulary:
    def __init__(self, threshold):
        self.word2idx = {}
        self.idx2word = {}
        self.threshold = threshold
        self.build_vocab()

    def build_vocab(self):
        # load Flickr8k.token.txt and tokenize
        with open("dataset/captions.txt", 'r') as f:
            lines = f.readlines()

        captions = []
        for line in lines:
            line = line.strip().split('\t')
            if len(line) == 2:
                tokens = nltk.tokenize.word_tokenize(line[1].lower())
                captions.extend(tokens)

        counter = Counter(captions)
        words = [word for word in counter if counter[word] >= self.threshold]

        self.word2idx['<pad>'] = 0
        self.word2idx['<start>'] = 1
        self.word2idx['<end>'] = 2
        self.word2idx['<unk>'] = 3

        for i, word in enumerate(words, 4):
            self.word2idx[word] = i
            self.idx2word[i] = word

    def __len__(self):
        return len(self.word2idx)

