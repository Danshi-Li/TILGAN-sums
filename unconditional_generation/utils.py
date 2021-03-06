import os
import torch
import numpy as np
import random
import shutil
import json
import math
from transformers import BertTokenizer, GPT2Tokenizer, T5Tokenizer, BartTokenizer

def load_kenlm():
    global kenlm
    import kenlm


def to_gpu(gpu, var):
    if gpu:
        return var.cuda()
    return var


class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.word2idx['<pad>'] = 0
        self.word2idx['<sos>'] = 1
        self.word2idx['<eos>'] = 2
        self.word2idx['<oov>'] = 3
        self.wordcounts = {}

    # to track word counts
    def add_word(self, word):
        if word not in self.wordcounts:
            self.wordcounts[word] = 1
        else:
            self.wordcounts[word] += 1

    # prune vocab based on count k cutoff or most frequently seen k words
    def prune_vocab(self, k=5, cnt=False):
        # get all words and their respective counts
        vocab_list = [(word, count) for word, count in self.wordcounts.items()]
        if cnt:
            # prune by count
            # self.pruned_vocab = \
            #         {pair[0]: pair[1] for pair in vocab_list if pair[1] > k}
            self.pruned_vocab = \
                    [pair[0] for pair in vocab_list if pair[1] > k]
        else:
            # prune by most frequently seen words
            vocab_list.sort(key=lambda x: (x[1], x[0]), reverse=True)
            k = min(k, len(vocab_list))
            self.pruned_vocab = [pair[0] for pair in vocab_list[:k]]
        # sort to make vocabulary determistic
        self.pruned_vocab.sort()

        # add all chosen words to new vocabulary/dict
        for word in self.pruned_vocab:
            if word not in self.word2idx:
                self.word2idx[word] = len(self.word2idx)
        print("original vocab {}; pruned to {}".
              format(len(self.wordcounts), len(self.word2idx)))
        self.idx2word = {v: k for k, v in self.word2idx.items()}

    def __len__(self):
        return len(self.word2idx)


class Corpus(object):
    def __init__(self, path, maxlen, vocab_size=11000, lowercase=False, bert=False, gpt=False, T5=False, BART=False):
        self.dictionary = Dictionary()
        self.maxlen = maxlen
        self.lowercase = lowercase
        self.vocab_size = vocab_size
        self.train_path = os.path.join(path, 'train.txt')
        self.test_path = os.path.join(path, 'test.txt')

        # make the vocabulary from training set
        self.make_vocab()

        if T5 == False:
            #In T5 or bert+gpt setting, do not need transformer tokenizer
            self.train = self.tokenize(self.train_path)
            self.test = self.tokenize(self.test_path)

        if bert == True:
            self.bertTokenizer = BertTokenizer.from_pretrained('bert-base-cased')
            self.train_bert = self.tokenize_bert(self.train_path)
            self.test_bert = self.tokenize_bert(self.test_path)
            

        if gpt == True:
            self.gptTokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            self.train_gpt = self.tokenize_gpt(self.train_path)
            self.test_gpt = self.tokenize_gpt(self.test_path)

        if T5 == True:
            self.T5Tokenizer = T5Tokenizer.from_pretrained("t5-small")
            self.train_T5 = self.tokenize_T5(self.train_path)
            self.test_T5 = self.tokenize_T5(self.test_path)

        if BART == True:
            self.BARTTokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
            self.train_BART = self.tokenize_BART(self.train_path)
            self.test_BART = self.tokenize_BART(self.test_path)
            

    def make_vocab(self):
        assert os.path.exists(self.train_path)
        # Add words to the dictionary
        with open(self.train_path, 'r') as f:
            for line in f:
                if self.lowercase:
                    # -1 to get rid of \n character
                    words = line[:-1].lower().split(" ")
                else:
                    words = line[:-1].split(" ")
                for word in words:
                    self.dictionary.add_word(word)

        # prune the vocabulary
        self.dictionary.prune_vocab(k=self.vocab_size, cnt=True)

    def tokenize(self, path):
        """Tokenizes a text file."""
        dropped = 0
        with open(path, 'r') as f:
            linecount = 0
            lines = []
            for line in f:
                linecount += 1
                if self.lowercase:
                    words = line[:-1].lower().strip().split(" ")
                else:
                    words = line[:-1].strip().split(" ")
                if len(words) > self.maxlen:
                    dropped += 1
                    continue
                words = ['<sos>'] + words
                words += ['<eos>']
                # vectorize
                vocab = self.dictionary.word2idx
                unk_idx = vocab['<oov>']
                indices = [vocab[w] if w in vocab else unk_idx for w in words]
                lines.append(indices)

        print("Transformer tokenizer: Number of sentences dropped from {}: {} out of {} total".
              format(path, dropped, linecount))
        return lines

    def tokenize_bert(self, path):
        """Tokenizes a text file."""
        dropped = 0
        with open(path, 'r') as f:
            linecount = 0
            dropped = 0
            lines = []
            for line in f:
                linecount += 1
                line=line.strip()
                if len(line.split()) < 1:
                    dropped += 1
                    continue
                if len(line.split()) > self.maxlen:
                    dropped += 1
                    continue
                # vectorize
                lines.append(line)
            indices = self.bertTokenizer(lines, max_length=self.maxlen, padding=True,truncation=True)['input_ids']

        print("BERT tokenizer: Number of sentences dropped from {}: {} out of {} total".
              format(path, dropped, linecount))
        return indices

    def tokenize_gpt(self, path):
        dropped = 0
        with open(path, 'r') as f:
            linecount = 0
            dropped = 0
            lines = []
            for line in f:
                linecount += 1
                if self.lowercase:
                    words = line[:-1].lower().strip().split(" ")
                else:
                    words = line[:-1].strip().split(" ")
                if len(words) > self.maxlen:
                    dropped += 1
                    continue
                if len(words) < 1:
                    dropped += 1
                    continue
                words = ['<|endoftext|>'] + words
                words += ['<|endoftext|>']
                # vectorize
                lines.append(" ".join(words))
            indices = self.gptTokenizer(lines, max_length=self.maxlen, padding=False,truncation=True)['input_ids']
        print("GPT tokenizer: Number of sentences dropped from {}: {} out of {} total".
              format(path, dropped, linecount))
        return indices

    def tokenize_T5(self,path):
        """Tokenizes a text file."""
        dropped = 0
        with open(path, 'r') as f:
            linecount = 0
            dropped = 0
            lines = []
            for line in f:
                linecount += 1
                line=line.strip()
                if len(line.split()) < 1:
                    dropped += 1
                    continue
                if len(line.split()) > self.maxlen:
                    dropped += 1
                    continue
                # vectorize
                lines.append(line)
            indices = self.T5Tokenizer(lines,padding=False,truncation=True).input_ids

        print("T5 tokenizer: Number of sentences dropped from {}: {} out of {} total".
              format(path, dropped, linecount))
        return indices

    def tokenize_BART(self,path):
        """Tokenizes a text file."""
        dropped = 0
        with open(path, 'r') as f:
            linecount = 0
            dropped = 0
            lines = []
            for line in f:
                linecount += 1
                line=line.strip()
                if len(line.split()) < 1:
                    dropped += 1
                    continue
                if len(line.split()) > self.maxlen:
                    dropped += 1
                    continue
                # vectorize
                lines.append(line)
            indices = self.BARTTokenizer(lines,padding=False,truncation=True).input_ids

        print("BART tokenizer: Number of sentences dropped from {}: {} out of {} total".
              format(path, dropped, linecount))
        return indices



def batchify(data, bsz, max_len, shuffle=False, gpu=False, GPT=False):
    if shuffle:
        random.shuffle(data)
    nbatch = len(data) // bsz
    batches = []

    for i in range(nbatch):
        maxlen = max_len+1
        # Pad batches to maximum sequence length in batch
        batch = data[i*bsz:(i+1)*bsz]
        # subtract 1 from lengths b/c includes BOTH starts & end symbols
        lengths = [len(x)-1 for x in batch]

        # sort items by length (decreasing)
        batch, lengths = length_sort(batch, lengths)

        # source has no end symbol
        source = [x[:-1] for x in batch]
        # target has no start symbol
        target = [x[1:] for x in batch]
        for x, y in zip(source, target):
            if GPT:
                zeros = (maxlen-len(x))*[50256]
            else:
                zeros = (maxlen-len(x))*[0]
            x += zeros
            y += zeros
        source = torch.LongTensor(np.int64(source))
        target = torch.LongTensor(np.int64(target)).view(-1)

        if gpu:
            source = source.cuda()
            target = target.cuda()

        batches.append((source, target, lengths))

    return batches


def length_sort(items, lengths, descending=True):
    """In order to use pytorch variable length sequence package"""
    items = list(zip(items, lengths))
    items.sort(key=lambda x: x[1], reverse=True)
    items, lengths = zip(*items)
    return list(items), list(lengths)


def train_ngram_lm(kenlm_path, data_path, output_path, N):
    """
    Trains a modified Kneser-Ney n-gram KenLM from a text file.
    Creates a .arpa file to store n-grams.
    """
    # create .arpa file of n-grams
    curdir = os.path.abspath(os.path.curdir)
    #
    command = "bin/lmplz -o "+str(N)+" <"+os.path.join(curdir, data_path) + \
              " >"+os.path.join(curdir, output_path)
    os.system("cd "+os.path.join(kenlm_path, 'build')+" && "+command)

    load_kenlm()
    # create language model
    assert(output_path)  # captured by try..except block outside
    model = kenlm.Model(output_path)

    return model


def get_ppl(lm, sentences):
    """
    Assume sentences is a list of strings (space delimited sentences)
    """
    total_nll = 0
    total_wc = 0
    for sent in sentences:
        words = sent.strip().split()
        nll = np.sum([- math.log(math.pow(10.0, score)) for score, _, _ in lm.full_scores(sent, bos=True, eos=False)])
        word_count = len(words)
        total_wc += word_count
        total_nll += nll
    ppl = np.exp(total_nll / total_wc)
    return ppl


def create_exp_dir(path, scripts_to_save=None, dict=None, options=None):
    if not os.path.exists(path):
        os.mkdir(path)
    else:
        shutil.rmtree(path)
        os.mkdir(path)

    print('Experiment dir : {}'.format(path))
    if scripts_to_save is not None:
        if not os.path.exists(os.path.join(path, 'scripts')):
            os.mkdir(os.path.join(path, 'scripts'))
        for script in scripts_to_save:
            dst_file = os.path.join(path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)

    # dump the dictionary
    if dict is not None:
        with open(os.path.join(path, 'vocab.json'), 'w') as f:
            json.dump(dict, f)

    # dump the args
    if options is not None:
        with open(os.path.join(path, 'options.json'), 'w') as f:
            json.dump(vars(options), f)

