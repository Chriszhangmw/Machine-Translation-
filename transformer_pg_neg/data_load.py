# -*- coding: utf-8 -*-
#/usr/bin/python3
'''
date: 2019/5/21
mail: cally.maxiong@gmail.com
page: http://www.cnblogs.com/callyblog/
'''

import tensorflow as tf
from utils import calc_num_batches

SENTENCE_START = '<s>'
SENTENCE_END = '</s>'

PAD_TOKEN = 'PAD'
UNKNOWN_TOKEN = 'UNK'
START_DECODING = 'START'
STOP_DECODING = 'STOP'
class Vocab:
    def __init__(self, vocab_file, max_size):
        self.word2id = {UNKNOWN_TOKEN: 0, PAD_TOKEN: 1, START_DECODING: 2, STOP_DECODING: 3}
        self.id2word = {0: UNKNOWN_TOKEN, 1: PAD_TOKEN, 2: START_DECODING, 3: STOP_DECODING}
        self.count = 4

        with open(vocab_file, 'r',encoding='utf-8') as f:
            for line in f:
                pieces = line.split()
                if len(pieces) != 2:
                    print('Warning : incorrectly formatted line in vocabulary file : %s\n' % line)
                    continue

                w = pieces[0]
                # if w in [SENTENCE_START, SENTENCE_END, UNKNOWN_TOKEN, PAD_TOKEN, START_DECODING, STOP_DECODING]:
                #     raise Exception(r'<s>, </s>, [UNK], [PAD], [START] and [STOP] shouldn\'t be in the vocab file, but %s is' % w)
                #
                # if w in self.word2id:
                #     raise Exception('Duplicated word in vocabulary file: %s' % w)

                self.word2id[w] = self.count
                self.id2word[self.count] = w
                self.count += 1
                if max_size != 0 and self.count >= max_size:
                    print("max_size of vocab was specified as %i; we now have %i words. Stopping reading." % (max_size, self.count))
                    break

        print("Finished constructing vocabulary of %i total words. Last word added: %s" % (self.count, self.id2word[self.count - 1]))

    def word_to_id(self, word):
        if word not in self.word2id:
            return self.word2id[UNKNOWN_TOKEN]
        return self.word2id[word]

    def id_to_word(self, word_id):
        if word_id not in self.id2word:
            raise ValueError('Id not found in vocab: %d' % word_id)
        return self.id2word[word_id]

    def size(self):
        return self.count

def _load_data(f_x,f_y):
    '''Loads source and target data and filters out too lengthy samples.
    fpath1: source file path. string.
    fpath2: target file path. string.
    maxlen1: source sent maximum length. scalar.
    maxlen2: target sent maximum length. scalar.

    Returns
    sents1: list of source sents
    sents2: list of target sents
    '''
    sents1, sents2 = [], []
    with open(f_x,'r',encoding='utf-8') as f1:
        datax = f1.readlines()
    for line in datax:
        line = line.strip()
        sents1.append(line)
    with open(f_y,'r',encoding='utf-8') as f2:
        datay = f2.readlines()
    for line in datay:
        line = line.strip()
        sents2.append(line)

    return sents1, sents2

def _encode(inp, token2idx, maxlen, type):
    '''Converts string to number. Used for `generator_fn`.
    inp: 1d byte array.
    type: "x" (source side) or "y" (target side)
    dict: token2idx dictionary

    Returns
    list of numbers
    '''
    inp = inp.decode('utf-8')
    if type == 'x':
        tokens = ['START'] + list(inp) + ['STOP']
        if len(tokens) >= maxlen:
            tokens = tokens[:maxlen]
        else:
            while len(tokens) < maxlen:
                tokens.append('PAD')
        return [token2idx.get(token, token2idx['UNK']) for token in tokens],tokens

    else:
        inputs = ['START'] + list(inp)
        target = list(inp) + ['STOP']
        if len(target) >= maxlen:
            inputs =inputs[:maxlen]
            target = target[:maxlen]
        else:
            while len(target) < maxlen:
                inputs.append('PAD')
                target.append('PAD')
        return [token2idx.get(token, token2idx['UNK']) for token in inputs], [token2idx.get(token, token2idx['UNK']) for token in target]

def get_dic(path):
    vocab = Vocab(path, 50000)
    token2idx = vocab.word2id
    return token2idx

import pickle
import numpy as np
def load_obj():
    with open('./data/frequence_neg.pkl', 'rb') as f:
        return pickle.load(f)
fre = load_obj()
a = min(fre.values())
alpha = 1e-2
word_freqs = np.array(list(fre.values())).sum()
max_fre = max(fre.values())
# print(fre.get('维修',a)/word_freqs.sum())
def calcu_dis(sent1):
    # weights = alpha / (alpha + fre.get(word, max_fre))
    dis = [ alpha / (alpha + fre.get(w, max_fre)) for w in sent1]
    return dis

def encoder_sif(sent1):
    # weights = alpha / (alpha + fre.get(word, max_fre))
    dis = [ alpha / (alpha + fre.get(w, max_fre)) for w in sent1]
    return dis

def _generator_fn(sents1, sents2, vocab_path, maxlen1, maxlen2):
    '''Generates training / evaluation data
    sents1: list of source sents
    sents2: list of target sents
    vocab_fpath: string. vocabulary file path.
    yields
    xs: tuple of
        x: list of source token ids in a sent
        x_seqlen: int. sequence length of x
        sent1: str. raw source (=input) sentence
    labels: tuple of
        decoder_input: decoder_input: list of encoded decoder inputs
        y: list of target token ids in a sent
        y_seqlen: int. sequence length of y
        sent2: str. target sentence
    '''
    token2idx = get_dic(vocab_path)
    # idx2token = vocab.id2word
    for sent1, sent2 in zip(sents1, sents2):
        x,tokens = _encode(sent1, token2idx, maxlen1, "x")

        fre = encoder_sif(tokens)

        inputs, targets = _encode(sent2, token2idx, maxlen2, "y")

        yield (x, sent1.decode('utf-8')), (fre,sent1.decode('utf-8')),(inputs, targets, sent2.decode('utf-8'))

def _input_fn(sents1, sents2,hp, batch_size, gpu_nums, shuffle=False):
    '''Batchify data
    sents1: list of source sents
    sents2: list of target sents
    vocab_fpath: string. vocabulary file path.
    batch_size: scalar
    shuffle: boolean

    Returns
    xs: tuple of
        x: int32 tensor. (N, T1)
        x_seqlens: int32 tensor. (N,)
        sents1: str tensor. (N,)
    ys: tuple of
        decoder_input: int32 tensor. (N, T2)
        y: int32 tensor. (N, T2)
        y_seqlen: int32 tensor. (N, )
        sents2: str tensor. (N,)
    '''
    maxlen1 = hp['max_enc_len']
    maxlen2 = hp['max_dec_len']
    shapes = (([maxlen1], ()),
              ([maxlen1], ()),
              ([maxlen2], [maxlen2], ()))
    types = ((tf.int32, tf.string),
             (tf.float32, tf.string),
             (tf.int32, tf.int32, tf.string))

    dataset = tf.data.Dataset.from_generator(
        _generator_fn,
        output_shapes=shapes,
        output_types=types,
        args=(sents1, sents2, hp['vocab_path'], maxlen1, maxlen2))  # <- arguments for generator_fn. converted to np string arrays

    if shuffle: # for training
        dataset = dataset.shuffle(128*batch_size*gpu_nums)

    dataset = dataset.repeat()  # iterate forever
    dataset = dataset.batch(batch_size*gpu_nums)

    return dataset

def get_batch(f_x,f_y, hp, batch_size, gpu_nums,shuffle=False):

    sents1, sents2 = _load_data(f_x,f_y)
    batches = _input_fn(sents1, sents2, hp, batch_size, gpu_nums,shuffle=shuffle)
    num_batches = calc_num_batches(len(sents1), batch_size*gpu_nums)
    return batches, num_batches, len(sents1)
