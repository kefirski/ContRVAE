import collections
import os
import numpy as np
from six.moves import cPickle
from .functions import *
import torch as t
from torch.autograd import Variable


class BatchLoader:
    def __init__(self, path=''):

        path += 'data/'

        self.preprocessing_path = path + 'preprocessings/'
        if not os.path.exists(self.preprocessing_path):
            os.makedirs(self.preprocessing_path)

        self.data_files = [path + 'train.txt',
                           path + 'test.txt']

        self.idx_file = self.preprocessing_path + 'words_vocab.pkl'

        self.tensor_files = [self.preprocessing_path + 'train_word_tensor.npy',
                             self.preprocessing_path + 'valid_word_tensor.npy']

        self.pad_token = '_'
        self.go_token = '>'
        self.end_token = '|'

        idx_exists = os.path.exists(self.idx_file)
        tensors_exists = all([os.path.exists(target) for target in self.tensor_files])

        if idx_exists and tensors_exists:
            self.load_preprocessed(self.data_files,
                                   self.idx_file,
                                   self.tensor_files)
            print('preprocessed data was found and loaded')
        else:
            self.preprocess(self.data_files,
                            self.idx_file,
                            self.tensor_files)
            print('data have preprocessed')

        self.word_embedding_index = 0

    def build_vocab(self, sentences):
        """
        build_vocab(self, sentences) -> vocab_size, idx_to_word, word_to_idx
            vocab_size - number of unique words in corpus
            idx_to_word - array of shape [vocab_size] containing ordered list of unique words
            word_to_idx - dictionary of shape [vocab_size]
                such that idx_to_char[idx_to_word[some_word]] = some_word
                where some_word is such that idx_to_word contains it
        """

        word_counts = collections.Counter(sentences)

        idx_to_word = [x[0] for x in word_counts.most_common()]
        idx_to_word = list(sorted(idx_to_word)) + [self.pad_token, self.go_token, self.end_token]

        word_to_idx = {x: i for i, x in enumerate(idx_to_word)}

        words_vocab_size = len(idx_to_word)

        return words_vocab_size, idx_to_word, word_to_idx

    def preprocess(self, data_files, idx_file, tensor_paths):

        data = [open(file, "r").read() for file in data_files]

        data_words = [[line.split() for line in target.split('\n')] for target in data]
        self.max_seq_len = np.amax([len(line) for target in data_words for line in target])
        self.num_lines = [len(target) for target in data_words]

        '''split whole data and build vocabulary from it'''
        merged_data_words = (data[0] + '\n' + data[1]).split()
        self.vocab_size, self.idx_to_word, self.word_to_idx = self.build_vocab(merged_data_words)
        self.max_word_len = np.amax([len(word) for word in self.idx_to_word])

        with open(idx_file, 'wb') as f:
            cPickle.dump(self.idx_to_word, f)

        self.data_tensor = np.array([[list(map(self.word_to_idx.get, line)) for line in target]
                                     for target in data_words])

        self.words_freq = collections.Counter([idx for line in self.data_tensor[0] for idx in line])
        self.words_freq = [self.words_freq[i] + 1 / self.vocab_size for i in range(self.vocab_size)]

        for target, path in enumerate(tensor_paths):
            np.save(path, self.data_tensor[target])

            '''uses to pick up data pairs for embedding learning'''
        self.embed_pairs = np.array([pair for line in self.data_tensor[0] for pair in BatchLoader.bag_window(line, 12)])

    def load_preprocessed(self, data_files, idx_file, tensor_paths):

        data = [open(file, "r").read() for file in data_files]
        data_words = [[line.split() for line in target.split('\n')] for target in data]

        self.max_seq_len = np.amax([len(line) for target in data_words for line in target])
        self.num_lines = [len(target) for target in data_words]

        self.idx_to_word = cPickle.load(open(idx_file, "rb"))
        self.word_to_idx = dict(zip(self.idx_to_word, range(len(self.idx_to_word))))
        self.vocab_size = len(self.idx_to_word)

        self.max_word_len = np.amax([len(word) for word in self.idx_to_word])

        self.data_tensor = np.array([np.load(target) for target in tensor_paths])

        self.words_freq = collections.Counter([idx for line in self.data_tensor[0] for idx in line])
        self.words_freq = [self.words_freq[i] + 1 / self.vocab_size for i in range(self.vocab_size)]

        self.embed_pairs = np.array([pair for line in self.data_tensor[0] for pair in BatchLoader.bag_window(line, 12)])

    def next_seq(self, batch_size, target: str, use_cuda: bool):
        """
        :param batch_size: number of selected data elements 
        :param target: whether to use train or valid data source
        :return: target tensor
        """

        target = 0 if target == 'train' else 1

        indexes = np.array(np.random.randint(self.num_lines[target], size=batch_size))

        '''
        I've spended for about 3.5 hours fixing this bug caused by nympy's by-defauld mutability
        
        '''
        encoder_input = [np.copy(self.data_tensor[target][index]).tolist() for index in indexes]
        input_seq_len = [len(line) for line in encoder_input]
        max_input_seq_len = max(input_seq_len)

        decoder_input = [[self.word_to_idx[self.go_token]] + line for line in encoder_input]
        decoder_target = [line + [self.word_to_idx[self.end_token]] for line in encoder_input]

        to_add = [max_input_seq_len - len(encoder_input[i]) for i in range(batch_size)]

        for i in range(batch_size):

            encoder_input[i] += [self.word_to_idx[self.pad_token]] * to_add[i]
            decoder_input[i] += [self.word_to_idx[self.pad_token]] * to_add[i]
            decoder_target[i] += [self.word_to_idx[self.pad_token]] * to_add[i]

        input = [np.array(var) for var in [encoder_input, decoder_input, decoder_target]]
        input = [Variable(t.from_numpy(var)).long() for var in input]
        if use_cuda:
            input = [var.cuda() for var in input]

        return input

    def next_embedding_seq(self, batch_size):

        embed_len = len(self.embed_pairs)

        seq = np.array([self.embed_pairs[i % embed_len]
                        for i in np.arange(self.word_embedding_index, self.word_embedding_index + batch_size)])

        self.word_embedding_index = (self.word_embedding_index + batch_size) % embed_len

        return seq[:, 0], seq[:, 1:]

    @staticmethod
    def bag_window(seq, window=3):

        assert window > 1 and isinstance(window, int)

        context = []
        seq_len = len(seq)
        num_slices = seq_len - window + 1

        for i in range(num_slices):
            context += [seq[i:i + window]]

        return [[value] + c[:i] + c[i + 1:] for c in context for i, value in enumerate(c)]

    def go_input(self, batch_size, use_cuda):
        go_input = np.array([[self.word_to_idx[self.go_token]]] * batch_size)
        go_input = Variable(t.from_numpy(go_input)).long()
        if use_cuda:
            go_input = go_input.cuda()
        return go_input

    def encode_word(self, word):

        idx = self.word_to_idx[word]
        result = np.zeros(self.vocab_size)
        result[idx] = 1
        return result

    def decode_word(self, distribution):

        return np.random.choice(range(self.vocab_size), p=distribution.ravel())
