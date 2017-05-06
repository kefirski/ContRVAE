from .functions import *


class Parameters:
    def __init__(self, max_word_len, max_seq_len, word_vocab_size, char_vocab_size):
        self.max_word_len = int(max_word_len)
        self.max_seq_len = int(max_seq_len) + 1  # go or eos token

        self.word_vocab_size = int(word_vocab_size)
        self.char_vocab_size = int(char_vocab_size)

        self.word_embed_size = 300
        self.char_embed_size = 15

        self.encoder_rnn_size = 600
        self.encoder_num_layers = 1

        self.latent_variable_size = 1100

        self.decoder_rnn_size = 800
        self.decoder_num_layers = 2
