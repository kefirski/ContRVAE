from .functions import *


class Parameters:
    def __init__(self, max_seq_len, vocab_size):
        self.max_seq_len = int(max_seq_len) + 1  # go or eos token

        self.vocab_size = int(vocab_size)

        self.word_embed_size = 300

        self.encoder_rnn_size = 180
        self.encoder_num_layers = 3

        self.latent_variable_size = 900

        self.decoder_rnn_size = 200
        self.decoder_num_layers = 3
