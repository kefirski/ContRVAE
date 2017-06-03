from .functions import *


class Parameters:
    def __init__(self, max_seq_len, vocab_size):
        self.max_seq_len = int(max_seq_len) + 1  # go or eos token

        self.vocab_size = int(vocab_size)

        self.word_embed_size = 35

        self.encoder_size = 20
        self.encoder_num_layers = 2

        self.latent_variable_size = 20

        self.decoder_size = 25
        self.decoder_num_layers = 2
