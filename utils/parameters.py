from .functions import *


class Parameters:
    def __init__(self, max_seq_len, vocab_size):
        self.max_seq_len = int(max_seq_len) + 1  # go or eos token

        self.vocab_size = int(vocab_size)

        self.word_embed_size = 300

        self.encoder_size = 250
        self.encoder_num_layers = 3

        self.latent_variable_size = 200

        self.decoder_size = 120
        self.decoder_num_layers = 4
