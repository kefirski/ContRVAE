import os
import numpy as np
import torch as t
import torch.nn as nn
from torch.nn import Parameter


class EmbeddingLockup(nn.Module):
    def __init__(self, params, path_prefix='../../../'):
        super(EmbeddingLockup, self).__init__()

        self.params = params
        word_embeddings_path = path_prefix + 'data/preprocessings/word_embeddings.npy'
        assert os.path.exists(word_embeddings_path), 'Word embeddings not found'

        embeddings = np.load(word_embeddings_path)

        self.embeddings = nn.Embedding(self.params.vocab_size, self.params.word_embed_size)
        self.embeddings.weight = Parameter(t.from_numpy(embeddings).float(), requires_grad=False)

    def forward(self, input):
        """
        :param input: [batch_size, seq_len] tensor of Long type
        :return: input embedding with shape of [batch_size, seq_len, word_embed_size]
        """

        return self.embeddings(input)

    def similarity(self, input):
        """
        :param input: An tensor with shape of [word_embed_size] 
        :return: An tensor with shape [vocab_size] with estimated similarity values
        """

        input = input.unsqueeze(0).repeat(self.params.vocab_size, 1)

        return t.pow(self.embeddings.weight - input, 2).mean(1).squeeze()


