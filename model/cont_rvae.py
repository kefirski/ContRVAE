import numpy as np
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from .generator import Generator
from .inference import Inference
from selfModules.embedding import EmbeddingLockup
from selfModules.softargmax import SoftArgmax
from utils.functions import kld_coef, fold


class ContRVAE(nn.Module):
    def __init__(self, params):
        super(ContRVAE, self).__init__()

        self.params = params

        self.embedding = EmbeddingLockup(self.params, '')

        self.inference = Inference(self.params)

        self.context_to_mu = nn.Linear(self.params.encoder_size * 2, self.params.latent_variable_size)
        self.context_to_logvar = nn.Linear(self.params.encoder_size * 2, self.params.latent_variable_size)

        self.generator = Generator(self.params)

    def forward(self, drop_prob,
                encoder_input=None,
                decoder_input=None,
                z=None, initial_state=None):
        """
        :param drop_prob: probability of an element of decoder input to be dropped out
        :param encoder_input: An tensor with shape of [batch_size, seq_len] of Long type
        :param decoder_input: An tensor with shape of [batch_size, max_seq_len + 1] of Long type
        :param z: context if sampling is performing
        :param initial_state: initial state of decoder rnn if sampling is performing
        :return: mu of N(mu, var) of sentence words distribution probabilities
                    with shape of [batch_size, seq_len, word_embed_size]
                 final rnn state with shape of [num_layers, batch_size, decoder_rnn_size]
        """

        assert z is None and fold(lambda acc, parameter: acc and parameter is not None,
                                  [encoder_input, decoder_input],
                                  True) \
               or (z is not None and decoder_input is not None), \
            "Invalid input. If z is None then encoder and decoder inputs should be passed as arguments"

        if z is None:
            ''' Get context from encoder and sample z ~ N(mu, std)
            '''
            [batch_size, _] = encoder_input.size()

            encoder_input = self.embedding(encoder_input)

            context = self.inference(encoder_input)

            mu = self.context_to_mu(context)
            logvar = self.context_to_logvar(context)
            std = t.exp(0.5 * logvar)

            z = Variable(t.randn([batch_size, self.params.latent_variable_size]))
            if encoder_input.is_cuda:
                z = z.cuda()

            z = z * std + mu

            kld = (-0.5 * t.sum(logvar - t.pow(mu, 2) - t.exp(logvar) + 1, 1)).mean()
        else:
            kld = None

        decoder_input = self.embedding(decoder_input)
        decoder_input = F.dropout(decoder_input, drop_prob, training=z is None)
        out, final_state = self.generator(decoder_input, z, initial_state)

        return out, final_state, kld

    def learnable_parameters(self):

        """word_embedding is constant parameter thus it must be dropped from list of parameters for optimizer"""
        return [p for p in self.parameters() if p.requires_grad]

    def trainer(self, optimizer, batch_loader):
        def train(i, batch_size, use_cuda, drop_prob):
            [encoder_input, decoder_input, decoder_target] = batch_loader.next_seq(batch_size, 'train', use_cuda)
            output, _, kld = self(drop_prob, encoder_input, decoder_input)

            decoder_target = self.embedding(decoder_target).view(batch_size, -1)
            output = output.view(batch_size, -1)

            error = t.pow(output - decoder_target, 2).sum() / batch_size

            '''
            loss is constructed from error formed from squared error between output and target
            and KL Divergence between p(z) and q(z|x) averaged over whole batches
            '''
            loss = error + kld_coef(i) * kld

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            return error, kld, kld_coef(i)

        return train

    def validater(self, batch_loader):
        def validate(batch_size, use_cuda):
            [encoder_input, decoder_input, decoder_target] = batch_loader.next_seq(batch_size, 'validation', use_cuda)

            output, _, kld = self(0, encoder_input, decoder_input)

            decoder_target = self.embedding(decoder_target).view(batch_size, -1)
            output = output.view(batch_size, -1)

            error = t.pow(output - decoder_target, 2).sum() / batch_size

            return error, kld

        return validate

    def sample(self, batch_loader, seq_len, z, use_cuda):

        z = Variable(t.from_numpy(z).float())
        if use_cuda:
            z = z.cuda()

        decoder_input = batch_loader.go_input(1, use_cuda)

        result = []

        initial_state = None

        for i in range(seq_len):
            output, initial_state, _ = self(0., None,
                                            decoder_input,
                                            z, initial_state)

            output = output.squeeze()
            similarity = self.embedding.similarity(output)
            _, idx = t.max(similarity, 0)
            idx = idx.data.cpu().numpy()[0]

            word = batch_loader.idx_to_word[idx]

            if word == batch_loader.end_token:
                break

            result += [word]

            decoder_input = np.array([[idx]])
            decoder_input = Variable(t.from_numpy(decoder_input).long())

            if use_cuda:
                decoder_input = decoder_input.cuda()

        return ' '.join(result)
