import argparse
import os
import numpy as np
import torch as t
from torch.optim import Adam
from utils.batch_loader import BatchLoader
from utils.parameters import Parameters
from model.cont_rvae import ContRVAE

if __name__ == "__main__":

    if not os.path.exists('data/preprocessings/word_embeddings.npy'):
        raise FileNotFoundError("word embeddings file was't found")

    parser = argparse.ArgumentParser(description='ContRVAE')
    parser.add_argument('--num-iterations', type=int, default=120000, metavar='NI',
                        help='num iterations (default: 120000)')
    parser.add_argument('--batch-size', type=int, default=40, metavar='BS',
                        help='batch size (default: 40)')
    parser.add_argument('--use-cuda', type=bool, default=True, metavar='CUDA',
                        help='use cuda (default: True)')
    parser.add_argument('--learning-rate', type=float, default=0.0005, metavar='LR',
                        help='learning rate (default: 0.0005)')
    parser.add_argument('--dropout', type=float, default=0.25, metavar='DR',
                        help='dropout (default: 0.25)')
    parser.add_argument('--use-trained', type=bool, default=False, metavar='UT',
                        help='load pretrained model (default: False)')
    parser.add_argument('--se-result', default='', metavar='SE',
                        help='square error result path (default: '')')
    parser.add_argument('--kld-result', default='', metavar='KLD',
                        help='kld result path (default: '')')

    args = parser.parse_args()

    batch_loader = BatchLoader('')
    parameters = Parameters(batch_loader.max_seq_len,
                            batch_loader.vocab_size)

    cont_rvae = ContRVAE(parameters)
    if args.use_trained:
        cont_rvae.load_state_dict(t.load('trained_ContRVAE'))
    if args.use_cuda:
        cont_rvae = cont_rvae.cuda()

    optimizer = Adam(cont_rvae.learnable_parameters(), args.learning_rate)

    train = cont_rvae.trainer(optimizer, batch_loader)
    validate = cont_rvae.validater(batch_loader)

    se_result = []
    kld_result = []

    for iteration in range(args.num_iterations):

        error, kld, coef = train(iteration, args.batch_size, args.use_cuda, args.dropout)

        if iteration % 10 == 0:
            print('\n')
            print('------------TRAIN-------------')
            print('----------ITERATION-----------')
            print(iteration)
            print('--------SQUARE ERROR----------')
            print(error.data.cpu().numpy()[0])
            print('-------------KLD--------------')
            print(kld.data.cpu().numpy()[0])
            print('-----------KLD-coef-----------')
            print(coef)
            print('------------------------------')

        if iteration % 10 == 0:
            error, kld = validate(args.batch_size, args.use_cuda)

            error = error.data.cpu().numpy()[0]
            kld = kld.data.cpu().numpy()[0]

            print('\n')
            print('------------VALID-------------')
            print('--------SQUARE ERROR----------')
            print(error)
            print('-------------KLD--------------')
            print(kld)
            print('------------------------------')

            se_result += [error]
            kld_result += [kld]

        if iteration % 20 == 0:
            seed = np.random.normal(size=[1, parameters.latent_variable_size])

            sample = cont_rvae.sample(batch_loader, 50, seed, args.use_cuda)

            print('\n')
            print('------------SAMPLE------------')
            print('------------------------------')
            print(sample)
            print('------------------------------')

    t.save(cont_rvae.state_dict(), 'trained_ContRVAE')

    np.save('ce_result_{}.npy'.format(args.ce_result), np.array(se_result))
    np.save('kld_result_{}.npy'.format(args.kld_result), np.array(kld_result))
