import argparse
import torch
import torch.nn as nn
import unidecode
import string
import time

from utils import char_tensor, random_training_set, time_since, CHUNK_LEN, log_experiment
from language_model import plot_loss, diff_temp, custom_train, train, generate
from model.model import LSTM


def main():
    parser = argparse.ArgumentParser(
        description='Train LSTM'
    )

    parser.add_argument(
        '--default_train', dest='default_train',
        help='Train LSTM with default hyperparameter',
        action='store_true'
    )

    parser.add_argument(
        '--custom_train', dest='custom_train',
        help='Train LSTM while tuning hyperparameter',
        action='store_true'
    )

    parser.add_argument(
        '--plot_loss', dest='plot_loss',
        help='Plot losses chart with different learning rates',
        action='store_true'
    )

    parser.add_argument(
        '--diff_temp', dest='diff_temp',
        help='Generate strings by using different temperature',
        action='store_true'
    )

    args = parser.parse_args()

    all_characters = string.printable
    n_characters = len(all_characters)

    if args.default_train:
        n_epochs = 3000
        print_every = 100
        plot_every = 10
        hidden_size = 128
        n_layers = 2

        lr = 0.005
        decoder = LSTM(n_characters, hidden_size, n_characters, n_layers)
        decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=lr)

        start = time.time()
        all_losses = []
        loss_avg = 0

        for epoch in range(1, n_epochs+1):
            loss = train(decoder, decoder_optimizer, *random_training_set())
            loss_avg += loss

            if epoch % print_every == 0:
                print('[{} ({} {}%) {:.4f}]'.format(time_since(start), epoch, epoch/n_epochs * 100, loss))
                print(generate(decoder, 'A', 100), '\n')

            if epoch % plot_every == 0:
                all_losses.append(loss_avg / plot_every)
                loss_avg = 0

    if args.custom_train:
        # YOUR CODE HERE
        #     TODO:
        #         1) Fill in `hyperparam_list` with dictionary of hyperparameters
        #         that you want to try.
        ####################### STUDENT SOLUTION ###############################
        hyperparam_list = [
            # Baseline
            #{'hidden_size': 128, 'lr': 0.005, 'n_layers': 2, 'temperature': 0.8},
            
            # Vary hidden_size
            {'hidden_size': 64, 'lr': 0.005, 'n_layers': 2, 'temperature': 0.8},
            {'hidden_size': 256, 'lr': 0.005, 'n_layers': 2, 'temperature': 0.8},
            
            # Vary learning rate
            {'hidden_size': 128, 'lr': 0.001, 'n_layers': 2, 'temperature': 0.8},
            {'hidden_size': 128, 'lr': 0.01, 'n_layers': 2, 'temperature': 0.8},
            
            # Vary layers
            {'hidden_size': 128, 'lr': 0.005, 'n_layers': 1, 'temperature': 0.8},
            {'hidden_size': 128, 'lr': 0.005, 'n_layers': 3, 'temperature': 0.8},
            
            # Vary temperature
            {'hidden_size': 128, 'lr': 0.005, 'n_layers': 2, 'temperature': 0.5},
            {'hidden_size': 128, 'lr': 0.005, 'n_layers': 2, 'temperature': 1.0}
        ]
        ########################################################################
        bpc, all_losses = custom_train(hyperparam_list)

        for keys, values in bpc.items():
            print("BPC {}: {}".format(keys, values))
        
        # Log experiments for each model
        for i, params in enumerate(hyperparam_list, 1):
            model_key = f'model_{i}'
            min_loss = min(all_losses[model_key])
            log_experiment(params, min_loss, bpc[model_key])

    if args.plot_loss:
        # YOUR CODE HERE
        #     TODO:
        #         1) Fill in `lr_list` with learning rates that you want to try.
        ######################### STUDENT SOLUTION #############################
        lr_list = [0.0001, 0.001, 0.005, 0.01, 0.1]
        ########################################################################
        plot_loss(lr_list)

    if args.diff_temp:
        # YOUR CODE HERE
        #     TODO:
        #         1) Fill in `temp_list` with temperatures that you want to try.
        ########################### STUDENT SOLUTION ###########################
        temp_list = [0.1, 0.5, 1.0, 2.0]
        ########################################################################
        diff_temp(temp_list)


if __name__ == "__main__":
    main()
