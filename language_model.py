import torch
import torch.nn as nn
import string
import time
import unidecode
import matplotlib.pyplot as plt

from utils import char_tensor, random_training_set, time_since, random_chunk, CHUNK_LEN
from evaluation import compute_bpc
from model.model import LSTM


def generate(decoder, prime_str='A', predict_len=100, temperature=0.8):
    hidden, cell = decoder.init_hidden()
    prime_input = char_tensor(prime_str)
    predicted = prime_str
    all_characters = string.printable

    # Use priming string to "build up" hidden state
    for p in range(len(prime_str) - 1):
        _, (hidden, cell) = decoder(prime_input[p], (hidden, cell)) 
    inp = prime_input[-1]

    for p in range(predict_len):
        output, (hidden, cell) = decoder(inp, (hidden, cell))

        # Sample from the network as a multinomial distribution
        output_dist = output.data.view(-1).div(temperature).exp()
        top_i = torch.multinomial(output_dist, 1)[0]

        # Add predicted character to string and use as next input
        predicted_char = all_characters[top_i]
        predicted += predicted_char
        inp = char_tensor(predicted_char)

    return predicted


def train(decoder, decoder_optimizer, inp, target):
    hidden, cell = decoder.init_hidden()
    decoder.zero_grad()
    loss = 0
    criterion = nn.CrossEntropyLoss()

    for c in range(CHUNK_LEN):
        output, (hidden, cell) = decoder(inp[c], (hidden, cell))
        loss += criterion(output, target[c].view(1))

    loss.backward()
    decoder_optimizer.step()

    return loss.item() / CHUNK_LEN


def tuner(n_epochs=3000, print_every=100, plot_every=10, hidden_size=128, n_layers=2,
          lr=0.005, start_string='A', prediction_length=100, temperature=0.8):
        # YOUR CODE HERE
        #     TODO:
        #         1) Implement a `tuner` that wraps over the training process (i.e. part
        #            of code that is ran with `default_train` flag) where you can
        #            adjust the hyperparameters
        #         2) This tuner will be used for `custom_train`, `plot_loss`, and
        #            `diff_temp` functions, so it should also accomodate function needed by
        #            those function (e.g. returning trained model to compute BPC and
        #            losses for plotting purpose).

        ################################### STUDENT SOLUTION #######################
        # Initialize model and optimizer
        n_characters = len(string.printable)
        decoder = LSTM(n_characters, hidden_size, n_characters, n_layers)
        decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=lr)
        
        # Training tracking
        start = time.time()
        all_losses = []
        loss_avg = 0
        
        # Training loop
        for epoch in range(1, n_epochs + 1):
            loss = train(decoder, decoder_optimizer, *random_training_set())
            loss_avg += loss
            
            if epoch % print_every == 0:
                print('[{} ({} {}%) {:.4f}]'.format(time_since(start), epoch, epoch/n_epochs * 100, loss))
                print(generate(decoder, start_string, prediction_length, temperature), '\n')
                
            if epoch % plot_every == 0:
                all_losses.append(loss_avg / plot_every)
                loss_avg = 0
                
        return decoder, all_losses
        ############################################################################

def plot_loss(lr_list):
    """Plot training losses for different learning rates"""
    # YOUR CODE HERE
    #     TODO:
    #         1) Using `tuner()` function, train X models where X is len(lr_list),
    #         and plot the training loss of each model on the same graph.
    #         2) Don't forget to add an entry for each experiment to the legend of the graph.
    #         Each graph should contain no more than 10 experiments.
    ###################################### STUDENT SOLUTION ##########################
    # Calculate number of graphs needed (max 10 experiments per graph)
    n_graphs = (len(lr_list) - 1) // 10 + 1
    
    # Create plots
    for graph_idx in range(n_graphs):
        plt.figure(figsize=(12, 8))
        start_idx = graph_idx * 10
        end_idx = min(start_idx + 10, len(lr_list))
        
        # Train and plot models for this graph
        for lr in lr_list[start_idx:end_idx]:
            # Train model and get loss history
            _, losses = tuner(lr=lr, n_epochs=3000, plot_every=10)
            # Plot loss curve
            epochs = range(len(losses))
            plt.plot(epochs, losses, label=f'lr={lr}')
        
        # Configure plot
        plt.title(f'Training Loss vs Epochs (Learning Rates Set {graph_idx + 1})')
        plt.xlabel('Epochs (x10)')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # Save plot
        plt.savefig(f'loss_plot_{graph_idx + 1}.png')
        plt.close()
    ##################################################################################

def diff_temp(temp_list):
    """Generate text using different temperature values"""
    # YOUR CODE HERE
    #     TODO:
    #         1) Using `tuner()` function, try to generate strings by using different temperature
    #         from `temp_list`.
    #         2) In order to do this, create chunks from the test set (with 200 characters length)
    #         and take first 10 characters of a randomly chosen chunk as a priming string.
    #         3) What happen with the output when you increase or decrease the temperature?
    ################################ STUDENT SOLUTION ################################
    # Load test data and get random chunk
    TEST_PATH = './data/dickens_test.txt'
    test_string = unidecode.unidecode(open(TEST_PATH, 'r').read())
    chunk = random_chunk()
    prime_str = chunk[:10]
    
    # Train base model
    model, _ = tuner()
    
    print(f"Prime string: {prime_str}\n")
    print("Generated text with different temperatures:")
    print("-" * 50)
    
    # Generate text with different temperatures
    for temp in temp_list:
        print(f"\nTemperature = {temp}:")
        generated = generate(model, prime_str=prime_str, predict_len=100, temperature=temp)
        print(generated)
        print("-" * 50)
    ##################################################################################

def custom_train(hyperparam_list):
    """
    Train model with X different set of hyperparameters, where X is 
    len(hyperparam_list).

    Args:
        hyperparam_list: list of dict of hyperparameter settings

    Returns:
        bpc_dict: dict of bpc score for each set of hyperparameters.
    """
    TEST_PATH = './data/dickens_test.txt'
    string = unidecode.unidecode(open(TEST_PATH, 'r').read())
    # YOUR CODE HERE
    #     TODO:
    #         1) Using `tuner()` function, train X models with different
    #         set of hyperparameters and compute their BPC scores on the test set.

    ################################# STUDENT SOLUTION ##########################
    # Initialize results dictionary
    bpc_dict = {}
    
    # Train and evaluate each model configuration
    for i, params in enumerate(hyperparam_list, 1):
        # Train model using tuner
        model, _ = tuner(
            hidden_size=params['hidden_size'],
            lr=params['lr'],
            n_layers=params['n_layers'],
            temperature=params['temperature']
        )
        
        # Compute BPC score on test data
        bpc = compute_bpc(model, string)
        bpc_dict[f'model_{i}'] = bpc
        
    return bpc_dict
    #############################################################################