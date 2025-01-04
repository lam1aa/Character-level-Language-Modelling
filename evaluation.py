import torch

from utils import char_tensor


def compute_bpc(model, string):
    """
    Given a model and a string of characters, compute bits per character
    (BPC) using that model.

    Args:
        model: RNN-based model (RNN, LSTM, GRU, etc.)
        string: string of characters

    Returns:
        BPC for that set of string.
    """
    ################# STUDENT SOLUTION ################################
    # Convert string to input and target tensors
    input_tensor = char_tensor(string[:-1])
    target_tensor = char_tensor(string[1:])
    
    # Initialize model states
    hidden, cell = model.init_hidden()
    
    # Initialize total loss
    total_log_prob = 0
    sequence_length = len(string) - 1
    
    # Process each character
    for i in range(sequence_length):
        # Get model output
        output, (hidden, cell) = model(input_tensor[i], (hidden, cell))
        
        # Convert output to probabilities
        output_probs = torch.nn.functional.softmax(output, dim=1)
        
        # Get probability of correct character
        correct_prob = output_probs[0][target_tensor[i]]
        
        # Add log probability (convert to base 2)
        total_log_prob += torch.log2(correct_prob)
    
    # Calculate bits per character
    bpc = -total_log_prob.item() / sequence_length
    
    return bpc
    ###################################################################

