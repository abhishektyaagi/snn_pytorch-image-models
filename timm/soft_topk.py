import torch
import torch.nn.functional as F
import numpy as np
import pdb

def soft_topk_with_temperature_noscaling(x, k, temperature=1e-2, device='cpu'):
    """
    Approximates the top-k function using softmax with temperature scaling in a fully differentiable manner.

    Args:
        x (torch.Tensor): Input tensor of shape (n,).
        k (int): Number of top elements to select.
        temperature (float): Temperature parameter for softmax.
        device (str): Device to perform computations on ('cpu' or 'cuda').

    Returns:
        torch.Tensor: A tensor of the same shape as x, containing soft values that approximate
                      the top-k selection in a fully differentiable way.
    """
    # Ensure the input is on the correct device
    x = x.to(device)
    
    # Scale the input by the inverse temperature
    scaled_x = x / (temperature)

    # Apply the softmax function
    softmax_probs = F.softmax(scaled_x, dim=0)

    # Compute a "soft" top-k by emphasizing the largest k probabilities
    # We do not perform any hard masking but instead rely on the natural
    # behavior of softmax to distribute most of the probability mass on the top elements
    # Multiply the softmax output by k to ensure that approximately k elements contribute
    #soft_topk_output = k * softmax_probs
    soft_topk_output = softmax_probs

    # Clip the probabilities to [0, 1] range to make it similar to the hard top-k
    #soft_topk_output = torch.clamp(soft_topk_output, 0.0, 1.0)

    return soft_topk_output

def soft_topk_with_temperature(x, k, temperature=1e-2, device='cpu'):
    """
    Approximates the top-k function using softmax with temperature scaling in a fully differentiable manner.

    Args:
        x (torch.Tensor): Input tensor of shape (n,).
        k (int): Number of top elements to select.
        temperature (float): Temperature parameter for softmax.
        device (str): Device to perform computations on ('cpu' or 'cuda').

    Returns:
        torch.Tensor: A tensor of the same shape as x, containing soft values that approximate
                      the top-k selection in a fully differentiable way.
    """
    pdb.set_trace()
    # Ensure the input is on the correct device
    x = x.to(device)
    
    # Scale the input by the inverse temperature
    scaled_x = x / (temperature)

    # Apply the softmax function
    softmax_probs = F.softmax(scaled_x, dim=0)

    # Compute a "soft" top-k by emphasizing the largest k probabilities
    # We do not perform any hard masking but instead rely on the natural
    # behavior of softmax to distribute most of the probability mass on the top elements
    # Multiply the softmax output by k to ensure that approximately k elements contribute
    soft_topk_output = k * softmax_probs

    # Clip the probabilities to [0, 1] range to make it similar to the hard top-k
    soft_topk_output = torch.clamp(soft_topk_output, 0.0, 1.0)

    return soft_topk_output