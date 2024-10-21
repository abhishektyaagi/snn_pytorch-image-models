import numpy as np
import time
import matplotlib.pyplot as plt
import math
import torch
import pdb
import random

""" def get_mask_pseudo_diagonal_numpy(mask_shape, sparsity,file_name=None,experimentType="randomWithZero", 
                                   layerNum = None, numDiag = None, diag_pos = None, currLayer=1, debug=0, vector =  []):

  # Create an array of zeros with the specified shape
  mask = np.zeros(mask_shape)
  # Calculate the length of the diagonals
  #diag_length = min(mask_shape[0], mask_shape[1])
  diag_length = max(mask_shape[0], mask_shape[1])

  #TODO: Change it to depend on the sparsity
  #r = 6
  r = []

  start_positions = []
  used_rows = set()
  start_row = 0
  start_col = 0

  used_rows.add(0)
  #start_positions.append((0,0)) 
  np.random.seed(int(time.time()))
  start_row = int(diag_pos)
  start_positions.append((start_row,start_col))

  #Check for the dimension of elements in start_positions
  r = [start_positions[i][0] for i in range(len(start_positions))]

  for start_row in r:
    current_row, current_col = (start_row)% mask_shape[0], 0
    for i in range(diag_length):
      #mask[current_row % mask_shape[0], current_col % mask_shape[1]] = 1
      mask[current_row % mask_shape[0], current_col % mask_shape[1]] = 1
      current_row += 1
      current_col += 1

  return mask """

def get_mask_pseudo_diagonal_numpy(mask_shape, sparsity, file_name=None, experimentType="randomWithZero", 
                                   layerNum=None, numDiag=None, diag_pos=None, currLayer=1, debug=0):
    """Creates a pseudo-diagonal mask with the specified sparsity.
    Args:
        mask_shape: list, used to obtain shape of the random mask.
        sparsity: float, between 0 and 1.
    Returns:
        numpy.ndarray
    """
    # Create an array of zeros with the specified shape
    mask = np.zeros(mask_shape)
    diag_length = max(mask_shape[0], mask_shape[1])
    
    # Ensure reproducibility
    np.random.seed(int(time.time()))
    
    start_row = int(diag_pos)
    
    # Vectorized operation to create the pseudo-diagonal mask
    rows = (np.arange(diag_length) + start_row) % mask_shape[0]
    cols = np.arange(diag_length) % mask_shape[1]
    
    mask[rows, cols] = 1

    return mask

def get_mask_pseudo_diagonal_torch(mask_shape, sparsity, diag_pos, experimentType="random",  device='cuda'):
    """Creates a pseudo-diagonal mask with the specified sparsity using PyTorch.
    Args:
        mask_shape: tuple, used to obtain shape of the random mask.
        sparsity: float, between 0 and 1.
        diag_pos: int, starting position of the diagonal.
        device: str, 'cpu' or 'cuda' for GPU.
    Returns:
        torch.Tensor
    """
    # Create an array of zeros with the specified shape
    mask = torch.zeros(mask_shape, device=device)
    diag_length = max(mask_shape[0], mask_shape[1])

    start_row = int(diag_pos)

    # Vectorized operation to create the pseudo-diagonal mask
    rows = (torch.arange(diag_length, device=device) + start_row) % mask_shape[0]
    cols = torch.arange(diag_length, device=device) % mask_shape[1]

    mask[rows, cols] = 1

    return mask