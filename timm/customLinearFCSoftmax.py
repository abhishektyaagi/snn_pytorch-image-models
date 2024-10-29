import torch
import torch.nn as nn
import torch.nn.functional as F
import types
import numpy as np
import math
import time
import pdb

torch.set_printoptions(linewidth=120)  # Adjust line width to control line breaks

def soft_topk_with_temperature(x, k, temperature=1e-2, device='gpu'):
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
    soft_topk_output = k * softmax_probs

    # Clip the probabilities to [0, 1] range to make it similar to the hard top-k
    soft_topk_output = torch.clamp(soft_topk_output, 0.0, 1.0)

    return soft_topk_output

def get_mask_pseudo_diagonal_torch(mask_shape, sparsity, diag_pos, experimentType="random", device='cuda'):

    # Create an array of zeros with the specified shape and boolean type
    mask = torch.zeros(mask_shape, dtype=torch.bool, device=device)
    num_rows, num_cols = mask_shape

    if num_rows >= num_cols:
        # Case when there are more rows than columns
        diag_length = num_cols
        start_row = int(diag_pos)
        rows = (torch.arange(diag_length, device=device) + start_row) % num_rows
        cols = torch.arange(diag_length, device=device) % num_cols
    else:
        # Case when there are more columns than rows
        diag_length = num_rows
        start_col = int(diag_pos)
        rows = torch.arange(diag_length, device=device) % num_rows
        cols = (torch.arange(diag_length, device=device) + start_col) % num_cols

    mask[rows, cols] = True

    return mask


class CustomFullyConnectedLayerSoftmax(nn.Module):
    def __init__(self, in_features, out_features, device=None, sparsity=0.1, alphaLR=0.01):
        super(CustomFullyConnectedLayerSoftmax, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.total_permutations = max(in_features, out_features)
        self.diag_length = min(in_features, out_features)

        torch.manual_seed(0)

        num_params = in_features * out_features
        req_params = int((1 - sparsity) * num_params)
        K = math.ceil(req_params / min(in_features, out_features))

        self.K = K
        self.topkLR = alphaLR
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.V = nn.Parameter(torch.empty(self.total_permutations, self.diag_length, device=self.device, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.V, a=math.sqrt(5))

        self.alpha = nn.Parameter(torch.empty(self.total_permutations, device=self.device))
        
        """ self.alpha.data.fill_(0)
        non_zero_indices = torch.randperm(self.total_permutations)[:4]
        self.alpha.data[non_zero_indices] = 1 """

        nn.init.constant_(self.alpha, 1 / self.in_features)

        assert torch.all(self.alpha >= 0)

    def forward(self, x):
        x = x.to(self.device)

        # Compute alpha_topk
        self.alpha_topk = soft_topk_with_temperature(self.alpha, self.K, temperature=1, device=self.device)
        
        non_zero_alpha_indices = torch.nonzero(self.alpha_topk, as_tuple=False).squeeze()

        if non_zero_alpha_indices.dim() == 0:
            non_zero_alpha_indices = non_zero_alpha_indices.unsqueeze(0)

        if len(non_zero_alpha_indices) == 0:
            # If no non-zero alphas, output zero tensor
            output = torch.zeros(x.size(0), self.out_features, device=self.device)
            return output

        # Compute V_scaled
        V_scaled = self.V[non_zero_alpha_indices] * self.alpha_topk[non_zero_alpha_indices].unsqueeze(1)

        diag_pos_list = non_zero_alpha_indices
        diag_length = self.diag_length
        num_rows = self.out_features
        num_cols = self.in_features
        N = len(diag_pos_list)

        # Generate indices
        if num_rows >= num_cols:
            start_row = diag_pos_list.unsqueeze(1)
            rows = (torch.arange(diag_length, device=self.device).unsqueeze(0) + start_row) % num_rows
            cols = torch.arange(diag_length, device=self.device).unsqueeze(0).expand(N, -1)
        else:
            start_col = diag_pos_list.unsqueeze(1)
            rows = torch.arange(diag_length, device=self.device).unsqueeze(0).expand(N, -1)
            cols = (torch.arange(diag_length, device=self.device).unsqueeze(0) + start_col) % num_cols

        # Flatten indices and values
        indices_i = rows.reshape(-1)
        indices_j = cols.reshape(-1)
        values = V_scaled.reshape(-1)

        # Compute the output directly
        x_t = x.transpose(0, 1)
        output = torch.zeros(x.size(0), self.out_features, device=self.device)

        # Use index_add_ to compute the output
        multiplied_values = x_t[indices_j] * values.unsqueeze(1)
        output.index_add_(1, indices_i, multiplied_values.transpose(0, 1))
        return output

    def update_alpha_lr(self, new_alpha_lr):
        self.topkLR = new_alpha_lr


class CustomFullyConnectedLayerGoogleTopK(nn.Module):
    def __init__(self, in_features, out_features, device=None, sparsity = 0.1, diagPos=[], alphaLR=0.01):
        super(CustomFullyConnectedLayerGoogleTopK, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.total_permutations = max(in_features, out_features)
        self.diag_length = min(in_features, out_features)
        
        #Set a seed for initialization
        torch.manual_seed(0)

        #print("Sparsity is: ", sparsity)    
        num_params = in_features * out_features
        req_params = int((1-sparsity) * num_params)
        K = math.ceil(req_params/min(in_features, out_features))
        self.K = K
        self.topkLR = alphaLR
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.V = nn.Parameter(torch.empty(self.total_permutations, self.diag_length, device=self.device, dtype=torch.float32, requires_grad=True))
        nn.init.kaiming_uniform_(self.V, a=math.sqrt(5))
        #print(self.V)

        self.alpha = nn.Parameter(torch.empty(self.total_permutations, device=self.device, requires_grad=True))

        """ self.alpha.data.fill_(0)
        non_zero_indices = torch.randperm(self.total_permutations)[:4]
        self.alpha.data[non_zero_indices] = 1
        """
                
        nn.init.constant_(self.alpha, 1/self.in_features)
        
        #pdb.set_trace()
        assert torch.all(self.alpha >= 0)

        #Precompute the masks
        # Precompute the masks in sparse COO format
        self.precomputed_masks = self.precompute_masks()

    def precompute_masks(self):
        masks = []
        for i in range(self.total_permutations):
            mask = get_mask_pseudo_diagonal_torch(
                (self.out_features, self.in_features), 
                sparsity=0.99967,  # Adjust this value as needed
                diag_pos=i, 
                experimentType="randDiagOneLayer", 
                device=self.device
            )
            masks.append(mask)
        return masks

    def compute_weights(self):

        #self.alpha_topk = sparse_soft_topk_mask_dykstra(self.alpha, self.K, l=self.topkLR, num_iter=50).to(self.device)
        self.alpha_topk = soft_topk_with_temperature(self.alpha, self.K, temperature=1, device=self.device)

        # Find non-zero indices in alpha_topk
        non_zero_alpha_indices = torch.nonzero(self.alpha_topk, as_tuple=False).squeeze()

        if non_zero_alpha_indices.dim() == 0:
            non_zero_alpha_indices = non_zero_alpha_indices.unsqueeze(0)

        #print("Length of non_zero_alpha_indices: ", len(non_zero_alpha_indices))

        # Calculate the sparsity in weight matrix
        """ start = time.time()
        sparsity = 1 - (min(self.in_features, self.out_features) * len(non_zero_alpha_indices)) / (self.in_features * self.out_features)
        print("Sparsity in weight matrix is: ", sparsity)
        sparsity_time = time.time()-start
        print("Time to calculate sparsity: ", sparsity_time) """

        # Initialize WSum
        WSum = torch.zeros((self.out_features, self.in_features), device=self.device)
    
        if len(non_zero_alpha_indices) > 0:
            # Compute V_scaled in parallel
            V_scaled = self.V[non_zero_alpha_indices] * self.alpha_topk[non_zero_alpha_indices].unsqueeze(1)
            
            #Measure execution time

            diag_pos_list = non_zero_alpha_indices  # Assuming diag_pos corresponds to indices
            diag_length = self.diag_length

            num_rows = self.out_features
            num_cols = self.in_features

            N = len(diag_pos_list)

            # Generate i and j indices based on the mask generation logic

            if num_rows >= num_cols:
                # Case when there are more rows than columns
                # diag_length = num_cols
                start_row = diag_pos_list.unsqueeze(1)  # Shape: (N, 1)
                rows = (torch.arange(diag_length, device=self.device).unsqueeze(0) + start_row) % num_rows  # Shape: (N, diag_length)
                cols = torch.arange(diag_length, device=self.device).unsqueeze(0).expand(N, -1)  # Shape: (N, diag_length)
            else:
                # Case when there are more columns than rows
                # diag_length = num_rows
                start_col = diag_pos_list.unsqueeze(1)  # Shape: (N, 1)
                rows = torch.arange(diag_length, device=self.device).unsqueeze(0).expand(N, -1)  # Shape: (N, diag_length)
                cols = (torch.arange(diag_length, device=self.device).unsqueeze(0) + start_col) % num_cols  # Shape: (N, diag_length)

            # Flatten indices and values
            indices_i = rows.reshape(-1)
            indices_j = cols.reshape(-1)
            values = V_scaled.reshape(-1)

            # Accumulate values into WSum
            WSum.index_put_((indices_i, indices_j), values, accumulate=True)

        """ # Print timing information
        total_time = alpha_topk_time + nnz_time + unsqueeze_time + sparsity_time + wsum_time + vscale_time + loop_time + reshape_time + index_time
        print(f"Total time for all operations: {total_time:.6f} seconds")
        print(f"Total time of the compute_weight function: {time.time()-start_alpha_topk:.6f} seconds")"""

        return WSum


    @property
    def weights(self):
        return self.compute_weights()

    def forward(self, x):
        x = x.to(self.device)
        W = self.weights
        print(W)
        #pdb.set_trace()    

        out = F.linear(x, W)
        return out

    def update_alpha_lr(self, new_alpha_lr):
        self.topkLR = new_alpha_lr
        #print("New learning rate for alpha is: ", self.topkLR)

#Debug
""" rows = 10
cols = 10

layer1 = CustomFullyConnectedLayerSoftmax(rows, cols, device='cuda', sparsity=0.9, alphaLR=0.01)

#Fix the seed for torch randn
torch.manual_seed(0)
randInput = torch.randn(1, rows, device='cuda')
print(randInput)

out1 = layer1(randInput)
print(out1)
# Do a backward pass
back1 = out1.sum().backward()
grad1 = layer1.V.grad
print("_________________________________________________________")

layer2 = CustomFullyConnectedLayerGoogleTopK(rows, cols, device='cuda', sparsity=0.9, alphaLR=0.01)

out2 = layer2(randInput)
print(out2)
# Do a backward pass
back2 = out2.sum().backward()
grad2 = layer2.V.grad
print("_________________________________________________________") """