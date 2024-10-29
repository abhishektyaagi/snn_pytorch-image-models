import torch
import torch.nn as nn
import torch.nn.functional as F
import types
import numpy as np
import math
import time
import pdb

#from timm.maskGenerator1DiagRect import get_mask_pseudo_diagonal_numpy, get_mask_pseudo_diagonal_torch
from timm.torch_sparse_soft_topk_google.isotonic_dykstra import isotonic_dykstra_mask
from timm.torch_sparse_soft_topk_google.topk import sparse_soft_topk_mask_dykstra
from timm.torch_sparse_soft_topk_google.isotonic_pav import sparse_soft_topk_mask_pav
torch.set_printoptions(linewidth=200)

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
    print("Temperature: ", temperature)
    # Ensure the input is on the correct device
    #Measure the execution time
    start = time.time()
    x = x.to(device)
    move_time = time.time()-start
    print("Time to move x to device: ", move_time)
    
    # Scale the input by the inverse temperature
    #Measure the execution time

    start = time.time()
    scaled_x = x / (temperature)
    scaled_time = time.time()-start
    print("Time to scale x: ", scaled_time)

    # Apply the softmax function
    #Measure the execution time
    start = time.time()
    softmax_probs = F.softmax(scaled_x, dim=0)
    softmax_time = time.time()-start
    print("Time to apply softmax: ", softmax_time)

    # Compute a "soft" top-k by emphasizing the largest k probabilities
    # We do not perform any hard masking but instead rely on the natural
    # behavior of softmax to distribute most of the probability mass on the top elements
    # Multiply the softmax output by k to ensure that approximately k elements contribute
    #Measure the execution time
    start = time.time()
    soft_topk_output = k * softmax_probs
    out_time = time.time()-start
    print("Time to compute soft topk output: ", out_time)

    # Clip the probabilities to [0, 1] range to make it similar to the hard top-k
    #Measure the execution time
    start = time.time()
    soft_topk_output = torch.clamp(soft_topk_output, 0.0, 1.0)
    clamp_time = time.time()-start
    print("Time to clip soft topk output: ", clamp_time)
    #pdb.set_trace()
    
    #Total time
    print("Total time for soft_topk_with_temperature: ", move_time+scaled_time+softmax_time+out_time+clamp_time)

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
        print(K)
        self.K = K
        self.topkLR = alphaLR
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.V = nn.Parameter(torch.empty(self.total_permutations, self.diag_length, device=self.device, dtype=torch.float32, requires_grad=True))
        nn.init.kaiming_uniform_(self.V, a=math.sqrt(5))

        self.alpha = nn.Parameter(torch.empty(self.total_permutations, device=self.device, requires_grad=True))
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

    """ def compute_weights(self):

        # Profile how long it takes to get alpha_topk
        start_alpha_topk = time.time()
        #self.alpha_topk = sparse_soft_topk_mask_dykstra(self.alpha, self.K, l=self.topkLR, num_iter=50).to(self.device)
        self.alpha_topk = soft_topk_with_temperature(self.alpha, self.K, temperature=1, device=self.device)
        alpha_topk_time = time.time() - start_alpha_topk

        print("Alpha Topk time is: ", alpha_topk_time)

        # Find non-zero indices in alpha_topk
        non_zero_alpha_indices = torch.nonzero(self.alpha_topk, as_tuple=False).squeeze()

        if non_zero_alpha_indices.dim() == 0:
            non_zero_alpha_indices = non_zero_alpha_indices.unsqueeze(0)

        print("Length of non_zero_alpha_indices: ", len(non_zero_alpha_indices))

        #Calculate the sparsity in weight matrix
        sparsity = 1 - (min(self.in_features, self.out_features)*len(non_zero_alpha_indices))/(self.in_features*self.out_features)
        print("Sparsity in weight matrix is: ", sparsity)

        # Prepare to time different operations
        mask_time_total = 0
        matmul_time_total = 0

        # Initialize WSum
        WSum = torch.zeros((self.out_features, self.in_features), device=self.device)

        if len(non_zero_alpha_indices) > 0:
            # Stack masks in parallel
            start_mask = time.time()
            masks = [self.precomputed_masks[i].to_dense().unsqueeze(0) for i in non_zero_alpha_indices]
            r_time = time.time()-start_mask
            print("Time to retrieve masks: ", r_time)
            start_mask = time.time()
            masks = torch.cat(masks, dim=0)  # Shape: (batch_size, out_features, in_features)
            s_time = time.time()-start_mask
            print("Time to stack masks: ", s_time)
            mask_time_total = s_time + r_time

            # Compute V_scaled in parallel
            V_scaled = self.V[non_zero_alpha_indices] * self.alpha_topk[non_zero_alpha_indices].unsqueeze(1)

            # Perform matrix operations in parallel
            start_matmul = time.time()
            with torch.cuda.amp.autocast(enabled=True):
                if self.out_features >= self.in_features:
                    # Case 1: Broadcast along columns
                    WSum += (masks.to(V_scaled.dtype) * V_scaled.unsqueeze(1)).sum(dim=0)
                else:
                    # Case 2: Transpose and broadcast along columns
                    masks = masks.transpose(1, 2)  # Shape: (batch_size, in_features, out_features)
                    WSum += (masks.to(V_scaled.dtype) * V_scaled.unsqueeze(1)).sum(dim=0).T

            matmul_time_total += time.time() - start_matmul

        # Print timing information
        print(f"Time for Google TopK: {alpha_topk_time:.6f} seconds")
        print(f"Time for get_mask_pseudo_diagonal_torch: {mask_time_total:.6f} seconds")
        print(f"Time for matmul operations: {matmul_time_total:.6f} seconds")

        # Add all the times up together
        total_time = alpha_topk_time + mask_time_total + matmul_time_total
        print(f"Total time for all operations: {total_time:.6f} seconds")

        return WSum """
    def compute_weights(self):

        # Profile how long it takes to get alpha_topk
        start_alpha_topk = time.time()
        #self.alpha_topk = sparse_soft_topk_mask_dykstra(self.alpha, self.K, l=self.topkLR, num_iter=50).to(self.device)
        self.alpha_topk = soft_topk_with_temperature(self.alpha, self.K, temperature=1, device=self.device)
        alpha_topk_time = time.time() - start_alpha_topk

        print("Alpha Topk time is: ", alpha_topk_time)

        # Find non-zero indices in alpha_topk
        start = time.time()
        non_zero_alpha_indices = torch.nonzero(self.alpha_topk, as_tuple=False).squeeze()
        nnz_time = time.time()-start
        print("Time to find non-zero indices: ", nnz_time)

        start = time.time() 
        if non_zero_alpha_indices.dim() == 0:
            non_zero_alpha_indices = non_zero_alpha_indices.unsqueeze(0)
        unsqueeze_time = time.time()-start
        print("Time to unsqueeze non-zero indices: ", unsqueeze_time)

        print("Length of non_zero_alpha_indices: ", len(non_zero_alpha_indices))

        # Calculate the sparsity in weight matrix
        start = time.time()
        sparsity = 1 - (min(self.in_features, self.out_features) * len(non_zero_alpha_indices)) / (self.in_features * self.out_features)
        print("Sparsity in weight matrix is: ", sparsity)
        sparsity_time = time.time()-start
        print("Time to calculate sparsity: ", sparsity_time)

        # Initialize WSum
        start = time.time()
        WSum = torch.zeros((self.out_features, self.in_features), device=self.device)
        wsum_time = time.time()-start
        print("Time to initialize WSum: ", wsum_time)

        if len(non_zero_alpha_indices) > 0:
            # Compute V_scaled in parallel
            start = time.time()
            V_scaled = self.V[non_zero_alpha_indices] * self.alpha_topk[non_zero_alpha_indices].unsqueeze(1)
            vscale_time = time.time()-start
            print("Time to compute V_scaled: ", vscale_time)

            #Measure execution time

            diag_pos_list = non_zero_alpha_indices  # Assuming diag_pos corresponds to indices
            diag_length = self.diag_length

            num_rows = self.out_features
            num_cols = self.in_features

            N = len(diag_pos_list)

            # Generate i and j indices based on the mask generation logic
            start = time.time()
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
            loop_time = time.time()-start
            print("Time to generate i and j indices: ", loop_time)


            # Flatten indices and values
            start = time.time()
            indices_i = rows.reshape(-1)
            indices_j = cols.reshape(-1)
            values = V_scaled.reshape(-1)
            reshape_time = time.time()-start
            print("Time to reshape indices and values: ", reshape_time)

            # Accumulate values into WSum
            start = time.time()
            WSum.index_put_((indices_i, indices_j), values, accumulate=True)
            index_time = time.time()-start
            print("Time to accumulate values into WSum: ", index_time)

        # Print timing information
        total_time = alpha_topk_time + nnz_time + unsqueeze_time + sparsity_time + wsum_time + vscale_time + loop_time + reshape_time + index_time
        print(f"Total time for all operations: {total_time:.6f} seconds")
        print(f"Total time of the compute_weight function: {time.time()-start_alpha_topk:.6f} seconds")

        return WSum


    @property
    def weights(self):
        return self.compute_weights()

    def forward(self, x):
        x = x.to(self.device)
        start_out = 0
        start_out = time.time()
        W = self.weights
        out_time = time.time() - start_out
        print(f"Execution time of W calculations(GoogleTopK): {out_time:.6f} seconds")
        #pdb.set_trace()    

        start_out = 0
        start_out = time.time()
        out = F.linear(x, W)
        out_time = time.time() - start_out
        print(f"Execution time of linear layer(GoogleTopK): {out_time:.6f} seconds")
        return out

    def update_alpha_lr(self, new_alpha_lr):
        self.topkLR = new_alpha_lr
        #print("New learning rate for alpha is: ", self.topkLR)
#_______________________________________________________________________________________________________________________

class CustomFullyConnectedLayerGoogleTopK2(nn.Module):
    def __init__(self, in_features, out_features, device=None, sparsity = 0.1, diagPos=[], alphaLR=0.01):
        super(CustomFullyConnectedLayerGoogleTopK2, self).__init__()
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
        print(K)
        self.K = K
        self.topkLR = alphaLR
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.V = nn.Parameter(torch.empty(self.total_permutations, self.diag_length, device=self.device, dtype=torch.float32, requires_grad=True))
        nn.init.kaiming_uniform_(self.V, a=math.sqrt(5))

        self.alpha = nn.Parameter(torch.empty(self.total_permutations, device=self.device, requires_grad=True))
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

        # Profile how long it takes to get alpha_topk
        start_alpha_topk = time.time()
        self.alpha_topk = sparse_soft_topk_mask_dykstra(self.alpha, self.K, l=self.topkLR, num_iter=50).to(self.device)
        alpha_topk_time = time.time() - start_alpha_topk

        print("Alpha Topk time is: ", alpha_topk_time)

        # Find non-zero indices in alpha_topk
        non_zero_alpha_indices = torch.nonzero(self.alpha_topk, as_tuple=False).squeeze()

        if non_zero_alpha_indices.dim() == 0:
            non_zero_alpha_indices = non_zero_alpha_indices.unsqueeze(0)

        print("Length of non_zero_alpha_indices: ", len(non_zero_alpha_indices))

        #Calculate the sparsity in weight matrix
        sparsity = 1 - (min(self.in_features, self.out_features)*len(non_zero_alpha_indices))/(self.in_features*self.out_features)
        print("Sparsity in weight matrix is: ", sparsity)

        # Prepare to time different operations
        mask_time_total = 0
        matmul_time_total = 0

        # Initialize WSum
        WSum = torch.zeros((self.out_features, self.in_features), device=self.device)

        if len(non_zero_alpha_indices) > 0:
            # Stack masks in parallel
            start_mask = time.time()
            masks = [self.precomputed_masks[i].to_dense().unsqueeze(0) for i in non_zero_alpha_indices]
            masks = torch.cat(masks, dim=0)  # Shape: (batch_size, out_features, in_features)
            mask_time_total += time.time() - start_mask

            # Compute V_scaled in parallel
            V_scaled = self.V[non_zero_alpha_indices] * self.alpha_topk[non_zero_alpha_indices].unsqueeze(1)

            # Perform matrix operations in parallel
            start_matmul = time.time()
            with torch.cuda.amp.autocast(enabled=True):
                if self.out_features >= self.in_features:
                    # Case 1: Broadcast along columns
                    WSum += (masks.to(V_scaled.dtype) * V_scaled.unsqueeze(1)).sum(dim=0)
                else:
                    # Case 2: Transpose and broadcast along columns
                    masks = masks.transpose(1, 2)  # Shape: (batch_size, in_features, out_features)
                    WSum += (masks.to(V_scaled.dtype) * V_scaled.unsqueeze(1)).sum(dim=0).T

            matmul_time_total += time.time() - start_matmul

        # Print timing information
        print(f"Time for Google TopK: {alpha_topk_time:.6f} seconds")
        print(f"Time for get_mask_pseudo_diagonal_torch: {mask_time_total:.6f} seconds")
        print(f"Time for matmul operations: {matmul_time_total:.6f} seconds")

        # Add all the times up together
        total_time = alpha_topk_time + mask_time_total + matmul_time_total
        print(f"Total time for all operations: {total_time:.6f} seconds")

        return WSum

    @property
    def weights(self):
        return self.compute_weights()

    def forward(self, x):
        x = x.to(self.device)
        start_out = 0
        start_out = time.time()
        W = self.weights
        out_time = time.time() - start_out
        print(f"Execution time of W calculations(GoogleTopK): {out_time:.6f} seconds")
        #pdb.set_trace()    

        start_out = 0
        start_out = time.time()
        out = F.linear(x, W)
        out_time = time.time() - start_out
        print(f"Execution time of linear layer(GoogleTopK): {out_time:.6f} seconds")
        return out

    def update_alpha_lr(self, new_alpha_lr):
        self.topkLR = new_alpha_lr
        #print("New learning rate for alpha is: ", self.topkLR)

class CustomFullyConnectedLayerGoogleTopKNew(nn.Module):
    def __init__(self, in_features, out_features, device=None, sparsity=0.1, alphaLR=0.01):
        super(CustomFullyConnectedLayerGoogleTopKNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.total_permutations = max(in_features, out_features)
        self.diag_length = min(in_features, out_features)

        torch.manual_seed(0)

        num_params = in_features * out_features
        req_params = int((1 - sparsity) * num_params)
        K = math.ceil(req_params / min(in_features, out_features))
        print(K)
        self.K = K
        self.topkLR = alphaLR
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.V = nn.Parameter(torch.empty(self.total_permutations, self.diag_length, device=self.device, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.V, a=math.sqrt(5))

        self.alpha = nn.Parameter(torch.empty(self.total_permutations, device=self.device))
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


rows = 4
cols = 4

layer1 = CustomFullyConnectedLayerGoogleTopK(rows, cols, device='cuda', sparsity=0.9, diagPos=0, alphaLR=0.01)
layer2 = CustomFullyConnectedLayerGoogleTopKNew(rows, cols, device='cuda', sparsity=0.9, alphaLR=0.01)
#layer2 = CustomFullyConnectedLayerElemWise(rows, cols, device='cuda', sparsity=0.9, diagPos=0, alphaLR=0.01)
layer3 = nn.Linear(rows, cols, device='cuda')

#Use the above layer for in_features=768 and out_features=2304
start_out = 0
out_time = 0
out_time2 = 0

#Fix the seed for torch randn
torch.manual_seed(0)
randInput = torch.randn(1, rows, device='cuda')
start_out = time.time()
#print(randInput)

times_soft_topk = []
times_soft_topk2 = []

for _ in range(10):
    start_out = time.time()
    out1 = layer1(randInput)
    out_time = time.time() - start_out
    print("_________________________________________________________")
    # Do a backward pass
    back1 = out1.sum().backward()
    grad1 = layer1.V.grad
    out_time2 = time.time() - start_out

    times_soft_topk.append(out_time)
    times_soft_topk2.append(out_time2)
    randInput = torch.randn(1, rows, device='cuda')

#Plot the times on a single curve with  times_soft_topk labelled as time (forward pass) and times_soft_topk2 labelled as time (backward pass)
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(times_soft_topk, label='Time (Forward Pass)', marker='o')
plt.plot(times_soft_topk2, label='Time (Backward Pass)', marker='x')
plt.xlabel('Iteration')
plt.ylabel('Time (seconds)')
#use log scale

plt.yscale('log')
#Add annotation to each point uptyo 6 decimal points
for i, txt in enumerate(times_soft_topk):
    plt.annotate(f'{txt:.6f}', (i, times_soft_topk[i]), textcoords="offset points", xytext=(0,10), ha='center')
for i, txt in enumerate(times_soft_topk2):
    plt.annotate(f'{txt:.6f}', (i, times_soft_topk2[i]), textcoords="offset points", xytext=(0,10), ha='center')

plt.title('Execution Time for Forward and Backward Passes using Softmax based Top-K')
plt.legend()
plt.grid(True)
plt.savefig('execution_time_soft.png')

#print(f"Execution time (SoftTopk): {out_time:.6f} seconds")
#print(f"Execution time (SoftTopk): {out_time2:.6f} seconds")
print("_________________________________________________________")

start_out = 0
out_time = 0
out_time2 = 0
start_out = time.time()
torch.manual_seed(0)

times_soft_topk = []
times_soft_topk2 = []

for _ in range(10):
    start_out = time.time()
    out1 = layer2(randInput)
    out_time = time.time() - start_out
    print("_________________________________________________________")
    # Do a backward pass
    back1 = out1.sum().backward()
    grad1 = layer1.V.grad
    out_time2 = time.time() - start_out

    times_soft_topk.append(out_time)
    times_soft_topk2.append(out_time2)
    randInput = torch.randn(1, rows, device='cuda')

#Plot the times on a single curve with  times_soft_topk labelled as time (forward pass) and times_soft_topk2 labelled as time (backward pass)
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(times_soft_topk, label='Time (Forward Pass)', marker='o')
plt.plot(times_soft_topk2, label='Time (Backward Pass)', marker='x')
plt.xlabel('Iteration')
plt.ylabel('Time (seconds)')
#use log scale

plt.yscale('log')
#Add annotation to each point uptyo 6 decimal points
for i, txt in enumerate(times_soft_topk):
    plt.annotate(f'{txt:.6f}', (i, times_soft_topk[i]), textcoords="offset points", xytext=(0,10), ha='center')
for i, txt in enumerate(times_soft_topk2):
    plt.annotate(f'{txt:.6f}', (i, times_soft_topk2[i]), textcoords="offset points", xytext=(0,10), ha='center')

plt.title('Execution Time for Forward and Backward Passes using Softmax based Top-K')
plt.legend()
plt.grid(True)
plt.savefig('execution_time_soft_new.png')

#print(f"Execution time (SoftTopk): {out_time:.6f} seconds")
#print(f"Execution time (SoftTopk): {out_time2:.6f} seconds")
print("_________________________________________________________")



start_out = 0
out_time = 0
out_time2 = 0
start_out = time.time()
torch.manual_seed(0)
#print(randInput)
times_soft_topk = []
times_soft_topk2 = []

for _ in range(10):
    start_out = time.time()
    out1 = layer3(randInput)
    out_time = time.time() - start_out
    print("_________________________________________________________")
    # Do a backward pass
    back1 = out1.sum().backward()
    grad1 = layer1.V.grad
    out_time2 = time.time() - start_out

    times_soft_topk.append(out_time)
    times_soft_topk2.append(out_time2)
    randInput = torch.randn(1, rows, device='cuda')

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(times_soft_topk, label='Time (Forward Pass)', marker='o')
plt.plot(times_soft_topk2, label='Time (Backward Pass)', marker='x')
plt.xlabel('Iteration')
plt.ylabel('Time (seconds)')
#Add annotation to each point uptyo 6 decimal points
for i, txt in enumerate(times_soft_topk):
    plt.annotate(f'{txt:.6f}', (i, times_soft_topk[i]), textcoords="offset points", xytext=(0,10), ha='center')
for i, txt in enumerate(times_soft_topk2):
    plt.annotate(f'{txt:.6f}', (i, times_soft_topk2[i]), textcoords="offset points", xytext=(0,10), ha='center')


plt.yscale('log')
plt.title('Execution Time for Forward and Backward Passes using Google TopK')
plt.legend()
plt.grid(True)
plt.savefig('execution_time_linear.png')

#print(f"Execution time(nn.Linear): {out_time:.6f} seconds")
#print(f"Execution time(nn.Linear): {out_time2:.6f} seconds")
print("_________________________________________________________")

import pdb
pdb.set_trace()