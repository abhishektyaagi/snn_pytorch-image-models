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

""" def get_mask_pseudo_diagonal_torch(mask_shape, sparsity, diag_pos, experimentType="random", device='cuda'):
    
    # Create an array of zeros with the specified shape
    mask = torch.zeros(mask_shape, device=device)
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

    mask[rows, cols] = 1

    return mask """

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

class CustomFullyConnectedLayer(nn.Module):
    def __init__(self, in_features, out_features, device=None, sparsity = 0.1, diagPos=[], alphaLR=0.01):
        super(CustomFullyConnectedLayer, self).__init__()
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

        #Profile how long it takes to get alpha_topk
        start_alpha_topk = time.time()
        self.alpha_topk = sparse_soft_topk_mask_dykstra(self.alpha, self.K, l=self.topkLR, num_iter=50).to(self.device)

        #Make alpha_topk to be random values between 0 and 1
        #self.alpha_topk = torch.rand_like(self.alpha, device=self.device)
        alpha_topk_time = time.time() - start_alpha_topk

        #pdb.set_trace()
        print("Alpha Topk time is: ",alpha_topk_time)

        non_zero_alpha_indices = torch.nonzero(self.alpha_topk, as_tuple=False).squeeze()
        
        #print("memory after alpha_topk:{}MB ".format(torch.cuda.memory_allocated()/(1024**2)))

        if non_zero_alpha_indices.dim() == 0:
            non_zero_alpha_indices = non_zero_alpha_indices.unsqueeze(0) 

        print("Length of non_zero_alpha_indices: ", len(non_zero_alpha_indices))

        WSum = torch.zeros((self.out_features, self.in_features), device=self.device)

        mask_time_total = 0
        matmul_time_total = 0
        for i in non_zero_alpha_indices:
            #print("Iteration: ", i)
            start_mask = time.time()
            #mask1 = get_mask_pseudo_diagonal_torch((self.out_features, self.in_features), sparsity=0.99967, experimentType="randDiagOneLayer", diag_pos=i)
            mask1 = self.precomputed_masks[i].to_dense()
            mask_time_total += time.time() - start_mask

            #print("Mask Time: ",time.time() - start_mask)
            V_scaled = self.V[i] * self.alpha_topk[i]
            #print("Memory after V_scaled:{}MB ".format(torch.cuda.memory_allocated()/(1024**2)))
            
            with torch.cuda.amp.autocast(enabled=True):
                start_matmul = time.time()
                if self.out_features > self.in_features:
                    WSum += torch.matmul(mask1.to(V_scaled.dtype), torch.diag(V_scaled).to(self.device))
                else:
                    mask1 = mask1.T
                    WSum += torch.matmul(mask1.to(V_scaled.dtype), torch.diag(V_scaled)).T.to(self.device)
                matmul_time_total += time.time() - start_matmul
            
        #alpha_time, mask_time, matmul_time = compute_weights(your_class_instance)
        print(f"Time for sparse_soft_topk_mask_dykstra: {alpha_topk_time:.6f} seconds")
        print(f"Time for get_mask_pseudo_diagonal_torch: {mask_time_total:.6f} seconds")
        print(f"Time for each get_mask_pseudo_diagonal_torch:{mask_time_total/int(len(non_zero_alpha_indices)):.6f}")
        print(f"Time for matmul operations: {matmul_time_total:.6f} seconds")
        
        #Add all the times up together
        total_time = alpha_topk_time + mask_time_total + matmul_time_total
        print(f"Total time for all operations: {total_time:.6f} seconds")

        print(WSum)
        return WSum """

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

        # Prepare to time different operations
        mask_time_total = 0
        matmul_time_total = 0

        # Initialize WSum
        WSum = torch.zeros((self.out_features, self.in_features), device=self.device)

        if len(non_zero_alpha_indices) > 0:
            # Create a batch of masks and V_scaled tensors for parallel computation
            start_mask = time.time()
            masks = [self.precomputed_masks[i].to_dense().unsqueeze(0) for i in non_zero_alpha_indices]
            masks = torch.cat(masks, dim=0)  # Stack all the masks (shape: batch_size, out_features, in_features)
            mask_time_total += time.time() - start_mask

            # Compute V_scaled in parallel
            V_scaled = self.V[non_zero_alpha_indices] * self.alpha_topk[non_zero_alpha_indices].unsqueeze(1)

            # Perform batched matrix multiplication in parallel
            start_matmul = time.time()
            if self.out_features > self.in_features:
                # Broadcast and perform batched matrix multiplication
                V_scaled_diag = torch.diag_embed(V_scaled)  # Shape: (batch_size, in_features, in_features)
                WSum_batch = torch.bmm(masks.to(V_scaled_diag.dtype), V_scaled_diag.to(self.device))
            else:
                # Transpose masks and perform batched matrix multiplication
                masks = masks.transpose(1, 2)  # Transpose masks to shape: (batch_size, in_features, out_features)
                V_scaled_diag = torch.diag_embed(V_scaled)  # Shape: (batch_size, out_features, out_features)
                WSum_batch = torch.bmm(masks.to(V_scaled_diag.dtype), V_scaled_diag.to(self.device)).transpose(1, 2)

            # Sum over the batch dimension to get the final WSum
            WSum += WSum_batch.sum(dim=0)
            matmul_time_total += time.time() - start_matmul

        # Print timing information
        print(f"Time for sparse_soft_topk_mask_dykstra: {alpha_topk_time:.6f} seconds")
        print(f"Time for get_mask_pseudo_diagonal_torch: {mask_time_total:.6f} seconds")
        print(f"Time for matmul operations: {matmul_time_total:.6f} seconds")

        # Add all the times up together
        total_time = alpha_topk_time + mask_time_total + matmul_time_total
        print(f"Total time for all operations: {total_time:.6f} seconds")

        print(WSum)
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
        print(f"Execution time of W calculations: {out_time:.6f} seconds")
        #pdb.set_trace()    

        start_out = 0
        start_out = time.time()
        out = F.linear(x, W)
        out_time = time.time() - start_out
        print(f"Execution time of linear layer: {out_time:.6f} seconds")
        return out

    def update_alpha_lr(self, new_alpha_lr):
        self.topkLR = new_alpha_lr
        #print("New learning rate for alpha is: ", self.topkLR)


class CustomFullyConnectedLayerElemWise(nn.Module):
    def __init__(self, in_features, out_features, device=None, sparsity = 0.1, diagPos=[], alphaLR=0.01):
        super(CustomFullyConnectedLayerElemWise, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.total_permutations = max(in_features, out_features)
        self.diag_length = min(in_features, out_features)
        
        #print("Sparsity is: ", sparsity)    
        num_params = in_features * out_features
        req_params = int((1-sparsity) * num_params)
        K = math.ceil(req_params/min(in_features, out_features))
        print(K)
        self.K = K
        self.topkLR = alphaLR
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        #Set a seed for initialization
        torch.manual_seed(0)

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
        start_alpha_topk = time.time()
        self.alpha_topk = sparse_soft_topk_mask_dykstra(self.alpha, self.K, l=self.topkLR, num_iter=50).to(self.device)
        
        alpha_topk_time = time.time() - start_alpha_topk

        print("Alpha Topk time is: ",alpha_topk_time)

        non_zero_alpha_indices = torch.nonzero(self.alpha_topk, as_tuple=False).squeeze()

        if non_zero_alpha_indices.dim() == 0:
            non_zero_alpha_indices = non_zero_alpha_indices.unsqueeze(0) 

        WSum = torch.zeros((self.out_features, self.in_features), device=self.device)

        #pdb.set_trace()
        mask_time_total = 0
        matmul_time_total = 0
        with torch.cuda.amp.autocast(enabled=True):
            for i in non_zero_alpha_indices:
                start_mask = time.time()
                mask1 = self.precomputed_masks[i].to_dense()
                mask_time_total += time.time() - start_mask

                V_scaled = self.V[i] * self.alpha_topk[i]  # Shape: (diag_length,)
                start_matmul = time.time()
                if self.out_features >= self.in_features:
                    # Case 1
                    # mask1 shape: (out_features, in_features)
                    # V_scaled shape: (in_features,)
                    WSum += mask1.to(V_scaled.dtype) * V_scaled  # Broadcasting along columns
                else:
                    # Case 2
                    mask1 = mask1.T  # Shape: (in_features, out_features)
                    # V_scaled shape: (out_features,)
                    result = mask1.to(V_scaled.dtype) * V_scaled  # Broadcasting along columns
                    WSum += result.T  # Transpose back to (out_features, in_features)
                matmul_time_total += time.time() - start_matmul
            
        #alpha_time, mask_time, matmul_time = compute_weights(your_class_instance)
        print(f"Time for sparse_soft_topk_mask_dykstra: {alpha_topk_time:.6f} seconds")
        print(f"Time for get_mask_pseudo_diagonal_torch: {mask_time_total:.6f} seconds")
        print(f"Time for each get_mask_pseudo_diagonal_torch:{mask_time_total/int(len(non_zero_alpha_indices)):.6f}")
        print(f"Time for matmul operations: {matmul_time_total:.6f} seconds")
        
        #Add all the times up together
        total_time = alpha_topk_time + mask_time_total + matmul_time_total
        print(f"Total time for all operations: {total_time:.6f} seconds")
        print(WSum)
        return WSum """

    def compute_weights(self):
        start_alpha_topk = time.time()
        self.alpha_topk = sparse_soft_topk_mask_dykstra(self.alpha, self.K, l=self.topkLR, num_iter=50).to(self.device)
        alpha_topk_time = time.time() - start_alpha_topk

        print("Alpha Topk time is: ", alpha_topk_time)

        # Find non-zero indices in alpha_topk
        non_zero_alpha_indices = torch.nonzero(self.alpha_topk, as_tuple=False).squeeze()

        if non_zero_alpha_indices.dim() == 0:
            non_zero_alpha_indices = non_zero_alpha_indices.unsqueeze(0)

        print("Length of non_zero_alpha_indices: ", len(non_zero_alpha_indices))

        # Initialize timing variables
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
                    # masks: (batch_size, out_features, in_features)
                    # V_scaled: (batch_size, in_features) -> needs to broadcast along in_features
                    WSum += (masks.to(V_scaled.dtype) * V_scaled.unsqueeze(1)).sum(dim=0)
                else:
                    # Case 2: Transpose and broadcast along columns
                    masks = masks.transpose(1, 2)  # Shape: (batch_size, in_features, out_features)
                    WSum += (masks.to(V_scaled.dtype) * V_scaled.unsqueeze(1)).sum(dim=0).T

            matmul_time_total += time.time() - start_matmul

        # Print timing information
        print(f"Time for sparse_soft_topk_mask_dykstra: {alpha_topk_time:.6f} seconds")
        print(f"Time for get_mask_pseudo_diagonal_torch: {mask_time_total:.6f} seconds")
        print(f"Time for each get_mask_pseudo_diagonal_torch: {mask_time_total/int(len(non_zero_alpha_indices)):.6f}")
        print(f"Time for matmul operations: {matmul_time_total:.6f} seconds")

        # Add all the times up together
        total_time = alpha_topk_time + mask_time_total + matmul_time_total
        print(f"Total time for all operations: {total_time:.6f} seconds")

        print(WSum)
        return WSum



    @property
    def weights(self):
        return self.compute_weights()

    def forward(self, x):
        x = x.to(self.device)
        start_out = time.time()
        W = self.weights
        out_time = time.time() - start_out
        print(f"Execution time of W calculations: {out_time:.6f} seconds")
        #pdb.set_trace()    

        start_out = time.time()
        out = F.linear(x, W)
        out_time = time.time() - start_out
        print(f"Execution time of linear layer: {out_time:.6f} seconds")
        return out

    def update_alpha_lr(self, new_alpha_lr):
        self.topkLR = new_alpha_lr
        #print("New learning rate for alpha is: ", self.topkLR)


rows = 768
cols = 2304

layer1 = CustomFullyConnectedLayer(rows, cols, device='cuda', sparsity=0.9, diagPos=0, alphaLR=0.01)
layer2 = CustomFullyConnectedLayerElemWise(rows, cols, device='cuda', sparsity=0.9, diagPos=0, alphaLR=0.01)
layer3 = nn.Linear(rows, cols, device='cuda')

#Use the above layer for in_features=768 and out_features=2304
start_out = 0
out_time = 0
out_time2 = 0

#Fix the seed for torch randn
torch.manual_seed(0)
randInput = torch.randn(1, rows, device='cuda')
start_out = time.time()
print(randInput)
out1 = layer1(randInput)
out_time= time.time() - start_out
#Do a backward pass
back1 = out1.sum().backward()
grad1 = layer1.V.grad
out_time2= time.time() - start_out
print(f"Execution time: {out_time:.6f} seconds")
print(f"Execution time: {out_time2:.6f} seconds")

start_out = 0
out_time = 0
out_time2 = 0
torch.manual_seed(0)
randInput = torch.randn(1, rows, device='cuda')
start_out = time.time()
print(randInput)
out2 = layer2(randInput)
out_time= time.time() - start_out
#Do a backward pass
back2 = out2.sum().backward()
grad2 = layer2.V.grad
out_time2= time.time() - start_out
print(f"Execution time (Elemwise): {out_time:.6f} seconds")
print(f"Execution time (Elemwise): {out_time2:.6f} seconds")


start_out = 0
out_time = 0
out_time2 = 0
start_out = time.time()
torch.manual_seed(0)
print(randInput)
out3 = layer3(randInput)
out_time= time.time() - start_out
#Do a backward pass
back3 = out3.sum().backward()
grad3 = layer3.weight.grad
out_time2= time.time() - start_out
print(f"Execution time(nn.Linear): {out_time:.6f} seconds")
print(f"Execution time(nn.Linear): {out_time2:.6f} seconds")

import pdb
pdb.set_trace()