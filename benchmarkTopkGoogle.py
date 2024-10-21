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



#A random vector of size N
N = [10,50,100,500,1000,2000, 5000]
K = [1,2,4,8,16,32,64,128,256,512,1024,2048]
topkLR = [0.1,0.01,0.001,0.0001]

device = 'cuda'

resultsDict = {}

#alpha_warmup = torch.rand(11, device='cuda', dtype=torch.float32)
#alpha_topk_warmup = sparse_soft_topk_mask_dykstra(alpha_warmup, 4, l=0.0001, num_iter=500).to(device)

alpha_ex = torch.tensor([3.24, 3.22, 3.25, 3.29, 3.23], device='cuda', dtype=torch.float32)
alpha_ex_topk = sparse_soft_topk_mask_dykstra(alpha_ex, 3, l=1, num_iter=50).to(device)
#print(alpha_warmup)
print(alpha_ex_topk)
print("----------------------------------------------------")
alpha_ex_topk = sparse_soft_topk_mask_dykstra(alpha_ex, 3, l=0.01, num_iter=50).to(device)
print(alpha_ex_topk)
print("----------------------------------------------------")
print("----------------------------------------------------")

alpha_ex = torch.tensor([2, 5, 1, 4, 7], device='cuda', dtype=torch.float32)
alpha_ex_topk = sparse_soft_topk_mask_dykstra(alpha_ex, 3, l=1, num_iter=50).to(device)
#print(alpha_warmup)
print(alpha_ex_topk)
print("----------------------------------------------------")
alpha_ex_topk = sparse_soft_topk_mask_dykstra(alpha_ex, 3, l=0.01, num_iter=50).to(device)
print(alpha_ex_topk)
print("----------------------------------------------------")
print("----------------------------------------------------")

alpha_ex = torch.tensor([2, 2, 2, 2, 2], device='cuda', dtype=torch.float32)
alpha_ex_topk = sparse_soft_topk_mask_dykstra(alpha_ex, 3, l=1, num_iter=50).to(device)
#print(alpha_warmup)
print(alpha_ex_topk)
print("----------------------------------------------------")
alpha_ex_topk = sparse_soft_topk_mask_dykstra(alpha_ex, 3, l=0.001, num_iter=50).to(device)
print(alpha_ex_topk)
print("----------------------------------------------------")
""" pdb.set_trace() 
#Go through all the different combinations of N, K and topkLR
for n in N:
    for k in K:
        for topk in topkLR:
            alpha = torch.rand(n, device='cuda', dtype=torch.float32)
            if k <= n:
                print("N: ", n, "K: ", k, "topkLR: ", topk)
                start_alpha_topk = 0
                start_alpha_topk = time.time()
                alpha_topk = sparse_soft_topk_mask_dykstra(alpha, k, l=topk, num_iter=50).to(device)
                alpha_topk_time = time.time() - start_alpha_topk
                print("Alpha Topk time is: ", alpha_topk_time)
                print("----------------------------------------------------")
                resultsDict[(n,k,topk)] = alpha_topk_time


#Save the dictionary to a text file
with open("benchmarkTopkGoogle.txt", "w") as file:
    file.write(str(resultsDict))
    file.close()    
 """
#Plot the results