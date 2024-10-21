import torch
import torch.nn as nn
import torch.nn.functional as F
import types
import numpy as np
import math
import pdb
#from maskGenerator1Diag import get_mask_pseudo_diagonal_numpy, get_mask_pseudo_diagonal_torch
from timm.maskGenerator1DiagRect import get_mask_pseudo_diagonal_numpy, get_mask_pseudo_diagonal_torch
from timm.torch_sparse_soft_topk_google.isotonic_dykstra import isotonic_dykstra_mask
from timm.torch_sparse_soft_topk_google.topk import sparse_soft_topk_mask_dykstra
from timm.torch_sparse_soft_topk_google.isotonic_pav import sparse_soft_topk_mask_pav
from torch.utils.checkpoint import checkpoint   

seed = 5
torch.manual_seed(seed)

class CustomFullyConnectedLayer(nn.Module):
    def __init__(self, in_features, out_features, device=None, sparsity = 0.1, diagPos=[], alphaLR=0.01):
        super(CustomFullyConnectedLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.total_permutations = max(in_features, out_features)
        self.diag_length = min(in_features, out_features)
        
        print("Sparsity is: ", sparsity)    
        num_params = in_features * out_features
        req_params = int((1-sparsity) * num_params)
        K = math.ceil(req_params/min(in_features, out_features))

        self.K = K
        self.topkLR = alphaLR
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.V = nn.Parameter(torch.empty(self.total_permutations, self.diag_length, device=self.device, dtype=torch.float32, requires_grad=True))
        nn.init.kaiming_uniform_(self.V, a=math.sqrt(5))

        self.alpha = nn.Parameter(torch.empty(self.total_permutations, device=self.device, requires_grad=True))
        nn.init.constant_(self.alpha, 1/self.in_features)
        #pdb.set_trace()
        assert torch.all(self.alpha >= 0)

    def compute_weights(self):
        self.alpha_topk = sparse_soft_topk_mask_dykstra(self.alpha, self.K, l=self.topkLR, num_iter=50).to(self.device)
        non_zero_alpha_indices = torch.nonzero(self.alpha_topk, as_tuple=False).squeeze()
        
        if non_zero_alpha_indices.dim() == 0:
            non_zero_alpha_indices = non_zero_alpha_indices.unsqueeze(0) 
        
        WSum = torch.zeros((self.out_features, self.in_features), device=self.device)
        
        for i in non_zero_alpha_indices:
            mask1 = get_mask_pseudo_diagonal_torch((self.out_features, self.in_features), sparsity=0.99967, experimentType="randDiagOneLayer", diag_pos=i)
            #mask1 = mask1.detach()

            V_scaled = self.V[i] * self.alpha_topk[i]

            with torch.cuda.amp.autocast(enabled=False):
                if self.out_features >= self.in_features:
                    #mask2 = mask1.T==1
                    WSum += torch.where(mask1.T,V_scaled.unsqueeze(1).expand_as(mask1.T),0).T
                else:
                    #mask2 = mask1==1
                    WSum += torch.where(mask1,V_scaled.unsqueeze(1).expand_as(mask1),0)
        #Print percentage of non-zeros in WSum
        print("Percentage of non-zeros in WSum: ", 100*(torch.count_nonzero(WSum).item()/WSum.numel()))
        return WSum

    @property
    def weights(self):
        return self.compute_weights()

    def forward(self, x):
        x = x.to(self.device)
        W = self.weights
        #pdb.set_trace()    

        out = F.linear(x, W)
        return out

    def update_alpha_lr(self, new_alpha_lr):
        self.topkLR = new_alpha_lr
        #print("New learning rate for alpha is: ", self.topkLR) 

