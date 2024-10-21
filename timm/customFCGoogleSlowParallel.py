import torch
from torch import nn
import torch.jit as jit
import math
#from maskGenerator1DiagRect import get_mask_pseudo_diagonal_torch
#from maskGenerator1Diag import get_mask_pseudo_diagonal_numpy, get_mask_pseudo_diagonal_torch
from timm.maskGenerator1DiagRect import get_mask_pseudo_diagonal_numpy, get_mask_pseudo_diagonal_torch
from timm.torch_sparse_soft_topk_google.isotonic_dykstra import isotonic_dykstra_mask
from timm.torch_sparse_soft_topk_google.topk import sparse_soft_topk_mask_dykstra
from timm.torch_sparse_soft_topk_google.isotonic_pav import sparse_soft_topk_mask_pav
from torch.profiler import profile, record_function, ProfilerActivity

class CustomFullyConnectedLayer(nn.Module):
    def __init__(self, in_features, out_features, device=None, sparsity=0.1, diagPos=[], alphaLR=0.05):
        super(CustomFullyConnectedLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.total_permutations = max(in_features, out_features)
        self.diag_length = min(in_features, out_features)

        num_params = in_features * out_features
        req_params = int((1-sparsity) * num_params)
        self.K = math.ceil(req_params / min(in_features, out_features))
        print("Number of diagonals is ", self.K)
        print("Shape is ", in_features, out_features)

        self.topkLR = alphaLR
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.V = nn.Parameter(torch.empty(self.total_permutations, self.diag_length, device=self.device, dtype=torch.float32, requires_grad=True))
        nn.init.kaiming_uniform_(self.V, a=math.sqrt(5))

        self.alpha = nn.Parameter(torch.empty(self.total_permutations, device=self.device, requires_grad=True))
        nn.init.constant_(self.alpha, 1 / self.in_features)
        assert torch.all(self.alpha >= 0)

        # Precompute masks and store them
        self.precomputed_masks = self.precompute_masks()

    def precompute_masks(self):
        masks = []
        for i in range(self.total_permutations):
            mask = get_mask_pseudo_diagonal_torch((self.out_features, self.in_features), sparsity=0.99967, experimentType="randDiagOneLayer", diag_pos=i)
            masks.append(mask)
        return torch.stack(masks).to(self.device)

    """ def compute_weights(self):
        self.alpha_topk = sparse_soft_topk_mask_dykstra(self.alpha, self.K, l=self.topkLR, num_iter=50).to(self.device)
        non_zero_alpha_indices = torch.nonzero(self.alpha_topk, as_tuple=False).squeeze()

        if non_zero_alpha_indices.dim() == 0:
            non_zero_alpha_indices = non_zero_alpha_indices.unsqueeze(0)

        WSum = torch.zeros((self.out_features, self.in_features), device=self.device)

        V_diag = torch.stack([torch.diag(self.V[i]) for i in non_zero_alpha_indices])
        masks = self.precomputed_masks[non_zero_alpha_indices]

        if self.out_features > self.in_features:
            #results = self.alpha_topk[non_zero_alpha_indices].unsqueeze(-1).unsqueeze(-1) * torch.matmul(masks, V_diag)
            results = self.alpha_topk[non_zero_alpha_indices].unsqueeze(-1).unsqueeze(-1) * torch.sparse.mm(masks, V_diag)
        else:
            masks = masks.transpose(2, 1)
            #results = self.alpha_topk[non_zero_alpha_indices].unsqueeze(-1).unsqueeze(-1) * torch.matmul(masks, V_diag).transpose(2, 1)
            results = self.alpha_topk[non_zero_alpha_indices].unsqueeze(-1).unsqueeze(-1) * torch.sparse.mm(masks, V_diag).transpose(2, 1)

        WSum += results.sum(dim=0)

        return WSum """

    def compute_weights(self):
        self.alpha_topk = sparse_soft_topk_mask_dykstra(self.alpha, self.K, l=self.topkLR, num_iter=50).to(self.device)
        non_zero_alpha_indices = torch.nonzero(self.alpha_topk, as_tuple=False).squeeze()

        if non_zero_alpha_indices.dim() == 0:
            non_zero_alpha_indices = non_zero_alpha_indices.unsqueeze(0)

        WSum = torch.zeros((self.out_features, self.in_features), device=self.device)

        # Loop over non-zero alpha indices to perform the sparse operation
        for idx in non_zero_alpha_indices:
            V_diag = torch.diag(self.V[idx])
            mask = self.precomputed_masks[idx].to_dense()  # Convert to dense only for the required slice

            if self.out_features > self.in_features:
                result = self.alpha_topk[idx] * torch.matmul(mask, V_diag)
            else:
                result = self.alpha_topk[idx] * torch.matmul(mask.T, V_diag).T

            WSum += result

        return WSum

    @property
    def weights(self):
        return self.compute_weights()

    def forward(self, x):
        x = x.to(self.device)
        W = self.weights
        out = nn.functional.linear(x, W)
        return out

    def update_alpha_lr(self, new_alpha_lr):
        self.topkLR = new_alpha_lr