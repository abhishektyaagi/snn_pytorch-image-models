""" import torch
import torch.nn.functional as F
#import isotonic_dykstra_mask """

'''def isotonic_dykstra_mask(s, num_iter=500):'''
"""Solves an isotonic regression problem using Dykstra's projection algorithm.

    Formally, it approximates the solution of

    argmin_{v_1 >= ... >= v_n} 0.5 ||v - s||^2.

    Args:
        s: input to isotonic regression, 1d-array
        num_iter: the number of alternate steps

    Returns:
        sol: the solution, an array of the same size as y.
    """
""" def f(v):
        # Here we assume that v's length is even.
        d = v[1::2] - v[::2]
        a = v[1::2] + v[::2]
        mask = (d < 0).repeat_interleave(2)
        mean = a.repeat_interleave(2) / 2.0
        return v * mask + mean * (~mask)

    def body_fn(vpq):
        xk, pk, qk = vpq
        yk = torch.cat([f(xk[:-1] + pk[:-1]), xk[-1:] + pk[-1:]])
        p = xk + pk - yk
        v = torch.cat([xk[:1] + qk[0:1], f(yk[1:] + qk[1:])])
        q = yk + qk - v
        return v, p, q

    # Ensure that the length is odd.
    n = s.shape[0]
    if n % 2 == 0:
        minv = s.min().item() - 1
        s = torch.cat([s, torch.tensor([minv], dtype=s.dtype)])

    v = s.clone()
    p = torch.zeros_like(s)
    q = torch.zeros_like(s)
    vpq = (v, p, q)
    for _ in range(num_iter // 2):
        vpq = body_fn(vpq)
    sol = vpq[0]

    return sol if n % 2 != 0 else sol[:-1] """

'''def sparse_soft_topk_mask_dykstra(x, k, l=1e-1, num_iter=500):'''
"""Returns a differentiable approximation of the top-k mask operator of x using Dykstra's algorithm.

    Args:
        x: input to the top-k mask, a 1d-array.
        k: int k for the top k.
        l: the regularization parameter l that trades sparsity for smoothness.
        num_iter: int, number of iterations in Dykstra's projection algorithm.

    Returns:
        sol: the relaxed top-k mask of x.
    """
"""     n = x.shape[0]
    perm = torch.argsort(-x)
    P = F.one_hot(perm, n).float()
    s = P @ x
    s_w = s - l * torch.cat([torch.ones(k), torch.zeros(n - k)])
    out_dykstra = isotonic_dykstra_mask(s_w, num_iter=num_iter)
    out = (s - out_dykstra) / l
    return P.T @ out  """

import torch
import torch.nn.functional as F

def isotonic_dykstra_mask(s, num_iter=500, device='cuda'):
    def f(v):
        # Here we assume that v's length is even.
        d = v[1::2] - v[::2]
        a = v[1::2] + v[::2]
        mask = (d < 0).repeat_interleave(2)
        mean = a.repeat_interleave(2) / 2.0
        return v * mask + mean * (~mask)

    def body_fn(vpq):
        xk, pk, qk = vpq
        yk = torch.cat([f(xk[:-1] + pk[:-1]), xk[-1:] + pk[-1:]])
        p = xk + pk - yk
        v = torch.cat([xk[:1] + qk[0:1], f(yk[1:] + qk[1:])])
        q = yk + qk - v
        return v, p, q

    # Ensure that the length is odd.
    n = s.shape[0]
    if n % 2 == 0:
        minv = s.min().item() - 1
        s = torch.cat([s, torch.tensor([minv], dtype=s.dtype, device=device)])

    v = s.clone()
    p = torch.zeros_like(s)
    q = torch.zeros_like(s)
    vpq = (v, p, q)
    for _ in range(num_iter // 2):
        vpq = body_fn(vpq)
    sol = vpq[0]

    return sol if n % 2 != 0 else sol[:-1]

def sparse_soft_topk_mask_dykstra(x, k, l=1e-1, num_iter=500, device='cuda'):
    #print(x)
    n = x.shape[0]
    perm = torch.argsort(-x)
    P = F.one_hot(perm, n).float().to(device)
    #print(P)
    s = P @ x
    #print(s)
    s_w = s - l * torch.cat([torch.ones(k, device=device), torch.zeros(n - k, device=device)])
    #print(s_w)
    out_dykstra = isotonic_dykstra_mask(s_w, num_iter=num_iter, device=device)
    #print(out_dykstra)
    out = (s - out_dykstra) / l
    #print(s-out_dykstra)
    return P.T @ out
