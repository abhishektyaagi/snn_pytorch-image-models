import torch
import torch.nn.functional as F


def isotonic_dykstra_mask(s, num_iter=500):
    """Solves an isotonic regression problem using Dykstra's projection algorithm.

    Formally, it approximates the solution of

    argmin_{v_1 >= ... >= v_n} 0.5 ||v - s||^2.

    Args:
        s: input to isotonic regression, 1d-array
        num_iter: the number of alternate steps

    Returns:
        sol: the solution, an array of the same size as y.
    """
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
        s = torch.cat([s, torch.tensor([minv], dtype=s.dtype)])

    v = s.clone()
    p = torch.zeros_like(s)
    q = torch.zeros_like(s)
    vpq = (v, p, q)
    for _ in range(num_iter // 2):
        vpq = body_fn(vpq)
    sol = vpq[0]

    return sol if n % 2 != 0 else sol[:-1]