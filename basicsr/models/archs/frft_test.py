#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 16:49:33 2024

@author: dl
"""

import torch
from math import ceil
#   = torch. ("cuda:0" if (torch.cuda.is_available()) else "cpu")

def FRFT2D(x: torch.Tensor, a: float) -> torch.Tensor:
    # Apply 1D DFRFT along the last dimension (columns)
    x_transformed = dfrft(x, a, dim=-1)
    # Apply 1D DFRFT along the second to last dimension (rows)
    x_transformed = dfrft(x_transformed, a, dim=-2)
    return x_transformed

def IFRFT2D(x: torch.Tensor, a: float) -> torch.Tensor:
    # Apply 1D IDFRFT along the last dimension (columns)
    x_transformed = idfrft(x, a, dim=-1)
    # Apply 1D IDFRFT along the second to last dimension (rows)
    x_transformed = idfrft(x_transformed, a, dim=-2)
    return x_transformed

def dfrft(x: torch.Tensor, a: float, *, dim: int = -1) -> torch.Tensor:
    dfrft_matrix = dfrftmtx(x.size(dim), a)
    dtype = torch.promote_types(dfrft_matrix.dtype, x.dtype)
    return torch.einsum(
        _get_dfrft_einsum_str(len(x.shape), dim),
        dfrft_matrix.type(dtype),
        x.type(dtype),
    )

def idfrft(x: torch.Tensor, a: float, *, dim: int = -1) -> torch.Tensor:
    return dfrft(x, -a, dim=dim)

def _get_dfrft_einsum_str(dim_count: int, req_dim: int) -> str:
    if req_dim < -dim_count or req_dim >= dim_count:
        raise ValueError("Dimension size error.")
    dim = torch.remainder(req_dim, torch.tensor(dim_count))
    diff = dim_count - dim
    remaining_str = "".join([chr(num) for num in range(98, 98 + diff)])
    return f"ab,...{remaining_str}->...{remaining_str.replace('b', 'a', 1)}"

def dfrftmtx(N: int, a: float, *, approx_order: int = 2) -> torch.Tensor:
    if N < 1 or approx_order < 2:
        raise ValueError("Necessary conditions for integers: N > 1 and approx_order >= 2.")
    evecs = _get_dfrft_evecs(N, approx_order=approx_order).type(torch.complex64)
    idx = _dfrft_index(N)
    evals = torch.exp(-1j * a * (torch.pi / 2) * idx).type(torch.complex64)
    dfrft_matrix = torch.einsum("ij,j,kj->ik", evecs, evals, evecs)
    return dfrft_matrix

def _get_dfrft_evecs(N: int, *, approx_order: int = 2) -> torch.Tensor:
    if N < 1 or approx_order < 2:
        raise ValueError("Necessary conditions for integers: N > 1 and approx_order >= 2.")
    H = _create_hamiltonian(N, approx_order=approx_order)
    evals, evecs = torch.linalg.eigh(H)
    return evecs

def _circulant(x: torch.Tensor) -> torch.Tensor:
    N = x.numel()
    x = x.flip(0).tile(N)
    return x[torch.arange(N).unsqueeze(1) + torch.arange(N)]

def _dfrft_index(N: int) -> torch.Tensor:
    return torch.cat((torch.arange(ceil(N / 2)), torch.arange(-N // 2, 0)))

def _conv1d_full(vector: torch.Tensor, kernel1d: torch.Tensor) -> torch.Tensor:
    padding_size = kernel1d.size(0) - 1
    padded_input = torch.nn.functional.pad(vector, (padding_size, padding_size), mode="constant", value=0)
    conv_output = torch.conv1d(padded_input.view(1, 1, -1), kernel1d.view(1, 1, -1).flip(-1))
    return conv_output.reshape(-1)

def _create_hamiltonian(N: int, *, approx_order: int = 2) -> torch.Tensor:
    if N < 1 or approx_order < 2:
        raise ValueError("Necessary conditions for integers: N > 1 and approx_order >= 2.")
    order = approx_order // 2
    dum0 = torch.tensor([1.0, -2.0, 1.0] )
    dum = dum0.clone()
    s = torch.zeros(1)
    for k in range(1, order + 1):
        coefficient = (2 * (-1) ** (k - 1) * torch.prod(torch.arange(1, k)) ** 2 / torch.prod(torch.arange(1, 2 * k + 1)))
        s = (coefficient * torch.cat((torch.zeros(1), dum[k + 1 : 2 * k + 1], torch.zeros(N - 1 - 2 * k), dum[:k])) + s)
        dum = _conv1d_full(dum, dum0)
    return _circulant(s) + torch.diag(torch.real(torch.fft.fft(s)))

def _create_odd_even_decomp_matrix(N: int) -> torch.Tensor:
    if N < 1:
        raise ValueError("N must be positive integer.")
    x1 = torch.ones(1 + N // 2, dtype=torch.float32,  )
    x2 = -torch.ones(N - N // 2 - 1, dtype=torch.float32,)
    diagonal = torch.diag(torch.cat((x1, x2)))
    anti = torch.diag(torch.ones(N - 1), -1).rot90()
    P = (diagonal + anti) / torch.sqrt(torch.tensor(2.0))
    P[0, 0] = 1
    if N % 2 == 0:
        P[N // 2, N // 2] = 1
    return P