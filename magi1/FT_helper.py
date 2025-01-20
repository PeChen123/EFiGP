import torch
import gpytorch
import torch.fft as ft
from linear_operator import to_linear_operator 
torch.set_default_dtype(torch.double)

def FourierGP(matrix, k):
        pre = ft.fft(matrix)
        pre1 = torch.fft.fft(pre.t().conj()).real
        pre2 = torch.fft.fft(pre.t()).real
        pre3 = torch.fft.fft(pre.t().conj()).imag
        pre4 = torch.fft.fft(pre.t()).imag
        final1t = 0.5 * (pre1 + pre2)[:k, :k]         # top left
        final2t = 0.5 * (pre1 - pre2)[1:k, 1:k]       # bottom right
        final3t = 0.5 * (pre3 + pre4)[:k, 1:k]       # top right
        final4t = -0.5 * (pre3 - pre4)[1:k, :k]
        truncated_matrix = torch.cat((torch.cat((final1t, final4t), 0), 
                                    torch.cat((final3t, final2t), 0)), 1)
        new_matrix = to_linear_operator(truncated_matrix)
        return new_matrix

def EigenGP(matrix, z_t):
    eval, evec = torch.linalg.eigh(matrix)
    eval_tilde = eval[-z_t:]
    evec_tilde= evec[:, -z_t:]
    eval_sqrt = torch.diag(torch.sqrt(eval_tilde)) # 161x161
    return eval_sqrt, evec_tilde
    
def Fourier_vetor(vector, k):
    xr_ft = ft.fft(vector)[:k]
    rp,ip = xr_ft.real,xr_ft.imag[1:k]
    xr_ft = torch.cat((rp,ip),0)
    return xr_ft