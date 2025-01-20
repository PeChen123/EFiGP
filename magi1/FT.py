import torch
import torch.fft as ft
from linear_operator import operators, to_linear_operator 

torch.set_default_dtype(torch.double)

def FT_matrix(matrix, k):
        pre = ft.fft(LK)
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
        LK = to_linear_operator(truncated_matrix)
        
        return LK
