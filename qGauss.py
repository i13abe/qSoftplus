import torch
import torch.nn as nn
import torch.nn.functional as F


def qexp(x, q):
    x = 1 + (1.-q)*(x)
    x = F.relu(x).clamp(min=1e-8)
    return x**(1/(1.-q))


def qlog(x, q):
    x = F.relu(x).clamp(min=1e-8)
    x = x**(1.-q)
    x = x - 1.
    return x*(1./(1. - q))



class qExp(nn.Module):
    def __init__(self, q):
        super(qExp, self).__init__()
        self.q = q
        
        
    def forward(self, x):
        return qexp(x, self.q)
    
    
    
class qLog(nn.Module):
    def __init__(self, q):
        super(qLog, self).__init__()
        self.q = q
        
        
    def forward(self, x):
        return qlog(x, self.q)