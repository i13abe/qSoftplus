import torch
import torch.nn as nn
from .qGauss import qexp, qlog

class qSoftplus(nn.Module):
    def __init__(self, q = 0.99):
        super(qSoftplus, self).__init__()
        self.q = q

        
    def forward(self, x):
        x = qlog(1 + qexp(x, self.q), self.q)
        return x
    
    
    
class ShiftedqSoftplus(nn.Module):
    def __init__(self, q = 0.0):
        super(ShiftedqSoftplus, self).__init__()
        self.q = q

        
    def forward(self, x):
        x = qlog(1 + qexp(x-(1./(1.-self.q)), self.q), self.q)
        return x