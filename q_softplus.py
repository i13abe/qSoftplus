import torch
import torch.nn as nn

EPS = 1e-6

def qexp(x, q):
    m = nn.ReLU()
    y = m(1. + (1.-q)*x)
    y = (y + EPS)**(1./(1.-q))
    return y

def qlog(x, q):
    m = nn.ReLU()
    y = m(x)
    y = (y + EPS)**(1.-q)
    y = y - 1.
    y = (1./(1. - q))*y
    return y

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