# qSoftplus
q-Softplus Function: Extensions of Activation Function and Loss Function by using q-Space

The q-Softplus is an extensions of softplus function by using q-space.
This work is accepted ACPR 2021.
The detail for the q-Softplus can be found in xxxxxx.

In this GitHub, we provide the implementation of q-Softplus and shifted q-Softpus on PyTorch.

# Instllation
Requirements:
+ pytorch 1.5+

this requirments is just my development enviroment.

Please manual install to get this package:
```
git clone https://github.com/i13abe/qSoftplus.git
```


# How to use the q-Softplus
We provide the qsoftplus_tutorial.ipynb in this repository.
If you can use jupyter notebook or jupter lab, please use this demonstrate file.

Also you can use soon like below
```
from q_softplus import qSoftplus, ShiftedqSoftplus

qs = qSoftplus(q=0.1)
input = torch.randn(2)
output = qs(input)

sqs = ShiftedqSoftplus(q=0.1)
input = torch.randn(2)
output = sqs(input)
```
If you want to use our proposal from ReLU in any networks, you can use same as ReLU.
