# PyTorch-Reparam-Module
Reparameterize your PyTorch modules

## Requirements

+ [PyTorch](https://pytorch.org) with version at least `1.0.1`
+ Python 3

## Example

```py
import torch
import torchvision
from torchreparam import ReparamModule

dev = torch.device('cuda')

reparam_vgg11 = ReparamModule(
    torchvision.models.vgg11().to(dev),
    example_input=(torch.randn(1, 3, 224, 224, device=dev),)
)

def maml_loss(param, input, loss_fn, lr=0.01):
    out1 = reparam_vgg11(input, flat_param=param)
    gparam, = torch.autograd.grad(loss_fn(out1), param, create_graph=True)
    updated_param = param - lr * gparam
    return loss_fn(reparam_vgg11(input, flat_param=updated_param))


trained_param = torch.randn(reparam_vgg11.param_numel, device=dev).mul_(0.001).requires_grad_()
input = torch.randn(1, 3, 224, 224, device=dev)
l = maml_loss(trained_param, input, loss_fn=torch.norm)
torch.autograd.grad(l, trained_param)
```

## Installation

```sh
python setup.py install
```
