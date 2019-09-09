# PyTorch-Reparam-Module
Reparameterize your PyTorch modules

## Requirements

+ [PyTorch](https://pytorch.org) with version at least `1.2.0`
+ Python 3

## Example

```py
import torch
import torchvision
from torchreparam import ReparamModule

device = torch.device('cuda')

reparam_vgg11 = ReparamModule(
    torchvision.models.vgg11().to(device),
    example_input=(torch.randn(1, 3, 224, 224, device=device),)
)

def maml_loss(param, input, loss_fn, lr=0.01):
    out1 = reparam_vgg11(input, flat_param=param)
    gparam, = torch.autograd.grad(loss_fn(out1), param, create_graph=True)
    updated_param = param - lr * gparam
    return loss_fn(reparam_vgg11(input, flat_param=updated_param))


trained_param = torch.randn_like(reparam_vgg11.flat_param).mul_(0.001).requires_grad_()
input = torch.randn(1, 3, 224, 224, device=device)
l = maml_loss(trained_param, input, loss_fn=torch.norm)
l.backward()

print(trained_param.grad)
```

## Installation

```sh
python setup.py install
```

## Documentation

For a `ReparamModule`, the following fields are available:

+ `.flat_param`: a flattened parameter vector representing all parameteres of the wrapped module.
+ `.param_numel`: the total number of parameters, i.e., the size of `.flat_param`.

A `ReparamModule` can be called with the following signatire:

```py
reparam_module(self, *inputs, flat_param=None, buffers=None)
```

where
+ `inputs` will be passed over as inputs to the inner module.
+ `flat_param` will be used as the parameter of this forward pass, if specified. Note that this allows you easily activate a network on an entirely different set of parameters, and backprop to them.
+ `buffers` will be used as the buffers for this forward pass, if specified (experimental).

Note
+ `ReparamModule` currently does not work properly with Batch Normalization layers.
+ The inner module must be moved to desired device and/or dtype, before being wrapped in `ReparamModule`.
