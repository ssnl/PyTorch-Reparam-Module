# PyTorch-Reparam-Module
Reparameterize your PyTorch modules

## Requirements

+ [PyTorch](https://pytorch.org) `>= 1.2.0`
+ Python 3

## Example

```py
import torch
import torch.nn.functional as F
import torchvision
from torchreparam import ReparamModule

device = torch.device('cuda')

# A regular network
net = torchvision.models.resnet18().to(device)

# Reparametrize it!
reparam_net = ReparamModule(net)

print(f"reparam_net has {reparam_net.param_numel} parameters")

assert tuple(reparam_net.parameters()) == (reparam_net.flat_param,)
print(f"reparam_net now has **only one** vector parameter of shape {reparam_net.flat_param.shape}")

# The reparametrized module is equivalent with the original one.
# In fact, the weights share storage.
dummy_input_image = torch.randn(1, 3, 224, 224, device=device)
print(f'original net output for class 746: {net(dummy_input_image)[0, 746]}')
print(f'reparam_net output for class 746: {reparam_net(dummy_input_image)[0, 746]}')

# We can optionally trace the forward method with PyTorch JIT so it runs faster.
# To do so, we can call `.trace` on the reparamtrized module with dummy inputs
# expected by the module.
# Comment out this following line if you do not want to trace.
reparam_net = reparam_net.trace(dummy_input_image)


# Example on a MAML loss that
#   1. Train `theta_0` on `inner_train` for `num_gd_steps` gradient descent steps with `lr`.
#   2. Compute the loss of the updated parameter on `inner_val`.
#
# This assumes classification with cross entropy loss, but can be easily adapted
# to other loss functions.
def maml_loss(reparam_net, theta_0, inner_train, inner_val, num_gd_steps=5, lr=0.01):
    # train stage
    train_data, train_label = inner_train
    theta = theta_0
    for _ in range(num_gd_steps):
        # perform GD update on (data, label) w.r.t. theta
        loss = F.cross_entropy(reparam_net(train_data, flat_param=theta), train_label)
        gtheta, = torch.autograd.grad(loss, theta, create_graph=True)  # create_graph=True for backprop through this
        # update
        theta = theta - lr * gtheta

    # val stage
    # theta is now the final updated set of parameters
    val_data, val_label = inner_val
    return F.cross_entropy(reparam_net(val_data, flat_param=theta), val_label)

# Let's use the above function:

# Initialize our theta_0 that we want to train
theta_0 = torch.randn_like(reparam_net.flat_param).mul_(0.001).requires_grad_()
# Make dummy data
inner_train = (
    torch.randn(2, 3, 224, 224, device=device),                 # input
    torch.randint(low=0, high=1000, size=(2,), device=device),  # label
)
inner_val = (
    torch.randn(5, 3, 224, 224, device=device),                 # input
    torch.randint(low=0, high=1000, size=(5,), device=device),  # label
)
l = maml_loss(reparam_net, theta_0, inner_train, inner_val)
l.backward()

# Here, easily backprop-able by autograd.
print(f'MAML loss gradient for theta_0:\n{theta_0.grad}')
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
+ `ReparamModule` currently does not work properly with Batch Normalization layers with the default `track_running_stats=True`.
