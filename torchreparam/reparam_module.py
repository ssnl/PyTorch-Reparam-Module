import torch
import torch.nn as nn
import warnings
import types
from collections import namedtuple
from contextlib import contextmanager


class ReparamModule(nn.Module):
    def __init__(self, module):
        super(ReparamModule, self).__init__()
        self.module = module

        param_infos = []
        params = []
        param_numels = []
        param_shapes = []
        for m in self.modules():
            for n, p in m.named_parameters(recurse=False):
                if p is not None:
                    param_infos.append((m, n))
                    params.append(p.detach())
                    param_numels.append(p.numel())
                    param_shapes.append(p.size())

        assert len(set(p.dtype for p in params)) <= 1, \
            "expects all parameters in module to have same dtype"

        # store the info for unflatten
        self._param_infos = tuple(param_infos)
        self._param_numels = tuple(param_numels)
        self._param_shapes = tuple(param_shapes)

        # flatten
        flat_param = nn.Parameter(torch.cat([p.reshape(-1) for p in params], 0))
        self.register_parameter('flat_param', flat_param)
        self.param_numel = flat_param.numel()
        del params

        # deregister the names as parameters
        for m, n in self._param_infos:
            delattr(m, n)

        # register the views as plain attributes
        self._unflatten_param(self.flat_param)

        # now buffers
        # they are not reparametrized. just store info as (module, name, buffer)
        buffer_infos = []
        for m in self.modules():
            for n, b in m.named_buffers(recurse=False):
                if b is not None:
                    buffer_infos.append((m, n, b))

        self._buffer_infos = tuple(buffer_infos)
        self._traced_self = None

    def trace(self, example_input, **trace_kwargs):
        assert self._traced_self is None, 'This ReparamModule is already traced'

        if isinstance(example_input, torch.Tensor):
            example_input = (example_input,)
        example_input = tuple(example_input)
        example_param = (self.flat_param.detach().clone(),)
        example_buffers = (tuple(b.detach().clone() for _, _, b in self._buffer_infos),)

        self._traced_self = torch.jit.trace_module(
            self,
            inputs=dict(
                _forward_with_param=example_param + example_input,
                _forward_with_param_and_buffers=example_param + example_buffers + example_input,
            ),
            **trace_kwargs,
        )

        # replace forwards with traced versions
        self._forward_with_param = self._traced_self._forward_with_param
        self._forward_with_param_and_buffers = self._traced_self._forward_with_param_and_buffers
        return self

    def clear_views(self):
        for m, n in self._param_infos:
            setattr(m, n, None)  # This will set as plain attr

    def _apply(self, *args, **kwargs):
        if self._traced_self is not None:
            self._traced_self._apply(*args, **kwargs)
            return self
        return super(ReparamModule, self)._apply(*args, **kwargs)

    def _unflatten_param(self, flat_param):
        ps = (t.view(s) for (t, s) in zip(flat_param.split(self._param_numels), self._param_shapes))
        for (m, n), p in zip(self._param_infos, ps):
            setattr(m, n, p)  # This will set as plain attr

    @contextmanager
    def unflattened_param(self, flat_param):
        saved_views = [getattr(m, n) for m, n in self._param_infos]
        self._unflatten_param(flat_param)
        yield
        # Why not just `self._unflatten_param(self.flat_param)`?
        # 1. because of https://github.com/pytorch/pytorch/issues/17583
        # 2. slightly faster since it does not require reconstruct the split+view
        #    graph
        for (m, n), p in zip(self._param_infos, saved_views):
            setattr(m, n, p)

    @contextmanager
    def replaced_buffers(self, buffers):
        for (m, n, _), new_b in zip(self._buffer_infos, buffers):
            setattr(m, n, new_b)
        yield
        for m, n, old_b in self._buffer_infos:
            setattr(m, n, old_b)

    def _forward_with_param_and_buffers(self, flat_param, buffers, *inputs, **kwinputs):
        with self.unflattened_param(flat_param):
            with self.replaced_buffers(buffers):
                return self.module(*inputs, **kwinputs)

    def _forward_with_param(self, flat_param, *inputs, **kwinputs):
        with self.unflattened_param(flat_param):
            return self.module(*inputs, **kwinputs)

    def forward(self, *inputs, flat_param=None, buffers=None, **kwinputs):
        if flat_param is None:
            flat_param = self.flat_param
        if buffers is None:
            return self._forward_with_param(flat_param, *inputs, **kwinputs)
        else:
            return self._forward_with_param_and_buffers(flat_param, tuple(buffers), *inputs, **kwinputs)
