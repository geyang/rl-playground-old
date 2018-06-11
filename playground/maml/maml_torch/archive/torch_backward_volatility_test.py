import torch
from moleskin import Moleskin
from torch_helpers import varify, volatile

from debug import graph

M = Moleskin()

x = varify(torch.randn(4, 2))
loss = x.sum()
assert x.grad is None

loss.backward(varify(torch.ones(1)), retain_graph=True)
assert x.grad.volatile is False, "gradient is never volatile"

def test_pytorch_grad():
    """NOTE: volatile can only be set on leaf variables. pyTorch enforces this."""
    try:
        x.grad.volatile = True
    except RuntimeError as e:
        assert str(e) == "volatile can only be set on leaf variables"
        return
    raise Exception('pyTorch did not enforce gradient non-volatility.')

test_pytorch_grad()

# However, there is a way to get around it.
x.grad = volatile(torch.ones(1).expand_as(x))
assert x.grad.volatile is True
loss.backward(varify(torch.ones(1)), retain_graph=True)
assert x.grad.volatile is True

# So in order to avoid setting volatile=True on the gradient, always operate
# on the data attribute instead of the gradient operator itself.
#
# Directly operating on the data attribute has problems too, because it preserves the subgraph that leads
# to the variable.
# The best way is to reinstantiate a variable using the data.

# M.debug(x.grad)
# graph(x.grad, block=True)
