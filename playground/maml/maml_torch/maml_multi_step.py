"""
reports a RuntimeError: a leaf Variable that requires grad has been used in an in-place operation.

The reported testing should be ran with 5 new data points that are different from the 5-shot adaptation data.
"""
from collections import OrderedDict
import torch as t
import numpy as np
from params_proto import cli_parse, BoolFlag, Proto
from waterbear import Bear, DefaultBear
from moleskin import moleskin as M
from ml_logger import logger, os
import torch_helpers as h
import torch.nn.functional as F
from torch.autograd import Variable


class FunctionalGRU:
    """
    reference: https://pytorch.org/docs/stable/nn.html?highlight=lstm#torch.nn.GRUCell
    """
    h0 = None
    is_recurrent = True

    def __init__(self, input_n, output_n, hidden_n=None):
        bias_n = 10
        input_n += bias_n
        self.input_n = input_n
        self.output_n = output_n
        self.hidden_n = hidden_n or output_n
        self.params = Bear(
            bias_var=Variable(t.empty(1, bias_n, dtype=t.double), requires_grad=True),
            # combine the weights for different gates
            w_ir=Variable(t.DoubleTensor(3 * hidden_n, input_n), requires_grad=True),
            b_ir=Variable(t.DoubleTensor(3 * hidden_n), requires_grad=True),
            w_hh=Variable(t.DoubleTensor(3 * hidden_n, hidden_n), requires_grad=True),
            b_hh=Variable(t.DoubleTensor(3 * hidden_n), requires_grad=True),
            w_out=Variable(t.DoubleTensor(output_n, hidden_n), requires_grad=True),
            b_out=Variable(t.DoubleTensor(output_n), requires_grad=True),
        )
        self.reset_parameters()

    criteria = t.nn.MSELoss()

    def state_dict(self):
        return vars(self.params)

    def reset_parameters(self):
        for name, p in self.named_parameters():
            if name.startswith('w') or name.startswith('u'):
                init.xavier_normal_(p)
                print(name, 'xavier', f'max: {p.max().item():.2f}, min: {p.min().item():.2f}')
            else:
                init.uniform_(p)
                print(name, 'uniform', f'max: {p.max().item():.2f}, min: {p.min().item():.2f}')

    def h0_init(self):
        if self.h0 is None:
            # LSTM as h and c concatenated.
            self.h0 = t.zeros(1, self.hidden_n, dtype=t.double)
        return self.h0

    def step_forward(self, x, h=None):
        """run the forward inference by one timestep"""
        params = self.params
        if h is None:
            h = t.zeros(self.hidden_n).unsqueeze(-1)
        _x = t.cat([x, params.bias_var.repeat(x.shape[0], 1)], dim=-1)
        s = _x @ params.w_ih.t() + params.b_ih + h @ params.w_hh.t() + params.b_hh
        _ = F.sigmoid(s[:, :self.hidden_n * 2])
        r = _[:, :self.hidden_n]
        z = _[:, self.hidden_n:]
        n = F.tanh(s[:, self.hidden_n * 2:])
        h = (1 - z) * n + z * h
        # note: only works with Shape(n,) inputs. No squeezing
        # to respect the signature (y, h).
        return F.tanh(h @ params.w_out.t() + params.b_out), h

    def __call__(self, x, h0):
        """

        :param x: Size(seq_len, batch, input_size)
        :param h0: Size(batch, hidden_size)
        :return: Size(seq_len, batch, output_size)
        """
        assert len(x.shape) >= 3, "x need to be of the shape (seq_len, batch, input_size)"
        assert len(h0.shape) >= 2, "h need to be of the shape (batch, input_size)"
        h, ys = h0, []
        for i, x_ in enumerate(x):
            y, h = self.step_forward(x_, h)
            ys.append(y.unsqueeze(0))
        return t.cat(ys), h

    def parameters(self):
        return (v for k, v in self.params.items())

    def named_parameters(self):
        return self.params.items()


class FunctionalAutoGRU(FunctionalGRU):
    is_autoregressive = True
    is_eval = False

    def eval(self):
        self.is_eval = True

    def train(self):
        self.is_eval = False

    def __init__(self, input_n, output_n, hidden_n):
        super(FunctionalAutoGRU, self).__init__(input_n + output_n, output_n, hidden_n)

    # noinspection PyMethodOverriding
    def step_forward(self, x, h, y_tm):
        _ = t.cat([x, y_tm], dim=-1)
        return super(FunctionalAutoGRU, self).step_forward(_, h)

    # noinspection PyMethodOverriding
    def __call__(self, x, h0, label=None):
        """

        :param x: Size(seq_len, batch, input_size)
        :param h0: Size(batch, hidden_size)
        :param y: Size(seq_len, batch, output_size) or None
        :return: Size(seq_len, batch, output_size)
        """
        assert len(x.shape) >= 3, "x need to be of the shape (seq_len, batch, input_size)"
        assert len(h0.shape) >= 2, "h need to be of the shape (batch, input_size)"
        # y_0 == 0 is the unique starting token.
        y_t, h, ys = t.zeros(1, 1, dtype=t.double), h0, []
        for i, x_t in enumerate(x):
            # teacher forcing: always use label data
            if self.is_eval or label is None or i == 0:
                y_t, h = self.step_forward(x_t, h, y_t)
            else:
                y_t, h = self.step_forward(x_t, h, label[i - 1])
            ys.append(y_t.unsqueeze(0))
        return t.cat(ys), h


class FunctionalLSTM:
    h0 = None
    is_recurrent = True

    def __init__(self, input_n, output_n, hidden_n=None):
        bias_n = 10
        input_n += bias_n
        self.input_n = input_n
        self.output_n = output_n
        self.hidden_n = hidden_n or output_n
        self.params = Bear(
            bias_var=Variable(t.empty(1, bias_n, dtype=t.double), requires_grad=True),
            # combine the weights for different gates
            w_ih=Variable(t.DoubleTensor(4 * hidden_n, input_n), requires_grad=True),
            b_ih=Variable(t.DoubleTensor(4 * hidden_n), requires_grad=True),
            w_hh=Variable(t.DoubleTensor(4 * hidden_n, hidden_n), requires_grad=True),
            b_hh=Variable(t.DoubleTensor(4 * hidden_n), requires_grad=True),
            w_out=Variable(t.DoubleTensor(output_n, hidden_n), requires_grad=True),
            b_out=Variable(t.DoubleTensor(output_n), requires_grad=True),
        )
        self.reset_parameters()

    criteria = t.nn.MSELoss()

    def state_dict(self):
        return vars(self.params)

    def reset_parameters(self):
        for name, p in self.named_parameters():
            if name.startswith('w') or name.startswith('u'):
                init.xavier_normal_(p)
                print(name, 'xavier', f'max: {p.max().item():.2f}, min: {p.min().item():.2f}')
            else:
                init.uniform_(p)
                print(name, 'uniform', f'max: {p.max().item():.2f}, min: {p.min().item():.2f}')

    def h0_init(self):
        if self.h0 is None:
            # LSTM as h and c concatenated.
            self.h0 = t.zeros(1, self.hidden_n * 2, dtype=t.double)
        return self.h0

    def step_forward(self, x, hc=None):
        """run the forward inference by one timestep"""
        params = self.params
        if hc is None:
            h = t.zeros(self.hidden_n).unsqueeze(-1)
            c = t.zeros(self.hidden_n).unsqueeze(-1)
        else:
            h = hc[:, :self.hidden_n]
            c = hc[:, self.hidden_n:]
        _x = t.cat([x, params.bias_var.repeat(x.shape[0], 1)], dim=-1)
        z = _x @ params.w_ih.t() + params.b_ih + h @ params.w_hh.t() + params.b_hh
        a = F.sigmoid(z[:, :self.hidden_n * 3])
        i = a[:, :self.hidden_n]
        f = a[:, self.hidden_n:self.hidden_n * 2]
        o = a[:, self.hidden_n * 2:]
        g = F.tanh(z[:, self.hidden_n * 3:self.hidden_n * 4])
        c = c * f + g * i
        h = F.tanh(c) * o
        # note: only works with Shape(n,) inputs. No squeezing
        # to respect the signature (y, h).
        return F.tanh(h @ params.w_out.t() + params.b_out), t.cat([h, c], dim=-1)

    def __call__(self, x, h0):
        """

        :param x: Size(seq_len, batch, input_size)
        :param h0: Size(batch, hidden_size)
        :return: Size(seq_len, batch, output_size)
        """
        assert len(x.shape) >= 3, "x need to be of the shape (seq_len, batch, input_size)"
        assert len(h0.shape) >= 2, "h need to be of the shape (batch, input_size)"
        h, ys = h0, []
        for i, x_ in enumerate(x):
            y, h = self.step_forward(x_, h)
            ys.append(y.unsqueeze(0))
        return t.cat(ys), h

    def parameters(self):
        return (v for k, v in self.params.items())

    def named_parameters(self):
        return self.params.items()


class FunctionalAutoLSTM(FunctionalLSTM):
    is_autoregressive = True
    is_eval = False

    def eval(self):
        self.is_eval = True

    def train(self):
        self.is_eval = False

    def __init__(self, input_n, output_n, hidden_n):
        super(FunctionalAutoLSTM, self).__init__(input_n + output_n, output_n, hidden_n)

    # noinspection PyMethodOverriding
    def step_forward(self, x, h, y_tm):
        _ = t.cat([x, y_tm], dim=-1)
        return super(FunctionalAutoLSTM, self).step_forward(_, h)

    # noinspection PyMethodOverriding
    def __call__(self, x, h0, label=None):
        """

        :param x: Size(seq_len, batch, input_size)
        :param h0: Size(batch, hidden_size)
        :param y: Size(seq_len, batch, output_size) or None
        :return: Size(seq_len, batch, output_size)
        """
        assert len(x.shape) >= 3, "x need to be of the shape (seq_len, batch, input_size)"
        assert len(h0.shape) >= 2, "h need to be of the shape (batch, input_size)"
        # y_0 == 0 is the unique starting token.
        y_t, h, ys = t.zeros(1, 1, dtype=t.double), h0, []
        for i, x_t in enumerate(x):
            # teacher forcing: always use label data
            if self.is_eval or label is None or i == 0:
                y_t, h = self.step_forward(x_t, h, y_t)
            else:
                y_t, h = self.step_forward(x_t, h, label[i - 1])
            ys.append(y_t.unsqueeze(0))
        return t.cat(ys), h


class FunctionalRNN:
    h0 = None
    is_recurrent = True

    def __init__(self, input_n, output_n, hidden_n):
        self.input_n = input_n
        self.output_n = output_n
        self.hidden_n = hidden_n
        bias_n = 10
        self.params = Bear(
            bias_var=Variable(t.empty(1, bias_n, dtype=t.double), requires_grad=True),
            w1=Variable(t.DoubleTensor(100, input_n + hidden_n + bias_n), requires_grad=True),
            b1=Variable(t.DoubleTensor(100), requires_grad=True),
            w2=Variable(t.DoubleTensor(100, 100), requires_grad=True),
            b2=Variable(t.DoubleTensor(100), requires_grad=True),
            w3=Variable(t.DoubleTensor(output_n + hidden_n, 100), requires_grad=True),
            b3=Variable(t.DoubleTensor(output_n + hidden_n), requires_grad=True),
        )
        self.reset_parameters()

    criteria = t.nn.MSELoss()

    def state_dict(self):
        return vars(self.params)

    def reset_parameters(self):
        for name, p in self.named_parameters():
            if name.startswith('w'):
                init.xavier_normal_(p)
                print(name, 'xavier', f'max: {p.max().item():.2f}, min: {p.min().item():.2f}')
            else:
                init.uniform_(p)
                print(name, 'uniform', f'max: {p.max().item():.2f}, min: {p.min().item():.2f}')

    def h0_init(self):
        if self.h0 is None:
            self.h0 = t.zeros(1, self.hidden_n, dtype=t.double)
        return self.h0

    def step_forward(self, x, h):
        """run the forward inference by one timestep"""
        params = self.params
        _x = t.cat([x, h, params.bias_var.repeat(x.shape[0], 1)], dim=-1)
        o = F.linear(_x, params.w1, params.b1)
        o = F.relu(o)
        o = F.linear(o, params.w2, params.b2)
        o = F.relu(o)
        o = F.linear(o, params.w3, params.b3)
        # note: only works with Shape(n,) inputs. No squeezing
        return o[:, :self.output_n], o[:, self.output_n:]

    def __call__(self, x, h0):
        """

        :param x: Size(seq_len, batch, input_size)
        :param h0: Size(batch, hidden_size)
        :return: Size(seq_len, batch, output_size)
        """
        assert len(x.shape) >= 3, "x need to be of the shape (seq_len, batch, input_size)"
        assert len(h0.shape) >= 2, "h need to be of the shape (batch, input_size)"
        h, ys = h0, []
        for i, x_ in enumerate(x):
            y, h = self.step_forward(x_, h)
            ys.append(y.unsqueeze(0))
        return t.cat(ys), h

    def parameters(self):
        return (v for k, v in self.params.items())

    def named_parameters(self):
        return self.params.items()


class FunctionalAutoRNN(FunctionalRNN):
    """ Auto-regressive RNN """
    is_autoregressive = True
    is_eval = False

    def eval(self):
        self.is_eval = True

    def train(self):
        self.is_eval = False

    def __init__(self, input_n, output_n, hidden_n):
        super(FunctionalAutoRNN, self).__init__(input_n + output_n, output_n, hidden_n)

    # noinspection PyMethodOverriding
    def step_forward(self, x, h, y_tm):
        _ = t.cat([x, y_tm], dim=-1)
        return super(FunctionalAutoRNN, self).step_forward(_, h)

    # noinspection PyMethodOverriding
    def __call__(self, x, h0, label=None):
        """

        :param x: Size(seq_len, batch, input_size)
        :param h0: Size(batch, hidden_size)
        :param y: Size(seq_len, batch, output_size) or None
        :return: Size(seq_len, batch, output_size)
        """
        assert len(x.shape) >= 3, "x need to be of the shape (seq_len, batch, input_size)"
        assert len(h0.shape) >= 2, "h need to be of the shape (batch, input_size)"
        # y_0 == 0 is the unique starting token.
        y_t, h, ys = t.zeros(1, 1, dtype=t.double), h0, []
        for i, x_t in enumerate(x):
            # teacher forcing: always use label data
            if self.is_eval or label is None or i == 0:
                y_t, h = self.step_forward(x_t, h, y_t)
            else:
                y_t, h = self.step_forward(x_t, h, label[i - 1])
            ys.append(y_t.unsqueeze(0))
        return t.cat(ys), h


class FunctionalMLP:
    def __init__(self, input_n, output_n):
        self.params = Bear(
            bias_var=Variable(t.empty(1, 10, dtype=t.double), requires_grad=True),
            w1=Variable(t.DoubleTensor(100, input_n + 10), requires_grad=True),
            b1=Variable(t.DoubleTensor(100), requires_grad=True),
            w2=Variable(t.DoubleTensor(100, 100), requires_grad=True),
            b2=Variable(t.DoubleTensor(100), requires_grad=True),
            w3=Variable(t.DoubleTensor(output_n, 100), requires_grad=True),
            b3=Variable(t.DoubleTensor(output_n), requires_grad=True),
        )
        self.reset_parameters()

    criteria = t.nn.MSELoss()

    def state_dict(self):
        return vars(self.params)

    def reset_parameters(self):
        for name, p in self.named_parameters():
            if name.startswith('w'):
                init.xavier_normal_(p)
                print(name, 'xavier', f'max: {p.max().item():.2f}, min: {p.min().item():.2f}')
            else:
                init.uniform_(p)
                print(name, 'uniform', f'max: {p.max().item():.2f}, min: {p.min().item():.2f}')

    def __call__(self, x):
        params = self.params
        _x = t.cat([x, params.bias_var.repeat(x.shape[0], 1)], dim=-1)
        o = F.linear(_x, params.w1, params.b1)
        o = F.relu(o)
        o = F.linear(o, params.w2, params.b2)
        o = F.relu(o)
        o = F.linear(o, params.w3, params.b3)
        return o

    def parameters(self):
        return (v for k, v in self.params.items())

    def named_parameters(self):
        return self.params.items()


from datetime import datetime

now = datetime.now()


@cli_parse
class G:
    # log_dir = "http://54.71.92.65:8081"
    log_dir = os.path.realpath(f'../../../../../ins-runs/')
    # log_dir = "/tmp/maml_torch"
    log_prefix = f'{now:%Y-%m-%d}/maml-debug'
    seed = 19024
    alpha = 0.001
    beta = 0.001
    debug = False
    n_epochs = 80000
    task_batch_n = 25
    k_shot = 10
    n_gradient_steps = 5
    test_grad_steps = Proto([0, 1, 2, 3, 4, 5],
                            help='run test_fn when the grad_ind matches the element inside this list.')
    test_interval = Proto(5, help="The frequency at which we run the test function `test_fn`")
    save_interval = Proto(100, dtype=int, help="interval (of epochs) to save the network weights.")
    test_mode = BoolFlag(True, help="boolean flag for test model. False by default.")


import torch.nn as nn
import torch.nn.init as init


class StandardMLP(nn.Module):
    def __init__(self, input_n, output_n, **_):
        super(StandardMLP, self).__init__()
        self.add_module('model', nn.Sequential(
            nn.Linear(input_n, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, output_n), ), )
        self.reset_parameters()

    def gradients(self):
        """generator for parameter gradients """
        for p in self.parameters():
            yield p.grad

    @staticmethod
    def _weight_init(m):
        if isinstance(m, nn.Linear):
            init.xavier_normal_(m.weight)
            init.uniform_(m.bias)

    def reset_parameters(self):
        self.apply(self._weight_init)

    def forward(self, *args):
        return self.model(*args)


def sgd_baseline(lr=0.001):
    from playground.maml.maml_torch.tasks import Sine
    task = Sine()
    model = StandardMLP(1, 1) if G.debug else FunctionalMLP(1, 1)

    adam = t.optim.Adam([p for p in model.parameters()], lr=lr)
    mse = t.nn.MSELoss()
    for ep_ind in range(1000):
        xs, labels = h.const(task.proper())
        ys = model(xs.unsqueeze(-1))
        loss = mse(ys, labels.unsqueeze(-1))
        logger.log(ep_ind, loss=loss.item(), silent=ep_ind % 50)
        adam.zero_grad()
        loss.backward()
        adam.step()
    logger.flush()


def reptile(model=None, test_fn=None):
    from playground.maml.maml_torch.tasks import Sine

    model = model or FunctionalMLP(1, 1)

    meta_optimizer = t.optim.Adam(model.parameters(), lr=G.beta)
    mse = t.nn.MSELoss()

    M.tic('epoch')
    for ep_ind in range(G.n_epochs):
        M.split('epoch')
        original_ps = OrderedDict(model.named_parameters())

        tasks = [Sine() for _ in range(G.task_batch_n)]

        for task_ind, task in enumerate(tasks):
            if task_ind != 0:
                model.params.update(original_ps)
            xs, labels = t.DoubleTensor(task.samples(G.k_shot))
            _silent = task_ind != 0
            for grad_ind in range(G.n_gradient_steps):
                if hasattr(model, "is_autoregressive") and model.is_autoregressive:
                    h0 = model.h0_init()
                    ys, ht = model(xs.view(G.k_shot, 1, 1), h0, labels.view(G.k_shot, 1, 1))
                    ys = ys.squeeze(-1)  # ys:Size(5, batch_n:1, 1).
                elif hasattr(model, "is_recurrent") and model.is_recurrent:
                    h0 = model.h0_init()
                    ys, ht = model(xs.view(G.k_shot, 1, 1), h0)
                    ys = ys.squeeze(-1)  # ys:Size(5, batch_n:1, 1).
                else:
                    ys = model(xs.unsqueeze(-1))
                    ht = None
                loss = mse(ys, labels.unsqueeze(-1))
                with t.no_grad():
                    logger.log_keyvalue(ep_ind, f"{task_ind}-grad-{grad_ind}-loss", loss.item(), silent=_silent)
                    if callable(test_fn) and ep_ind % G.test_interval == 0 and grad_ind in G.test_grad_steps:
                        test_fn(model, task, task_id=task_ind, epoch=ep_ind, grad_step=grad_ind, h0=ht, silent=_silent)
                dps = t.autograd.grad(loss, model.parameters())
                with t.no_grad():
                    for (name, p), dp in zip(model.named_parameters(), dps):
                        model.params[name] = p - G.alpha * dp
                        model.params[name].requires_grad = True

            grad_ind = G.n_gradient_steps
            with t.no_grad():
                # domain adaptation
                if hasattr(model, "is_autoregressive") and model.is_autoregressive:
                    h0 = model.h0_init()
                    ys, ht = model(xs.view(G.k_shot, 1, 1), h0, labels.view(G.k_shot, 1, 1))
                    ys = ys.squeeze(-1)  # ys:Size(5, batch_n:1, 1).
                elif hasattr(model, "is_recurrent") and model.is_recurrent:
                    h0 = model.h0_init()
                    ys, ht = model(xs.view(G.k_shot, 1, 1), h0)
                    ys = ys.squeeze(-1)  # ys:Size(5, batch_n:1, 1).
                else:
                    ys = model(xs.unsqueeze(-1))
                    ht = None
                loss = mse(ys, labels.unsqueeze(-1))
                logger.log_keyvalue(ep_ind, f"{task_ind}-grad-{grad_ind}-loss", loss.item(), silent=_silent)

                if callable(test_fn) and \
                        ep_ind % G.test_interval == 0 and grad_ind in G.test_grad_steps:
                    test_fn(model, task, task_id=task_ind, epoch=ep_ind, grad_step=grad_ind, h0=ht, silent=_silent)

            # Compute REPTILE 1st-order gradient
            with t.no_grad():
                for name, p in original_ps.items():
                    # let's do the division at the end.
                    p.grad = (0 if p.grad is None else p.grad) + (p - model.params[name])  # / G.task_batch_n

        with t.no_grad():
            for name, p in original_ps.items():
                # let's do the division at the end.
                p.grad /= G.task_batch_n

        model.params.update(original_ps)
        meta_optimizer.step()
        meta_optimizer.zero_grad()

        if G.save_interval and ep_ind % G.save_interval == 0:
            logger.log_module(ep_ind, **{type(model).__name__: model})

    logger.flush()


def standard_sine_test(model, task, task_id, epoch, grad_step, silent=False, h0=None):
    with t.no_grad():
        xs, labels = t.DoubleTensor(task.samples(G.k_shot))
        # xs, labels = t.DoubleTensor(task.samples(G.k_shot))
        if hasattr(model, "is_recurrent") and model.is_recurrent:
            h0 = model.h0_init() if h0 is None else h0
            ys, ht = model(xs.view(G.k_shot, 1, 1), h0)
            ys = ys.squeeze(-1)  # ys:Size(5, batch_n:1, 1).
        else:
            ys = model(xs.unsqueeze(-1))
        mse_acc = model.criteria(ys, labels.unsqueeze(-1))
        logger.log_keyvalue(epoch, f"{task_id}-grad-{grad_step}-test-loss", mse_acc.item(), silent=silent)


def maml(model=None, test_fn=None):
    from playground.maml.maml_torch.tasks import Sine

    model = model or (StandardMLP(1, 1) if G.debug else FunctionalMLP(1, 1))
    meta_optimizer = t.optim.Adam(model.parameters(), lr=G.beta)
    mse = t.nn.MSELoss()

    M.tic('start')
    M.tic('epoch')
    for ep_ind in range(G.n_epochs):
        dt = M.split('epoch', silent=True)
        dt_ = M.toc('start', silent=True)
        print(f"epoch {ep_ind} @ {dt:.4f}sec/ep, {dt_:.1f} sec from start")
        original_ps = OrderedDict(model.named_parameters())

        tasks = [Sine() for _ in range(G.task_batch_n)]

        for task_ind, task in enumerate(tasks):

            if task_ind != 0:
                model.params.update(original_ps)
            if G.test_mode:
                _gradient = original_ps['bias_var'].grad
                if task_ind == 0:
                    assert _gradient is None or _gradient.sum().item() == 0, f"{_gradient} is not zero or None, epoch {ep_ind}."
                else:
                    assert _gradient.sum().item() != 0, f"{_gradient} should be non-zero"
                assert (original_ps['bias_var'] == model.params[
                    'bias_var']).all().item() == 1, 'the two parameters should be the same'

            xs, labels = t.DoubleTensor(task.samples(G.k_shot))

            _silent = task_ind != 0

            for grad_ind in range(G.n_gradient_steps):
                if hasattr(model, "is_autoregressive") and model.is_autoregressive:
                    h0 = model.h0_init()
                    ys, ht = model(xs.view(G.k_shot, 1, 1), h0, labels.view(G.k_shot, 1, 1))
                    ys = ys.squeeze(-1)  # ys:Size[5, batch_n: 1, 1]
                elif hasattr(model, "is_recurrent") and model.is_recurrent:
                    h0 = model.h0_init()
                    ys, ht = model(xs.view(G.k_shot, 1, 1), h0)
                    ys = ys.squeeze(-1)  # ys:Size[5, batch_n: 1, 1]
                else:
                    ys = model(xs.unsqueeze(-1))
                    ht = None

                loss = mse(ys, labels.unsqueeze(-1))
                logger.log_keyvalue(ep_ind, f"{task_ind}-grad-{grad_ind}-loss", loss.item(), silent=_silent)
                if callable(test_fn) and \
                        ep_ind % G.test_interval == 0 and grad_ind in G.test_grad_steps:
                    test_fn(model, task=task, task_id=task_ind, epoch=ep_ind, grad_step=grad_ind, silent=_silent, h0=ht)
                dps = t.autograd.grad(loss, model.parameters(), create_graph=True, retain_graph=True)
                # 1. update parameters, use updated theta'.
                # 2. run forward, get direct gradient to update the network
                for (name, p), dp in zip(model.named_parameters(), dps):
                    model.params[name] = p - G.alpha * dp

            grad_ind = G.n_gradient_steps
            # meta gradient
            if hasattr(model, "is_autoregressive") and model.is_autoregressive:
                h0 = model.h0_init()
                ys, ht = model(xs.view(G.k_shot, 1, 1), h0, labels.view(G.k_shot, 1, 1))
                ys = ys.squeeze(-1)  # ys:Size[5, batch_n: 1, 1]
            elif hasattr(model, "is_recurrent") and model.is_recurrent:
                h0 = model.h0_init()
                ys, ht = model(xs.view(G.k_shot, 1, 1), h0)
                ys = ys.squeeze(-1)  # ys:Size[5, batch_n: 1, 1]
            else:
                ys = model(xs.unsqueeze(-1))
                ht = None
            loss = mse(ys, labels.unsqueeze(-1))
            logger.log_keyvalue(ep_ind, f"{task_ind}-grad-{grad_ind}-loss", loss.item(), silent=_silent)

            if callable(test_fn) and \
                    ep_ind % G.test_interval == 0 and grad_ind in G.test_grad_steps:
                test_fn(model, task=task, task_id=task_ind, epoch=ep_ind, grad_step=grad_ind, silent=_silent, h0=ht)
            meta_dps = t.autograd.grad(loss, original_ps.values())
            with t.no_grad():
                for (name, p), meta_dp in zip(original_ps.items(), meta_dps):
                    p.grad = (0 if p.grad is None else p.grad) + meta_dp

        # normalize the gradient.
        with t.no_grad():
            for (name, p) in original_ps.items():
                p.grad /= G.task_batch_n

        model.params.update(original_ps)
        meta_optimizer.step()
        meta_optimizer.zero_grad()

        if G.save_interval and ep_ind % G.save_interval == 0:
            logger.log_module(ep_ind, **{type(model).__name__: model})

    logger.flush()


def launch_reptile_mlp(log_prefix=None, **_G):
    G.log_prefix = log_prefix or f'{now:%Y-%m-%d}/debug-maml-baselines/sinusoid-reptile-mlp'
    G.update(_G)

    logger.configure(log_directory=G.log_dir, prefix=G.log_prefix)
    logger.log_params(G=vars(G))

    np.random.seed(G.seed)
    t.manual_seed(G.seed)
    t.cuda.manual_seed(G.seed)

    reptile(test_fn=standard_sine_test)


def launch_reptile_rnn(log_prefix=None, **_G):
    G.log_prefix = log_prefix or f'{now:%Y-%m-%d}/debug-maml-baselines/sinusoid-reptile-rnn'
    G.update(_G)

    logger.configure(log_directory=G.log_dir, prefix=G.log_prefix)
    logger.log_params(G=vars(G))

    np.random.seed(G.seed)
    t.manual_seed(G.seed)
    t.cuda.manual_seed(G.seed)

    rnn = FunctionalRNN(1, 1, 10)
    reptile(model=rnn, test_fn=standard_sine_test)


def launch_maml_mlp(log_prefix=None, **_G):
    G.log_prefix = log_prefix or f'{now:%Y-%m-%d}/debug-maml-baselines/sinusoid-maml-mlp'
    G.update(_G)

    logger.configure(log_directory=G.log_dir, prefix=G.log_prefix)
    logger.log_params(G=vars(G))

    np.random.seed(G.seed)
    t.manual_seed(G.seed)
    t.cuda.manual_seed(G.seed)

    maml(test_fn=standard_sine_test)


def launch_maml_rnn(log_prefix=None, **_G):
    G.log_prefix = log_prefix or f'{now:%Y-%m-%d}/debug-maml-baselines/sinusoid-maml-rnn'
    G.update(_G)

    logger.configure(log_directory=G.log_dir, prefix=G.log_prefix)
    logger.log_params(G=vars(G))

    np.random.seed(G.seed)
    t.manual_seed(G.seed)
    t.cuda.manual_seed(G.seed)

    rnn = FunctionalRNN(1, 1, 10)
    maml(model=rnn, test_fn=standard_sine_test)


def launch_maml_auto_rnn(log_prefix=None, **_G):
    G.log_prefix = log_prefix or f'{now:%Y-%m-%d}/debug-maml-baselines/sinusoid-maml-auto-rnn'
    G.update(_G)

    logger.configure(log_directory=G.log_dir, prefix=G.log_prefix)
    logger.log_params(G=vars(G))

    np.random.seed(G.seed)
    t.manual_seed(G.seed)
    t.cuda.manual_seed(G.seed)

    auto_rnn = FunctionalAutoRNN(1, 1, 10)
    maml(model=auto_rnn, test_fn=standard_sine_test)


def launch_reptile_auto_rnn(log_prefix=None, **_G):
    G.log_prefix = log_prefix or f'{now:%Y-%m-%d}/debug-maml-baselines/sinusoid-reptile-auto-rnn'
    G.update(_G)

    logger.configure(log_directory=G.log_dir, prefix=G.log_prefix)
    logger.log_params(G=vars(G))

    np.random.seed(G.seed)
    t.manual_seed(G.seed)
    t.cuda.manual_seed(G.seed)

    auto_rnn = FunctionalAutoRNN(1, 1, 10)
    reptile(model=auto_rnn, test_fn=standard_sine_test)


def launch_reptile_lstm(log_prefix=None, **_G):
    G.log_prefix = log_prefix or f'{now:%Y-%m-%d}/debug-maml-baselines/sinusoid-reptile-lstm'
    G.update(_G)

    logger.configure(log_directory=G.log_dir, prefix=G.log_prefix)
    logger.log_params(G=vars(G))

    np.random.seed(G.seed)
    t.manual_seed(G.seed)
    t.cuda.manual_seed(G.seed)

    auto_rnn = FunctionalLSTM(1, 1, 10)
    reptile(model=auto_rnn, test_fn=standard_sine_test)


def launch_maml_lstm(log_prefix=None, **_G):
    G.log_prefix = log_prefix or f'{now:%Y-%m-%d}/debug-maml-baselines/sinusoid-maml-lstm'
    G.update(_G)

    logger.configure(log_directory=G.log_dir, prefix=G.log_prefix)
    logger.log_params(G=vars(G))

    np.random.seed(G.seed)
    t.manual_seed(G.seed)
    t.cuda.manual_seed(G.seed)

    auto_rnn = FunctionalLSTM(1, 1, 10)
    maml(model=auto_rnn, test_fn=standard_sine_test)


def launch_reptile_auto_lstm(log_prefix=None, **_G):
    G.log_prefix = log_prefix or f'{now:%Y-%m-%d}/debug-maml-baselines/sinusoid-reptile-auto-lstm'
    G.update(_G)

    logger.configure(log_directory=G.log_dir, prefix=G.log_prefix)
    logger.log_params(G=vars(G))

    np.random.seed(G.seed)
    t.manual_seed(G.seed)
    t.cuda.manual_seed(G.seed)

    auto_rnn = FunctionalAutoLSTM(1, 1, 10)
    reptile(model=auto_rnn, test_fn=standard_sine_test)


def launch_maml_auto_lstm(log_prefix=None, **_G):
    G.log_prefix = log_prefix or f'{now:%Y-%m-%d}/debug-maml-baselines/sinusoid-maml-auto-lstm'
    G.update(_G)

    logger.configure(log_directory=G.log_dir, prefix=G.log_prefix)
    logger.log_params(G=vars(G))

    np.random.seed(G.seed)
    t.manual_seed(G.seed)
    t.cuda.manual_seed(G.seed)

    auto_rnn = FunctionalAutoLSTM(1, 1, 10)
    maml(model=auto_rnn, test_fn=standard_sine_test)


def launch_reptile_gru(log_prefix=None, **_G):
    G.log_prefix = log_prefix or f'{now:%Y-%m-%d}/debug-maml-baselines/sinusoid-reptile-gru'
    G.update(_G)

    logger.configure(log_directory=G.log_dir, prefix=G.log_prefix)
    logger.log_params(G=vars(G))

    np.random.seed(G.seed)
    t.manual_seed(G.seed)
    t.cuda.manual_seed(G.seed)

    auto_rnn = FunctionalGRU(1, 1, 10)
    reptile(model=auto_rnn, test_fn=standard_sine_test)


def launch_maml_gru(log_prefix=None, **_G):
    G.log_prefix = log_prefix or f'{now:%Y-%m-%d}/debug-maml-baselines/sinusoid-maml-gru'
    G.update(_G)

    logger.configure(log_directory=G.log_dir, prefix=G.log_prefix)
    logger.log_params(G=vars(G))

    np.random.seed(G.seed)
    t.manual_seed(G.seed)
    t.cuda.manual_seed(G.seed)

    auto_rnn = FunctionalGRU(1, 1, 10)
    maml(model=auto_rnn, test_fn=standard_sine_test)


def launch_reptile_auto_gru(log_prefix=None, **_G):
    G.log_prefix = log_prefix or f'{now:%Y-%m-%d}/debug-maml-baselines/sinusoid-reptile-auto-gru'
    G.update(_G)

    logger.configure(log_directory=G.log_dir, prefix=G.log_prefix)
    logger.log_params(G=vars(G))

    np.random.seed(G.seed)
    t.manual_seed(G.seed)
    t.cuda.manual_seed(G.seed)

    auto_rnn = FunctionalAutoGRU(1, 1, 10)
    reptile(model=auto_rnn, test_fn=standard_sine_test)


def launch_maml_auto_gru(log_prefix=None, **_G):
    G.log_prefix = log_prefix or f'{now:%Y-%m-%d}/debug-maml-baselines/sinusoid-maml-auto-gru'
    G.update(_G)

    logger.configure(log_directory=G.log_dir, prefix=G.log_prefix)
    logger.log_params(G=vars(G))

    np.random.seed(G.seed)
    t.manual_seed(G.seed)
    t.cuda.manual_seed(G.seed)

    auto_rnn = FunctionalAutoGRU(1, 1, 10)
    maml(model=auto_rnn, test_fn=standard_sine_test)


if __name__ == "__main__":
    # sgd_baseline()
    # launch_maml_mlp(log_prefix='local-debug-double')
    # launch_maml_auto_rnn(log_prefix='local-debug-auto-rnn')
    # launch_reptile_auto_rnn(log_prefix='local-debug-reptile-auto-rnn')
    # launch_maml_lstm(log_prefix='local-debug-maml-lstm')
    launch_maml_gru(log_prefix='local-debug-maml-gru')
    # launch_reptile_mlp(log_prefix='local-debug')
    # launch_rnn()
