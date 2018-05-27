import os

import torch
from ml_logger import logger
from params_proto import cli_parse
from torch import nn, optim
import torch_helpers as h



class Conv2d(nn.Module):
    def __init__(self):
        super(Conv2d, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 20, kernel_size=5),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(20, 50, kernel_size=5),
            nn.MaxPool2d(2, 2),
            h.Flatten(),
            nn.Linear(4 * 4 * 50, 500),
            nn.Linear(500, 10),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.model(x)


class Mlp(nn.Module):
    def __init__(self):
        super(Mlp, self).__init__()
        self.model = nn.Sequential(
            h.Flatten(),
            nn.Linear(28 * 28, 64),
            nn.ReLU(),
            nn.Dropout(0.7),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.7),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.7),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.model(x)


@cli_parse
class G:
    log_dir = os.path.realpath("./outputs")
    data_dir = "./__dataset"
    model = 'lenet'
    batch_size = 1000
    n_epochs = 20
    test_interval = 10
    learning_rate = 0.1


def train():
    from moleskin import moleskin as M

    M.tic('Full Run')
    if G.model == "lenet":
        model = Conv2d()
    elif G.model == 'mlp':
        model = Mlp()
    else:
        raise NotImplementedError('only lenet and mlp are allowed')
    model.train()
    print(model)

    G.log_prefix = f"mnist_{type(model).__name__}"
    logger.configure(log_directory=G.log_dir, prefix=G.log_prefix)
    logger.log_params(G=vars(G), Model=dict(architecture=str(model)))

    from torchvision import datasets, transforms

    trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
    train_set = datasets.MNIST(root=G.data_dir, train=True, transform=trans, download=True)
    test_set = datasets.MNIST(root=G.data_dir, train=False, transform=trans, download=True)
    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=G.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=G.batch_size, shuffle=False)

    celoss = nn.CrossEntropyLoss()
    adam = optim.SGD(model.parameters(), lr=G.learning_rate, momentum=0.9)
    for epoch in range(G.n_epochs):
        for it, (x, target) in enumerate(train_loader):
            adam.zero_grad()
            ys = model(x)
            loss = celoss(ys, target)
            loss.backward()
            adam.step()

            if it % G.test_interval == 0:
                with h.Eval(model), torch.no_grad():
                    accuracy = h.Average()
                    for x, label in test_loader:
                        acc = h.cast(h.one_hot_to_int(model(x).detach()) == label, float).sum() / len(x)
                        accuracy.add(acc.detach().numpy())
                logger.log(float(epoch) + it / len(train_loader), accuracy=accuracy.value)

        M.split("epoch")
        # logger.log(epoch, it=it, loss=loss.detach().numpy())
    M.toc('Full Run')


if __name__ == "__main__":
    train()
