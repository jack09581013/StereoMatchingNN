import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.autograd import Function

def disparity_regression(x):
    disp = torch.arange(0, x.size(1)).unsqueeze(0)
    disp = disp.repeat(x.size(0), 1)
    return torch.sum(x*disp, dim=1).unsqueeze(1)

def entropy(x, epsilon=1e-6):
    return - torch.sum(x * torch.log(x + epsilon))

class Model(nn.Module):
    def __init__(self, max_disparity):
        super(Model, self).__init__()
        self.w1 = nn.Linear(max_disparity, 64)
        self.w2 = nn.Linear(64, 32)
        self.w3 = nn.Linear(32, 64)
        self.w4 = nn.Linear(64, max_disparity)

    def forward(self, x):
        x = self.w1(x)
        x = F.relu(x)
        x = self.w2(x)
        x = F.relu(x)
        x = self.w3(x)
        # x = F.normalize(x, dim=1, p=1)
        # x = x / x.sum()
        # x = torch.sigmoid(x)
        x = F.relu(x)
        x = self.w4(x)
        return x

def softmax(x):
    e = torch.exp(x - x.max(dim=1)[0].unsqueeze(1))
    return e / torch.sum(e, dim=1).unsqueeze(1)

def cross_entropy(y, t, epsilon=1e-6):
    return - torch.sum(t * torch.log(y + epsilon)) / y.size(0)

class SoftmaxWithLoss(Function):
    @staticmethod
    def forward(ctx, x, t):
        y = softmax(x)
        ctx.save_for_backward(y, t)
        return cross_entropy(y, t)

    @staticmethod
    def backward(ctx, grad):
        y, t = ctx.saved_tensors
        return grad * (y - t), None

class Regression(nn.Module):
    def __init__(self, max_disparity):
        super(Regression, self).__init__()
        self.w = Model(max_disparity)

    def forward(self, x, t):
        cost = self.w(x)
        # x = F.normalize(x, dim=1, p=1)
        cost = F.softmax(cost, dim=1)
        y = disparity_regression(cost)
        loss = F.mse_loss(y, t)
        return loss, cost, y

class CrossEntropy(nn.Module):
    def __init__(self, max_disparity):
        super(CrossEntropy, self).__init__()
        self.w = Model(max_disparity)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, t_value):
        cost = self.w(x)
        y = torch.argmax(cost, dim=1)
        t = torch.tensor(int(t_value), dtype=torch.long).unsqueeze(0)
        loss = self.loss(cost, t)
        cost = F.softmax(cost, dim=1).squeeze()
        return loss, cost, float(y)

class CrossEntropyMulti(nn.Module):
    def __init__(self, max_disparity):
        super(CrossEntropyMulti, self).__init__()
        self.max_disparity = max_disparity
        self.w = Model(max_disparity)

    def forward(self, x, t_value):
        cost = self.w(x)
        t = self.get_t(t_value)
        loss = SoftmaxWithLoss.apply(cost, t)
        cost = F.softmax(cost, dim=1)
        y = disparity_regression(cost).float()
        return loss, cost, y

    def get_t(self, t_value):
        t = torch.zeros((t_value.size(0), self.max_disparity), dtype=torch.float)
        long_t = t_value.long()
        mid = t_value - long_t
        for b in range(t_value.size(0)):
            t[b, long_t[b]] = 1 - mid[b]
            t[b, long_t[b] + 1] = mid[b]
        return t.unsqueeze(0)


class CrossEntropyRegression(nn.Module):
    def __init__(self, max_disparity):
        super(CrossEntropyRegression, self).__init__()
        self.w = Model(max_disparity)

    def forward(self, x, t_value):
        cost = self.w(x)
        y = disparity_regression(cost)

        t1 = torch.tensor(t_value, dtype=torch.float)
        t2 = torch.tensor(t_value, dtype=torch.long).unsqueeze(0)

        loss_regression = F.mse_loss(y, t1)
        loss_cross_entropy = F.cross_entropy(cost.unsqueeze(0), t2)

        loss = loss_cross_entropy + loss_regression
        return loss, cost, y

class State2Regression(nn.Module):
    def __init__(self, max_disparity):
        super(State2Regression, self).__init__()
        self.w = Model(max_disparity)

    def forward(self, x, t_value, first):
        cost = self.w(x)
        # print(cost)

        if first:
            y = torch.argmax(cost, dim=0).float()
            t = torch.tensor(int(t_value), dtype=torch.long).unsqueeze(0)
            loss = F.cross_entropy(cost.unsqueeze(0), t)

        else:
            y = disparity_regression(cost)
            t = torch.tensor(t_value, dtype=torch.float)
            loss = F.mse_loss(y, t)

        return loss, cost, y

batch = 5
max_disparity = 128

if batch == 1:
    t = torch.tensor([80.6]).view(-1, 1)
elif batch == 2:
    t = torch.tensor([20.3, 80.6]).view(-1, 1)
elif batch == 5:
    t = torch.tensor([20.3, 80.6, 30.7, 60.33, 5.7]).view(-1, 1)

converge_i = []

for r in range(100):
    print(f'round: {r}')
    x = torch.randn(batch, max_disparity)
    # model = Regression(max_disparity)
    # model = CrossEntropy(max_disparity)
    model = CrossEntropyMulti(max_disparity)
    # model = CrossEntropyRegression(max_disparity)
    # model = State2Regression(max_disparity)

    optimizer = optim.Adam(model.parameters(), lr=0.1, betas=(0.9, 0.999))

    i = 0
    y = torch.zeros((batch, max_disparity), dtype=torch.float)
    cost = None
    while torch.all(torch.abs(y - t) >= 1e-03):
    # while i < 200:
        # if i > 10:
        #     t = torch.tensor([40.8, 30.6, 50.7, 15.33, 7.7]).view(-1, 1)
        optimizer.zero_grad()
        loss, cost, y = model(x, t)
        # loss, cost, y = model(x, t, i < 10)
        # print(f'[{i}], y = {y.view(-1).data.numpy()}, loss = {loss:.3f}')
        loss.backward()
        optimizer.step()
        i += 1

    converge_i.append(i)
    # fig = plt.figure()
    # for b in range(batch):
    #     plt.subplot(batch, 1, b+1)
    #     plt.plot(cost[b].data.numpy())
    #     plt.axvline(float(y[b]), color='k', linestyle='--', label=f'y({float(y[b]):.3f})')
    #     plt.axvline(float(t[b]), color='r', linestyle='--', label=f't({float(t[b]):.3f})')
    #     plt.legend()
    # plt.show()
    # plt.close(fig)

converge_i = torch.tensor(converge_i, dtype=torch.float)
print(f'avg i = {converge_i.mean():.3f}')
