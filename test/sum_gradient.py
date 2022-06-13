import torch
import torch.optim as optim
import torch.nn.functional as F

def sum(x):
    return x / x.sum()

def sum_grad(x, y):
    grad = 2/len(y) * (y - 0)
    sum = x.sum()
    dsum = grad / sum
    dxk = (grad * (-x) / sum**2).sum()
    grad = dsum + dxk
    return grad

def test_show():
    x = torch.randn((5,))
    x1 = x.clone()
    x.requires_grad = True

    y = sum(x)
    y1 = sum(x1)

    print(y)
    print(y1)

    loss = (y - 0).pow(2).mean()
    loss.backward()

    print(x.grad)
    print(sum_grad(x1, y1))

def test_loss():
    x = torch.randn((50,))
    xt = x.clone()
    y = torch.randn((50,))

    optimizar = optim.Adam([x], lr=0.01)
    loss = None

    for i in range(100):
        optimizar.zero_grad()
        y1 = sum(x)
        loss = (y1 - y).pow(2).mean()
        x.grad = sum_grad(x, y1)
        optimizar.step()

    print('loss_c = {:.3f}'.format(loss))

    x = xt
    x.requires_grad = True

    optimizar = optim.Adam([x], lr=0.01)

    for i in range(100):
        optimizar.zero_grad()
        y1 = sum(x)
        loss = (y1 - y).pow(2).mean()
        loss.backward()
        optimizar.step()

    print('loss_p = {:.3f}'.format(loss))

test_show()
# test_loss()