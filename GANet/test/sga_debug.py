import torch
from torch.autograd import Function
import ganet_lib
import tools
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# [SGA forward]
# 	cost mean: -4.433936e-01
# 	cost_aggregation mean: 1.344887e-01
# 	weight mean: 1.113268e-02
# [LGA forward]
# 	cost mean: 1.157863e+01
# 	weight mean: 8.460315e-03
# 	output_cost mean: 8.419344e+00
# [LGA forward]
# 	cost mean: 5.208335e-03
# 	weight mean: 3.818816e-03
# 	output_cost mean: 1.484113e-03
# [LGA backward]
# 	cost mean: 5.208335e-03
# 	weight mean: 3.818816e-03
# 	cost_grad mean: 2.232139e-05
# 	weight_grad mean: -5.774160e-07
# 	grad_output mean: 9.478718e-05
# [LGA backward]
# 	cost mean: 1.157863e+01
# 	weight mean: 8.460315e-03
# 	cost_grad mean: 1.277093e-10
# 	weight_grad mean: -2.143900e-06
# 	grad_output mean: 3.496733e-16
# [SGA backward]
# 	cost mean: -4.433936e-01
# 	cost_aggregation mean: 1.345039e-01
# 	weight mean: 1.113268e-02
# 	cost_grad mean: nan
# 	weight_grad mean: nan
# 	grad_output mean: -8.289666e-15

def backward():
    cost, cost_aggregation, weight, max_index, cost_grad, weight_grad, grad_output = tools.load('../log/error-data.np')
    # cost: torch.Size([4, 32, 48, 64, 128])

    assert grad_output.is_contiguous()
    with torch.cuda.device_of(cost):
        weight_grad = cost.new().resize_(weight.shape).zero_()
        cost_grad = cost.new().resize_(cost.shape).zero_()
        grad_aggregation = cost.new().resize_(cost.shape).zero_()
        ganet_lib.cuda_sga_backward(cost, cost_aggregation.to(cost.device), weight, max_index,
                                    cost_grad, weight_grad, grad_output, grad_aggregation)
        disp, height, width = cost_grad.size()[2:5]

        cost_grad = cost_grad[3, 28, :, :, 124]
        grad_output = grad_output[3, 28, :, :, :, 124]
        w0 = weight[3, 28, :, 0, :, 124]
        # print(torch.isnan(cost_grad.view(-1)).sum())  # 2501

        grad_output = grad_output.abs().sum(dim=0)
        w0 = w0.data.abs()
        cost_grad = cost_grad.abs()
        cost_grad_low = cost_grad <= 10
        # print(cost_grad.view(-1).max())
        # cost_grad /= cost_grad.view(-1).max()

        # cost_grad = cost_grad.view(disp, height)
        # cost_grad_nan = torch.isnan(cost_grad).sum(dim=0)
        # cost_grad_sum = cost_grad.sum(dim=0)
        # weight_nan = torch.isnan(weight_grad.view(-1, height, width)).sum(dim=0)

        # print(cost_nan.view(-1).sum())
        # print(torch.isnan(weight_grad.view(-1)).sum())
        # print(torch.isnan(cost_grad).nonzero())

        plt.figure()

        plt.subplot(221)
        plt.title('cost_grad abs')
        plt.imshow(cost_grad.cpu())

        plt.subplot(222)
        plt.title('cost_grad abs < 10')
        plt.imshow(cost_grad_low.cpu(), vmin=False, vmax=True)

        plt.subplot(223)
        plt.title('grad_output abs')
        plt.imshow(grad_output.cpu())

        plt.subplot(224)
        plt.title('w0 abs')
        plt.imshow(w0.cpu())

        plt.show()

        # print('[SGA backward]')
        # print('\tcost mean: {:e}'.format(cost.view(-1).mean()))
        # print('\tcost_aggregation mean: {:e}'.format(cost_aggregation.view(-1).mean()))
        #
        # mean_weight_grad = weight_grad.view(-1).mean()
        # print('\tweight mean: {:e}'.format(weight.view(-1).mean()))
        # print('\tcost_grad mean: {:e}'.format(cost_grad.view(-1).mean()))
        # print('\tweight_grad mean: {:e}'.format(mean_weight_grad))
        # print('\tgrad_output mean: {:e}'.format(grad_output.view(-1).mean()))
    return cost_grad, weight_grad

backward()