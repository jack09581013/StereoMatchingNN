import torch
import torch.nn.functional as F
from torch.autograd import Function
import ganet_lib
import tools
import math

DEBUG = False

class SgaFunction(Function):
    @staticmethod
    def forward(ctx, cost, weight):
        assert cost.is_contiguous() and weight.is_contiguous()
        with torch.cuda.device_of(cost):
            batch, channels, max_disparity, height, width = cost.size()
            direction = weight.size()[2]
            cost_aggregation = cost.new().resize_((batch, channels, direction, max_disparity, height, width)).zero_()
            max_index = torch.zeros((batch, channels, direction, height, width), dtype=torch.uint8).to(cost.device)
            ganet_lib.cuda_sga_forward(cost, cost_aggregation, weight, max_index)
            ctx.save_for_backward(cost, cost_aggregation.to('cpu'), weight, max_index)

            if DEBUG:
                print('[SGA forward]')
                print('\tcost mean: {:e}'.format(cost.view(-1).mean()))
                print('\tcost_aggregation mean: {:e}'.format(cost_aggregation.view(-1).mean()))
                print('\tweight mean: {:e}'.format(weight.view(-1).mean()))

        return cost_aggregation

    @staticmethod
    def backward(ctx, grad_output):
        cost, cost_aggregation, weight, max_index = ctx.saved_tensors

        assert grad_output.is_contiguous()
        with torch.cuda.device_of(cost):
            weight_grad = cost.new().resize_(weight.shape).zero_()
            cost_grad = cost.new().resize_(cost.shape).zero_()
            grad_aggregation = cost.new().resize_(cost.shape).zero_()
            ganet_lib.cuda_sga_backward(cost, cost_aggregation.to(cost.device), weight, max_index,
                                        cost_grad, weight_grad, grad_output, grad_aggregation)
            if DEBUG:
                print('[SGA backward]')
                print('\tcost mean: {:e}'.format(cost.view(-1).mean()))
                print('\tcost_aggregation mean: {:e}'.format(cost_aggregation.view(-1).mean()))

                mean_weight_grad = weight_grad.view(-1).mean()
                print('\tweight mean: {:e}'.format(weight.view(-1).mean()))
                print('\tcost_grad mean: {:e}'.format(cost_grad.view(-1).mean()))
                print('\tweight_grad mean: {:e}'.format(mean_weight_grad))
                print('\tgrad_output mean: {:e}'.format(grad_output.view(-1).mean()))

                if torch.isnan(mean_weight_grad):
                    tools.save((cost, cost_aggregation, weight, max_index, cost_grad, weight_grad, grad_output),
                               '../log/error-data.np')
                    exit(2)
        return cost_grad, weight_grad

class LgaFunction(Function):
    @staticmethod
    def forward(ctx, cost, weight):
        assert cost.is_contiguous() and weight.is_contiguous()
        with torch.cuda.device_of(cost):
            output_cost = cost.new().resize_(cost.shape).zero_()
            ganet_lib.cuda_lga_forward(cost, output_cost, weight)
            ctx.save_for_backward(cost, weight)
            if DEBUG:
                print('[LGA forward]')
                print('\tcost mean: {:e}'.format(cost.view(-1).mean()))
                print('\tweight mean: {:e}'.format(weight.view(-1).mean()))
                print('\toutput_cost mean: {:e}'.format(output_cost.view(-1).mean()))
        return output_cost

    @staticmethod
    def backward(ctx, grad_output):
        cost, weight = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        assert grad_output.is_contiguous()

        with torch.cuda.device_of(cost):
            weight_grad = cost.new().resize_(weight.shape).zero_()
            cost_grad = cost.new().resize_(cost.shape).zero_()
            ganet_lib.cuda_lga_backward(cost, weight,
                                       cost_grad, weight_grad, grad_output)
            if DEBUG:
                print('[LGA backward]')
                print('\tcost mean: {:e}'.format(cost.view(-1).mean()))
                print('\tweight mean: {:e}'.format(weight.view(-1).mean()))
                print('\tcost_grad mean: {:e}'.format(cost_grad.view(-1).mean()))
                print('\tweight_grad mean: {:e}'.format(weight_grad.view(-1).mean()))
                print('\tgrad_output mean: {:e}'.format(grad_output.view(-1).mean()))
        return cost_grad, weight_grad


class GDF4_Function(Function):
    @staticmethod
    def forward(ctx, cost, g0, filter):
        assert cost.is_contiguous() and g0.is_contiguous() and filter.is_contiguous()
        with torch.cuda.device_of(cost):
            batch, channels, max_disparity, height, width = cost.size()
            direction = g0.size()[2]
            assert direction == 4
            cost_aggregation = cost.new().resize_((batch, channels, direction, max_disparity, height, width)).zero_()
            ganet_lib.cuda_df4_forward(cost, cost_aggregation, g0, filter)
            ctx.save_for_backward(cost, cost_aggregation, g0, filter)
        return cost_aggregation

    @staticmethod
    def backward(ctx, grad_output):
        cost, cost_aggregation, g0, filter = ctx.saved_tensors

        assert grad_output.is_contiguous()
        with torch.cuda.device_of(cost):
            g0_grad = cost.new().resize_(g0.shape).zero_()
            filter_grad = cost.new().resize_(filter.shape).zero_()
            cost_grad = cost.new().resize_(cost.shape).zero_()
            grad_aggregation = cost.new().resize_(cost.shape).zero_()
            ganet_lib.cuda_df4_backward(cost, cost_aggregation, g0, filter,
                                        cost_grad, g0_grad, filter_grad, grad_aggregation, grad_output)

        return cost_grad, g0_grad, filter_grad

class GDF6_Function(Function):
    @staticmethod
    def forward(ctx, cost, g0, filter):
        assert cost.is_contiguous() and g0.is_contiguous() and filter.is_contiguous()
        with torch.cuda.device_of(cost):
            batch, channels, max_disparity, height, width = cost.size()
            direction = g0.size()[2]
            assert direction == 6
            cost_aggregation = cost.new().resize_((batch, channels, direction, max_disparity, height, width)).zero_()
            ganet_lib.cuda_df6_forward(cost, cost_aggregation, g0, filter)
            ctx.save_for_backward(cost, cost_aggregation, g0, filter)
        return cost_aggregation

    @staticmethod
    def backward(ctx, grad_output):
        cost, cost_aggregation, g0, filter = ctx.saved_tensors

        assert grad_output.is_contiguous()
        with torch.cuda.device_of(cost):
            g0_grad = cost.new().resize_(g0.shape).zero_()
            filter_grad = cost.new().resize_(filter.shape).zero_()
            cost_grad = cost.new().resize_(cost.shape).zero_()
            grad_aggregation = cost.new().resize_(cost.shape).zero_()
            ganet_lib.cuda_df6_backward(cost, cost_aggregation, g0, filter,
                                        cost_grad, g0_grad, filter_grad, grad_aggregation, grad_output)

        return cost_grad, g0_grad, filter_grad

def softmax(x):
    e = torch.exp(x - x.max(dim=0)[0].unsqueeze(0))
    return e / torch.sum(e, dim=0).unsqueeze(0)

def cross_entropy(y, t, epsilon=1e-6):
    return torch.sum(- t * torch.log(y + epsilon), dim=0).mean()

class SoftmaxWithLoss(Function):
    @staticmethod
    def forward(ctx, cost, t):
        cost = F.softmax(cost, dim=0)
        ctx.save_for_backward(cost, t)
        return cross_entropy(cost, t)

    @staticmethod
    def backward(ctx, grad):
        cost, t = ctx.saved_tensors
        return grad * (cost - t), None

class MinimumKernel(Function):
    @staticmethod
    def forward(ctx, cost, kernel_size: int):
        assert cost.is_contiguous()
        with torch.cuda.device_of(cost):
            cost_grad = cost.new().resize_(cost.shape).zero_()
            min_cost = cost.new().resize_(cost.shape).zero_()
            ganet_lib.cuda_cost_minimum_conv(cost, cost_grad, min_cost, kernel_size)
            ctx.save_for_backward(cost_grad)
        return min_cost

    @staticmethod
    def backward(ctx, grad):
        cost_grad = ctx.saved_tensors[0]
        return grad * cost_grad, None

class FlipCost(Function):
    @staticmethod
    def forward(ctx, cost):
        assert cost.is_contiguous()
        with torch.cuda.device_of(cost):
            flip_cost = cost.new().resize_(cost.shape).zero_()
            ganet_lib.cuda_flip_cost_forward(cost, flip_cost)
        return flip_cost

    @staticmethod
    def backward(ctx, grad):
        grad = grad.contiguous()
        with torch.cuda.device_of(grad):
            cost_grad = grad.new().resize_(grad.shape).zero_()
            ganet_lib.cuda_flip_cost_backward(cost_grad, grad)
        return cost_grad