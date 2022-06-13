#include <torch/extension.h>

void sga_forward(at::Tensor cost, at::Tensor cost_aggregation, at::Tensor weight, at::Tensor max_index);

void sga_backward(at::Tensor cost, at::Tensor cost_aggregation, at::Tensor weight, at::Tensor max_index,
                  at::Tensor cost_gradient, at::Tensor weight_gradient, at::Tensor grad_output, at::Tensor grad_aggregation);

void lga_forward(at::Tensor cost, at::Tensor output_cost, at::Tensor weight);

void lga_backward(at::Tensor cost, at::Tensor weight,
                  at::Tensor cost_gradient, at::Tensor weight_gradient, at::Tensor grad_output);

void calc_cost(at::Tensor left_image, at::Tensor right_image, at::Tensor cost, int kernel_size, int min_disparity, int cost_type);

void sgm(at::Tensor cost, at::Tensor cost_aggregation, float p1, float p2);

void left_right_consistency_check(at::Tensor left_disparity, at::Tensor right_disparity, float max_diff);

void df4_forward(at::Tensor cost, at::Tensor cost_agg, at::Tensor g0, at::Tensor filter);

void df4_backward(at::Tensor cost, at::Tensor cost_agg, at::Tensor g0, at::Tensor filter,
        at::Tensor cost_gradient, at::Tensor g0_gradient, at::Tensor filter_gradient, at::Tensor grad_aggregation, at::Tensor grad_output);

void df6_forward(at::Tensor cost, at::Tensor cost_agg, at::Tensor g0, at::Tensor filter);

void df6_backward(at::Tensor cost, at::Tensor cost_agg, at::Tensor g0, at::Tensor filter,
        at::Tensor cost_gradient, at::Tensor g0_gradient, at::Tensor filter_gradient, at::Tensor grad_aggregation, at::Tensor grad_output);

void cspn3d_forward(at::Tensor cost, at::Tensor k0, at::Tensor filter);
void cspn3d_backward(at::Tensor cost, at::Tensor k0, at::Tensor filter,
                     at::Tensor cost_grad, at::Tensor k0_grad, at::Tensor filter_grad);

void cost_mask(at::Tensor cost, at::Tensor mask, at::Tensor disp);
void cost_minimum_conv(at::Tensor cost, at::Tensor cost_grad, at::Tensor min_cost, int kernel_size);

void flip_cost_forward(at::Tensor cost, at::Tensor fcost);
void flip_cost_backward(at::Tensor cost_grad, at::Tensor grad_output);

void test(at::Tensor cost_aggregation, at::Tensor cost, at::Tensor sga, at::Tensor max_index,
          at::Tensor one_channel_cost, at::Tensor lga);
