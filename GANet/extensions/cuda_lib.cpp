#include <torch/extension.h>
#include "cuda_kernel.h"
#include <iostream>
using namespace std;

void cuda_sga_forward(at::Tensor cost, at::Tensor cost_aggregation, at::Tensor weight, at::Tensor max_index){
    sga_forward(cost, cost_aggregation, weight, max_index);
}

void cuda_sga_backward(at::Tensor cost, at::Tensor cost_aggregation, at::Tensor weight, at::Tensor max_index,
                       at::Tensor cost_gradient, at::Tensor weight_gradient, at::Tensor grad_output, at::Tensor grad_aggregation){
    sga_backward(cost, cost_aggregation, weight, max_index,
                cost_gradient, weight_gradient, grad_output, grad_aggregation);
}

void cuda_lga_forward(at::Tensor cost, at::Tensor output_cost, at::Tensor weight){
    lga_forward(cost, output_cost, weight);

}

void cuda_lga_backward(at::Tensor cost, at::Tensor weight,
                      at::Tensor cost_gradient, at::Tensor weight_gradient, at::Tensor grad_output){
    lga_backward(cost, weight,
                cost_gradient, weight_gradient, grad_output);
}

void cuda_calc_cost(at::Tensor left_image, at::Tensor right_image, at::Tensor cost, int kernel_size, int min_disparity, int cost_type){
    calc_cost(left_image, right_image, cost, kernel_size, min_disparity, cost_type);
}

void cuda_sgm(at::Tensor cost, at::Tensor cost_aggregation, float p1, float p2){
    sgm(cost, cost_aggregation, p1, p2);
}

void cuda_left_right_consistency_check(at::Tensor left_disparity, at::Tensor right_disparity, float max_diff){
    left_right_consistency_check(left_disparity, right_disparity, max_diff);
}

void cuda_df4_forward(at::Tensor cost, at::Tensor cost_agg, at::Tensor g0, at::Tensor filter){
    df4_forward(cost, cost_agg, g0, filter);
}

void cuda_df4_backward(at::Tensor cost, at::Tensor cost_agg, at::Tensor g0, at::Tensor filter,
        at::Tensor cost_gradient, at::Tensor g0_gradient, at::Tensor filter_gradient, at::Tensor grad_aggregation, at::Tensor grad_output){
    df4_backward(cost, cost_agg, g0, filter,
        cost_gradient, g0_gradient,filter_gradient, grad_aggregation, grad_output);
}

void cuda_df6_forward(at::Tensor cost, at::Tensor cost_agg, at::Tensor g0, at::Tensor filter){
    df6_forward(cost, cost_agg, g0, filter);
}

void cuda_df6_backward(at::Tensor cost, at::Tensor cost_agg, at::Tensor g0, at::Tensor filter,
        at::Tensor cost_gradient, at::Tensor g0_gradient, at::Tensor filter_gradient, at::Tensor grad_aggregation, at::Tensor grad_output){
    df6_backward(cost, cost_agg, g0, filter,
        cost_gradient, g0_gradient,filter_gradient, grad_aggregation, grad_output);
}

void cuda_cspn3d_forward(at::Tensor cost, at::Tensor k0, at::Tensor filter){
    cspn3d_forward(cost, k0, filter);
}

void cuda_cspn3d_backward(at::Tensor cost, at::Tensor k0, at::Tensor filter,
                          at::Tensor cost_grad, at::Tensor k0_grad, at::Tensor filter_grad){
    cspn3d_backward(cost, k0, filter,
                    cost_grad, k0_grad, filter_grad);
}

void cuda_cost_mask(at::Tensor cost, at::Tensor mask, at::Tensor disp){
    cost_mask(cost, mask, disp);
}

void cuda_cost_minimum_conv(at::Tensor cost, at::Tensor cost_grad, at::Tensor min_cost, int kernel_size){
    cost_minimum_conv(cost, cost_grad, min_cost, kernel_size);
}

void cuda_flip_cost_forward(at::Tensor cost, at::Tensor fcost){
    flip_cost_forward(cost, fcost);
}

void cuda_flip_cost_backward(at::Tensor cost_grad, at::Tensor grad_output){
    flip_cost_backward(cost_grad, grad_output);
}

void cuda_test(at::Tensor cost_aggregation, at::Tensor cost, at::Tensor sga, at::Tensor max_index,
               at::Tensor one_channel_cost, at::Tensor lga){
    test(cost_aggregation, cost, sga, max_index,
        one_channel_cost, lga);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	m.def("cuda_sga_forward", &cuda_sga_forward, "cuda_sga_forward");
	m.def("cuda_sga_backward", &cuda_sga_backward, "cuda_sga_backward");
	m.def("cuda_lga_forward", &cuda_lga_forward, "cuda_lga_forward");
	m.def("cuda_lga_backward", &cuda_lga_backward, "cuda_lga_backward");
	m.def("cuda_calc_cost", &cuda_calc_cost, "cuda_calc_cost");
	m.def("cuda_sgm", &cuda_sgm, "cuda_sgm");
	m.def("cuda_left_right_consistency_check", &cuda_left_right_consistency_check, "cuda_left_right_consistency_check");
	m.def("cuda_df4_forward", &cuda_df4_forward, "cuda_df4_forward");
	m.def("cuda_df4_backward", &cuda_df4_backward, "cuda_df4_backward");
	m.def("cuda_df6_forward", &cuda_df6_forward, "cuda_df6_forward");
	m.def("cuda_df6_backward", &cuda_df6_backward, "cuda_df6_backward");
	m.def("cuda_cost_mask", &cuda_cost_mask, "cuda_cost_mask");
	m.def("cuda_cost_minimum_conv", &cuda_cost_minimum_conv, "cuda_cost_minimum_conv");
	m.def("cuda_flip_cost_forward", &cuda_flip_cost_forward, "cuda_flip_cost_forward");
	m.def("cuda_flip_cost_backward", &cuda_flip_cost_backward, "cuda_flip_cost_backward");
	m.def("cuda_test", &cuda_test, "cuda_test");
}
