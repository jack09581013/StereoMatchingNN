# pragma once

#include <torch/extension.h>
#include "tensor.h"

namespace gdf4_kernel{
    
    // guided diffusion kernel forward
    __global__ void forward(int n, int shift, int direction, CostTensor cost, CostAggregationTensor cost_agg, GDF4_G0Tensor g0, GDF4_FilterTensor filter){
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index >= n){
            return;
        }

        int r, c;        
        if (direction == 0){
            // left to right
            r = index % cost.height;
            index /= cost.height;
            c = shift;
        }
        else if (direction == 1){
            // right to left
            r = index % cost.height;
            index /= cost.height;
            c = cost.width - 1 - shift;
            
        }
        else if (direction == 2){
            // up to down
            r = shift;
            c = index % cost.width;
            index /= cost.width;
            
        }
        else if (direction == 3){
            // down to up
            r = cost.height - 1 - shift;
            c = index % cost.width;
            index /= cost.width;            
        }        

        int d = index % cost.max_disparity;
        index /= cost.max_disparity;

        int channel = index % cost.channel_size;
        index /= cost.channel_size;

        int batch = index;

        int mid = filter.kernel_size/2;
        bool first = shift == 0;

        if(first){
            cost_agg.at(batch, channel, direction, d, r, c) =
                    cost.at(batch, channel, d, r, c) * g0.at(batch, channel, direction, r, c);
        }
        else{
            int pre_r;
            int pre_c;

            if (direction == 0){
                // left to right
                pre_r = r;
                pre_c = c - 1;
            }
            else if (direction == 1){
                // right to left
                pre_r = r;
                pre_c = c + 1;
            }
            else if (direction == 2){
                // up to down
                pre_r = r - 1;
                pre_c = c;
            }
            else if (direction == 3){
                // down to up
                pre_r = r + 1;
                pre_c = c;
            }

            float &sum = cost_agg.at(batch, channel, direction, d, r, c);
            sum = cost.at(batch, channel, d, r, c) * g0.at(batch, channel, direction, r, c);

            for(int k1 = 0; k1 < filter.kernel_size; k1++)
                for(int k2 = 0; k2 < filter.kernel_size; k2++){
                    int kd = d + k1 - mid;
                    int kr = pre_r;
                    int kc = pre_c;

                    if (direction == 0){
                        // left to right
                        kr += k2 - mid;
                    }
                    else if (direction == 1){
                        // right to left
                        kr += k2 - mid;    
                    }
                    else if (direction == 2){
                        // up to down
                        kc += k2 - mid;  
                    }
                    else if (direction == 3){
                        // down to up
                        kc += k2 - mid;          
                    }

                    if (0 <= kr && kr < cost.height &&
                        0 <= kc && kc < cost.width &&
                        0 <= kd && kd < cost.max_disparity){
                            sum += cost_agg.at(batch, channel, direction, kd, kr, kc) * filter.at(batch, channel, direction, k1, k2, r, c);
                        }
                }
        }
    }

    // guided diffusion kernel backward
    __global__ void backward(int n, int shift, int direction, bool last,
        CostTensor cost, CostAggregationTensor cost_agg, GDF4_G0Tensor g0, GDF4_FilterTensor filter,
        CostTensor cost_gradient, GDF4_G0Tensor g0_gradient, GDF4_FilterTensor filter_gradient, CostTensor grad_aggregation, CostAggregationTensor grad_output){    
        
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index >= n){
            return;
        }

        int r, c;        
        if (direction == 0){
            // left to right
            r = index % cost.height;
            index /= cost.height;
            c = cost.width - 1 - shift;
        }
        else if (direction == 1){
            // right to left
            r = index % cost.height;
            index /= cost.height;
            c = shift;
            
        }
        else if (direction == 2){
            // up to down
            r = cost.height - 1 - shift;
            c = index % cost.width;
            index /= cost.width;
            
        }
        else if (direction == 3){
            // down to up
            r = shift;
            c = index % cost.width;
            index /= cost.width;            
        }

        int d = index % cost.max_disparity;
        index /= cost.max_disparity;

        int channel = index % cost.channel_size;
        index /= cost.channel_size;

        int batch = index;
        int mid = filter.kernel_size/2;
        
        const float direct_grad = grad_output.at(batch, channel, direction, d, r, c);
        const float agg_grad = grad_aggregation.at(batch, channel, d, r, c);
        const float total_grad = direct_grad + agg_grad;

        if(last){
            cost_gradient.at(batch, channel, d, r, c) += total_grad * g0.at(batch, channel, direction, r, c);
            g0_gradient.at(batch, channel, direction, r, c) += total_grad * cost.at(batch, channel, d, r, c);
        }
        else{
            // update grad_aggregation value
            int pre_r;
            int pre_c;

            if (direction == 0){
                // left to right
                pre_r = r;
                pre_c = c - 1;
            }
            else if (direction == 1){
                // right to left
                pre_r = r;
                pre_c = c + 1;
            }
            else if (direction == 2){
                // up to down
                pre_r = r - 1;
                pre_c = c;
            }
            else if (direction == 3){
                // down to up
                pre_r = r + 1;
                pre_c = c;
            }

            for(int k1 = 0; k1 < filter.kernel_size; k1++)
                for(int k2 = 0; k2 < filter.kernel_size; k2++){
                    int kd = d + k1 - mid;
                    int kr = pre_r;
                    int kc = pre_c;

                    if (direction == 0){
                        // left to right
                        kr += k2 - mid;
                    }
                    else if (direction == 1){
                        // right to left
                        kr += k2 - mid;    
                    }
                    else if (direction == 2){
                        // up to down
                        kc += k2 - mid;  
                    }
                    else if (direction == 3){
                        // down to up
                        kc += k2 - mid;          
                    }

                    if (0 <= kr && kr < cost.height &&
                        0 <= kc && kc < cost.width &&
                        0 <= kd && kd < cost.max_disparity){
                            grad_aggregation.at(batch, channel, kd, kr, kc) +=
                                total_grad * filter.at(batch, channel, direction, k1, k2, r, c);
                        }
                }

            // update cost and weight gradient
            cost_gradient.at(batch, channel, d, r, c) += total_grad * g0.at(batch, channel, direction, r, c);
            g0_gradient.at(batch, channel, direction, r, c) += total_grad * cost.at(batch, channel, d, r, c);

            for(int k1 = 0; k1 < filter.kernel_size; k1++)
                for(int k2 = 0; k2 < filter.kernel_size; k2++){
                    int kd = d + k1 - mid;
                    int kr = pre_r;
                    int kc = pre_c;

                    if (direction == 0){
                        // left to right
                        kr += k2 - mid;
                    }
                    else if (direction == 1){
                        // right to left
                        kr += k2 - mid;    
                    }
                    else if (direction == 2){
                        // up to down
                        kc += k2 - mid;  
                    }
                    else if (direction == 3){
                        // down to up
                        kc += k2 - mid;          
                    }

                    if (0 <= kr && kr < cost.height &&
                        0 <= kc && kc < cost.width &&
                        0 <= kd && kd < cost.max_disparity){
                            filter_gradient.at(batch, channel, direction, k1, k2, r, c) += 
                                total_grad * cost_agg.at(batch, channel, direction, kd, kr, kc);
                        }
                }
        }
    }

    __global__ void cost_mask_kernel(int n, NoChannelCostTensor cost, CostMaskTensor mask, DisparityTensor disp){
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index >= n){
            return;
        }

        int col = index % cost.width;
        index /= cost.width;

        int row = index % cost.height;
        index /= cost.height;

        int batch = index;
        int disp_v = disp.at(batch, row, col);

        int d = disp_v;

        while (d >= 1){
            float gradient = cost.at(batch, d, row, col) - cost.at(batch, d - 1, row, col);
            if (gradient > 0)
                mask.at(batch, d, row, col) = 1;
            else break;
            d--;
        }

        d = disp_v;
        while (d < cost.max_disparity - 1){
            float gradient = cost.at(batch, d + 1, row, col) - cost.at(batch, d, row, col);
            if (gradient < 0)
                mask.at(batch, d, row, col) = 1;
            else break;
            d++;
        }
    }

    __global__ void minimum_conv_kernel(int n, NoChannelCostTensor cost, NoChannelCostTensor cost_grad, NoChannelCostTensor min_cost, int kernel_size){
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index >= n){
            return;
        }

        int col = index % cost.width;
        index /= cost.width;

        int row = index % cost.height;
        index /= cost.height;

        int d = index % cost.max_disparity;
        index /= cost.max_disparity;

        int batch = index;

        float min_v = cost.at(batch, d, row, col);
        int min_d = d;
        int mid = kernel_size / 2;

        for (int k = 0; k < kernel_size; k++){
            int kd = d + k - mid;
            if (kd >= 0 && kd < cost.max_disparity){
                float temp = cost.at(batch, kd, row, col);
                if (temp < min_v){
                    min_v = temp;
                    min_d = kd;
                }
            }
        }
        min_cost.at(batch, d, row, col) = min_v;
        cost_grad.at(batch, min_d, row, col) += 1;
    }

    __global__ void flip_cost_forward_kernel (int n, NoChannelCostTensor cost, NoChannelCostTensor flip_cost){
         int index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index >= n){
            return;
        }

        int col = index % cost.width;
        index /= cost.width;

        int row = index % cost.height;
        index /= cost.height;

        int d = index % cost.max_disparity;
        index /= cost.max_disparity;

        int batch = index;
        int q = cost.width - 1 - col + d;

        if (q >= cost.width)
            return;

        flip_cost.at(batch, d, row, col) = cost.at(batch, d, row, q);
    }

    __global__ void flip_cost_backward_kernel (int n, NoChannelCostTensor cost_grad, NoChannelCostTensor grad_output){
         int index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index >= n){
            return;
        }

        int col = index % cost_grad.width;
        index /= cost_grad.width;

        int row = index % cost_grad.height;
        index /= cost_grad.height;

        int d = index % cost_grad.max_disparity;
        index /= cost_grad.max_disparity;

        int batch = index;
        int q = cost_grad.width - 1 - col + d;

        if (q >= cost_grad.width)
            return;

        cost_grad.at(batch, d, row, q) += grad_output.at(batch, d, row, col);
    }
}

namespace gdf6_kernel{
    
    // guided diffusion kernel forward
    __global__ void forward(int n, int shift, int direction, CostTensor cost, CostAggregationTensor cost_agg, GDF6_G0Tensor g0, GDF6_FilterTensor filter){
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index >= n){
            return;
        }

        int c, r, d;       
        if (direction == 0){
            // left to right
            c = shift;
            r = index % cost.height;
            index /= cost.height;
            d = index % cost.max_disparity;
            index /= cost.max_disparity;
        }
        else if (direction == 1){
            // right to left
            c = cost.width - 1 - shift;
            r = index % cost.height;
            index /= cost.height;
            d = index % cost.max_disparity;
            index /= cost.max_disparity;
        }
        else if (direction == 2){
            // up to down
            c = index % cost.width;
            index /= cost.width;   
            r = shift;
            d = index % cost.max_disparity;
            index /= cost.max_disparity;         
        }
        else if (direction == 3){
            // down to up
            c = index % cost.width;
            index /= cost.width;  
            r = cost.height - 1 - shift;
            d = index % cost.max_disparity;
            index /= cost.max_disparity;          
        }
        else if (direction == 4){
            // top to bottom
            c = index % cost.width;
            index /= cost.width;
            r = index % cost.height;
            index /= cost.height;
            d = cost.max_disparity - 1 - shift;
        }   
        else if (direction == 5){
            // bottom to top
            c = index % cost.width;
            index /= cost.width;
            r = index % cost.height;
            index /= cost.height;
            d = shift;
        }

        int channel = index % cost.channel_size;
        index /= cost.channel_size;

        int batch = index;

        int mid = filter.kernel_size/2;
        bool first = shift == 0;

        if(first){
            cost_agg.at(batch, channel, direction, d, r, c) =
                    cost.at(batch, channel, d, r, c) * g0.at(batch, channel, direction, r, c);
        }
        else{
            int pre_d;
            int pre_r;
            int pre_c;

            if (direction == 0){
                // left to right
                pre_r = r;
                pre_c = c - 1;
                pre_d = d;
            }
            else if (direction == 1){
                // right to left
                pre_r = r;
                pre_c = c + 1;
                pre_d = d;
            }
            else if (direction == 2){
                // up to down
                pre_r = r - 1;
                pre_c = c;
                pre_d = d;
            }
            else if (direction == 3){
                // down to up
                pre_r = r + 1;
                pre_c = c;
                pre_d = d;
            }
            else if (direction == 4){
                // top to bottom
                pre_r = r;
                pre_c = c;
                pre_d = d + 1;
            }   
            else if (direction == 5){
                // bottom to top
                pre_r = r;
                pre_c = c;
                pre_d = d - 1;
            }
            

            float &sum = cost_agg.at(batch, channel, direction, d, r, c);
            sum = cost.at(batch, channel, d, r, c) * g0.at(batch, channel, direction, r, c);

            for(int k1 = 0; k1 < filter.kernel_size; k1++)
                for(int k2 = 0; k2 < filter.kernel_size; k2++){
                    int kd = pre_d;
                    int kr = pre_r;
                    int kc = pre_c;

                    if (direction == 0){
                        // left to right
                        kd += k1 - mid;
                        kr += k2 - mid;
                    }
                    else if (direction == 1){
                        // right to left
                        kd += k1 - mid;
                        kr += k2 - mid;
                    }
                    else if (direction == 2){
                        // up to down
                        kd += k1 - mid;
                        kc += k2 - mid;  
                    }
                    else if (direction == 3){
                        // down to up
                        kd += k1 - mid;
                        kc += k2 - mid;          
                    }
                    else if (direction == 4){
                        // top to bottom
                        kr += k1 - mid;
                        kc += k2 - mid;  
                    }   
                    else if (direction == 5){
                        // bottom to top
                        kr += k1 - mid;
                        kc += k2 - mid;  
                    }

                    if (0 <= kr && kr < cost.height &&
                        0 <= kc && kc < cost.width &&
                        0 <= kd && kd < cost.max_disparity){
                            sum += cost_agg.at(batch, channel, direction, kd, kr, kc) * filter.at(batch, channel, direction, k1, k2, r, c);
                        }
                }
        }
    }

    // guided diffusion kernel backward
    __global__ void backward(int n, int shift, int direction, bool last,
        CostTensor cost, CostAggregationTensor cost_agg, GDF6_G0Tensor g0, GDF6_FilterTensor filter,
        CostTensor cost_gradient, GDF6_G0Tensor g0_gradient, GDF6_FilterTensor filter_gradient, CostTensor grad_aggregation, CostAggregationTensor grad_output){    
        
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index >= n){
            return;
        }

        int c, r, d;
        if (direction == 0){
            // left to right
            c = cost.width - 1 - shift;
            r = index % cost.height;
            index /= cost.height;
            d = index % cost.max_disparity;
            index /= cost.max_disparity;
        }
        else if (direction == 1){
            // right to left
            c = shift;
            r = index % cost.height;
            index /= cost.height;
            d = index % cost.max_disparity;
            index /= cost.max_disparity;            
        }
        else if (direction == 2){
            // up to down
            c = index % cost.width;
            index /= cost.width;
            r = cost.height - 1 - shift;
            d = index % cost.max_disparity;
            index /= cost.max_disparity;            
        }
        else if (direction == 3){
            // down to up
            c = index % cost.width;
            index /= cost.width;
            r = shift;   
            d = index % cost.max_disparity;
            index /= cost.max_disparity;         
        }
        else if (direction == 4){
            // top to bottom
            c = index % cost.width;
            index /= cost.width;
            r = index % cost.height;
            index /= cost.height;
            d = shift;
        }   
        else if (direction == 5){
            // bottom to top
            c = index % cost.width;
            index /= cost.width;
            r = index % cost.height;
            index /= cost.height;
            d = cost.max_disparity - 1 - shift;
        }

        int channel = index % cost.channel_size;
        index /= cost.channel_size;

        int batch = index;
        int mid = filter.kernel_size/2;
        
        const float direct_grad = grad_output.at(batch, channel, direction, d, r, c);
        const float agg_grad = grad_aggregation.at(batch, channel, d, r, c);
        const float total_grad = direct_grad + agg_grad;

        if(last){
            cost_gradient.at(batch, channel, d, r, c) += total_grad * g0.at(batch, channel, direction, r, c);
            g0_gradient.at(batch, channel, direction, r, c) += total_grad * cost.at(batch, channel, d, r, c);
        }
        else{
            // update grad_aggregation value
            int pre_d;
            int pre_r;
            int pre_c;

            if (direction == 0){
                // left to right
                pre_r = r;
                pre_c = c - 1;
                pre_d = d;
            }
            else if (direction == 1){
                // right to left
                pre_r = r;
                pre_c = c + 1;
                pre_d = d;
            }
            else if (direction == 2){
                // up to down
                pre_r = r - 1;
                pre_c = c;
                pre_d = d;
            }
            else if (direction == 3){
                // down to up
                pre_r = r + 1;
                pre_c = c;
                pre_d = d;
            }
            else if (direction == 4){
                // top to bottom
                pre_r = r;
                pre_c = c;
                pre_d = d + 1;
            }   
            else if (direction == 5){
                // bottom to top
                pre_r = r;
                pre_c = c;
                pre_d = d - 1;
            }

            for(int k1 = 0; k1 < filter.kernel_size; k1++)
                for(int k2 = 0; k2 < filter.kernel_size; k2++){
                    int kd = pre_d;
                    int kr = pre_r;
                    int kc = pre_c;

                    if (direction == 0){
                        // left to right
                        kd += k1 - mid;
                        kr += k2 - mid;
                    }
                    else if (direction == 1){
                        // right to left
                        kd += k1 - mid;
                        kr += k2 - mid;
                    }
                    else if (direction == 2){
                        // up to down
                        kd += k1 - mid;
                        kc += k2 - mid;  
                    }
                    else if (direction == 3){
                        // down to up
                        kd += k1 - mid;
                        kc += k2 - mid;          
                    }
                    else if (direction == 4){
                        // top to bottom
                        kr += k1 - mid;
                        kc += k2 - mid;  
                    }   
                    else if (direction == 5){
                        // bottom to top
                        kr += k1 - mid;
                        kc += k2 - mid;  
                    }

                    if (0 <= kr && kr < cost.height &&
                        0 <= kc && kc < cost.width &&
                        0 <= kd && kd < cost.max_disparity){
                            grad_aggregation.at(batch, channel, kd, kr, kc) +=
                                total_grad * filter.at(batch, channel, direction, k1, k2, r, c);
                        }
                }

            // update cost and weight gradient
            cost_gradient.at(batch, channel, d, r, c) += total_grad * g0.at(batch, channel, direction, r, c);
            g0_gradient.at(batch, channel, direction, r, c) += total_grad * cost.at(batch, channel, d, r, c);

            for(int k1 = 0; k1 < filter.kernel_size; k1++)
                for(int k2 = 0; k2 < filter.kernel_size; k2++){
                    int kd = pre_d;
                    int kr = pre_r;
                    int kc = pre_c;

                    if (direction == 0){
                        // left to right
                        kd += k1 - mid;
                        kr += k2 - mid;
                    }
                    else if (direction == 1){
                        // right to left
                        kd += k1 - mid;
                        kr += k2 - mid;
                    }
                    else if (direction == 2){
                        // up to down
                        kd += k1 - mid;
                        kc += k2 - mid;  
                    }
                    else if (direction == 3){
                        // down to up
                        kd += k1 - mid;
                        kc += k2 - mid;          
                    }
                    else if (direction == 4){
                        // top to bottom
                        kr += k1 - mid;
                        kc += k2 - mid;  
                    }   
                    else if (direction == 5){
                        // bottom to top
                        kr += k1 - mid;
                        kc += k2 - mid;  
                    }

                    if (0 <= kr && kr < cost.height &&
                        0 <= kc && kc < cost.width &&
                        0 <= kd && kd < cost.max_disparity){
                            filter_gradient.at(batch, channel, direction, k1, k2, r, c) += 
                                total_grad * cost_agg.at(batch, channel, direction, kd, kr, kc);
                        }
                }
        }
    }
}



