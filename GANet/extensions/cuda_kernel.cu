#include "cuda_kernel.h"
#include "tensor.h"
#include "direction.h"
#include "guided_diffusion.h"
#include <iostream>
using namespace std;

#define CUDA_NUM_THREADS 256

template<class T>
__device__ T min(T* m, int length){
    T min_value = m[0];
    for (int i = 1; i < length; i++)
        if (min_value > m[i])
            min_value = m[i];
    return min_value;
}

int get_thread_size(int n){
    return (n + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
}

// forward sga kernel definition
__global__ void sga_kernel_forward(int n, CostTensor cost, CostAggregationTensor cost_aggregation, SgaWeightTensor weight, MaxTensor max_index)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= n){
        return;
    }

    int dir_code_1 = index % 2;
    index /= 2;

    int dir_code_2 = index % (cost.height + cost.width);
    index /= cost.height + cost.width;

    int channel = index % cost.channel_size;
    index /= cost.channel_size;

    int batch = index;

    int base_index = 0;

    if(dir_code_2 < cost.width){        
        base_index = dir_code_2;
    }
    else{
        base_index = dir_code_2 - cost.width;
    }

    DirectionInfo* dir_info = 0;

    if (dir_code_1 == 0){

        // up to down forward
        if(dir_code_2 < cost.width){
            dir_info = new UpToDown(cost.height, cost.width);
        }

        // left to right forward
        else {
            dir_info = new LeftToRight(cost.height, cost.width);
        }

    }
    else if(dir_code_1 == 1){

        // down to up forward
        if(dir_code_2 < cost.width){            
            dir_info = new DownToUp(cost.height, cost.width);
        }

        // right to left forward
        else {            
            dir_info = new RightToLeft(cost.height, cost.width);
        }
    }

    int row = dir_info->start_row(base_index, DirectionInfo::Type::FORWARD);
    int col = dir_info->start_col(base_index, DirectionInfo::Type::FORWARD);
    int direction = dir_info->direction;

    int r, c;
    bool first;

    for (int shift = 0; shift < dir_info->shift_limit; shift++){
        r = row + dir_info->row_offset * shift;
        c = col + dir_info->col_offset * shift;
        first = shift == 0;  

        if(first){
            for (int d = 0; d < cost.max_disparity; d++){
                cost_aggregation.at(batch, channel, direction, d, r, c) =
                    cost.at(batch, channel, d, r, c) * weight.at(batch, channel, direction, 0, r, c);
            }
        }
        else{
            int pre_r = r - dir_info->row_offset;
            int pre_c = c - dir_info->col_offset;
            const uchar &max_idx = max_index.at(batch, channel, direction, pre_r, pre_c);
            const float &max_cost_aggregation = cost_aggregation.at(batch, channel, direction, max_idx, pre_r, pre_c);

            for (int d = 0; d < cost.max_disparity; d++){
                float sum = 0;

                sum += cost.at(batch, channel, d, r, c) * weight.at(batch, channel, direction, 0, r, c);
                sum += cost_aggregation.at(batch, channel, direction, d, pre_r, pre_c) * weight.at(batch, channel, direction, 1, r, c);

                if (d > 0)
                    sum += cost_aggregation.at(batch, channel, direction, d - 1, pre_r, pre_c) * weight.at(batch, channel, direction, 2, r, c);

                if (d < cost.max_disparity - 1)
                    sum += cost_aggregation.at(batch, channel, direction, d + 1, pre_r, pre_c) * weight.at(batch, channel, direction, 3, r, c);
                
                sum += max_cost_aggregation * weight.at(batch, channel, direction, 4, r, c);

                cost_aggregation.at(batch, channel, direction, d, r, c) = sum;
            }
        }
        max_index.at(batch, channel, direction, r, c) = cost_aggregation.max_index(batch, channel, direction, r, c);
    }
    delete dir_info;
}

// backward sga kernel definition
__global__ void sga_kernel_backward(int n, CostTensor cost, CostAggregationTensor cost_aggregation, SgaWeightTensor weight, MaxTensor max_index,
                                  CostTensor cost_gradient, SgaWeightTensor weight_gradient, CostAggregationTensor grad_output,
                                  CostTensor grad_aggregation,
                                  int direction)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= n){
        return;
    }

    DirectionInfo* dir_info = 0;

    if (direction == 0)
        dir_info = new LeftToRight(cost.height, cost.width);
    else if (direction == 1)
        dir_info = new RightToLeft(cost.height, cost.width);
    else if (direction == 2)
        dir_info = new UpToDown(cost.height, cost.width);
    else if (direction == 3)
        dir_info = new DownToUp(cost.height, cost.width);

    int base_index = index % dir_info->base_width;
    index /= dir_info->base_width;

    int channel = index % cost.channel_size;
    index /= cost.channel_size;

    int batch = index;
    int row = dir_info->start_row(base_index, DirectionInfo::Type::BACKWARD);
    int col = dir_info->start_col(base_index, DirectionInfo::Type::BACKWARD);

    int r, c;
    bool last;

    for (int shift = 0; shift < dir_info->shift_limit; shift++){
        // NOTE: inverse direction !
        r = row - dir_info->row_offset * shift;
        c = col - dir_info->col_offset * shift;
        last = ((shift + 1) == dir_info->shift_limit);
        const float &w0 = weight.at(batch, channel, direction, 0, r, c);
        const float &w1 = weight.at(batch, channel, direction, 1, r, c);
        const float &w2 = weight.at(batch, channel, direction, 2, r, c);
        const float &w3 = weight.at(batch, channel, direction, 3, r, c);
        const float &w4 = weight.at(batch, channel, direction, 4, r, c);
        float &w0_grad = weight_gradient.at(batch, channel, direction, 0, r, c);
        float &w1_grad = weight_gradient.at(batch, channel, direction, 1, r, c);
        float &w2_grad = weight_gradient.at(batch, channel, direction, 2, r, c);
        float &w3_grad = weight_gradient.at(batch, channel, direction, 3, r, c);
        float &w4_grad = weight_gradient.at(batch, channel, direction, 4, r, c);

        if(last){
            for (int d = 0; d < cost.max_disparity; d++){
                const float direct_grad = grad_output.at(batch, channel, direction, d, r, c);
                const float agg_grad = grad_aggregation.at(batch, channel, d, r, c);
                const float total_grad = direct_grad + agg_grad;
                cost_gradient.at(batch, channel, d, r, c) += total_grad * w0;
                w0_grad += total_grad * cost.at(batch, channel, d, r, c);
            }
        }
        else{
            int pre_r = r - dir_info->row_offset;
            int pre_c = c - dir_info->col_offset;
            uchar max_idx = max_index.at(batch, channel, direction, pre_r, pre_c);
            float &max_grad_aggregation = grad_aggregation.at(batch, channel, max_idx, pre_r, pre_c);
            const float &max_cost_aggregation = cost_aggregation.at(batch, channel, direction, max_idx, pre_r, pre_c);

            for (int d = 0; d < cost.max_disparity; d++){
                const float direct_grad = grad_output.at(batch, channel, direction, d, r, c);
                const float agg_grad = grad_aggregation.at(batch, channel, d, r, c);
                const float total_grad = direct_grad + agg_grad;

                // update grad_aggregation value
                grad_aggregation.at(batch, channel, d, pre_r, pre_c) += total_grad * w1;

                if (d > 0)
                    grad_aggregation.at(batch, channel, d - 1, pre_r, pre_c) += total_grad * w2;

                if (d < cost.max_disparity - 1)
                    grad_aggregation.at(batch, channel, d + 1, pre_r, pre_c) += total_grad * w3;

                max_grad_aggregation += total_grad * w4;

                // update cost and weight gradient
                cost_gradient.at(batch, channel, d, r, c) += total_grad * w0;
                w0_grad += total_grad * cost.at(batch, channel, d, r, c);
                w1_grad += total_grad * cost_aggregation.at(batch, channel, direction, d, pre_r, pre_c);

                if (d > 0)
                    w2_grad += total_grad * cost_aggregation.at(batch, channel, direction, d - 1, pre_r, pre_c);

                if (d < cost.max_disparity - 1)
                    w3_grad += total_grad * cost_aggregation.at(batch, channel, direction, d + 1, pre_r, pre_c);

                w4_grad += total_grad * max_cost_aggregation;
            }
        }
    }
    delete dir_info;
}

// forward lga kernel definition
__global__ void lga_kernel_forward(int n, NoChannelCostTensor cost, NoChannelCostTensor output_cost, LgaWeightTensor weight){

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= n){
        return;
    }

    int col = index % cost.width;
    index /= cost.width;

    int row = index % cost.height;
    index /= cost.height;

    int disparity = index % cost.max_disparity;
    index /= cost.max_disparity;

    int batch = index;
    int mid = weight.kernel_size/2;

    float &sum = output_cost.at(batch, disparity, row, col);

    for(int kr = 0; kr < weight.kernel_size; kr++)
        for(int kc = 0; kc < weight.kernel_size; kc++){
            int cost_row = row + kr - mid;
            int cost_col = col + kc - mid;

            if (0 <= cost_row && cost_row < cost.height &&
                0 <= cost_col && cost_col < cost.width){
                    const float &w0 = weight.at(batch, 0, kr, kc, row, col);
                    const float &w1 = weight.at(batch, 1, kr, kc, row, col);
                    const float &w2 = weight.at(batch, 2, kr, kc, row, col);

                    sum += w0 * cost.at(batch, disparity, cost_row, cost_col);
                    if (disparity > 0)
                        sum += w1 * cost.at(batch, disparity - 1, cost_row, cost_col);
                    if (disparity < cost.max_disparity - 1)
                        sum += w2 * cost.at(batch, disparity + 1, cost_row, cost_col);
                }
        }
}

// backward lga kernel definition
__global__ void lga_kernel_backward(int n, NoChannelCostTensor cost, LgaWeightTensor weight,
                                    NoChannelCostTensor cost_gradient, LgaWeightTensor weight_gradient, NoChannelCostTensor grad_output){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= n){
        return;
    }

    int col = index % cost.width;
    index /= cost.width;

    int row = index % cost.height;
    index /= cost.height;

    int disparity = index % cost.max_disparity;
    index /= cost.max_disparity;

    int batch = index;
    int mid = weight.kernel_size/2;

    const float &direct_grad = grad_output.at(batch, disparity, row, col);    

    for(int kr = 0; kr < weight.kernel_size; kr++)
        for(int kc = 0; kc < weight.kernel_size; kc++){
            int cost_row = row + kr - mid;
            int cost_col = col + kc - mid;

            if (0 <= cost_row && cost_row < cost.height &&
                0 <= cost_col && cost_col < cost.width){                    
                const float &w0 = weight.at(batch, 0, kr, kc, row, col);
                const float &w1 = weight.at(batch, 1, kr, kc, row, col);
                const float &w2 = weight.at(batch, 2, kr, kc, row, col);
                float &w0_grad = weight_gradient.at(batch, 0, kr, kc, row, col);
                float &w1_grad = weight_gradient.at(batch, 1, kr, kc, row, col);
                float &w2_grad = weight_gradient.at(batch, 2, kr, kc, row, col);

                w0_grad += direct_grad * cost.at(batch, disparity, cost_row, cost_col);
                cost_gradient.at(batch, disparity, cost_row, cost_col) += direct_grad * w0;

                if (disparity > 0){
                    w1_grad += direct_grad * cost.at(batch, disparity - 1, cost_row, cost_col);
                    cost_gradient.at(batch, disparity - 1, cost_row, cost_col) += direct_grad * w1;
                }

                if (disparity < cost.max_disparity - 1){
                    w2_grad += direct_grad * cost.at(batch, disparity + 1, cost_row, cost_col);
                    cost_gradient.at(batch, disparity + 1, cost_row, cost_col) += direct_grad * w2;
                }                    
            }
        }
}

__global__ void calc_cost_kernel(int n, ImageTensor left_image, ImageTensor right_image, NoChannelCostTensor cost, int kernel_size, int min_disparity, int cost_type){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= n){
        return;
    }

    int col = index % cost.width;
    index /= cost.width;

    int row = index % cost.height;
    index /= cost.height;

    int disparity = index % cost.max_disparity + min_disparity;
    int batch = index / cost.max_disparity;

    if (col - disparity < 0 || col - disparity >= cost.width)
        return;

    int mid = kernel_size / 2;
    float &cost_value = cost.at(batch, disparity - min_disparity, row, col);

    // Sum Absolute Difference
    if (cost_type == 0){
        for(int i = -mid; i <= mid; i++)
            for(int j = -mid; j <= mid; j++){
                int lr = row + i;
                int lc = col + j;
                int rr = row + i;
                int rc = col + j - disparity;

                if (lr < 0 || lr >= cost.height) continue;
                if (lc < 0 || lc >= cost.width) continue;
                if (rr < 0 || rr >= cost.height) continue;
                if (rc < 0 || rc >= cost.width) continue;
                
                for (int channel = 0; channel < left_image.channel_size; channel++){                
                    cost_value += abs(left_image.at(batch, channel, lr, lc) - right_image.at(batch, channel, rr, rc));                
                }
            }
    }

    // Census (Average Center)
    else if (cost_type == 1){
        float average_left[3] = {};
        float average_right[3] = {};
        int count = 0;

        for(int i = -mid; i <= mid; i++)
            for(int j = -mid; j <= mid; j++){
                int lr = row + i;
                int lc = col + j;
                int rr = row + i;
                int rc = col + j - disparity;

                if (lr < 0 || lr >= cost.height) continue;
                if (lc < 0 || lc >= cost.width) continue;
                if (rr < 0 || rr >= cost.height) continue;
                if (rc < 0 || rc >= cost.width) continue;
                
                for (int channel = 0; channel < left_image.channel_size; channel++){
                    average_left[channel] += left_image.at(batch, channel, lr, lc);
                    average_right[channel] += right_image.at(batch, channel, rr, rc);
                }
                count++;
            }
        
        for (int channel = 0; channel < left_image.channel_size; channel++){
            average_left[channel] /= count;
            average_right[channel] /= count;
        }

        for(int i = -mid; i <= mid; i++)
            for(int j = -mid; j <= mid; j++){
                int lr = row + i;
                int lc = col + j;
                int rr = row + i;
                int rc = col + j - disparity;

                if (lr < 0 || lr >= cost.height) continue;
                if (lc < 0 || lc >= cost.width) continue;
                if (rr < 0 || rr >= cost.height) continue;
                if (rc < 0 || rc >= cost.width) continue;
                
                for (int channel = 0; channel < left_image.channel_size; channel++){
                    bool bl = left_image.at(batch, channel, lr, lc) > average_left[channel];
                    bool br = right_image.at(batch, channel, rr, rc) > average_right[channel];
                    
                    if (bl != br){
                        cost_value += 1;
                    }
                }
            }
    }

    // Census (Fix Center)
    else if (cost_type == 2){
        float center_left[3];
        float center_right[3];

        for (int channel = 0; channel < left_image.channel_size; channel++){
            center_left[channel] = left_image.at(batch, channel, row, col);
            center_right[channel] = right_image.at(batch, channel, row, col - disparity);
        }

        for(int i = -mid; i <= mid; i++)
            for(int j = -mid; j <= mid; j++){
                int lr = row + i;
                int lc = col + j;
                int rr = row + i;
                int rc = col + j - disparity;

                if (lr < 0 || lr >= cost.height) continue;
                if (lc < 0 || lc >= cost.width) continue;
                if (rr < 0 || rr >= cost.height) continue;
                if (rc < 0 || rc >= cost.width) continue;
                
                for (int channel = 0; channel < left_image.channel_size; channel++){
                    bool bl = left_image.at(batch, channel, lr, lc) > center_left[channel];
                    bool br = right_image.at(batch, channel, rr, rc) > center_right[channel];
                    
                    if (bl != br){
                        cost_value += 1;
                    }
                }
            }
    }

    // NCC
    else if (cost_type == 3){
        float average_left[3] = {};
        float average_right[3] = {};
        int count = 0;

        for(int i = -mid; i <= mid; i++)
            for(int j = -mid; j <= mid; j++){
                int lr = row + i;
                int lc = col + j;
                int rr = row + i;
                int rc = col + j - disparity;

                if (lr < 0 || lr >= cost.height) continue;
                if (lc < 0 || lc >= cost.width) continue;
                if (rr < 0 || rr >= cost.height) continue;
                if (rc < 0 || rc >= cost.width) continue;
                
                for (int channel = 0; channel < left_image.channel_size; channel++){
                    average_left[channel] += left_image.at(batch, channel, lr, lc);
                    average_right[channel] += right_image.at(batch, channel, rr, rc);
                }
                count++;
            }
        
        for (int channel = 0; channel < left_image.channel_size; channel++){
            average_left[channel] /= count;
            average_right[channel] /= count;
        }

        float cov[3] = {};
        float stdx[3] = {};
        float stdy[3] = {};

        for(int i = -mid; i <= mid; i++)
            for(int j = -mid; j <= mid; j++){
                int lr = row + i;
                int lc = col + j;
                int rr = row + i;
                int rc = col + j - disparity;

                if (lr < 0 || lr >= cost.height) continue;
                if (lc < 0 || lc >= cost.width) continue;
                if (rr < 0 || rr >= cost.height) continue;
                if (rc < 0 || rc >= cost.width) continue;
                
                for (int channel = 0; channel < left_image.channel_size; channel++){
                    float dx = left_image.at(batch, channel, lr, lc) - average_left[channel];
                    float dy = right_image.at(batch, channel, rr, rc) - average_right[channel];
                    cov[channel] += dx * dy;
                    stdx[channel] += dx * dx;
                    stdy[channel] += dy * dy;
                }
            }

        float r[3];
        for (int channel = 0; channel < left_image.channel_size; channel++){
            float stdxy = sqrt(stdx[channel]) * sqrt(stdy[channel]);
            if (stdxy != 0) r[channel] = cov[channel] / stdxy;                
            else r[channel] = 0;
        }

        float sum_r = 0;
        for (int channel = 0; channel < left_image.channel_size; channel++){
            sum_r += r[channel];
        }
        cost_value = - sum_r / left_image.channel_size;
    }

}

__global__ void sgm_kernel(int n, NoChannelCostTensor cost, SGMCostAggTensor cost_agg, float p1, float p2){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= n){
        return;
    }

    int direction = index % cost_agg.direction;
    index /= cost_agg.direction;

    int dir_code = index % (cost.height + cost.width);
    index /= cost.height + cost.width;

    int batch = index;

    SGMDirection *dir = 0;
    if (direction == 0)
        dir = new SGM_D0();
    else if (direction == 1)
        dir = new SGM_D1();
    else if (direction == 2)
        dir = new SGM_D2();
    else if (direction == 3)
        dir = new SGM_D3();
    else if (direction == 4)
        dir = new SGM_D4();  
    else if (direction == 5)
        dir = new SGM_D5();      
    else if (direction == 6)
        dir = new SGM_D6();
    else if (direction == 7)
        dir = new SGM_D7();
    else return;

    int row = dir->start_row(dir_code, cost.height, cost.width);
    int col = dir->start_col(dir_code, cost.height, cost.width);

    if (row == -1 || col == -1){
        delete dir;
        return;
    }

    // Initializing
    for (int d = 0; d < cost.max_disparity; d++)
        cost_agg.at(batch, direction, d, row, col) = cost.at(batch, d, row, col);
    
    row += dir->row_offset;
    col += dir->col_offset;    

    // Aggregating
    while (row >= 0 && row < cost.height && col >= 0 && col < cost.width)
    {
        int pre_r = row - dir->row_offset;
        int pre_c = col - dir->col_offset;
        float min_cost_agg = cost_agg.min(batch, direction, pre_r, pre_c);

        for (int d = 0; d < cost.max_disparity; d++)
        {
            float min_cost = cost_agg.at(batch, direction, d, pre_r, pre_c);

            if (d + 1 < cost.max_disparity){
                float temp = cost_agg.at(batch, direction, d + 1, pre_r, pre_c) + p1;
                if (temp < min_cost)
                    min_cost = temp;
            }

            if (d - 1 >= 0){
                float temp = cost_agg.at(batch, direction, d - 1, pre_r, pre_c) + p1;
                if (temp < min_cost)
                    min_cost = temp;
            }

            if (min_cost_agg + p2 < min_cost)
                min_cost = min_cost_agg + p2;

            cost_agg.at(batch, direction, d, row, col) = cost.at(batch, d, row, col) + min_cost - min_cost_agg;
        }
        row += dir->row_offset;
        col += dir->col_offset;
    }

    delete dir;
}

__global__ void right_disparity_fill_kernel(int n, DisparityTensor left_disparity, DisparityTensor right_disparity, float max_diff){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= n){
        return;
    }

    int col = index % left_disparity.width;
    index /= left_disparity.width;

    int row = index % left_disparity.height;
    index /= left_disparity.height;

    int batch = index;

    const float &dr = right_disparity.at(batch, row, col);
    if ((int)dr == -1)
        return;

    int q = col - dr;
    if (0 <= q && q < left_disparity.width){
        float &dl = left_disparity.at(batch, row, q);
        if ((int)dl == -1) dl = -dr;
    }
}

__global__ void left_right_consistency_check_kernel(int n, DisparityTensor left_disparity, DisparityTensor right_disparity, float max_diff){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= n){
        return;
    }

    int col = index % left_disparity.width;
    index /= left_disparity.width;

    int row = index % left_disparity.height;
    index /= left_disparity.height;

    int batch = index;

    float &dl = left_disparity.at(batch, row, col);
    int q = col - dl;

    if (0 <= q && q < left_disparity.width){
        const float &dr = right_disparity.at(batch, row, q);

        if ((int)dl != -1 && (int)dr != -1
            && abs(dl + dr) >= max_diff){
            dl = -1;
        }
    }    
}

void sga_forward(at::Tensor cost, at::Tensor cost_aggregation, at::Tensor weight, at::Tensor max_index){
    CostTensor cost_t(cost);
    CostAggregationTensor cost_aggregation_t(cost_aggregation);
    SgaWeightTensor weight_t(weight);
    MaxTensor max_index_t(max_index);

    int n = cost_t.batch_size * cost_t.channel_size * (cost_t.height + cost_t.width) * 2;
    int threads = (n + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
    
    sga_kernel_forward <<<threads, CUDA_NUM_THREADS>>> (n, cost_t, cost_aggregation_t, weight_t, max_index_t);

    cost_t.free();
    cost_aggregation_t.free();
    weight_t.free();
    max_index_t.free();
}

void sga_backward(at::Tensor cost, at::Tensor cost_aggregation, at::Tensor weight, at::Tensor max_index,
                  at::Tensor cost_gradient, at::Tensor weight_gradient, at::Tensor grad_output, at::Tensor grad_aggregation){
    CostTensor cost_t(cost);
    CostTensor cost_gradien_t(cost_gradient);
    CostTensor grad_aggregation_t(grad_aggregation);

    CostAggregationTensor cost_aggregation_t(cost_aggregation);
    CostAggregationTensor grad_output_t(grad_output);

    SgaWeightTensor weight_t(weight);
    SgaWeightTensor weight_gradient_t(weight_gradient);

    MaxTensor max_index_t(max_index);

    int n, threads;

    n = cost_t.batch_size * cost_t.channel_size * cost_t.height;
    threads = (n + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
    // left to right backward
    sga_kernel_backward <<<threads, CUDA_NUM_THREADS>>> (n, cost_t, cost_aggregation_t, weight_t, max_index_t,
                                                        cost_gradien_t, weight_gradient_t, grad_output_t,
                                                        grad_aggregation_t, 0);

    cudaMemset (grad_aggregation.data<float>(), 0, sizeof (float) * grad_aggregation.numel());
    // right to left backward
    sga_kernel_backward <<<threads, CUDA_NUM_THREADS>>> (n, cost_t, cost_aggregation_t, weight_t, max_index_t,
                                                        cost_gradien_t, weight_gradient_t, grad_output_t,
                                                        grad_aggregation_t, 1);

    
    n = cost_t.batch_size * cost_t.channel_size * cost_t.width;
    threads = (n + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
    cudaMemset (grad_aggregation.data<float>(), 0, sizeof (float) * grad_aggregation.numel());
    // up to down backward
    sga_kernel_backward <<<threads, CUDA_NUM_THREADS>>> (n, cost_t, cost_aggregation_t, weight_t, max_index_t,
                                                        cost_gradien_t, weight_gradient_t, grad_output_t,
                                                        grad_aggregation_t, 2);

    cudaMemset (grad_aggregation.data<float>(), 0, sizeof (float) * grad_aggregation.numel());
    // down to up backward
    sga_kernel_backward <<<threads, CUDA_NUM_THREADS>>> (n, cost_t, cost_aggregation_t, weight_t, max_index_t,
                                                        cost_gradien_t, weight_gradient_t, grad_output_t,
                                                        grad_aggregation_t, 3);

    cost_t.free();
    cost_gradien_t.free();
    grad_aggregation_t.free();
    cost_aggregation_t.free();
    grad_output_t.free();
    weight_t.free();
    weight_gradient_t.free();
    max_index_t.free();
}

void lga_forward(at::Tensor cost, at::Tensor output_cost, at::Tensor weight){
    NoChannelCostTensor cost_t(cost);
    NoChannelCostTensor output_cost_t(output_cost);
    LgaWeightTensor weight_t(weight);

    int n = cost_t.batch_size * cost_t.height * cost_t.width * cost_t.max_disparity;
    int threads = (n + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;

    lga_kernel_forward <<<threads, CUDA_NUM_THREADS>>> (n, cost_t, output_cost_t, weight_t);

    cost_t.free();
    output_cost_t.free();
    weight_t.free();
}

void lga_backward(at::Tensor cost, at::Tensor weight,
                  at::Tensor cost_gradient, at::Tensor weight_gradient, at::Tensor grad_output){
    NoChannelCostTensor cost_t(cost);
    NoChannelCostTensor grad_output_t(grad_output);
    NoChannelCostTensor cost_gradient_t(cost_gradient);

    LgaWeightTensor weight_t(weight);
    LgaWeightTensor weight_gradient_t(weight_gradient);

    int n = cost_t.batch_size * cost_t.height * cost_t.width * cost_t.max_disparity;
    int threads = (n + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;

    lga_kernel_backward <<<threads, CUDA_NUM_THREADS>>> (n, cost_t, weight_t,
                                                        cost_gradient_t, weight_gradient_t, grad_output_t);

    cost_t.free();
    grad_output_t.free();
    cost_gradient_t.free();
    weight_t.free();
    weight_gradient_t.free();
}

void calc_cost(at::Tensor left_image, at::Tensor right_image, at::Tensor cost, int kernel_size, int min_disparity, int cost_type){
    ImageTensor left_imageT(left_image);
    ImageTensor right_imageT(right_image);
    NoChannelCostTensor costT(cost);

    int n = costT.batch_size * costT.max_disparity * costT.height * costT.width;
    int threads = get_thread_size(n);
    calc_cost_kernel <<<threads, CUDA_NUM_THREADS>>>(n, left_imageT, right_imageT, costT, kernel_size, min_disparity, cost_type);

    left_imageT.free();
    right_imageT.free();
    costT.free();
}

void sgm(at::Tensor cost, at::Tensor cost_aggregation, float p1, float p2){
    NoChannelCostTensor costT(cost);
    SGMCostAggTensor cost_aggregationT(cost_aggregation);

    int n = costT.batch_size * (costT.height + costT.width)*8;
    int threads = get_thread_size(n);
    sgm_kernel <<<threads, CUDA_NUM_THREADS>>>(n, costT, cost_aggregationT, p1, p2);

    costT.free();
    cost_aggregationT.free();
}


void left_right_consistency_check(at::Tensor left_disparity, at::Tensor right_disparity, float max_diff){
    DisparityTensor left_disparityT(left_disparity);
    DisparityTensor right_disparityT(right_disparity);

    int n = left_disparityT.batch_size * left_disparityT.height * left_disparityT.width;
    int threads = get_thread_size(n);
    right_disparity_fill_kernel <<<threads, CUDA_NUM_THREADS>>>(n, left_disparityT, right_disparityT, max_diff);
    left_right_consistency_check_kernel <<<threads, CUDA_NUM_THREADS>>>(n, left_disparityT, right_disparityT, max_diff);

    left_disparityT.free();
    right_disparityT.free();
}

void df4_forward(at::Tensor cost, at::Tensor cost_agg, at::Tensor g0, at::Tensor filter){
    CostTensor costT(cost);
    CostAggregationTensor cost_aggT(cost_agg);
    gdf4_kernel::GDF4_G0Tensor g0T(g0);
    gdf4_kernel::GDF4_FilterTensor filterT(filter);

    int n = costT.batch_size * costT.channel_size * costT.max_disparity * costT.height;
    int threads = get_thread_size(n);
    
    for (int shift = 0; shift < costT.width; shift++)
        gdf4_kernel::forward <<<threads, CUDA_NUM_THREADS>>> (n, shift, 0, costT, cost_aggT, g0T, filterT);

    for (int shift = 0; shift < costT.width; shift++)
        gdf4_kernel::forward <<<threads, CUDA_NUM_THREADS>>> (n, shift, 1, costT, cost_aggT, g0T, filterT);

    n = costT.batch_size * costT.channel_size * costT.max_disparity * costT.width;
    threads = get_thread_size(n);

    for (int shift = 0; shift < costT.height; shift++)
        gdf4_kernel::forward <<<threads, CUDA_NUM_THREADS>>> (n, shift, 2, costT, cost_aggT, g0T, filterT);

    for (int shift = 0; shift < costT.height; shift++)
        gdf4_kernel::forward <<<threads, CUDA_NUM_THREADS>>> (n, shift, 3, costT, cost_aggT, g0T, filterT);

    costT.free();
    cost_aggT.free();
    g0T.free();
    filterT.free();
}

void df4_backward(at::Tensor cost, at::Tensor cost_agg, at::Tensor g0, at::Tensor filter,
        at::Tensor cost_gradient, at::Tensor g0_gradient, at::Tensor filter_gradient, at::Tensor grad_aggregation, at::Tensor grad_output){
    CostTensor costT(cost);
    CostTensor cost_gradientT(cost_gradient);
    CostTensor grad_aggregationT(grad_aggregation);

    CostAggregationTensor cost_aggT(cost_agg);
    CostAggregationTensor grad_outputT(grad_output);

    gdf4_kernel::GDF4_G0Tensor g0T(g0);
    gdf4_kernel::GDF4_G0Tensor g0_gradientT(g0_gradient);

    gdf4_kernel::GDF4_FilterTensor filterT(filter);
    gdf4_kernel::GDF4_FilterTensor filter_gradientT(filter_gradient);

    int n = costT.batch_size * costT.channel_size * costT.max_disparity * costT.height;
    int threads = get_thread_size(n);

    for (int shift = 0; shift < costT.width; shift++)
        // left to right
        gdf4_kernel::backward <<<threads, CUDA_NUM_THREADS>>> (n, shift, 0, shift == costT.width - 1,
                                                              costT, cost_aggT, g0T, filterT,
                                                              cost_gradientT, g0_gradientT, filter_gradientT, grad_aggregationT, grad_outputT);
    
    cudaMemset (grad_aggregation.data<float>(), 0, sizeof (float) * grad_aggregation.numel());
    for (int shift = 0; shift < costT.width; shift++)
        // right to left
        gdf4_kernel::backward <<<threads, CUDA_NUM_THREADS>>> (n, shift, 1, shift == costT.width - 1,
                                                              costT, cost_aggT, g0T, filterT,
                                                              cost_gradientT, g0_gradientT, filter_gradientT, grad_aggregationT, grad_outputT);

    n = costT.batch_size * costT.channel_size * costT.max_disparity * costT.width;
    threads = get_thread_size(n);

    cudaMemset (grad_aggregation.data<float>(), 0, sizeof (float) * grad_aggregation.numel());
    for (int shift = 0; shift < costT.height; shift++)
        // up to down
        gdf4_kernel::backward <<<threads, CUDA_NUM_THREADS>>> (n, shift, 2, shift == costT.height - 1,
                                                              costT, cost_aggT, g0T, filterT,
                                                              cost_gradientT, g0_gradientT, filter_gradientT, grad_aggregationT, grad_outputT);

    cudaMemset (grad_aggregation.data<float>(), 0, sizeof (float) * grad_aggregation.numel());
    for (int shift = 0; shift < costT.height; shift++)
        // down to up
        gdf4_kernel::backward <<<threads, CUDA_NUM_THREADS>>> (n, shift, 3, shift == costT.height - 1,
                                                              costT, cost_aggT, g0T, filterT,
                                                              cost_gradientT, g0_gradientT, filter_gradientT, grad_aggregationT, grad_outputT);
        

    costT.free();
    cost_gradientT.free();
    grad_aggregationT.free();
    cost_aggT.free();
    grad_outputT.free();
    g0T.free();
    g0_gradientT.free();
    filterT.free();
    filter_gradientT.free();
}

void df6_forward(at::Tensor cost, at::Tensor cost_agg, at::Tensor g0, at::Tensor filter){
    CostTensor costT(cost);
    CostAggregationTensor cost_aggT(cost_agg);
    gdf6_kernel::GDF6_G0Tensor g0T(g0);
    gdf6_kernel::GDF6_FilterTensor filterT(filter);

    int n = costT.batch_size * costT.channel_size * costT.max_disparity * costT.height;
    int threads = get_thread_size(n);
    
    for (int shift = 0; shift < costT.width; shift++)
        // left to right
        gdf6_kernel::forward <<<threads, CUDA_NUM_THREADS>>> (n, shift, 0, costT, cost_aggT, g0T, filterT);

    for (int shift = 0; shift < costT.width; shift++)
        // right to left
        gdf6_kernel::forward <<<threads, CUDA_NUM_THREADS>>> (n, shift, 1, costT, cost_aggT, g0T, filterT);

    n = costT.batch_size * costT.channel_size * costT.max_disparity * costT.width;
    threads = get_thread_size(n);

    for (int shift = 0; shift < costT.height; shift++)
        // up to down
        gdf6_kernel::forward <<<threads, CUDA_NUM_THREADS>>> (n, shift, 2, costT, cost_aggT, g0T, filterT);

    for (int shift = 0; shift < costT.height; shift++)
        // down to up
        gdf6_kernel::forward <<<threads, CUDA_NUM_THREADS>>> (n, shift, 3, costT, cost_aggT, g0T, filterT);

    n = costT.batch_size * costT.channel_size * costT.height * costT.width;
    threads = get_thread_size(n);

    for (int shift = 0; shift < costT.max_disparity; shift++)
        // top to bottom
        gdf6_kernel::forward <<<threads, CUDA_NUM_THREADS>>> (n, shift, 4, costT, cost_aggT, g0T, filterT);

    for (int shift = 0; shift < costT.max_disparity; shift++)
        // bottom to up
        gdf6_kernel::forward <<<threads, CUDA_NUM_THREADS>>> (n, shift, 5, costT, cost_aggT, g0T, filterT);

    costT.free();
    cost_aggT.free();
    g0T.free();
    filterT.free();
}

void df6_backward(at::Tensor cost, at::Tensor cost_agg, at::Tensor g0, at::Tensor filter,
        at::Tensor cost_gradient, at::Tensor g0_gradient, at::Tensor filter_gradient, at::Tensor grad_aggregation, at::Tensor grad_output){
    CostTensor costT(cost);
    CostTensor cost_gradientT(cost_gradient);
    CostTensor grad_aggregationT(grad_aggregation);

    CostAggregationTensor cost_aggT(cost_agg);
    CostAggregationTensor grad_outputT(grad_output);

    gdf6_kernel::GDF6_G0Tensor g0T(g0);
    gdf6_kernel::GDF6_G0Tensor g0_gradientT(g0_gradient);

    gdf6_kernel::GDF6_FilterTensor filterT(filter);
    gdf6_kernel::GDF6_FilterTensor filter_gradientT(filter_gradient);

    int n = costT.batch_size * costT.channel_size * costT.max_disparity * costT.height;
    int threads = get_thread_size(n);

    for (int shift = 0; shift < costT.width; shift++)
        // left to right
        gdf6_kernel::backward <<<threads, CUDA_NUM_THREADS>>> (n, shift, 0, shift == costT.width - 1,
                                                              costT, cost_aggT, g0T, filterT,
                                                              cost_gradientT, g0_gradientT, filter_gradientT, grad_aggregationT, grad_outputT);
    
    cudaMemset (grad_aggregation.data<float>(), 0, sizeof (float) * grad_aggregation.numel());
    for (int shift = 0; shift < costT.width; shift++)
        // right to left
        gdf6_kernel::backward <<<threads, CUDA_NUM_THREADS>>> (n, shift, 1, shift == costT.width - 1,
                                                              costT, cost_aggT, g0T, filterT,
                                                              cost_gradientT, g0_gradientT, filter_gradientT, grad_aggregationT, grad_outputT);

    n = costT.batch_size * costT.channel_size * costT.max_disparity * costT.width;
    threads = get_thread_size(n);

    cudaMemset (grad_aggregation.data<float>(), 0, sizeof (float) * grad_aggregation.numel());
    for (int shift = 0; shift < costT.height; shift++)
        // up to down
        gdf6_kernel::backward <<<threads, CUDA_NUM_THREADS>>> (n, shift, 2, shift == costT.height - 1,
                                                              costT, cost_aggT, g0T, filterT,
                                                              cost_gradientT, g0_gradientT, filter_gradientT, grad_aggregationT, grad_outputT);

    cudaMemset (grad_aggregation.data<float>(), 0, sizeof (float) * grad_aggregation.numel());
    for (int shift = 0; shift < costT.height; shift++)
        // down to up
        gdf6_kernel::backward <<<threads, CUDA_NUM_THREADS>>> (n, shift, 3, shift == costT.height - 1,
                                                              costT, cost_aggT, g0T, filterT,
                                                              cost_gradientT, g0_gradientT, filter_gradientT, grad_aggregationT, grad_outputT);

    n = costT.batch_size * costT.channel_size * costT.height * costT.width;
    threads = get_thread_size(n);
    
    cudaMemset (grad_aggregation.data<float>(), 0, sizeof (float) * grad_aggregation.numel());
    for (int shift = 0; shift < costT.max_disparity; shift++)
        // top to bottom
        gdf6_kernel::backward <<<threads, CUDA_NUM_THREADS>>> (n, shift, 4, shift == costT.max_disparity - 1,
                                                              costT, cost_aggT, g0T, filterT,
                                                              cost_gradientT, g0_gradientT, filter_gradientT, grad_aggregationT, grad_outputT);
    
    cudaMemset (grad_aggregation.data<float>(), 0, sizeof (float) * grad_aggregation.numel());
    for (int shift = 0; shift < costT.max_disparity; shift++)
        // bottom to top
        gdf6_kernel::backward <<<threads, CUDA_NUM_THREADS>>> (n, shift, 5, shift == costT.max_disparity - 1,
                                                              costT, cost_aggT, g0T, filterT,
                                                              cost_gradientT, g0_gradientT, filter_gradientT, grad_aggregationT, grad_outputT);
        

    costT.free();
    cost_gradientT.free();
    grad_aggregationT.free();
    cost_aggT.free();
    grad_outputT.free();
    g0T.free();
    g0_gradientT.free();
    filterT.free();
    filter_gradientT.free();
}

void cspn3d_forward(at::Tensor cost, at::Tensor k0, at::Tensor filter){
    CostTensor costT(cost);
}

void cspn3d_backward(at::Tensor cost, at::Tensor k0, at::Tensor filter,
    at::Tensor cost_grad, at::Tensor k0_grad, at::Tensor filter_grad){
    
}

void cost_mask(at::Tensor cost, at::Tensor mask, at::Tensor disp){
    NoChannelCostTensor costT(cost);
    CostMaskTensor maskT(mask);
    DisparityTensor dispT(disp);

    int n = costT.batch_size * costT.height * costT.width;
    int threads = get_thread_size(n);

    gdf4_kernel::cost_mask_kernel <<<threads, CUDA_NUM_THREADS>>> (n, costT, maskT, dispT);

    costT.free();
    maskT.free();
    dispT.free();
}


void cost_minimum_conv(at::Tensor cost, at::Tensor cost_grad, at::Tensor min_cost, int kernel_size){
    NoChannelCostTensor costT(cost);
    NoChannelCostTensor cost_gradT(cost_grad);
    NoChannelCostTensor min_costT(min_cost);

    int n = costT.batch_size * costT.max_disparity * costT.height * costT.width;
    int threads = get_thread_size(n);

    gdf4_kernel::minimum_conv_kernel <<<threads, CUDA_NUM_THREADS>>> (n, costT, cost_grad, min_costT, kernel_size);

    costT.free();
    cost_gradT.free();
    min_costT.free();
}


void flip_cost_forward(at::Tensor cost, at::Tensor fcost){
    NoChannelCostTensor costT(cost);
    NoChannelCostTensor fcostT(fcost);

    int n = costT.batch_size * costT.max_disparity * costT.height * costT.width;
    int threads = get_thread_size(n);

    gdf4_kernel::flip_cost_forward_kernel <<<threads, CUDA_NUM_THREADS>>> (n, costT, fcostT);

    costT.free();
    fcostT.free();
}

void flip_cost_backward(at::Tensor cost_grad, at::Tensor grad_output){
    NoChannelCostTensor cost_gradT(cost_grad);
    NoChannelCostTensor grad_outputT(grad_output);

    int n = cost_gradT.batch_size * cost_gradT.max_disparity * cost_gradT.height * cost_gradT.width;
    int threads = get_thread_size(n);

    gdf4_kernel::flip_cost_backward_kernel <<<threads, CUDA_NUM_THREADS>>> (n, cost_gradT, grad_outputT);

    cost_gradT.free();
    grad_outputT.free();
}


__global__ void test_kernel(CostAggregationTensor cost_aggregation, CostTensor cost, SgaWeightTensor sga, MaxTensor max_index,
                            NoChannelCostTensor one_channel_cost, LgaWeightTensor lga){
    int b = blockIdx.x * blockDim.x + threadIdx.x;

    cost_aggregation.at(b, 2, 2, 2, 2, 2) = 2;
    cost.at(b, 2, 2, 2, 2) = 2;
    sga.at(b, 2, 2, 2, 2, 2) = 2;
    max_index.at(b, 2, 2, 2, 2) = 2;

    one_channel_cost.at(b, 2, 2, 2) = 2;
    lga.at(b, 2, 2, 2, 2, 2) = 2;
}

void test(at::Tensor cost_aggregation, at::Tensor cost, at::Tensor sga, at::Tensor max_index,
          at::Tensor one_channel_cost, at::Tensor lga){

    CostAggregationTensor cost_aggregation_t(cost_aggregation);
    CostTensor cost_t(cost);
    SgaWeightTensor sga_t(sga);
    MaxTensor max_index_t(max_index);

    NoChannelCostTensor one_channel_cost_t(one_channel_cost);
    LgaWeightTensor lga_t(lga);

    test_kernel <<<1, cost_t.batch_size>>> (cost_aggregation, cost, sga, max_index,
                                            one_channel_cost, lga);

    cost_aggregation_t.free();
    cost_t.free();
    sga_t.free();
    max_index_t.free();
    one_channel_cost_t.free();
    lga_t.free();
}
