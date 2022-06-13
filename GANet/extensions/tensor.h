# pragma once

#include <torch/extension.h>
#include <iostream>
using namespace std;

typedef unsigned char uchar;

template<class T>
class Tensor{
private:
    int* tensor_dims(at::Tensor &t){
        if (t.dim() < 2){
            cout << "t.size() < 2" << endl;
            return 0;
        }
    
        int* dims = new int[t.dim() - 1];
        int t_index = t.dim() - 1;
        int d_index = t.dim() - 2;
    
        dims[d_index] = t.size(t_index);
    
        d_index--;
        t_index--;
        while(d_index >= 0){
            dims[d_index] = dims[d_index + 1] * t.size(t_index);
            d_index--;
            t_index--;
        }
    
        int* cuda_dims;
        cudaMalloc(&cuda_dims, (t.dim() - 1) * sizeof(int)); 
        cudaMemcpy(cuda_dims, dims, (t.dim() - 1) * sizeof(int), cudaMemcpyHostToDevice);
        delete[] dims;
        return cuda_dims;
    }

protected:
    T* data;
    int* dims;
    int dim_size;

public:
    Tensor(at::Tensor &t){
        data = t.data<T>();
        dim_size = t.numel();
        dims = tensor_dims(t);
    }

    void free(){
        cudaFree(dims);
    }    
};

class CostAggregationTensor: public Tensor<float>{
public:
    int batch_size, channel_size, directions = 4, max_disparity, height, width;

    CostAggregationTensor(at::Tensor &t): Tensor(t){
        batch_size = t.size(0);
        channel_size = t.size(1);
        max_disparity = t.size(3);
        height = t.size(4);
        width = t.size(5);
    }

    __device__ float& at(int batch, int channel, int direction, int disparity, int row, int column){
        return data[(long)batch * dims[0] + channel * dims[1] + direction * dims[2] + disparity * dims[3] + row * dims[4] + column];
    }

    __device__ uchar max_index(int batch, int channel, int direction, int row, int column){
        float max_cost = at(batch, channel, direction, 0, row, column);
        uchar max_i = 0;

        for(uchar i = 1; i < max_disparity; i++){
            float cost = at(batch, channel, direction, i, row, column);
            if(cost > max_cost){
                max_cost = cost;
                max_i = i;
            }
        }
        return max_i;
    }
};

class CostTensor: public Tensor<float>{
public:
    int batch_size, channel_size, max_disparity, height, width;

    CostTensor(at::Tensor &t): Tensor(t){
        batch_size = t.size(0);
        channel_size = t.size(1);
        max_disparity = t.size(2);
        height = t.size(3);
        width = t.size(4);
    }

    __device__ float& at(int batch, int channel, int disparity, int row, int column){
        return data[(long)batch * dims[0] + channel * dims[1] + disparity * dims[2] + row * dims[3] + column];
    }
};

class NoChannelCostTensor: public Tensor<float>{
public:
    int batch_size, max_disparity, height, width;

    NoChannelCostTensor(at::Tensor &t): Tensor(t){
        batch_size = t.size(0);
        max_disparity = t.size(1);
        height = t.size(2);
        width = t.size(3);
    }

    __device__ float& at(int batch, int disparity, int row, int column){
        return data[(long)batch * dims[0] + disparity * dims[1] + row * dims[2] + column];
    }
};

class SgaWeightTensor: public Tensor<float>{
public:    
    int batch_size, channel_size, directions = 4, weight_size = 5, height, width;

    SgaWeightTensor(at::Tensor &t): Tensor(t){
        batch_size = t.size(0);
        channel_size = t.size(1);
        height = t.size(4);
        width = t.size(5);
    }

    __device__ float& at(int batch, int channel, int direction, int weight_number, int row, int column){
        return data[(long)batch * dims[0] + channel * dims[1] + direction * dims[2] + weight_number * dims[3] + row * dims[4] + column];
    }
};

class MaxTensor: public Tensor<uchar>{
public:
    int batch_size, channel_size, directions = 4,  height, width;

    MaxTensor(at::Tensor &t): Tensor(t){
        batch_size = t.size(0);
        channel_size = t.size(1);
        height = t.size(3);
        width = t.size(4);
    }

    __device__ uchar& at(int batch, int channel, int direction, int row, int column){
        return data[(long)batch * dims[0] + channel * dims[1] + direction * dims[2] + row * dims[3] + column];
    }
};

class LgaWeightTensor: public Tensor<float>{
public:
    int batch_size, weight_size, kernel_size, height, width;

    LgaWeightTensor(at::Tensor &t): Tensor(t){
        batch_size = t.size(0);
        weight_size = t.size(1);
        kernel_size = t.size(2);
        height = t.size(4);
        width = t.size(5);
    }

    __device__ float& at(int batch, int weight_number, int kernel_row, int kernel_column, int row, int column){
        return data[(long)batch * dims[0] + weight_number * dims[1] + kernel_row * dims[2] + kernel_column * dims[3]
                    + row * dims[4] + column];
    }
};

class ImageTensor: public Tensor<float>{
public:
    int batch_size, channel_size, height, width;

    ImageTensor(at::Tensor &t): Tensor(t){
        batch_size = t.size(0);
        channel_size = t.size(1);
        height = t.size(2);
        width = t.size(3);
    }

    __device__ float& at(int batch, int channel, int row, int column){
        return data[(long)batch * dims[0] + channel * dims[1] + row * dims[2] + column];
    }
};

class SGMCostAggTensor: public Tensor<float>{
public:
    int batch_size, direction = 8, max_disparity, height, width;

    SGMCostAggTensor(at::Tensor &t): Tensor(t){
        batch_size = t.size(0);
        max_disparity = t.size(2);
        height = t.size(3);
        width = t.size(4);
    }

    __device__ float& at(int batch, int direction, int disparity, int row, int column){
        return data[(long)batch * dims[0] + direction * dims[1] + disparity * dims[2] + row * dims[3] + column];
    }

    __device__ float& min(int batch, int direction, int row, int column){
        float min_value = at(batch, direction, 0, row, column);
        for (int i = 1; i < max_disparity; i++){
            float temp = at(batch, direction, i, row, column);
            if (min_value > temp)
                min_value = temp;
        }            
        return min_value;
    }
};

class DisparityTensor: public Tensor<float>{
public:
    int batch_size, height, width;

    DisparityTensor(at::Tensor &t): Tensor(t){
        batch_size = t.size(0);
        height = t.size(1);
        width = t.size(2);
    }

    __device__ float& at(int batch, int row, int column){
        return data[(long)batch * dims[0] + row * dims[1] + column];
    }
};

namespace gdf4_kernel{
    class GDF4_G0Tensor: public Tensor<float>{
    public:    
        int batch_size, channel_size, directions = 4, height, width;

        GDF4_G0Tensor(at::Tensor &t): Tensor(t){
            batch_size = t.size(0);
            channel_size = t.size(1);
            height = t.size(3);
            width = t.size(4);
        }

        __device__ float& at(int batch, int channel, int direction, int row, int column){
            return data[(long)batch * dims[0] + channel * dims[1] + direction * dims[2] + row * dims[3] + column];
        }
    };

    class GDF4_FilterTensor: public Tensor<float>{
    public:    
        int batch_size, channel_size, directions = 4, kernel_size, height, width;

        GDF4_FilterTensor(at::Tensor &t): Tensor(t){
            batch_size = t.size(0);
            channel_size = t.size(1);
            kernel_size = t.size(3);
            height = t.size(5);
            width = t.size(6);
        }

        __device__ float& at(int batch, int channel, int direction, int k1, int k2, int row, int column){
            return data[(long)batch * dims[0] + channel * dims[1] + direction * dims[2] + k1 * dims[3] + k2 * dims[4] + row * dims[5] + column];
        }
    };
}

namespace gdf6_kernel{
    class GDF6_G0Tensor: public Tensor<float>{
    public:    
        int batch_size, channel_size, directions = 6, height, width;

        GDF6_G0Tensor(at::Tensor &t): Tensor(t){
            batch_size = t.size(0);
            channel_size = t.size(1);
            height = t.size(3);
            width = t.size(4);
        }

        __device__ float& at(int batch, int channel, int direction, int row, int column){
            return data[(long)batch * dims[0] + channel * dims[1] + direction * dims[2] + row * dims[3] + column];
        }
    };

    class GDF6_FilterTensor: public Tensor<float>{
    public:    
        int batch_size, channel_size, directions = 6, kernel_size, height, width;

        GDF6_FilterTensor(at::Tensor &t): Tensor(t){
            batch_size = t.size(0);
            channel_size = t.size(1);
            kernel_size = t.size(3);
            height = t.size(5);
            width = t.size(6);
        }

        __device__ float& at(int batch, int channel, int direction, int k1, int k2, int row, int column){
            return data[(long)batch * dims[0] + channel * dims[1] + direction * dims[2] + k1 * dims[3] + k2 * dims[4] + row * dims[5] + column];
        }
    };
}

class CostMaskTensor: public Tensor<uchar>{
public:    
    int batch_size, max_disparity, height, width;

    CostMaskTensor(at::Tensor &t): Tensor(t){
        batch_size = t.size(0);
        max_disparity = t.size(1);
        height = t.size(2);
        width = t.size(3);
    }

    __device__ uchar& at(int batch, int disparity, int row, int column){
        return data[(long)batch * dims[0] + disparity * dims[1] + row * dims[2] + column];
    }
};

namespace cspn{
    class K0Tensor: public Tensor<float>{
    public:    
        int batch_size, channel_size, max_disparity, height, width;

        K0Tensor(at::Tensor &t): Tensor(t){
            batch_size = t.size(0);
            channel_size = t.size(1);
            max_disparity = t.size(2);
            height = t.size(3);
            width = t.size(4);
        }

        __device__ float& at(int batch, int channel, int disparity, int row, int column){
            return data[(long)batch * dims[0] + channel * dims[1] + disparity * dims[2] + row * dims[3] + column];
        }
    };

    // class FilterTensor: public Tensor<float>{
    // public:    
    //     int batch_size, channel_size, max_disparity, kernel_size, height, width;

    //     FilterTensor(at::Tensor &t): Tensor(t){
    //         batch_size = t.size(0);
    //         channel_size = t.size(1);
    //         max_disparity = t.size(2);
    //         height = t.size(3);
    //         width = t.size(4);
    //     }

    //     __device__ float& at(int batch, int channel, int direction, int k1, int k2, int row, int column){
    //         return data[(long)batch * dims[0] + channel * dims[1] + direction * dims[2] + k1 * dims[3] + k2 * dims[4] + row * dims[5] + column];
    //     }
    // };
    
}

