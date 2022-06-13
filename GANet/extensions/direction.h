# pragma once

#include <torch/extension.h>

class DirectionInfo{
public:
    enum Type {FORWARD, BACKWARD};

    int row_offset, col_offset;
    int base_width, shift_limit;
    int direction;

    __device__ virtual int start_row(int base_index, DirectionInfo::Type type);
    __device__ virtual int start_col(int base_index, DirectionInfo::Type type);
};

class LeftToRight: public DirectionInfo{
public:
    __device__ LeftToRight(int height, int width){
        base_width = height;
        shift_limit = width;
        row_offset = 0;
        col_offset = 1;
        direction = 0;
    }
    
    __device__ virtual int start_row(int base_index, DirectionInfo::Type type){
        return base_index;
    }
    __device__ virtual int start_col(int base_index, DirectionInfo::Type type){
        if (type == DirectionInfo::Type::FORWARD)
            return 0;
        else if (type == DirectionInfo::Type::BACKWARD)
            return shift_limit - 1;
    }
};

class RightToLeft: public DirectionInfo{
public:
    __device__ RightToLeft(int height, int width){
        base_width = height;
        shift_limit = width;
        row_offset = 0;
        col_offset = -1;
        direction = 1;
    }
    
    __device__ virtual int start_row(int base_index, DirectionInfo::Type type){
        return base_index;
    }
    __device__ virtual int start_col(int base_index, DirectionInfo::Type type){
        if (type == DirectionInfo::Type::FORWARD)
            return shift_limit - 1;
        else if (type == DirectionInfo::Type::BACKWARD)
            return 0;
    }
};

class UpToDown: public DirectionInfo{
public:
    __device__ UpToDown(int height, int width){
        base_width = width;
        shift_limit = height;
        row_offset = 1;
        col_offset = 0;
        direction = 2;
    }
    
    __device__ virtual int start_row(int base_index, DirectionInfo::Type type){
        if (type == DirectionInfo::Type::FORWARD)
            return 0;
        else if (type == DirectionInfo::Type::BACKWARD)
            return shift_limit - 1;
    }
    __device__ virtual int start_col(int base_index, DirectionInfo::Type type){
        return base_index;
    }
};

class DownToUp: public DirectionInfo{
public:
    __device__ DownToUp(int height, int width){
        base_width = width;
        shift_limit = height;
        row_offset = -1;
        col_offset = 0;
        direction = 3;
    }
    
    __device__ virtual int start_row(int base_index, DirectionInfo::Type type){
        if (type == DirectionInfo::Type::FORWARD)
            return shift_limit - 1;
        else if (type == DirectionInfo::Type::BACKWARD)
            return 0;
    }
    __device__ virtual int start_col(int base_index, DirectionInfo::Type type){
        return base_index;
    }
};

class SGMDirection{
public:
    int row_offset, col_offset;
    int direction;

    __device__ virtual int start_row(int dir_code, int height, int width);
    __device__ virtual int start_col(int dir_code, int height, int width);
};

class SGM_D0: public SGMDirection{
public:
    __device__ SGM_D0(){
        row_offset = -1;
        col_offset = 0;
        direction = 0;
    }

    __device__ virtual int start_row(int dir_code, int height, int width){
        if (dir_code < width)
            return height - 1;
        else return -1;
    }

    __device__ virtual int start_col(int dir_code, int height, int width){
        if (dir_code < width)
            return dir_code;
        else return -1;
    }
};

class SGM_D1: public SGMDirection{
public:
    __device__ SGM_D1(){
        row_offset = -1;
        col_offset = 1;
        direction = 1;
    }

    __device__ virtual int start_row(int dir_code, int height, int width){
        if (dir_code < width)
            return height - 1;
        else return dir_code - width;
    }

    __device__ virtual int start_col(int dir_code, int height, int width){
        if (dir_code < width)
            return dir_code;
        else return 0;
    }
};

class SGM_D2: public SGMDirection{
public:
    __device__ SGM_D2(){
        row_offset = 0;
        col_offset = 1;
        direction = 2;
    }

    __device__ virtual int start_row(int dir_code, int height, int width){
        if (dir_code < width)
            return -1;
        else return dir_code - width;
    }

    __device__ virtual int start_col(int dir_code, int height, int width){
        if (dir_code < width)
            return -1;
        else return 0;
    }
};

class SGM_D3: public SGMDirection{
public:
    __device__ SGM_D3(){
        row_offset = 1;
        col_offset = 1;
        direction = 3;
    }

    __device__ virtual int start_row(int dir_code, int height, int width){
        if (dir_code < width)
            return 0;
        else return dir_code - width;
    }

    __device__ virtual int start_col(int dir_code, int height, int width){
        if (dir_code < width)
            return dir_code;
        else return 0;
    }
};

class SGM_D4: public SGMDirection{
public:
    __device__ SGM_D4(){
        row_offset = 1;
        col_offset = 0;
        direction = 4;
    }

    __device__ virtual int start_row(int dir_code, int height, int width){
        if (dir_code < width)
            return 0;
        else return -1;
    }

    __device__ virtual int start_col(int dir_code, int height, int width){
        if (dir_code < width)
            return dir_code;
        else return -1;
    }
};

class SGM_D5: public SGMDirection{
public:
    __device__ SGM_D5(){
        row_offset = 1;
        col_offset = -1;
        direction = 5;
    }

    __device__ virtual int start_row(int dir_code, int height, int width){
        if (dir_code < width)
            return 0;
        else return dir_code - width;
    }

    __device__ virtual int start_col(int dir_code, int height, int width){
        if (dir_code < width)
            return dir_code;
        else return width - 1;
    }
};

class SGM_D6: public SGMDirection{
public:
    __device__ SGM_D6(){
        row_offset = 0;
        col_offset = -1;
        direction = 6;
    }

    __device__ virtual int start_row(int dir_code, int height, int width){
        if (dir_code < width)
            return -1;
        else return dir_code - width;
    }

    __device__ virtual int start_col(int dir_code, int height, int width){
        if (dir_code < width)
            return -1;
        else return width - 1;
    }
};

class SGM_D7: public SGMDirection{
public:
    __device__ SGM_D7(){
        row_offset = -1;
        col_offset = -1;
        direction = 7;
    }

    __device__ virtual int start_row(int dir_code, int height, int width){
        if (dir_code < width)
            return height - 1;
        else return dir_code - width;
    }

    __device__ virtual int start_col(int dir_code, int height, int width){
        if (dir_code < width)
            return dir_code;
        else return width - 1;
    }
};