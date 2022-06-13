# Dual-GDNet: Dual Guided-diffusion Network for Stereo Image Dense Matching
Neural Network Project for Ruei-Ping's Master Thesis

*Author: Ruei-Ping Wang (Jacky)*  
*Organizer: Yi-Chen Chen (Jack)*

## Prerequisite
* Windows 10 x64
* RTX 2080 Ti
* **Python 3.6**
* **PyTorch v1.6**
* **CUDA Toolkit v10.2**
* **cuDNN v8**
* **VS2019 (msvc)**

## Dependency Compilation in Windows (ganet-lib)
    cd ./GANet/extensions/
    rm -r build
replace `CppExtension` with `CUDAExtension` in line 5 of setup.py
    
    python ./setup.py install

## Installation
    .env/Scripts/activate.bat
    python -m pip install --upgrade pip
    pip install wheel
    pip install -r requirements.txt

## Issues
1. https://github.com/pytorch/pytorch/issues/42467
2. https://github.com/pytorch/pytorch/issues/11004#issuecomment-506721668
3. https://blog.csdn.net/tanmx219/article/details/100829920

## Modify pytorch package
修改 C:\Python37\Lib\site-packages\torch\include\c10\macros\Macros.h
CONSTEXPR_EXCEPT_WIN_CUDA -> CONSTEXPR_EXCEPT_WIN_CUDA const

修改 C:\Python37\Lib\site-packages\torch\include\pybind11\cast.h 1449行
explicit operator type&() { return *(this->value); ->
explicit operator type&() { *((type *)(this->value)); }

setup.py
使用CUDAExtension

# Neural Network for Stereo Image Matching
Dual-GANet

# Down-sampling rate

- height, width: 32
- disparity: 16
- KITTI 2015
  - height: 352
  - width: 1216
  - disparity: 144
## Original height width
1242 = 2*3^3*23
375 = 3*5^3
# Efficiency
Dual-GANet: 13 seconds per KITTI 2015 image with size 384, 1248



## Other
Disparity with float32 must multiply 256 for uint16 format