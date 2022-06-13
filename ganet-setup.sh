# export CUDA_HOME=/usr/local/cuda-10.0
# export CUDA_HOME=/usr/local/cuda-10.1
cd GANet/extensions
rm -r build
# /opt/conda/bin/python ./setup.py install --user
C:/Python39/python ./setup.py install --user
read -p "Press any key to continue..."