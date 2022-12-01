CUDA_DIR = /usr/local/cuda

all:
	g++ measure-speed.cpp -o main \
	-I../Time -I../Statistics \
	-I$(CUDA_DIR)/include -L$(CUDA_DIR)/lib64 -lcudart \
	-I$(CUDA_DIR)/include -L$(CUDA_DIR)/lib64 -lcudnn \
	-I/usr/include/opencv4 -lopencv_imgcodecs -lopencv_imgproc -lopencv_core
