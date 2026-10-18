# CUDA build of the SpikeNav golden model for the Jetson Orin Nano (sm_87); links the C99 model
# (golden_c/snn_golden.c) on the host for delays, generator seeds and the reference check.
NVCC ?= /usr/local/cuda/bin/nvcc
ARCH ?= sm_87
CC ?= gcc
all: snn_cuda
snn_golden.o: ../golden_c/snn_golden.c ../golden_c/snn_golden.h
	$(CC) -O3 -std=c99 -c -o $@ ../golden_c/snn_golden.c
snn_cuda: snn_cuda.cu snn_golden.o
	$(NVCC) -O3 -arch=$(ARCH) -lineinfo -o $@ snn_cuda.cu snn_golden.o -lm
clean:
	rm -f snn_cuda snn_golden.o
