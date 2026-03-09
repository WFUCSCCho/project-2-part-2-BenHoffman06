# Run with make run ARGS="grumpy.bmp"

SHELL := /bin/bash

CUDA_MODULE := nvidia/cuda12/cuda/12.8.1

all: blur

blur: blur.cu
	source /usr/share/Modules/init/bash && \
	module load $(CUDA_MODULE) && \
	nvcc -o blur blur.cu

run: blur
	./blur $(ARGS)

clean:
	rm -f blur

.PHONY: all blur run clean


