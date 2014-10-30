APP=summation
CUFILES=summation.cu utils.cu
CFLAGS=-g
LDFLAGS=
CUDA_PATH ?=/usr/local/cuda
NVCC=$(CUDA_PATH)/bin/nvcc
OBJS=$(patsubst %.cu,%.o,$(CUFILES))

all: $(APP)

%.o : %.cu
	$(NVCC) -c $< $(CFLAGS) -o $@

$(APP) : $(OBJS)
	$(NVCC) $^ $(LDFLAGS) -o $@

clean :
	rm -f $(OBJS) $(APP)

# Dependencies
summation.o : summation_kernel.cu
utils.o : utils.h
