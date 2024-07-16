CU_APPS=nestedHelloWorld nestedReduce nestedReduce2 nestedReduceNosync \
	    reduceInteger simpleDeviceQuery simpleDivergence sumMatrix
C_APPS=

all: ${C_APPS} ${CU_APPS}

%: %.cu
	nvcc -O3 -arch=sm_50 -o $@ $< -lcudadevrt --relocatable-device-code true
%: %.c
	gcc -O3 -std=c99 -o $@ $<
clean:
	rm -f ${CU_APPS} ${C_APPS}
