NVCC = nvcc
NVCC_FLAGS = -arch=sm_50

TARGET = tensor.cu

TC = tc

.PHONY: tc tcrun clean
tc:
	$(NVCC) $(NVCC_FLAGS) $(TARGET) -o $(TC) -lm

run:
	./$(TC)

clean:
	rm -f $(TC) *.o