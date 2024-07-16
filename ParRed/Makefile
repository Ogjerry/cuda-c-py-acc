# Makefile for CUDA Matrix Addition

# Compiler and flags
NVCC = nvcc
NVCC_FLAGS = -arch=sm_50
DIAGS = --ptxas-options=-v

#-maxrrrgcount=NUM # -arch flag represents the driver version which is a must in 
# cuda compilation. --ptxas-options=-v will print the registers
# per thread and shared memory per block resource usage.

# Target executable name
TARGET = 
PR = pr
RU = ru

# Source files
SRC = reduce_unrolling.cu
PRC = parallel_reduction.cu

.PHONY: $(PR) $(TARGET) $(RU)

all:
	$(NVCC) -O3 $(NVCC_FLAGS) $(SRC) -o $(RU) -lm && $(NVCC) -O3 $(NVCC_FLAGS) $(PRC) -o $(PR) -lm

# Reduce Loop Rolling
$(RU):
	$(NVCC) -O3 $(NVCC_FLAGS) $(SRC) -o $(RU) -lm

# Parallel Reduction
$(PR):
	$(NVCC) -O3 $(NVCC_FLAGS) $(PRC) -o $(PR) -lm


.PHONY: prrun

# RUN

runall:


prrun:
	./$(PR)

# remake && run pr
rpr:
	make clean && make && make prrun

#///////////////////////// cuda command line device check //////////////////////////#

## Memory Operations



# memory read efficiency
.PHONY: inst_warp effic load occ ef div freq

# instruction per warp (reduction as instance)
inst_warp:
	sudo nvprof --metrics inst_per_warp ./$(PR)
eff_red:
	sudo nvprof --metrics gld_throughput ./$(PR)




#///////////////////////// cuda command line device check //////////////////////////#


# Rule to clean the directory
.PHONY: clean
clean:
	rm -f $(TARGET) $(RU) $(PR) *.o
