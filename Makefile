# Makefile for CUDA Matrix Addition

# Compiler and flags
NVCC = nvcc
NVCC_FLAGS = -arch=sm_50
DIAGS = --ptxas-options=-v

#-maxrrrgcount=NUM # -arch flag represents the driver version which is a must in 
# cuda compilation. --ptxas-options=-v will print the registers
# per thread and shared memory per block resource usage.

# Target executable name
TARGET = chid
EX = exec
CAP = cap
GBC = gbc
PR = pr

# Source files
SRC = check_thread_id.cu
EXEC = execution_model.cu
CAPA = device_capacity_check.cu
GBCC = grid_block_combs.cu




.PHONY: $(GBC) $(CAP) $(EX) $(TARGET)

# Different grid and block configuration
$(GBC):
	$(NVCC) -O3 $(NVCC_FLAGS) $(DIAGS) $(GBCC) -o $(GBC)

# GPU device capacity check
$(CAP):
	$(NVCC) $(NVCC_FLAGS) $(DIAGS) $(CAPA) -o $(CAP)

# Warp Divergence denmenstration and fix #
$(EX):
	$(NVCC) -O3 -g -G $(NVCC_FLAGS) $(EXEC) -o $(EX)

# Rule to build the executable
$(TARGET):
	$(NVCC) $(NVCC_FLAGS) $(SRC) -o $(TARGET)



.PHONY: tarrun exrun cprun gbcrun prrun


# remake && run pr

# run chid
tarrun:
	./$(TARGET)

# run exec
exrun:
	./$(EX)
 
cprun:
	./$(CAP)

gbcrun:
	./$(GBC) $(ARGS)


#///////////////////////// cuda command line device check //////////////////////////#

## Memory Operations



# memory read efficiency
.PHONY: effic load occ ef div freq

effic:
	sudo nvprof --metrics gld_throughput ./$(GBC) $(ARGS)
# global memory load efficiency
load:
	sudo nvprof --metrics gld_efficiency ./$(GBC) $(ARGS)

# testing occupancy for different dimx and dimy configurations
occ:
	sudo nvprof --metrics achieved_occupancy ./$(GBC) $(ARGS)

## testing the branch divergence and thread utility
ef:
	sudo nvprof --metrics branch_efficiency ./$(EX)

# event counters for branch and divergent branch
div:
	sudo nvprof --events branch,divergent_branch ./$(EX)

# check device memory frequency
freq:
	nvidia-smi -a -a -d CLOCK | fgrep -A 3 "Max Clocks" | fgrep "Memory"


#///////////////////////// cuda command line device check //////////////////////////#


# Rule to clean the directory
.PHONY: clean
clean:
	rm -f $(TARGET) $(EX) $(CAP) $(GBC) $(PR) *.o