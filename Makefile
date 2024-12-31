# Makefile for a C++ and CUDA project
VERBOSE := @

# Compiler and flags
CXX := g++
NVCC := nvcc
CXXFLAGS := -std=c++17 -O2 -Iinclude
NVCCFLAGS := -std=c++17 -O2 -Iinclude -lineinfo

# Directories
SRC_DIR := src
OBJ_DIR := obj
BIN_DIR := bin
TEST_DIR := bin/test

# Files
CXX_SOURCES := $(SRC_DIR)/main.cpp $(SRC_DIR)/v0_cpu_naive.cpp
CUDA_SOURCES := $(SRC_DIR)/v1_cuda_DIA.cu $(SRC_DIR)/v2_cuda_ELL.cu $(SRC_DIR)/v3_cuda_CSR.cu $(SRC_DIR)/v4_cuda_COO.cu  
CXX_OBJECTS := $(patsubst $(SRC_DIR)/%.cpp, $(OBJ_DIR)/%.o, $(CXX_SOURCES))
CUDA_OBJECTS := $(patsubst $(SRC_DIR)/%.cu, $(OBJ_DIR)/%.o, $(CUDA_SOURCES))

OBJECTS := $(CXX_OBJECTS) $(CUDA_OBJECTS)
TARGET := $(BIN_DIR)/simple_sparse

# Rules
all: $(TARGET)

$(TARGET): $(OBJECTS)
	@mkdir -p $(BIN_DIR)
	$(NVCC) $(OBJECTS) -o $@

dia_test: $(OBJ_DIR)/test_dia.o $(OBJ_DIR)/v1_cuda_DIA.o $(OBJ_DIR)/v0_cpu_naive.o
	@mkdir -p $(TEST_DIR)
	$(NVCC) $^ -o $(TEST_DIR)/$@

segment_test: $(OBJ_DIR)/test_segment.o $(OBJ_DIR)/v4_cuda_COO.o $(OBJ_DIR)/v0_cpu_naive.o
	@mkdir -p $(TEST_DIR)
	$(NVCC) $^ -o $(TEST_DIR)/$@

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu
	@mkdir -p $(OBJ_DIR)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR)

.PHONY: all clean