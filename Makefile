# Makefile for a C++ and CUDA project
VERBOSE := @

# Compiler and flags
CXX := g++
NVCC := nvcc
CXXFLAGS := -std=c++11 -O2 -Iinclude
NVCCFLAGS := -O2 -Iinclude

# Directories
SRC_DIR := src
OBJ_DIR := obj
BIN_DIR := bin

# Files
CXX_SOURCES := $(wildcard $(SRC_DIR)/*.cpp)
CUDA_SOURCES := $(wildcard $(SRC_DIR)/*.cu)
CXX_OBJECTS := $(patsubst $(SRC_DIR)/%.cpp, $(OBJ_DIR)/%.o, $(CXX_SOURCES))
CUDA_OBJECTS := $(patsubst $(SRC_DIR)/%.cu, $(OBJ_DIR)/%.o, $(CUDA_SOURCES))

OBJECTS := $(CXX_OBJECTS) $(CUDA_OBJECTS)
TARGET := $(BIN_DIR)/simple_sparse

# Rules
all: $(TARGET)

$(TARGET): $(OBJECTS)
	@mkdir -p $(BIN_DIR)
	# $(NVCC) $(OBJECTS) -o $@
	$(CXX) $(OBJECTS) -o $@

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu
	@mkdir -p $(OBJ_DIR)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR)

.PHONY: all clean