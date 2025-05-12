#!/bin/bash

mkdir -p ./build-macOS && cd ./build-macOS
cmake -DOpenMP_C_FLAGS="-Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include" \
        -DOpenMP_C_LIB_NAMES="omp" \
        -DOpenMP_omp_LIBRARY=/opt/homebrew/opt/libomp/lib/libomp.dylib \
        -DOpenMP_CXX_FLAGS="-Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include" \
        -DOpenMP_CXX_LIB_NAMES="omp" \
        -G "Xcode" ..