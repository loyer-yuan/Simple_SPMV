#!/bin/bash

# mkdir -p ./build-linux && cd ./build-linux
# cmake -DCMAKE_BUILD_TYPE=Release ..
# make && make install

build_dir="./build-linux"

mkdir -p ${build_dir}
cmake -DCMAKE_BUILD_TYPE=Release -B ${build_dir}
cmake --build ${build_dir}
cmake --install ${build_dir}
# make && make install