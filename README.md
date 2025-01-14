# Simple SPMV

This project is a simple implementation of Sparse Matrix-Vector Multiplication (SPMV) with various sparse matrix formats on different devices (e.g., GPU and ARM). The project is implemented in C++ and CUDA.

## ToDoLists

GPU naive kernel:
- [X] DIA
- [X] ELL
- [X] CSR(scalar)
- [X] CSR(vector)
- [X] COO(segment)
- [ ] CSR-Stream
- [ ] CSR5

ARM naive kernel:
- [ ] DIA
- [ ] ELL
- [ ] CSR(scalar)
- [ ] CSR(vector)
- [ ] COO(segment)
- [ ] CSR-Stream
- [ ] CSR5

## Requirements

- CUDA Toolkit
- C++ compiler (e.g., g++)
- GNU make
- CMake

## Building

To build the project, you can use the following commands:

```sh
mkdir build && cd build
cmake ..
make -j8 && make install
```

## Usage

To run the program, use the following command:

```sh
# If you want to run the program on the NVIDIA GPU.
./bin/cuda_test_all <M> <K>
```

- `<M>`: The dimension M of the matrix.
- `<K>`: The dimension K of the matrix.

## License

This project is licensed under the MIT License.