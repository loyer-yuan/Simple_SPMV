# Simple SPMV

This project is a simple implementation of Sparse Matrix-Vector Multiplication (SPMV).

## ToDoLists

GPU kernel:
- [X] DIA
- [X] ELL
- [X] CSR(scalar)
- [X] CSR(vector)
- [X] COO(segment)

## Requirements

- CUDA Toolkit
- C++ compiler (e.g., g++)
- GNU make

## Building

To build the project, you can use the following commands:

```sh
make
```

## Usage

To run the program, use the following command:

```sh
./bin/simple_spmv <M> <K>
```

- `<M>`: The dimension M of the matrix.
- `<K>`: The dimension K of the matrix.

## License

This project is licensed under the MIT License.