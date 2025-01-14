# Try to find CUDA
find_package(CUDAToolkit QUIET)
if (CUDAToolkit_FOUND)
  message(STATUS "CUDAToolkit found, enabling CUDA support and building for CUDA")
  enable_language(CUDA)
  add_subdirectory(cuda)
else()
  message(WARNING "CUDAToolkit not found, skipping building for CUDA")
endif()