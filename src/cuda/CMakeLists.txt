set(CUDA_SRCS)

##
### Find all cuda files in this directory.
##

file(GLOB TMP k*.cu)
list(APPEND CUDA_SRCS ${TMP})
add_library(cuda_spmv STATIC ${CUDA_SRCS})
set_target_properties(cuda_spmv PROPERTIES 
  VERSION ${PROJECT_VERSION}
  CUDA_ARCHITECTURES native
)

set_source_files_properties(
  test_all.cpp
  test_dia.cpp
  test_segment.cpp
  PROPERTIES LANGUAGE CUDA
)
add_executable(cuda_test_all test_all.cpp)
target_link_libraries(cuda_test_all 
  cuda_spmv
)
set_target_properties(cuda_test_all PROPERTIES 
  VERSION ${PROJECT_VERSION}
  CUDA_ARCHITECTURES native
)

# Build the tests.
if (BUILD_TEST)
  add_executable(cuda_test_dia
    test_dia.cpp
    k1_cuda_DIA_v00.cu
  )
  set_target_properties(cuda_test_dia PROPERTIES 
    VERSION ${PROJECT_VERSION}
    CUDA_ARCHITECTURES native
  )

  add_executable(cuda_test_segment
    test_segment.cpp
    k4_cuda_COO_v00.cu
  )
  set_target_properties(cuda_test_segment PROPERTIES 
    VERSION ${PROJECT_VERSION}
    CUDA_ARCHITECTURES native
  )
endif()

##
### Install targets.
##
install(TARGETS cuda_test_all)
if (BUILD_TEST)
  install(TARGETS cuda_test_dia DESTINATION ${TEST_INSTALL_BASE})
  install(TARGETS cuda_test_segment DESTINATION ${TEST_INSTALL_BASE})
endif()