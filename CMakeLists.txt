cmake_minimum_required(VERSION 3.22)

project(SIMPLE_SPMV CXX C)

set(PROJECT_VERSION 0.0.1)
option(BUILD_TEST "Build test suite" OFF)

set(CMAKE_RULE_MESSAGES OFF)
set(CMAKE_VERBOSE_MAKEFILE ON)
##
### C++ and CUDA flags.
##

# Requires C++20.
set(CMAKE_CXX_STANDARD 20)

##
### Project build.
##

set(CMAKE_INSTALL_PREFIX "${PROJECT_SOURCE_DIR}")
set(TEST_INSTALL_BASE "${PROJECT_SOURCE_DIR}/bin/test")

include_directories("${PROJECT_SOURCE_DIR}/include")
add_subdirectory(src)