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
add_compile_options(
    $<$<AND:$<CONFIG:Release>,$<CXX_COMPILER_ID:GNU,Clang,AppleClang>>:-O3>
    $<$<AND:$<CONFIG:Release>,$<CXX_COMPILER_ID:MSVC>>:/O2>
)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/CMake")
find_package(ARMPL REQUIRED)

##
### Project build.
##

set(CMAKE_INSTALL_PREFIX "${PROJECT_SOURCE_DIR}")
set(TEST_INSTALL_BASE "${PROJECT_SOURCE_DIR}/bin/test")

include_directories("${PROJECT_SOURCE_DIR}/include")
include_directories("${PROJECT_SOURCE_DIR}/external")
add_subdirectory(external)
add_subdirectory(src)