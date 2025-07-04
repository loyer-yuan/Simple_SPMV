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

# 强制在 Release 模式保留调试符号
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -g")
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -g")

##
### Project build.
##

set(CMAKE_INSTALL_PREFIX "${PROJECT_SOURCE_DIR}")
set(TEST_INSTALL_BASE "${PROJECT_SOURCE_DIR}/bin/test")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH OFF)
set(CMAKE_SKIP_BUILD_RPATH TRUE)
set(CMAKE_SKIP_INSTALL_RPATH TRUE)
set(CMAKE_MACOSX_RPATH OFF)  # For macOS

include_directories("${PROJECT_SOURCE_DIR}/include")
include_directories("${PROJECT_SOURCE_DIR}/external")
add_subdirectory(external)
add_subdirectory(src)
add_subdirectory(Microbenchmarks)