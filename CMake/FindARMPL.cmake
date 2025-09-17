# FindARMPL.cmake
#
# Finds the ARM Performance Libraries (ARMPL) C/C++版本
# Usage: find_package(ARMPL REQUIRED)

include(FindPackageHandleStandardArgs)

# 1. 从环境变量获取ARMPL路径
if(NOT ARMPL_DIR)
    set(ARMPL_DIR $ENV{ARMPL_DIR})
endif()

# 2. 查找头文件
find_path(ARMPL_INCLUDE_DIR
    NAMES armpl.h
    HINTS ${ARMPL_DIR}
    PATH_SUFFIXES "include"
    DOC "ARMPL include directory"
)

find_package(OpenMP REQUIRED)

# 设置OpenMP静态链接
if(OpenMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# 3. 查找静态库版本
find_library(ARMPL_LIB
    NAMES "libarmpl_mp.a" "armpl_mp"  # 优先查找静态库
    HINTS ${ARMPL_DIR}
    PATH_SUFFIXES "lib" "lib64"
)

# 4. 设置库集合和静态链接选项
set(ARMPL_INCLUDE_DIRS ${ARMPL_INCLUDE_DIR})

# 静态链接ARMPL需要的完整库依赖
set(ARMPL_LIBRARIES
    ${ARMPL_LIB}
    m          # math库
    dl         # 动态加载库
    rt         # POSIX实时库
    OpenMP::OpenMP_C
)

# 设置静态链接标志
set(ARMPL_STATIC_LINK_FLAGS "-static-libgcc;-static-libstdc++")
set(ARMPL_LINK_OPTIONS "-Wl,--whole-archive;${ARMPL_LIB};-Wl,--no-whole-archive")

# 5. 验证必要组件
find_package_handle_standard_args(ARMPL
    REQUIRED_VARS
        ARMPL_INCLUDE_DIR
        ARMPL_LIB
)

mark_as_advanced(
    ARMPL_INCLUDE_DIR
    ARMPL_LIB
)