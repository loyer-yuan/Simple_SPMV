# FindKML.cmake
#
# Finds the Kunpeng Math Library (KML) with KSPBLAS support
# Usage: find_package(KML REQUIRED)

include(FindPackageHandleStandardArgs)

# 1. 从环境变量获取KML路径
# KML安装路径通常在 /opt/HPCKit/25.0.0/kml/gcc
if(NOT KML_DIR)
    set(KML_DIR "/opt/HPCKit/25.0.0/kml/gcc")
endif()

# 2. 查找头文件
find_path(KML_INCLUDE_DIR
    NAMES kspblas.h kml.h
    HINTS ${KML_DIR}
    PATH_SUFFIXES "include"
    DOC "KML include directory"
)

# 确保找到OpenMP支持
find_package(OpenMP REQUIRED)

# 设置OpenMP编译选项
if(OpenMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# 3. 查找KML KSPBLAS库
# 根据module load的LD_LIBRARY_PATH，KML库位于特定路径
find_library(KML_KSPBLAS_LIB
    NAMES libkspblas.so kspblas
    HINTS ${KML_DIR}
    PATH_SUFFIXES
        "lib/sve/kspblas/multi"
        "lib/kspblas/multi"
        "lib64/sve/kspblas/multi"
        "lib64/kspblas/multi"
        "lib/sve/kspblas/single"
        "lib/kspblas/single"
        "lib64/sve/kspblas/single"
        "lib64/kspblas/single"
        "lib"
        "lib64"
)

# 4. 设置库集合
set(KML_INCLUDE_DIRS ${KML_INCLUDE_DIR})

# KML库依赖
set(KML_LIBRARIES
    ${KML_KSPBLAS_LIB}
    m          # math库
    dl         # 动态加载库
    rt         # POSIX实时库
    OpenMP::OpenMP_C
)

# 5. 验证必要组件
find_package_handle_standard_args(KML
    REQUIRED_VARS
        KML_INCLUDE_DIR
        KML_KSPBLAS_LIB
)

mark_as_advanced(
    KML_INCLUDE_DIR
    KML_KSPBLAS_LIB
)
