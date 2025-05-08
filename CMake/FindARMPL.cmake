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

# 3. 仅查找C/C++需要的库
find_library(ARMPL_LIB
    NAMES "armpl_mp"  # "armpl"
    HINTS ${ARMPL_DIR}
    PATH_SUFFIXES "lib" "lib64"
)

# 4. 设置库集合
set(ARMPL_INCLUDE_DIRS ${ARMPL_INCLUDE_DIR})
set(ARMPL_LIBRARIES
    ${ARMPL_LIB}
    m  # math库
)

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