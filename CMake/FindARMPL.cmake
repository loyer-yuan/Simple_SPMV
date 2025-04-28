# FindARMPL.cmake - Finds Arm Performance Libraries installation

# 定义默认搜索路径
set(ARMPL_ROOT $ENV{ARMPL_DIR} $ENV{ARMPL_ROOT} CACHE PATH "ArmPL安装根目录")
set(ARMPL_ARCH "aarch64" CACHE STRING "目标架构 (e.g. aarch64)")
set(ARMPL_COMPILER "gcc" CACHE STRING "编译器类型 (gcc|clang)")
set(ARMPL_MULTITHREAD "mp" CACHE STRING "多线程支持 (mp|)")

# 生成可能的库目录后缀
# if(CMAKE_SIZEOF_VOID_P EQUAL 8)
#     set(_armpl_libdir_suffixes lib/${ARMPL_ARCH}_${ARMPL_COMPILER}_${ARMPL_MULTITHREAD})
# else()
#     set(_armpl_libdir_suffixes lib)
# endif()

# 搜索包含目录
find_path(ARMPL_INCLUDE_DIR
    NAMES armpl.h
    HINTS ${ARMPL_ROOT}
    PATH_SUFFIXES "include"
    DOC "ArmPL包含目录"
)

# 搜索库文件
find_library(ARMPL_LIBRARY
    NAMES "armpl_lp64_mp"
    HINTS ${ARMPL_ROOT}
    PATH_SUFFIXES "lib" "lib64"
    DOC "ArmPL主库路径"
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(ARMPL
    REQUIRED_VARS ARMPL_LIBRARY ARMPL_INCLUDE_DIR
)

# search dependency
find_library(ARMPL_AMATH_LIB NAMES "amath" HINTS ${ARMPL_ROOT} PATH_SUFFIXES "lib" "lib64")
find_library(ARMPL_M_LIB NAMES "m")

if(ARMPL_FOUND)
    if (NOT TARGET ARM::pl)
        add_library(ARM::pl INTERFACE IMPORTED)
    endif()

    find_package(OpenMP REQUIRED)

    set(ARMPL_LIBRARIES ${ARMPL_LIBRARY} ${ARMPL_AMATH_LIB} ${ARMPL_M_LIB})
    set(ARMPL_INCLUDE_DIRS ${ARMPL_INCLUDE_DIR})

    set_property(TARGET ARM::pl PROPERTY INTERFACE_LINK_LIBRARIES ${ARMPL_LIBRARIES})
    set_property(TARGET ARM::pl PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${ARMPL_INCLUDE_DIRS})

    mark_as_advanced(
        ARMPL_LIBRARIES
        ARMPL_INCLUDE_DIRS
    )
endif()