set(CPU_SRCS)

set(THREADS_PREFER_PTHREAD_FLAG ON)  # 强制优先使用 -pthread 标志
find_package(Threads REQUIRED)

# 添加ARM SVE支持
set(ARM_SVE_FLAGS "-march=armv8-a+sve")

##
### Find all cpu kernel files in this directory.
##

file(GLOB TMP k*.cpp)
list(APPEND CPU_SRCS ${TMP})
add_library(cpu_spmv STATIC ${CPU_SRCS})
set_target_properties(cpu_spmv PROPERTIES
    VERSION ${PROJECT_VERSION}
)
set_source_files_properties(
    test_all.cpp
    test_dia.cpp
    PROPERTIES LANGUAGE CXX
)

# 添加SVE编译选项
target_compile_options(cpu_spmv PRIVATE
    ${ARM_SVE_FLAGS}
    -save-temps=obj
)

add_executable(cpu_test_all
    test_all.cpp
)
target_include_directories(cpu_test_all PRIVATE
    ${ARMPL_INCLUDE_DIRS}
)
target_link_libraries(cpu_test_all PRIVATE
    cpu_spmv
    Threads::Threads
    mmio
    ${ARMPL_LIBRARIES}
)
set_target_properties(cpu_test_all PROPERTIES
    VERSION ${PROJECT_VERSION}
)
target_compile_options(cpu_test_all PRIVATE
    ${ARM_SVE_FLAGS}
    -save-temps=obj
)

add_executable(cpu_test_dia
    test_dia.cpp
    k1_cpu_DIA_v00.cpp
)
set_target_properties(cpu_test_dia PROPERTIES
    VERSION ${PROJECT_VERSION}
)

install(TARGETS cpu_test_dia DESTINATION ${TEST_INSTALL_BASE})
install(TARGETS cpu_test_all DESTINATION ${TEST_INSTALL_BASE})

# 检查ARM SVE支持的函数
include(CheckCXXSourceCompiles)
function(check_arm_sve_support var)
    set(CMAKE_REQUIRED_FLAGS "${ARM_SVE_FLAGS}")
    set(CMAKE_REQUIRED_INCLUDES "${CMAKE_CURRENT_SOURCE_DIR}")
    check_cxx_source_compiles("
        #include <arm_sve.h>
        int main() {
            svbool_t pg = svptrue_b32();
            return 0;
        }
    " ${var})
endfunction()

# 检查SVE支持
check_arm_sve_support(ARM_SVE_AVAILABLE)
if(ARM_SVE_AVAILABLE)
    message(STATUS "ARM SVE support detected")
else()
    message(WARNING "ARM SVE support not detected - SVE kernels will not be available")
endif()