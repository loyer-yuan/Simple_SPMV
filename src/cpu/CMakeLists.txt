set(CPU_SRCS)

set(THREADS_PREFER_PTHREAD_FLAG ON)  # 强制优先使用 -pthread 标志
find_package(Threads REQUIRED)

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


add_executable(cpu_test_dia
    test_dia.cpp
    k1_cpu_DIA_v00.cpp
)
set_target_properties(cpu_test_dia PROPERTIES
    VERSION ${PROJECT_VERSION}
)

install(TARGETS cpu_test_dia DESTINATION ${TEST_INSTALL_BASE})
install(TARGETS cpu_test_all DESTINATION ${TEST_INSTALL_BASE})
