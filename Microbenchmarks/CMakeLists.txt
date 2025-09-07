# 微基准测试构建配置
message(STATUS "Configuring Microbenchmarks with SVE support")

set(MB_SRCS)

file(GLOB TMP *.cpp)
list(APPEND MB_SRCS ${TMP})

# 检查编译器是否支持 SVE
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-march=armv8-a+sve" COMPILER_SUPPORTS_SVE)

if(NOT COMPILER_SUPPORTS_SVE)
    message(WARNING "Compiler does not support SVE extensions. Some benchmarks may not work correctly.")
endif()

foreach(mb ${MB_SRCS})
    get_filename_component(mb_name ${mb} NAME_WE)
    add_executable(${mb_name} ${mb})
    set_target_properties(${mb_name} PROPERTIES
        VERSION ${PROJECT_VERSION}
    )

    # 设置 C++ 标准
    set_property(TARGET ${mb_name} PROPERTY CXX_STANDARD 14)
    set_property(TARGET ${mb_name} PROPERTY CXX_STANDARD_REQUIRED ON)

    # 添加 SVE 编译选项
    if(COMPILER_SUPPORTS_SVE)
        target_compile_options(${mb_name} PRIVATE
            -march=armv8-a+sve      # 启用 SVE 指令集
            -O2                     # 优化级别
            -DENABLE_SVE            # 定义 SVE 宏
            -ffast-math             # 快速数学运算（可选）
            # -save-temps=obj       # 保存汇编文件（调试用）
            # -fverbose-asm         # 详细汇编输出（调试用）
        )
        message(STATUS "Building ${mb_name} with SVE support")
    else()
        target_compile_options(${mb_name} PRIVATE
            -O2                     # 基本优化
            -march=armv8-a          # 基本 ARMv8 支持
        )
        message(STATUS "Building ${mb_name} without SVE support")
    endif()

    # 添加调试信息（Debug 模式）
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        target_compile_options(${mb_name} PRIVATE -g -DDEBUG)
    endif()

    # 链接库（如果需要）
    # target_link_libraries(${mb_name} PRIVATE
    #     Threads::Threads
    #     mmio
    #     ${ARMPL_LIBRARIES}
    # )

    # 安装到指定目录
    install(TARGETS ${mb_name} DESTINATION ${PROJECT_SOURCE_DIR}/bin/benchmarks)

endforeach()

# 输出构建总结
message(STATUS "")
message(STATUS "=== Microbenchmarks Build Summary ===")
message(STATUS "SVE Support: ${COMPILER_SUPPORTS_SVE}")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ Standard: 14")
message(STATUS "Benchmarks found: ${MB_SRCS}")
message(STATUS "Install directory: ${PROJECT_SOURCE_DIR}/bin/benchmarks")
message(STATUS "=======================================")
