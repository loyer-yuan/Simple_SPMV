set(MB_SRCS)

file(GLOB TMP *.cpp)
list(APPEND MB_SRCS ${TMP})

foreach(mb ${MB_SRCS})
    get_filename_component(mb_name ${mb} NAME_WE)
    add_executable(${mb_name} ${mb})
    set_target_properties(${mb_name} PROPERTIES
        VERSION ${PROJECT_VERSION}
    )
    # target_compile_options(${mb_name} PRIVATE
    #     -save-temps=obj
    # )
    # target_link_libraries(${mb_name} PRIVATE
    #     Threads::Threads
    #     mmio
    #     ${ARMPL_LIBRARIES}
    # )
    install(TARGETS ${mb_name} DESTINATION ${PROJECT_SOURCE_DIR}/bin/benchmarks)

endforeach()
