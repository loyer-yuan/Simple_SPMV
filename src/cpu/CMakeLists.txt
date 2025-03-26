add_executable(cpu_test_dia
    test_dia.cpp
    k1_cpu_DIA_v00.cpp
)
set_target_properties(cpu_test_dia PROPERTIES 
    VERSION ${PROJECT_VERSION}
)

install(TARGETS cpu_test_dia DESTINATION ${TEST_INSTALL_BASE})
