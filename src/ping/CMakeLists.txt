
if( FELTOR_WITH_MPI)
    set( project ping)
    set( target ping)
    add_executable(${project}_${target} mpi-ping.cpp)
    set_target_properties( ${project}_${target} PROPERTIES OUTPUT_NAME ${target})
    set_target_properties( ${project}_${target} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/src/${project}")
    target_link_libraries(${project}_${target} PRIVATE MPI::MPI_CXX)
endif()

