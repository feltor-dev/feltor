
if( FELTOR_WITH_MPI)
    set( project ping)
    set( target ping)
    if(CCCL_THRUST_DEVICE_SYSTEM STREQUAL "CUDA" OR CCCL_THRUST_DEVICE_SYSTEM STREQUAL "")
        set_source_files_properties("mpi-ping.cpp" PROPERTIES LANGUAGE CUDA)
    endif()
    add_executable(${project}_${target} mpi-ping.cpp)
    set_target_properties( ${project}_${target} PROPERTIES OUTPUT_NAME ${target})
    set_target_properties( ${project}_${target} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/src/${project}")
    target_link_libraries(${project}_${target} PRIVATE MPI::MPI_CXX)
endif()

