
if(NOT FELTOR_WITH_MPI)
    set( project navier_stokes)
    set( target navier_stokes)
    set( executable "../feltor/feltor.cpp")
    set( PROJECT_HEADERS ../feltor/init.h navier_stokes.h)
    if(CCCL_THRUST_DEVICE_SYSTEM STREQUAL "CUDA" OR CCCL_THRUST_DEVICE_SYSTEM STREQUAL "")
        set_source_files_properties("${executable}" PROPERTIES LANGUAGE CUDA)
    endif()
    add_executable(${project}_${target} "${executable}" ${PROJECT_HEADERS})
    set_target_properties( ${project}_${target} PROPERTIES OUTPUT_NAME ${target})
    set_target_properties( ${project}_${target} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/src/${project}")
    # Add project ALIAS and add to feltor_projects custom target
    add_executable(feltor::${project}::${target} ALIAS ${project}_${target})
    add_dependencies( feltor_projects ${project}_${target})

    target_link_libraries(${project}_${target} PRIVATE feltor::dg::dg)
    target_link_libraries(${project}_${target} PRIVATE feltor::dg::file)
    target_link_libraries(${project}_${target} PRIVATE feltor::dg::geometries)
    if(FELTOR_WITH_GLFW)
        target_link_libraries(${project}_${target} PRIVATE draw::draw)
    else()
        target_compile_definitions(${project}_${target} PRIVATE WITHOUT_GLFW)
    endif()

endif()
