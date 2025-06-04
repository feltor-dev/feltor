# @brief Utility function to simplify common creation of src executables
#
# This function will add the target ${project}_${target} and its correct
# aliases in the output directory build/src/${project} The target will be added
# to the feltor_projects custom target
# @param project Name of the project the executable is part of
# @param executable Full path to the executable
# @param target Target to create
# @param PROJECT_HEADERS the list of headers the executable depends on (surround with "" when calling)
# @param If with_MPI is false then the target will not be created if FELTOR_WITH_MPI is true
# @param If with_geometries is true then the target will be linked to feltor::dg::geometries
function(feltor_add_executable
    project
    executable
    target
    PROJECT_HEADERS
    with_MPI
    with_geometries
    with_file
    with_matrix
    with_GLFW
)
    # Do not build targets in certain situations
    if(with_MPI AND NOT FELTOR_WITH_MPI)
        return()
    endif()
    if(NOT with_MPI AND FELTOR_WITH_MPI)
        return()
    endif()
    if(with_GLFW AND NOT FELTOR_WITH_GLFW)
        return()
    endif()
    #ignore GLFW when MPI is set
    if( with_GLFW AND with_MPI)
        set( with_GLFW OFF)
    endif()
    if(CCCL_THRUST_DEVICE_SYSTEM STREQUAL "CUDA" OR CCCL_THRUST_DEVICE_SYSTEM STREQUAL "")
        set_source_files_properties("${executable}" PROPERTIES LANGUAGE CUDA)
    endif()
    add_executable(${project}_${target} "${executable}")
    if( PROJECT_HEADERS)
        target_sources(${project}_${target} PRIVATE ${PROJECT_HEADERS})
    endif()
    # Change name and directory of executable
    set_target_properties( ${project}_${target} PROPERTIES OUTPUT_NAME ${target})
    set_target_properties( ${project}_${target} PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/src/${project}")
    # Add project ALIAS and add to feltor_projects custom target
    add_executable(feltor::${project}::${target} ALIAS ${project}_${target})
    add_dependencies( feltor_projects ${project}_${target})
    if(FELTOR_WITH_MPI)
        target_link_libraries(${project}_${target} PRIVATE MPI::MPI_CXX)
        target_compile_definitions(${project}_${target} PRIVATE WITH_MPI)
    endif()

    target_link_libraries(${project}_${target} PRIVATE feltor::dg::dg)
    if(with_geometries)
        target_link_libraries(${project}_${target} PRIVATE feltor::dg::geometries)
    endif()
    if(with_file)
        target_link_libraries(${project}_${target} PRIVATE feltor::dg::file)
    endif()
    if(with_matrix)
        target_link_libraries(${project}_${target} PRIVATE feltor::dg::matrix)
    endif()

    if(with_GLFW)
        target_link_libraries(${project}_${target} PRIVATE draw::draw)
        target_compile_definitions(${project}_${target} PRIVATE WITH_GLFW)
    else()
        # This is for the reconnection and poet project
        target_compile_definitions(${project}_${target} PRIVATE WITHOUT_GLFW)
    endif()
endfunction()
