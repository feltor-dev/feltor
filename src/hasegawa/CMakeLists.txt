
if((NOT FELTOR_WITH_MPI) AND FELTOR_WITH_GLFW)
    foreach ( target mima hw)
        set( project hasegawa)
        set( executable "${target}.cpp")
        if(CCCL_THRUST_DEVICE_SYSTEM STREQUAL "CUDA" OR CCCL_THRUST_DEVICE_SYSTEM STREQUAL "")
            set_source_files_properties("${executable}" PROPERTIES LANGUAGE CUDA)
        endif()
        add_executable(${project}_${target} "${executable}" "${target}.h")
        set_target_properties( ${project}_${target} PROPERTIES OUTPUT_NAME ${target})
        set_target_properties( ${project}_${target} PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/src/${project}")
        # Add project ALIAS and add to feltor_projects custom target
        add_executable(feltor::${project}::${target} ALIAS ${project}_${target})
        add_dependencies( feltor_projects ${project}_${target})

        target_link_libraries(${project}_${target} PRIVATE feltor::dg::dg)
        target_link_libraries(${project}_${target} PRIVATE feltor::dg::file)
        target_link_libraries(${project}_${target} PRIVATE draw::draw)
    endforeach()
endif()
