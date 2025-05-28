
if(NOT FELTOR_WITH_MPI)
    set( project heat)
    set( PROJECT_HEADERS heat.h parameters.h)
    if( FELTOR_WITH_GLFW)
        set( target heat)
        set( executable "heat.cpp")
        add_executable(${project}_${target} "${executable}" ${PROJECT_HEADERS})
        set_target_properties( ${project}_${target} PROPERTIES OUTPUT_NAME ${target})
        set_target_properties( ${project}_${target} PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/src/${project}")
        # Add project ALIAS and add to feltor_projects custom target
        add_executable(feltor::${project}::${target} ALIAS ${project}_${target})
        add_dependencies( feltor_projects ${project}_${target})

        target_link_libraries(${project}_${target} PRIVATE feltor::dg::dg)
        target_link_libraries(${project}_${target} PRIVATE feltor::dg::geometries)
        target_link_libraries(${project}_${target} PRIVATE feltor::dg::file)
        target_link_libraries(${project}_${target} PRIVATE draw::draw)
    endif()
    set( target heat_hpc)
    set( executable "heat_hpc.cpp")
    add_executable(${project}_${target} "${executable}" ${PROJECT_HEADERS})
    set_target_properties( ${project}_${target} PROPERTIES OUTPUT_NAME ${target})
    set_target_properties( ${project}_${target} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/src/${project}")
    # Add project ALIAS and add to feltor_projects custom target
    add_executable(feltor::${project}::${target} ALIAS ${project}_${target})
    add_dependencies( feltor_projects ${project}_${target})

    target_link_libraries(${project}_${target} PRIVATE feltor::dg::dg)
    target_link_libraries(${project}_${target} PRIVATE feltor::dg::geometries)
    target_link_libraries(${project}_${target} PRIVATE feltor::dg::file)
endif()
