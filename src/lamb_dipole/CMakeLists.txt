
if(NOT FELTOR_WITH_MPI)
    set( project lamb_dipole)
    set( target shu)
    set( executable "shu.cpp")
    set( PROJECT_HEADERS shu.h init.h)
    add_executable(${project}_${target} "${executable}" ${PROJECT_HEADERS})
    set_target_properties( ${project}_${target} PROPERTIES OUTPUT_NAME ${target})
    set_target_properties( ${project}_${target} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/src/${project}")
    # Add project ALIAS and add to feltor_projects custom target
    add_executable(feltor::${project}::${target} ALIAS ${project}_${target})
    add_dependencies( feltor_projects ${project}_${target})

    target_link_libraries(${project}_${target} PRIVATE feltor::dg::dg)
    target_link_libraries(${project}_${target} PRIVATE feltor::dg::file)
    if(FELTOR_WITH_GLFW)
        target_link_libraries(${project}_${target} PRIVATE draw::draw)
    else()
        target_compile_definitions(${project}_${target} PRIVATE WITHOUT_GLFW)
    endif()

endif()
