# ============================================
# Set up FELTOR
# TODO Should pdf documentation also be generated in cmake?

set(FELTOR_HEADERS
    feltor.h
    common.h
    feltordiag.h
    init_from_file.h
    init.h
    parameters.h
)
set( FELTOR_TARGETS
    feltor
    feltor_pol
    feltordiag
    interpolate_in_3d
    multigrid_b
    convert_b
)

set( FELTOR_MPI_TARGETS
    feltor
    feltor_pol
)

if(FELTOR_WITH_MPI)
    foreach ( target IN LISTS FELTOR_MPI_TARGETS)
        if( ${target} STREQUAL feltor_pol)
            add_executable(feltor_feltor_pol "feltor.cpp" ${FELTOR_HEADERS})
        else()
            add_executable(feltor_${target} "${target}.cpp" ${FELTOR_HEADERS})
        endif()
        # Add project ALIAS and add to feltor_projects custom target
        add_executable(feltor::feltor::${target} ALIAS feltor_${target})
        add_dependencies( feltor_projects feltor_${target})
        # Change name of executable
        set_target_properties( feltor_${target} PROPERTIES OUTPUT_NAME ${target})

        target_link_libraries(feltor_${target} PRIVATE feltor::dg::dg)
        target_link_libraries(feltor_${target} PRIVATE feltor::dg::geometries)
        target_link_libraries(feltor_${target} PRIVATE feltor::dg::file)

        target_link_libraries(feltor_${target} PRIVATE MPI::MPI_CXX)
        target_compile_definitions(feltor_${target} PRIVATE WITH_MPI)
    endforeach()
else()
    foreach( target IN LISTS FELTOR_TARGETS)
        # The executable will live in the current folder i.e. build/src/feltor/
        if( ${target} STREQUAL feltor_pol)
            add_executable(feltor_feltor_pol "feltor.cpp" ${FELTOR_HEADERS})
        else()
            add_executable(feltor_${target} "${target}.cpp" ${FELTOR_HEADERS})
        endif()
        # Add project ALIAS and add to feltor_projects custom target
        add_executable(feltor::feltor::${target} ALIAS feltor_${target})
        add_dependencies( feltor_projects feltor_${target})
        # Change name of executable
        set_target_properties( feltor_${target} PROPERTIES OUTPUT_NAME ${target})

        # target_include_directories(feltor PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

        # ============================================
        # Add dependencies

        target_link_libraries(feltor_${target} PRIVATE feltor::dg::dg)
        target_link_libraries(feltor_${target} PRIVATE feltor::dg::geometries)
        target_link_libraries(feltor_${target} PRIVATE feltor::dg::file)
    endforeach()
endif()


if(FELTOR_WITH_GLFW)
    target_link_libraries(feltor_feltor PRIVATE draw::draw)
else()
    target_compile_definitions(feltor_feltor PRIVATE WITHOUT_GLFW)
endif()
target_compile_definitions(feltor_feltor_pol PRIVATE WITHOUT_GLFW)
