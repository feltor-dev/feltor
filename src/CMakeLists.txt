add_custom_target( feltor_projects)
# =========================================
# First include all individual src projects

add_subdirectory("${PROJECT_SOURCE_DIR}/src/feltor")
add_subdirectory("${PROJECT_SOURCE_DIR}/src/geometry_diag")
add_subdirectory("${PROJECT_SOURCE_DIR}/src/hasegawa")
add_subdirectory("${PROJECT_SOURCE_DIR}/src/heat")
add_subdirectory("${PROJECT_SOURCE_DIR}/src/lamb_dipole")
add_subdirectory("${PROJECT_SOURCE_DIR}/src/navier_stokes")
add_subdirectory("${PROJECT_SOURCE_DIR}/src/ping")

# =========================================
# Second loop over toefl like projects
# Src pojects that are structured like the 2d toefl code
set( TOEFL_LIKE_SRC_PROJECTS
    toefl
    feltorSesol
    feltorSH
    feltorSHp
    feltorShw
    reco2D
)
# esol and poet depends on matrix
if( FELTOR_DG_WITH_MATRIX)
    list( APPEND TOEFL_LIKE_SRC_PROJECTS
        esol
        poet
    )
endif()

foreach( project IN LISTS TOEFL_LIKE_SRC_PROJECTS)
    # Good for toefl and esol and poet
    set( target ${project})
    set( executable "${project}.cpp")
    set( header "${project}.h")
    # Set the rest manually
    if( "${project}" STREQUAL feltorSesol)
        set( target "feltor")
        set( executable "feltor.cpp")
        set( header "feltor.h")
    elseif( "${project}" STREQUAL feltorSH)
        set( target "feltor")
        set( executable "feltor.cpp")
        set( header "feltor.h")
    elseif( "${project}" STREQUAL feltorSHp)
        set( target "feltor")
        set( executable "feltor.cpp")
        set( header "feltor.h")
    elseif( "${project}" STREQUAL feltorShw)
        set( target "feltor")
        set( executable "feltor.cpp")
        set( header "feltor.h")
    elseif( "${project}" STREQUAL reco2D)
        set( target "reconnection")
        set( executable "reconnection.cpp")
        set( header "reconnection.h")
    endif()

    set(PROJECT_HEADERS
        ${project}/diag.h
        ${project}/parameters.h
        ${project}/${header}
    )
    if( "${project}" STREQUAL poet)
        list( APPEND PROJECT_HEADERS ${project}/init.h ${project}/init_from_file.h)
    endif()
    add_executable(${project}_${target} "${project}/${executable}" ${PROJECT_HEADERS})
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
    target_link_libraries(${project}_${target} PRIVATE feltor::dg::geometries)
    target_link_libraries(${project}_${target} PRIVATE feltor::dg::file)
    if( "${project}" STREQUAL esol OR "${project}" STREQUAL poet)
        target_link_libraries(${project}_${target} PRIVATE feltor::dg::matrix)
    endif()

    if(FELTOR_WITH_GLFW)
        target_link_libraries(${project}_${target} PRIVATE draw::draw)
        target_compile_definitions(${project}_${target} PRIVATE WITH_GLFW)
    else()
        # This is for the reconnection and poet project
        target_compile_definitions(${project}_${target} PRIVATE WITHOUT_GLFW)
    endif()

endforeach()
