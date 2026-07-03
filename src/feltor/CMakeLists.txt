# ============================================
# Set up FELTOR
# TODO Should pdf documentation also be generated in cmake?

set(project feltor)

set(FELTOR_HEADERS
    feltor.h
    common.h
    feltordiag.h
    init_from_file.h
    init.h
    parameters.h
)
set( FELTOR_TARGETS
    feltordiag
    interpolate_in_3d
    multigrid_b
    convert_b
)


feltor_add_executable( ${project} "feltor.cpp" feltor_pol "${FELTOR_HEADERS}" ${FELTOR_WITH_MPI} ON ON OFF OFF)
target_compile_definitions(feltor_feltor_pol PRIVATE WRITE_POL_FILE)
feltor_add_executable( ${project} "feltor.cpp" feltor     "${FELTOR_HEADERS}" ${FELTOR_WITH_MPI} ON ON OFF ${FELTOR_WITH_GLFW})

foreach( target IN LISTS FELTOR_TARGETS)
    # The executable will live in the current folder i.e. build/src/feltor/
    feltor_add_executable( ${project} "${target}.cpp" ${target} "${FELTOR_HEADERS}" OFF ON ON OFF OFF)
endforeach()
