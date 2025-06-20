set( project esol)
set( target esol)
set( executable "esol.cpp")
set(PROJECT_HEADERS
    diag.h
    init.h
    init_from_file.h
    parameters.h
    esol.h
)
feltor_add_executable( ${project} "${executable}" ${target} "${PROJECT_HEADERS}"
    ${FELTOR_WITH_MPI} OFF ON ON ${FELTOR_WITH_GLFW})
