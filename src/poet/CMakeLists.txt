set( project poet)
set( target poet)
set( executable "poet.cpp")
set(PROJECT_HEADERS
    diag.h
    init.h
    init_from_file.h
    parameters.h
    poet.h
)
feltor_add_executable( ${project} "${executable}" ${target} "${PROJECT_HEADERS}" ${FELTOR_WITH_MPI} OFF ON ON ${FELTOR_WITH_GLFW})
