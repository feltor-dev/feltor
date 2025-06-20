set( project reco2D)
set( target reconnection)
set( executable "reconnection.cpp")
set(PROJECT_HEADERS
    diag.h
    init.h
    parameters.h
    reconnection.h
)
feltor_add_executable( ${project} "${executable}" ${target} "${PROJECT_HEADERS}" ${FELTOR_WITH_MPI} OFF ON OFF ${FELTOR_WITH_GLFW})


