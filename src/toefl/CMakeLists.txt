set( project toefl)
set( target toefl)
set( executable "toefl.cpp")
set(PROJECT_HEADERS
    diag.h
    parameters.h
    toefl.h
)
feltor_add_executable( ${project} "${executable}" ${target} "${PROJECT_HEADERS}" ${FELTOR_WITH_MPI} OFF ON OFF ${FELTOR_WITH_GLFW})
