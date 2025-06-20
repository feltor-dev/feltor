set( project feltorSHp)
set(PROJECT_HEADERS
    diag.h
    parameters.h
    feltor.h
)
feltor_add_executable( ${project} "feltor.cpp" "feltor" "${PROJECT_HEADERS}" ${FELTOR_WITH_MPI} OFF ON OFF ${FELTOR_WITH_GLFW})
