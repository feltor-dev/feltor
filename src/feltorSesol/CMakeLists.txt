
set( project feltorSesol)
set(PROJECT_HEADERS
    diag.h
    parameters.h
    feltor.h
)
feltor_add_executable( ${project} "feltor.cpp" "feltor" "${PROJECT_HEADERS}" ${FELTOR_WITH_MPI} OFF ON OFF ${FELTOR_WITH_GLFW})
# Add diag
feltor_add_executable( ${project} "feltorSesoldiag.cpp" "feltorSesoldiag" "parameters.h" OFF OFF ON OFF OFF)
