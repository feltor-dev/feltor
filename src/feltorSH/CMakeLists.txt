set( project feltorSH)
set(PROJECT_HEADERS
    diag.h
    parameters.h
    feltor.h
)
feltor_add_executable( ${project} "feltor.cpp" "feltor" "${PROJECT_HEADERS}" ${FELTOR_WITH_MPI} OFF ON OFF ${FELTOR_WITH_GLFW})
# Add diag
feltor_add_executable( ${project} "feltorSHdiag.cpp" "feltorSHdiag" "parameters.h" OFF OFF ON OFF OFF)
feltor_add_executable( ${project} "feltorSHdiag2.cpp" "feltorSHdiag2" "parameters.h" OFF OFF ON OFF OFF)
feltor_add_executable( ${project} "feltorSHvmaxdiag.cpp" "feltorSHvmaxdiag" "parameters.h" OFF OFF ON OFF OFF)

