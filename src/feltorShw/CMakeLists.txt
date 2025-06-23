set( project feltorShw)
set(PROJECT_HEADERS
    diag.h
    parameters.h
    feltor.h
)
set(PROJECT_EXES
    crosscoherencdiag
    feltorShwdiag
    feltorShwradstat
    #fftwdiag
    histdiag
    feltorShwmerger
    feltorShwstat
    #growthrate
    normdiag
)
feltor_add_executable( ${project} "feltor.cpp" "feltor" "${PROJECT_HEADERS}" ${FELTOR_WITH_MPI} OFF ON OFF ${FELTOR_WITH_GLFW})
foreach(exe IN LISTS PROJECT_EXES)
    feltor_add_executable( ${project} "${exe}.cpp" "${exe}" "${PROJECT_HEADERS}" OFF OFF ON OFF OFF)
endforeach()

