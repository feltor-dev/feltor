set( project lamb_dipole)
set( PROJECT_HEADERS shu.h init.h diag.h)
set( target shu)
feltor_add_executable( ${project} "${target}.cpp" ${target} "${PROJECT_HEADERS}" OFF OFF ON OFF ${FELTOR_WITH_GLFW})
