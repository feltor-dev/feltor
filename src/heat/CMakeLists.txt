set( project heat)
set( PROJECT_HEADERS heat.h parameters.h)
set( target heat)
feltor_add_executable( ${project} "${target}.cpp" ${target} "${PROJECT_HEADERS}" OFF ON ON OFF ON)

set( target heat_hpc)
feltor_add_executable( ${project} "${target}.cpp" ${target} "${PROJECT_HEADERS}" OFF ON ON OFF OFF)
