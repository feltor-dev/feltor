# ============================================
# Set up FELTOR
# TODO Should pdf documentation also be generated in cmake?

set(project thermal)

set(THERMAL_HEADERS
    collisions.h
    init_from_file.h
    init.h
    parallel.h
    parameters.h
    perpendicular.h
    solvers.h
    thermaldiag.h
    thermal.h
)
set( THERMAL_TARGETS
)

# Geometries: ON, File: ON, Matrix: ON, GLFW: OFF (There is no GLFW version)
feltor_add_executable( ${project} "thermal.cpp" thermal "${THERMAL_HEADERS}" ${FELTOR_WITH_MPI} ON ON ON OFF)

foreach( target IN LISTS THERMAL_TARGETS)
    # The executable will live in the current folder i.e. build/src/thermal/
    feltor_add_executable( ${project} "${target}.cpp" ${target} "${THERMAL_HEADERS}" OFF ON ON ON OFF)
endforeach()
