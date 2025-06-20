set( project hasegawa)

foreach ( target mima hw)
    set( PROJECT_HEADERS "${target}.h")
    feltor_add_executable( ${project} "${target}.cpp" ${target} "${PROJECT_HEADERS}" OFF OFF ON OFF ON)
endforeach()
