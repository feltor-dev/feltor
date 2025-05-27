#TODO This Cmake can probably be used for almost all 2d src projects
set(TOEFL_HEADERS
    diag.h
    parameters.h
    toefl.h
)
add_executable(feltor_toefl "toefl.cpp" ${TOEFL_HEADERS})
set_target_properties( feltor_toefl PROPERTIES OUTPUT_NAME toefl)
add_executable(feltor::toefl::toefl ALIAS feltor_toefl)
if(FELTOR_WITH_MPI)
    target_link_libraries(feltor_toefl PRIVATE MPI::MPI_CXX)
    target_compile_definitions(feltor_toefl PRIVATE WITH_MPI)
endif()

target_link_libraries(feltor_toefl PRIVATE feltor::dg::dg)
target_link_libraries(feltor_toefl PRIVATE feltor::dg::geometries)
target_link_libraries(feltor_toefl PRIVATE feltor::dg::file)

if(FELTOR_WITH_GLFW)
    target_link_libraries(feltor_toefl PRIVATE draw::draw)
    target_compile_definitions(feltor_toefl PRIVATE WITH_GLFW)
endif()

