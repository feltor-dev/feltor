# ============================================
# Set up FELTOR
# TODO Add other executables like feltordiag interpolate_in_3d feltor_pol multigrid_b and convert_b
# TODO Should pdf documentation also be generated in cmake?

set(FELTOR_HEADERS
    "feltor.h"
    "common.h"
    "feltordiag.h"
    "init_from_file.h"
    "init.h"
    "parameters.h"
)


add_executable(feltor "feltor.cpp" ${FELTOR_HEADERS})
add_executable(feltor::feltor ALIAS feltor)

if(CCCL_THRUST_DEVICE_SYSTEM STREQUAL "CUDA" OR CCCL_THRUST_DEVICE_SYSTEM STREQUAL "")
  set_source_files_properties("feltor.cpp" PROPERTIES LANGUAGE CUDA)
endif()

target_include_directories(feltor PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

if(FELTOR_WITH_MPI)
    find_package(MPI REQUIRED)
    target_link_libraries(feltor PRIVATE MPI::MPI_CXX)
    target_compile_definitions(feltor PRIVATE WITH_MPI)
endif()

# ============================================
# Add dependencies

target_link_libraries(feltor PRIVATE dg::dg)
target_link_libraries(feltor PRIVATE dg::geometries)
target_link_libraries(feltor PRIVATE dg::file)

if(FELTOR_WITH_GLFW)
  if(FELTOR_WITH_MPI)
    message(WARNING "GLFW is not compatible with MPI when building FELTOR")
    target_compile_definitions(feltor PRIVATE WITHOUT_GLFW)
  else()
    CPMAddPackage( NAME draw
        GIT_REPOSITORY https://github.com/feltor-dev/draw.git
        GIT_TAG master
    )
    if( draw_ADDED)
        message( STATUS "Found Draw library in ${draw_SOURCE_DIR}")
    endif()
    # Draw Links to GLFW and OpenGL
    # Must be installed separately, e.g. `sudo apt install libglfw3-dev`
    target_link_libraries(feltor PRIVATE draw::draw)
  endif()
else()
    target_compile_definitions(feltor PRIVATE WITHOUT_GLFW)
endif()
