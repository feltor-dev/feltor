# ===========================================
# Set up navier-stokes library

add_library(navier_stokes INTERFACE "navier_stokes/navier_stokes.h")
add_library(feltor::navier_stokes ALIAS navier_stokes)
target_link_libraries(navier_stokes INTERFACE dg::dg)

# ============================================
# Set up FELTOR
# TODO: Other executables

set(FELTOR_HEADERS
    "feltor/feltor.h"
    "feltor/init_from_file.h"
    "feltor/init.h"
    "feltor/parameters.h"
)

if(CCCL_THRUST_DEVICE_SYSTEM STREQUAL "CUDA" OR CCCL_THRUST_DEVICE_SYSTEM STREQUAL "")
  set_source_files_properties("feltor/feltor.cpp" PROPERTIES LANGUAGE CUDA)
endif()

add_executable(feltor "feltor/feltor.cpp" ${FELTOR_HEADERS})
add_executable(feltor::feltor ALIAS feltor)

target_include_directories(feltor PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

if(FELTOR_USE_MPI)
    find_package(MPI REQUIRED)
    target_link_libraries(feltor PRIVATE MPI::MPI_CXX)
    target_compile_definitions(feltor PRIVATE WITH_MPI)
endif()

# ============================================
# Add dependencies

target_link_libraries(feltor PRIVATE dg::dg)
target_link_libraries(feltor PRIVATE dg::file)

if(FELTOR_WITH_NAVIER_STOKES)
    target_link_libraries(feltor PRIVATE feltor::navier_stokes)
    target_compile_definitions(feltor PRIVATE WITH_NAVIER_STOKES)
endif()

if(FELTOR_WITH_GLFW)
  if(FELTOR_USE_MPI)
    message(WARNING "GLFW is not compatible with MPI when building FELTOR")
    target_compile_definitions(feltor PRIVATE WITHOUT_GLFW)
  else()
    if(FELTOR_FETCH_DRAW)
      fetch_draw()
    else()
      find_package(DRAW REQUIRED)
    endif()
    # Link to GLFW and OpenGL
    # Must be installed separately, e.g. `sudo apt install libglfw3-dev`
    find_package(glfw3 REQUIRED)
    target_link_libraries(draw::draw INTERFACE glfw)
    find_package(OpenGL REQUIRED)
    target_link_libraries(draw::draw INTERFACE OpenGL::GL)

    target_link_libraries(feltor PRIVATE draw::draw)
  endif()   
else()
    target_compile_definitions(feltor PRIVATE WITHOUT_GLFW)
endif()
