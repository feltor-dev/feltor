function(fetch_thrust)
  message(STATUS "Fetching Thrust 1.9.3")
  # Using old version of Thrust for compatibility with CUSP.
  # The CMake support in this version is lacking, and can't be used
  # to create an imported target via FetchContent_MakeAvailable.
  include(FetchContent)
  FetchContent_Declare(thrust)
  FetchContent_Populate(
    thrust
    GIT_REPOSITORY https://github.com/NVIDIA/thrust.git
    GIT_TAG 1.9.3
    GIT_SHALLOW TRUE
  )
  FetchContent_GetProperties(thrust)
  add_library(thrust::thrust INTERFACE IMPORTED)
  target_include_directories(thrust::thrust INTERFACE "${thrust_SOURCE_DIR}")
endfunction()

function(fetch_cusp)
  message(STATUS "Fetching CUSP")
  # CUSP is not a CMake project, can't use FetchContent_MakeAvailable.
  include(FetchContent)
  FetchContent_Declare(cusp)
  FetchContent_Populate(
    cusp
    GIT_REPOSITORY https://github.com/cusplibrary/cusplibrary.git
    GIT_SHALLOW TRUE
    SOURCE_DIR "${CMAKE_BINARY_DIR}/cusp"
  )
  FetchContent_GetProperties(cusp)
  add_library(cusp::cusp INTERFACE IMPORTED)
  target_include_directories(cusp::cusp INTERFACE "${cusp_SOURCE_DIR}")
endfunction()

function(fetch_vcl)
  message(STATUS "Fetching Vector class library")
  # Vector class library is not a CMake project, can't use FetchContent_MakeAvailable.
  # All header files are all in the root directory, need to use download directory as
  # the source directory.
  include(FetchContent)
  FetchContent_Declare(vcl)
  FetchContent_Populate(
    vcl
    GIT_REPOSITORY https://github.com/vectorclass/version1.git
    GIT_SHALLOW TRUE
    SOURCE_DIR "${CMAKE_BINARY_DIR}/vcl"
  )
  add_library(vcl::vcl INTERFACE IMPORTED)
  target_include_directories(vcl::vcl INTERFACE "${CMAKE_BINARY_DIR}")
endfunction()

function(fetch_draw)
  message(STATUS "Fetching DRAW")
  # Draw is not a CMake project, can't use FetchContent_MakeAvailable.
  # All header files are all in the root directory, need to use download directory as
  # the source directory.
  include(FetchContent)
  FetchContent_Declare(draw)
  FetchContent_Populate(
    draw
    GIT_REPOSITORY https://github.com/feltor-dev/draw.git
    GIT_SHALLOW TRUE
    SOURCE_DIR "${CMAKE_BINARY_DIR}/draw"
  )
  add_library(draw::draw INTERFACE IMPORTED)
  target_include_directories(draw::draw INTERFACE "${CMAKE_BINARY_DIR}")
endfunction()
