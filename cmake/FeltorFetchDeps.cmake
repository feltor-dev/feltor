function(fetch_thrust)
  message(STATUS "Fetching Thrust 1.9.3")
  # Using old version of Thrust for compatibility with CUSP.
  # The CMake support in this version is lacking, and can't be used
  # to create an imported target via FetchContent_MakeAvailable.
  # Instead, we'll have to use ExternalProject and set up the
  # target manually.
  include(ExternalProject)
  ExternalProject_Add(
    thrust
    GIT_REPOSITORY https://github.com/NVIDIA/thrust.git
    GIT_TAG 1.9.3
    GIT_SHALLOW TRUE
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
  )
  ExternalProject_Get_Property(thrust SOURCE_DIR)
  set(thrust_SOURCE_DIR ${SOURCE_DIR})
  add_library(thrust::thrust INTERFACE IMPORTED)
  target_include_directories(thrust::thrust INTERFACE ${thrust_SOURCE_DIR})
  # Add dependency to the external project. 
  # ExternalProject_Add will download during the build step, so we need to make
  # sure dg and all dependents are built after thrust is downloaded.
  add_dependencies(dg thrust)
endfunction()

function(fetch_cusp)
  message(STATUS "Fetching CUSP")
  # CUSP is not a CMake project, can't use FetchContent.
  include(ExternalProject)
  ExternalProject_Add(
    cusp
    GIT_REPOSITORY https://github.com/cusplibrary/cusplibrary.git
    GIT_SHALLOW TRUE
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
  )
  ExternalProject_Get_Property(cusp SOURCE_DIR)
  set(cusp_SOURCE_DIR ${SOURCE_DIR})
  add_library(cusp::cusp INTERFACE IMPORTED)
  target_include_directories(cusp::cusp INTERFACE ${cusp_SOURCE_DIR})
  add_dependencies(dg cusp)
endfunction()

function(fetch_vcl)
  message(STATUS "Fetching Vector class library")
  # Vector class library is not a CMake project, can't use FetchContent.
  include(ExternalProject)
  ExternalProject_Add(
    vcl
    GIT_REPOSITORY https://github.com/vectorclass/version1.git
    GIT_SHALLOW TRUE
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
  )
  # All header files are all in the root directory, need to use
  # download directory as the source directory.
  ExternalProject_Get_Property(vcl DOWNLOAD_DIR)
  set(vcl_SOURCE_DIR ${DOWNLOAD_DIR})
  add_library(vcl::vcl INTERFACE IMPORTED)
  target_include_directories(vcl::vcl INTERFACE ${vcl_SOURCE_DIR})
  add_dependencies(dg vcl)
endfunction()

function(fetch_draw)
  message(STATUS "Fetching DRAW")
  # Draw is not a CMake project, can't use FetchContent.
  include(ExternalProject)
  ExternalProject_Add(
    draw
    GIT_REPOSITORY https://github.com/feltor-dev/draw.git
    GIT_SHALLOW TRUE
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
  )
  # All header files are all in the root directory, need to use
  # download directory as the source directory.
  ExternalProject_Get_Property(draw DOWNLOAD_DIR)
  set(draw_SOURCE_DIR ${DOWNLOAD_DIR})
  add_library(draw::draw INTERFACE IMPORTED)
  target_include_directories(draw::draw INTERFACE ${draw_SOURCE_DIR})
  add_dependencies(dg draw)
endfunction()