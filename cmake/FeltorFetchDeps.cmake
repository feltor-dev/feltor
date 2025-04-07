function(fetch_cccl)
  message(STATUS "Fetching CCCL")
  include(FetchContent)
  FetchContent_Declare(
    cccl
    GIT_REPOSITORY https://github.com/NVIDIA/cccl
    GIT_TAG v2.8.0
    GIT_SHALLOW TRUE
  )
  FetchContent_MakeAvailable(cccl)
endfunction()

function(fetch_vcl)
  message(STATUS "Fetching Vector class library")
  # Vector class library is not a CMake project, can't use
  # FetchContent_MakeAvailable.  All header files are in the root directory,
  # so need to use download directory as the source directory.
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
  # Draw is not a CMake project, can't use FetchContent_MakeAvailable.  All
  # header files are all in the root directory, need to use download directory
  # as the source directory.
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

function(fetch_catch2)
  message(STATUS "Fetching Catch2")
  include(FetchContent)
  FetchContent_Declare(
    catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG v3.8.0
    GIT_SHALLOW TRUE
  )
  FetchContent_MakeAvailable(catch2)
endfunction()

function(fetch_json)
  message(STATUS "Fetching nlohmann/json")
  include(FetchContent)
  FetchContent_Declare(
    json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.11.3
    GIT_SHALLOW TRUE
  )
  FetchContent_MakeAvailable(json)
endfunction()
