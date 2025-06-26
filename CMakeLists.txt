cmake_minimum_required(VERSION 3.26)

# ============================================
# User Options

# Set these using -DFELTOR_WITH_MPI=ON, etc.
# By default, compiles single threaded on a CUDA.


option(FELTOR_WITH_MPI          "Configure all executables that allow MPI parallelization with MPI. Do not configure executables that do not allow MPI." OFF)

# option(FELTOR_INSTALL           "Create install rules" ${PROJECT_IS_TOP_LEVEL})
# Currently our install rules do not quite work (see related issue on github page)
set( FELTOR_INSTALL OFF)

option(FELTOR_FILE_WITH_JSONCPP "Use jsoncpp instead of nlohmann/json." OFF)
option(FELTOR_FILE_WITH_NETCDF  "Configure dg_file_netcdf and dg_file_file targets. Depends on NetCDF." ON)
option(FELTOR_DG_WITH_MATRIX    "Configure dg_matrix target. Depends on Boost and Lapack." ON)
option(FELTOR_WITH_GLFW         "Configure executables, which allow it, with plotting support. Ignored if MPI is used." ON)

# If using vcpkg to install dependencies
# this is a bit of a hack until a better solution
# reveals itself,
# because the options need to be defined here
# because vcpkg needs to called before project
# Or use e.g. vcpkg install --x-feature=netcdf
if(FELTOR_FILE_WITH_NETCDF)
    # Installing netcdf from scratch on windows takes about 20 min
    list(APPEND VCPKG_MANIFEST_FEATURES "netcdf")
endif()
if(FELTOR_DG_WITH_MATRIX)
    # Installing lapack from scratch on windows takes about 1.5 hours
    list( APPEND VCPKG_MANIFEST_FEATURES "matrix")
endif()
if(FELTOR_WITH_GLFW)
    # Installing glfw from scratch on windows takes about a minute
    list( APPEND VCPKG_MANIFEST_FEATURES "glfw")
endif()


# Project name and version
project(
    feltor
    VERSION         8.2.1 # Update every time a release (tag) is made
    DESCRIPTION     "Numerical methods for fluid simulations of magnetized plasmas"
    HOMEPAGE_URL    "https://feltor-dev.github.io"
    LANGUAGES       CXX
)

option(FELTOR_BUILD_PROJECTS    "Configure feltor project target feltor_projects." ${PROJECT_IS_TOP_LEVEL})
option(FELTOR_BUILD_TESTS       "Configure test target dg_tests." ${PROJECT_IS_TOP_LEVEL})
option(FELTOR_BUILD_BENCHMARKS  "Configure benchmark target dg_benchmarks." ${PROJECT_IS_TOP_LEVEL})

# Enable support for IDEs like Visual Studio
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# Add cmake modules (e.g. FindThrust.cmake)
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")



# ============================================
# Make cpm package manager visible here and in subdirectories
include(CPM)
# Make installation dirs available here and in subdirectories
include(GNUInstallDirs)

# ============================================
# Set up testing dependencies

if(FELTOR_BUILD_TESTS)
  enable_testing()
  CPMAddPackage(
    NAME Catch2
    GITHUB_REPOSITORY catchorg/Catch2
    VERSION 3.8.0
    SYSTEM ON
    EXCLUDE_FROM_ALL ON
  )
  # Make include(Catch2) available according to
  # https://github.com/catchorg/Catch2/blob/devel/docs/cmake-integration.md
  list(APPEND CMAKE_MODULE_PATH ${Catch2_SOURCE_DIR}/extras)
endif()


if(FELTOR_WITH_MPI)
  find_package(MPI REQUIRED)
  # target_link_libraries(dg INTERFACE MPI::MPI_CXX)
endif()

# =============================================
# On naming targets:
#https://stackoverflow.com/questions/67757157/change-exported-target-name-in-cmake-install-alias-to-a-target

# ============================================
# MW: CMake switches on -forward-unknown-to-host-compiler by default, so C++ flags
# may be set even when compiling for CUDA.

# We set warnings globally for this folder and below
# https://cmake.org/cmake/help/latest/command/add_compile_options.html
# Also note that "add_compile_options" only applies to targets defined after
# Set up compiler-specific flags
if (MSVC)
    # warning level 4
    add_compile_options(/W4 /permissive-)
else()
    # additional warnings
    if(CCCL_THRUST_DEVICE_SYSTEM STREQUAL "CUDA" OR CCCL_THRUST_DEVICE_SYSTEM STREQUAL "")
        # For CUDA pedantic spits out too many warnings
        add_compile_options(-Wall -Wextra)
    else()
        add_compile_options(-Wall -Wextra -Wpedantic -Werror)
    endif()
    if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
        add_compile_options( -Wno-c++20-extensions) # Necessary for Catch2 TEMPLATE_TEST_CASE
    endif()
endif()



# =============================================
# Set up DG header-only library
# Defines the targets:
# - dg library, alias dg::dg
# - dg file library, alias dg::file
# - dg_benchmarks
# - dg_tests

# This is a somewhat chicken-egg problem: The cuda language can only be enabled
# once in the project. But before including the cccl package the device macro
# is not defined. So we must assume the user defines CCCL_THRUST_DEVICE_SYSTEM
# and/or if they do not, then the default is also CUDA
if(CCCL_THRUST_DEVICE_SYSTEM STREQUAL "CUDA" OR CCCL_THRUST_DEVICE_SYSTEM STREQUAL "")
  #MW: For some reason enable_language should not be called in subprojects
  #https://gitlab.kitware.com/cmake/cmake/-/issues/26751
  if(${PROJECT_IS_TOP_LEVEL})
    enable_language(CUDA)
  endif()
  get_property(languages GLOBAL PROPERTY ENABLED_LANGUAGES)
  if( NOT "CUDA" IN_LIST languages)
      message( FATAL_ERROR "You have to use enable_language(CUDA) before fetching the feltor package with CCCL_THRUST_DEVICE_SYSTEM = CUDA.")
  endif()
  find_package(CUDAToolkit REQUIRED)
endif()
add_subdirectory("${PROJECT_SOURCE_DIR}/inc/dg")



# ============================================
# Add FELTOR executables

if(FELTOR_BUILD_PROJECTS)
  include(feltor-add-executable)
  if(FELTOR_WITH_GLFW)
    if(FELTOR_WITH_MPI)
      message(WARNING "GLFW is not compatible with MPI when building FELTOR and will be ignored")
    else()
      # Draw Links to GLFW and OpenGL
      # Must be installed separately, e.g. `sudo apt install libglfw3-dev`
      CPMAddPackage(
          NAME draw
          GIT_REPOSITORY https://github.com/feltor-dev/draw.git
          GIT_TAG master
          SYSTEM ON
          EXCLUDE_FROM_ALL ON
      )
      if( draw_ADDED)
          message( STATUS "Found Draw library in ${draw_SOURCE_DIR}")
      endif()
    endif()
  endif()
  add_custom_target( feltor_projects)
  # =========================================
  # Include all individual src projects

  add_subdirectory("${PROJECT_SOURCE_DIR}/src/esol")
  add_subdirectory("${PROJECT_SOURCE_DIR}/src/feltor")
  add_subdirectory("${PROJECT_SOURCE_DIR}/src/feltorSesol")
  add_subdirectory("${PROJECT_SOURCE_DIR}/src/feltorSH")
  add_subdirectory("${PROJECT_SOURCE_DIR}/src/feltorSHp")
  add_subdirectory("${PROJECT_SOURCE_DIR}/src/feltorShw")
  add_subdirectory("${PROJECT_SOURCE_DIR}/src/geometry_diag")
  add_subdirectory("${PROJECT_SOURCE_DIR}/src/hasegawa")
  add_subdirectory("${PROJECT_SOURCE_DIR}/src/heat")
  add_subdirectory("${PROJECT_SOURCE_DIR}/src/lamb_dipole")
  add_subdirectory("${PROJECT_SOURCE_DIR}/src/navier_stokes")
  add_subdirectory("${PROJECT_SOURCE_DIR}/src/ping")
  add_subdirectory("${PROJECT_SOURCE_DIR}/src/poet")
  add_subdirectory("${PROJECT_SOURCE_DIR}/src/reco2D")
  add_subdirectory("${PROJECT_SOURCE_DIR}/src/toefl")

endif()


# ============================================
# INSTALL

# PROVIDING A PACKAGE CONFIG FILE
# A config file is the preferred way for an installed project to make itself
# available for other CMake projects via find_package
# First, we generate the config files via the helper utilities in the current build directory
include(CMakePackageConfigHelpers)

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/feltor-config.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/feltor-config.cmake"
    INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/feltor"
    PATH_VARS PROJECT_NAME PROJECT_VERSION
)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/feltor-version.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

if(FELTOR_INSTALL)
    # Second, we need an install rule that copies the above two generated files
    # to the cmake install libdir
    install(
        FILES
            "${CMAKE_CURRENT_BINARY_DIR}/feltor-config.cmake"
            "${CMAKE_CURRENT_BINARY_DIR}/feltor-version.cmake"
        DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/feltor"
        COMPONENT feltor_dg
    )

    # ============================================
    # PACKAGING

    # Simply include CPack will generate CPack...cmake files
    include(CPack)
endif()
