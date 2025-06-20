# inc/dg/file/CMakeLists.txt

# Files depending on json but not netcdf:
set( JSON_HEADERS
  json_utilities.h
  ../../file/json_wrapper.h
  ../../file/json_probes.h
  ../../file/probes_params.h
)

# Files depending on netcdf (and dg) but not json:
set( NETCDF_HEADERS
  nc_utilities.h
  ../../file/easy_atts.h
  ../../file/easy_dims.h
  ../../file/easy_input.h
  ../../file/easy_output.h
  ../../file/nc_error.h
  ../../file/nc_file.h
  ../../file/nc_hyperslab.h
  ../../file/nc_mpi_file.h
  ../../file/records.h
)
# Files depending on both json and netcdf:
set(FILE_HEADERS
  file.h
  ../../file/probes.h
)

# find ../../file -name "*t.cpp" | sort
# Split out json tests
set(FILE_TESTS
  ../../file/easy_atts_t.cpp
  ../../file/easy_dims_t.cpp
  ../../file/easy_output_t.cpp
  ../../file/nc_file_t.cpp
  ../../file/nc_utilities_t.cpp
  ../../file/probes_t.cpp
)

set( JSON_TESTS
  ../../file/json_wrapper_t.cpp
  ../../file/probes_parser_t.cpp
)

# Find out with grep WITH_MPI *
set( JSON_MPI_TESTS
  ../../file/probes_parser_t.cpp
)
# Only easy_atts_t cannot be compiled for MPI
set(FILE_MPI_TESTS
  ../../file/easy_dims_t.cpp
  ../../file/easy_output_t.cpp
  ../../file/nc_file_t.cpp
  ../../file/nc_utilities_t.cpp
  ../../file/probes_t.cpp
)


# ============================================
# Create DG file library target
add_library(dg_file_json INTERFACE)
set_property(TARGET dg_file_json PROPERTY EXPORT_NAME json)
add_library(feltor::dg::file::json ALIAS dg_file_json)
target_sources( dg_file_json INTERFACE
    FILE_SET dg_file_json_headers TYPE HEADERS
    BASE_DIRS ${PROJECT_SOURCE_DIR}/inc
    FILES ${JSON_HEADERS}
)

if( FELTOR_FILE_WITH_NETCDF)
    add_library(dg_file INTERFACE)
    add_library(dg_file_netcdf INTERFACE)
    set_property(TARGET dg_file PROPERTY EXPORT_NAME file)
    set_property(TARGET dg_file_netcdf PROPERTY EXPORT_NAME netcdf)
    add_library(feltor::dg::file ALIAS dg_file)
    add_library(feltor::dg::file::netcdf ALIAS dg_file_netcdf)
    target_sources( dg_file INTERFACE
        FILE_SET dg_file_headers TYPE HEADERS
        BASE_DIRS ${PROJECT_SOURCE_DIR}/inc
        FILES ${FILE_HEADERS} ${NETCDF_HEADERS} ${JSON_HEADERS}
    )
    target_sources( dg_file_netcdf INTERFACE
        FILE_SET dg_file_netcdf_headers TYPE HEADERS
        BASE_DIRS ${PROJECT_SOURCE_DIR}/inc
        FILES ${NETCDF_HEADERS}
    )

    target_link_libraries(dg_file_netcdf INTERFACE dg_dg)
    target_link_libraries(dg_file INTERFACE dg_file_netcdf dg_file_json)
endif()

# ============================================
# Add dependencies

if(FELTOR_FILE_WITH_JSONCPP)
  CPMAddPackage(
    NAME jsoncpp
    GITHUB_REPOSITORY open-source-parsers/jsoncpp
    VERSION 1.9.5
    OPTIONS "JSONCPP_WITH_TESTS OFF"
    SYSTEM ON
    EXCLUDE_FROM_ALL ON
  )
  target_link_libraries( dg_file_json INTERFACE jsoncpp_lib)
else()
  CPMAddPackage(
    NAME nlohmann_json # Use name of the package that appears in "find_package"
    GITHUB_REPOSITORY "nlohmann/json"
    VERSION 3.11.0 # Update in sync with local package manager
    SYSTEM ON
    EXCLUDE_FROM_ALL ON
  )
  target_compile_definitions(dg_file_json INTERFACE DG_USE_JSONHPP)
  target_link_libraries( dg_file_json INTERFACE nlohmann_json::nlohmann_json)
endif()

if( FELTOR_FILE_WITH_NETCDF)
    # Must install separately, e.g. `sudo apt install libnetcdf-dev`
    find_package(netCDF REQUIRED)
    target_link_libraries(dg_file_netcdf INTERFACE netCDF::netcdf)


    # ============================================
    # Add compile definitions

    execute_process(
        COMMAND git rev-parse HEAD
        OUTPUT_VARIABLE GIT_HASH
        OUTPUT_STRIP_TRAILING_WHITESPACE
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    )
    message(STATUS "Git hash: ${GIT_HASH}")

    execute_process(
        COMMAND git branch --show-current
        OUTPUT_VARIABLE GIT_BRANCH
        OUTPUT_STRIP_TRAILING_WHITESPACE
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    )
    message(STATUS "Git branch: ${GIT_BRANCH}")

    string(TIMESTAMP COMPILE_TIME "%Y-%m-%dT%H:%M:%SZ" UTC)
    message(STATUS "Compile time: ${COMPILE_TIME}")

    target_compile_definitions(dg_file INTERFACE
        GIT_HASH="${GIT_HASH}"
        GIT_BRANCH="${GIT_BRANCH}"
        COMPILE_TIME="${COMPILE_TIME}"
    )
endif()

# ============================================
# Add file tests (there are no file benchmarks)

if(FELTOR_BUILD_TESTS)
    include(FeltorTests)
    if( FELTOR_WITH_MPI)
        foreach(test_path IN LISTS JSON_MPI_TESTS)
            add_dg_test("${test_path}" "file" ON target_name)
            target_link_libraries( ${target_name} PRIVATE dg_file_json)
        endforeach()
    else()
        foreach(test_path IN LISTS JSON_TESTS)
            add_dg_test("${test_path}" "file" OFF target_name)
            target_link_libraries( ${target_name} PRIVATE dg_file_json)
        endforeach()
    endif()
    if(FELTOR_FILE_WITH_NETCDF)
        if( FELTOR_WITH_MPI)
            foreach(test_path IN LISTS FILE_MPI_TESTS)
                add_dg_test("${test_path}" "file" ON target_name)
                target_link_libraries( ${target_name} PRIVATE dg_file)
            endforeach()
        else()
            foreach(test_path IN LISTS FILE_TESTS)
                add_dg_test("${test_path}" "file" OFF target_name)
                target_link_libraries( ${target_name} PRIVATE dg_file)
            endforeach()
        endif()
    endif()
endif()

# ============================================
# Add Install rules

if(FELTOR_INSTALL)
    install(
        TARGETS dg_file_json COMPONENT feltor_dg
        EXPORT feltor_dg
        FILE_SET dg_file_json_headers DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    )
    if( FELTOR_FILE_WITH_NETCDF)
        install(
            TARGETS dg_file COMPONENT feltor_dg
            EXPORT feltor_dg
            FILE_SET dg_file_headers DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        )
        install(
            TARGETS dg_file_netcdf COMPONENT feltor_dg
            EXPORT feltor_dg
            FILE_SET dg_file_netcdf_headers DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        )
    endif()
endif()
