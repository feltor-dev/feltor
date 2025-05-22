# inc/dg/file/CMakeLists.txt

# Files depending on json but not netcdf:
set( JSON_HEADERS
  json_utilities.h
  ../../json_wrapper.h
  ../../json_probes.h
  ../../probes_params.h
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
set(FILE_TESTS
  ../../file/easy_atts_t.cpp
  ../../file/easy_dims_t.cpp
  ../../file/easy_output_t.cpp
  ../../file/json_wrapper_t.cpp
  ../../file/nc_file_t.cpp
  ../../file/nc_utilities_t.cpp
  ../../file/probes_parser_t.cpp
  ../../file/probes_t.cpp
)

# ============================================
# Create DG file library target

add_library(file INTERFACE ${FILE_HEADERS} ${NETCDF_HEADERS} ${JSON_HEADERS})
add_library(file::json INTERFACE ${JSON_HEADERS})
add_library(file::netcdf INTERFACE ${NETCDF_HEADERS})
add_library(dg::file ALIAS file)
add_library(dg::file::json ALIAS file::json)
add_library(dg::file::netcdf ALIAS file::netcdf)

target_link_libraries(file::netcdf INTERFACE dg::dg)
target_link_libraries(file INTERFACE file::netcdf file::json)

# ============================================
# Add dependencies

if(FELTOR_USE_JSONCPP)
  # Must install separately, e.g. `sudo apt install libjsoncpp-dev`
  CPMAddPackage( NAME jsoncpp
    GITHUB_REPOSITORY open-source-parsers/jsoncpp
    GIT_TAG master
  )
  get_target_property(JSON_INC_PATH jsoncpp_lib INTERFACE_INCLUDE_DIRECTORIES)
  target_include_directories(file::json INTERFACE ${JSON_INC_PATH})
  target_link_libraries( file::json INTERFACE jsoncpp_lib)
else()
  CPMAddPackage( "gh:nlohmann/json@3.12.0")
  target_compile_definitions(file::json INTERFACE DG_USE_JSONHPP)
  target_link_libraries( file::json INTERFACE nlohmann_json::nlohmann_json)
endif()

# Must install separately, e.g. `sudo apt install libnetcdf-dev`
find_package(netCDF)
if(TARGET netCDF::netcdf)
  target_link_libraries(file::netcdf INTERFACE netCDF::netcdf)
else()
  message(WARNING "Failed to find netCDF, the target dg::file::netcdf is incomplete")
endif()

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

string(TIMESTAMP COMPILE_TIME "%Y-%m-%d %H:%M:%S %Z")
message(STATUS "Compile time: ${COMPILE_TIME}")

target_compile_definitions(file INTERFACE
    GIT_HASH="${GIT_HASH}"
    GIT_BRANCH="${GIT_BRANCH}"
    COMPILE_TIME="${COMPILE_TIME}"
)

# ============================================
# Add file tests

if(FELTOR_BUILD_TESTS)
  include(FeltorTests)
  foreach(test_path IN LISTS FILE_TESTS)
    add_file_test("${test_path}")
  endforeach()
endif()
