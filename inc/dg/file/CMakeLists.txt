# inc/dg/file/CMakeLists.txt

set(FILE_HEADERS
  file.h
  json_utilities.h
  nc_utilities.h
  ../../file/easy_atts.h
  ../../file/easy_dims.h
  ../../file/easy_input.h
  ../../file/easy_output.h
  ../../file/file.h
  ../../file/json_probes.h
  ../../file/json_utilities.h
  ../../file/json_wrapper.h
  ../../file/nc_error.h 
  ../../file/nc_file.h 
  ../../file/nc_hyperslab.h 
  ../../file/nc_mpi_file.h 
  ../../file/nc_utilities.h 
  ../../file/probes.h 
  ../../file/probes_params.h
  ../../file/records.h
)

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

add_library(file INTERFACE ${FILE_HEADERS})
add_library(dg::file ALIAS file)
target_link_libraries(file INTERFACE dg::dg)

# ============================================
# Add dependencies

include(FeltorFetchDeps)

if(FELTOR_USE_JSONCPP)
  # Must install separately, e.g. `sudo apt install libjsoncpp-dev`
  find_package(jsoncpp)
  if(jsoncpp_FOUND)
    target_link_libraries(file INTERFACE jsoncpp_lib)
  else()
    message(WARNING "Failed to find jsoncpp, the target dg::file is incomplete")
  endif()
else()
  if(FELTOR_FETCH_JSON)
    fetch_json()
  else()
    find_package(nlohmann_json REQUIRED)
  endif()
  if(TARGET nlohmann_json::nlohmann_json)
    target_link_libraries(file INTERFACE nlohmann_json::nlohmann_json)
  else()
    message(WARNING "Failed to find nlohmann_json, the target dg::file is incomplete")
  endif()
  target_compile_definitions(file INTERFACE DG_USE_JSONHPP)
endif()

# Must install separately, e.g. `sudo apt install libnetcdf-dev`
find_package(netCDF)
if(TARGET netCDF::netcdf)
  target_link_libraries(file INTERFACE netCDF::netcdf)
else()
  message(WARNING "Failed to find netCDF, the target dg::file is incomplete")
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

# FIXME this is not portable to Windows
execute_process(
    COMMAND "date" "+'%F %T %Z'"
    OUTPUT_VARIABLE COMPILE_TIME
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
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
