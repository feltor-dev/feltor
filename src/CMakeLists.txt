# ===========================================
# Set up navier-stokes library

add_library(navier_stokes INTERFACE "navier_stokes/navier_stokes.h")
add_library(feltor::navier_stokes ALIAS navier_stokes)
target_link_libraries(navier_stokes INTERFACE dg::dg)

# ============================================
# Set up FELTOR
#
# TODO: Other executables

set(FELTOR_HEADERS
    "feltor/feltor.h"
    "feltor/init_from_file.h"
    "feltor/init.h"
    "feltor/parameters.h"
)

add_executable(feltor "feltor/feltor.cpp" ${FELTOR_HEADERS})
add_executable(feltor::feltor ALIAS feltor)

target_include_directories(feltor PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# TODO: Replace following commands, bad for portability
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

execute_process(
    COMMAND "date" "+'%F %T %Z'"
    OUTPUT_VARIABLE COMPILE_TIME
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
message(STATUS "Compile time: ${COMPILE_TIME}")

target_compile_definitions(feltor PRIVATE 
    GIT_HASH="${GIT_HASH}"
    GIT_BRANCH="${GIT_BRANCH}"
    COMPILE_TIME="${COMPILE_TIME}"
)

# ============================================
# Add dependencies

target_link_libraries(feltor PRIVATE feltor::dg)

if(FELTOR_WITH_NAVIER_STOKES)
    target_link_libraries(feltor PRIVATE feltor::navier_stokes)
    target_compile_definitions(feltor PRIVATE WITH_NAVIER_STOKES)
endif()

if(FELTOR_WITH_GLFW)
  if(FELTOR_USE_MPI)
    message(WARNING "GLFW is not compatible with MPI when building FELTOR")
    target_compile_definitions(feltor PRIVATE WITHOUT_GLFW)
  else()
    target_link_libraries(feltor PRIVATE draw::draw)
  endif()   
else()
    target_compile_definitions(feltor PRIVATE WITHOUT_GLFW)
endif()
