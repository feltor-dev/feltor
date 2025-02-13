# FindCUSP
#
# Finds cusplibrary
# See https://github.com/cusplibrary/cusplibrary
# 
# Sets the variable CUSP_INCLUDE_DIR

find_path(CUSP_INCLUDE_DIR
  HINTS ./  
        /usr/include/
        /usr/local/include
        $ENV{HOME}/include
        $ENV{HOME}/include/cusplibrary
  NAMES cusp/version.h
  DOC "CUSP headers"
  )

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(CUSP REQUIRED_VARS CUSP_INCLUDE_DIR)
mark_as_advanced(CUSP_INCLUDE_DIR)

# Create an interface library target
if(CUSP_INCLUDE_DIR)
  add_library(cusp INTERFACE)
  add_library(cusp::cusp ALIAS cusp)
  target_include_directories(cusp INTERFACE ${CUSP_INCLUDE_DIR})
endif()