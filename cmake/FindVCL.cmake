# FindVCL
#
# Finds C++ vector class library, version 1
# https://github.com/vectorclass/vesion1
# 
# Sets the variable VCL_INCLUDE_DIR and creates an interface library target
# vcl::vcl.

find_path(VCL_INCLUDE_DIR
  HINTS ./  
        /usr/include/
        /usr/local/include
        $ENV{HOME}/include
  NAMES vcl/vectorclass.h
  DOC "Vector class headers"
  )

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(VCL REQUIRED_VARS VCL_INCLUDE_DIR)
mark_as_advanced(VCL_INCLUDE_DIR)

# Create an interface library target
if(VCL_INCLUDE_DIR)
  add_library(vcl INTERFACE)
  add_library(vcl::vcl ALIAS vcl)
  target_include_directories(vcl INTERFACE ${VCL_INCLUDE_DIR})
endif()
